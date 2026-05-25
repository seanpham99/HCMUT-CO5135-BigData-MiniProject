"""
Bizhub extractor
────────────────
Source: https://bizhub.vietnamnews.vn/news

Depth 1 — article listing:
  Page 1  : webclaw CLI (SSR HTML)
  Page 2+ : ASP.NET ASMX  LoadMoreArticle endpoint

Depth 2 — article body:
  webclaw CLI → clean plain-text body + structured-data metadata

news_id is extracted from the article URL with the pattern ``-post<id>.html``.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import subprocess
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from dags.etl_modules.extractors.base import NewsExtractor

logger = logging.getLogger(__name__)

NEWS_INDEX = "https://bizhub.vietnamnews.vn/news"
ASMX_URL = "https://bizhub.vietnamnews.vn/ajaxloads/servicedata.asmx/LoadMoreArticle"
ASMX_SITEID = 1
ASMX_CATECODE = "news"
ASMX_ISPARRENT = 0
ASMX_PAGESIZE = 20
REQUEST_DELAY = 2

_NEWS_ID_RE = re.compile(r"-post(\d+)", re.IGNORECASE)
_NOISE_LINE_RE = re.compile(
    r"""
    ^\s*-\s*$
    | ^\s*-\s*A[+\-]?\s*$
    | ^[-\s]*$
    | Photo\s+courtesy
    | —\s*Photo
    | —\s*BIZHUB\s*$
    | ^\+\s*Load\s+more
    """,
    re.VERBOSE | re.IGNORECASE,
)
_STOP_MARKERS = [
    "- share:",
    "- tags",
    "## comments",
    "## see also",
    "see also",
    "abc123@",
]
_ASMX_HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": NEWS_INDEX,
}


class _SSLContextAdapter(HTTPAdapter):
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
            **pool_kwargs,
        )


def _make_legacy_session() -> requests.Session:
    sess = requests.Session()
    try:
        ctx = ssl.create_default_context()
        if hasattr(ssl, "OP_LEGACY_SERVER_CONNECT"):
            ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        elif hasattr(ssl, "OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION"):
            ctx.options |= ssl.OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION
        adapter = _SSLContextAdapter(ssl_context=ctx)
        sess.mount("https://", adapter)
    except Exception:
        sess.verify = False
    return sess


def _webclaw_scrape(url: str) -> dict | None:
    try:
        proc = subprocess.run(
            ["webclaw", url, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.warning(
                "webclaw error [%s]: %s", url[:80], proc.stderr.strip()[:200]
            )
            return None
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        logger.warning("webclaw timeout: %s", url[:80])
        return None
    except json.JSONDecodeError as exc:
        logger.warning("webclaw JSON parse error [%s]: %s", url[:80], exc)
        return None
    except PermissionError:
        raise RuntimeError(
            "webclaw is not executable — ensure it has execute permissions in the Docker image"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "webclaw not found in PATH — ensure it is installed in the Docker image"
        )


def _extract_news_id(url: str) -> int | None:
    match = _NEWS_ID_RE.search(url)
    if match:
        return int(match.group(1))
    return None


def _clean_body(plain_text: str) -> str:
    lines = plain_text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*-\s*A\+\s*$", line):
            start = i + 1
            break

    body_lines: list[str] = []
    for line in lines[start:]:
        lower = line.strip().lower()
        if any(lower.startswith(m) or m in lower for m in _STOP_MARKERS):
            break
        body_lines.append(line.strip())

    body_lines = [l for l in body_lines if not _NOISE_LINE_RE.match(l)]
    deduped: list[str] = []
    prev = None
    for line in body_lines:
        if line == prev and line != "":
            continue
        deduped.append(line)
        prev = line

    text = "\n".join(deduped)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_metadata(raw: dict) -> dict:
    meta = raw.get("metadata", {})
    sd = raw.get("structured_data", [])
    news_sd = next(
        (s for s in sd if isinstance(s, dict) and s.get("@type") == "NewsArticle"), {}
    )
    links = raw.get("content", {}).get("links", [])
    tags = [l["text"] for l in links if "/tags/" in l.get("href", "") and l.get("text")]
    return {
        "title": news_sd.get("headline") or meta.get("title", ""),
        "description": news_sd.get("description") or meta.get("description", ""),
        "published_date": news_sd.get("datePublished")
        or meta.get("published_date", ""),
        "tags": tags,
    }


def _scrape_page1() -> list[dict]:
    raw = _webclaw_scrape(NEWS_INDEX)
    if not raw:
        return []

    markdown = raw.get("content", {}).get("markdown", "")
    pattern = re.compile(
        r"###\s+\[(?P<title>[^\]]+)\]\((?P<url>https://bizhub\.vietnamnews\.vn/[^\)]+)\)"
        r"(?:.*?\*\*(?P<date>[A-Za-z]+,\s+[A-Za-z]+\s+\d+,\s+\d+))?"
        r"(?:\n+(?P<teaser>[^\n#\-\*\[]{20,}))?",
        re.DOTALL,
    )

    stubs: list[dict] = []
    seen: set[str] = set()
    for m in pattern.finditer(markdown):
        url = m.group("url").strip()
        if url in seen:
            continue
        seen.add(url)
        stubs.append({"title": m.group("title").strip(), "url": url})

    return stubs


def _parse_stubs_from_html(html_fragment: str) -> list[dict]:
    soup = BeautifulSoup(html_fragment, "html.parser")
    stubs: list[dict] = []
    for h3 in soup.find_all("h3", class_="meta-data-tit"):
        a = h3.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if href.startswith("/"):
            href = "https://bizhub.vietnamnews.vn" + href
        stubs.append({"title": a.get_text(strip=True), "url": href})

    if not stubs:
        for h3 in soup.find_all("h3"):
            a = h3.find("a", href=re.compile(r"bizhub\.vietnamnews\.vn/.+-post\d+"))
            if a:
                stubs.append(
                    {"title": a.get_text(strip=True), "url": a["href"].strip()}
                )

    return stubs


def _fetch_asmx_page(
    session: requests.Session, page_number: int
) -> tuple[list[dict], int]:
    payload = {
        "siteid": ASMX_SITEID,
        "catecode": ASMX_CATECODE,
        "pagenumber": page_number,
        "pagesize": ASMX_PAGESIZE,
        "isparrent": ASMX_ISPARRENT,
    }

    def _parse(text: str) -> tuple[list[dict], int]:
        outer = json.loads(text)
        inner = json.loads(outer["d"])
        html_fragment = inner.get("listArticle", "")
        total_pages = int(inner.get("TotalPages", 1))
        return _parse_stubs_from_html(html_fragment), total_pages

    try:
        resp = session.post(ASMX_URL, headers=_ASMX_HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        stubs, total_pages = _parse(resp.text)
        return stubs, total_pages
    except Exception as exc:
        logger.warning("ASMX page %s failed: %s", page_number, exc)
        return [], 0


def _collect_stubs(max_articles: int = 200) -> list[dict]:
    all_stubs = _scrape_page1()
    seen = {s["url"] for s in all_stubs}

    session = _make_legacy_session()
    stubs_p2, total_pages = _fetch_asmx_page(session, 2)
    for s in stubs_p2:
        if s["url"] not in seen:
            seen.add(s["url"])
            all_stubs.append(s)

    for page in range(3, total_pages + 1):
        if len(all_stubs) >= max_articles:
            break
        time.sleep(REQUEST_DELAY)
        stubs, _ = _fetch_asmx_page(session, page)
        for s in stubs:
            if s["url"] not in seen:
                seen.add(s["url"])
                all_stubs.append(s)

    return all_stubs[:max_articles]


def _fetch_article(stub: dict) -> dict | None:
    raw = _webclaw_scrape(stub["url"])
    if not raw:
        return None

    plain = raw.get("content", {}).get("plain_text", "")
    meta = _extract_metadata(raw)
    title = meta["title"] or stub["title"]
    body = _clean_body(plain)

    news_id = _extract_news_id(stub["url"])
    if news_id is None:
        return None

    published_date = meta["published_date"]
    if not published_date:
        published_date = datetime.now(timezone.utc).isoformat()

    return {
        "news_id": news_id,
        "title": title,
        "news_content": body,
        "publish_date": published_date,
        "source_url": stub["url"],
    }


class BizhubExtractor(NewsExtractor):
    def extract(self) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stubs = _collect_stubs(max_articles=200)
        logger.info("BizhubExtractor: collected %s article stubs", len(stubs))

        articles: list[dict] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_fetch_article, stub): stub for stub in stubs}
            for future in as_completed(futures):
                try:
                    article = future.result()
                except Exception as exc:
                    logger.warning("BizhubExtractor article fetch error: %s", exc)
                    continue
                if article is None:
                    continue

                matched = self._match_tickers(
                    article["title"], article["news_content"] or ""
                )
                for ticker in matched:
                    articles.append(
                        {
                            "symbol": ticker["symbol"],
                            "asset_id": ticker["asset_id"],
                            "news_id": article["news_id"],
                            "title": article["title"],
                            "news_content": article["news_content"],
                            "publish_date": article["publish_date"],
                            "source": "bizhub",
                            "source_url": article["source_url"],
                        }
                    )

        logger.info(
            "BizhubExtractor: produced %s ticker-matched records", len(articles)
        )
        return articles
