"""
TheInvestor extractor
─────────────────────
Source: https://theinvestor.vn/companies/

Depth 1 — article listing:
  Page 1  : GET listing HTML, parse .row-three__left-item.py-20 cards
  Page 2+ : GET load-more endpoint with id_last_news cursor

Depth 2 — article body:
  webclaw CLI → clean plain-text body + structured-data metadata

news_id is taken from the ``id-news`` HTML attribute (preferred) or extracted
from the URL with the pattern ``-d<id>.html`` (fallback).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from dags.etl_modules.extractors.base import NewsExtractor

logger = logging.getLogger(__name__)

NEWS_INDEX = "https://theinvestor.vn/companies/"
LOAD_MORE_URL = "https://theinvestor.vn/"
CATE_ID = 3
REQUEST_DELAY = 1.2

_ARTICLE_URL_RE = re.compile(r"^https://theinvestor\.vn/.+-d(?P<article_id>\d+)\.html$")
_STOP_MARKER_RE = re.compile(
    r"^(Tags:?|Comments\s*\(|Latest\)\s*\|\s*Most liked\)|Share:?|Related news:?|See also:?|More in)",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(
    r"^(?:"
    r"-\s*Companies"
    r"|-\s*Executive Talk"
    r"|-\s*Bamboo Capital"
    r"|-\s*Consulting"
    r"|By"
    r"|[A-Z][a-z]{2},\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*\|"
    r"|Photo courtesy"
    r"|\*\*"
    r")$",
    re.IGNORECASE,
)
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": NEWS_INDEX,
}
_LOAD_MORE_HEADERS = {
    "User-Agent": _REQUEST_HEADERS["User-Agent"],
    "Accept": "*/*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": NEWS_INDEX,
}


def _webclaw_scrape(url: str) -> dict | None:
    try:
        proc = subprocess.run(
            ["webclaw", url, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=40,
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


def _extract_news_id_from_url(url: str) -> int | None:
    match = _ARTICLE_URL_RE.match(url.strip())
    if match:
        return int(match.group("article_id"))
    return None


def _clean_body(plain_text: str, title: str) -> str:
    lines = [line.strip() for line in plain_text.splitlines()]
    body_lines: list[str] = []
    started = False
    for line in lines:
        if not line:
            if started:
                body_lines.append("")
            continue
        if line == title:
            continue
        if _STOP_MARKER_RE.search(line):
            break
        if _NOISE_LINE_RE.search(line):
            continue
        if not started:
            if len(line) < 20:
                continue
            started = True
        body_lines.append(line)

    cleaned: list[str] = []
    prev: str | None = None
    for line in body_lines:
        if line == prev and line != "":
            continue
        cleaned.append(line)
        prev = line

    text = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_metadata(raw: dict) -> dict:
    metadata = raw.get("metadata", {})
    structured_data = raw.get("structured_data", [])
    news_article = next(
        (
            item
            for item in structured_data
            if isinstance(item, dict) and item.get("@type") == "NewsArticle"
        ),
        {},
    )
    author = metadata.get("author", "")
    sd_author = news_article.get("author")
    if isinstance(sd_author, dict):
        author = sd_author.get("name") or author

    links = raw.get("content", {}).get("links", [])
    tags = [
        link.get("text", "").strip()
        for link in links
        if isinstance(link, dict)
        and re.search(r"-tag\d+/?$", (link.get("href") or ""))
        and link.get("text")
    ]
    return {
        "title": news_article.get("headline") or metadata.get("title", ""),
        "published_date": news_article.get("datePublished")
        or metadata.get("published_date", ""),
        "tags": sorted(set(tags)),
    }


def _parse_listing_items(html_fragment: str) -> list[dict]:
    soup = BeautifulSoup(html_fragment, "html.parser")
    stubs: list[dict] = []
    for item in soup.select(".row-three__left-item.py-20"):
        id_attr = item.get("id-news")
        try:
            news_id = int(id_attr) if id_attr else None
        except (ValueError, TypeError):
            news_id = None

        title_anchor = item.select_one("h3 a[href]") or item.select_one("a[href]")
        if not title_anchor:
            continue

        href = title_anchor.get("href", "").strip()
        if href.startswith("/"):
            href = "https://theinvestor.vn" + href

        if not _ARTICLE_URL_RE.match(href):
            continue

        if news_id is None:
            url_match = _ARTICLE_URL_RE.match(href)
            if url_match:
                news_id = int(url_match.group("article_id"))

        if news_id is None:
            continue

        stubs.append(
            {
                "title": title_anchor.get_text(" ", strip=True),
                "url": href,
                "news_id": news_id,
            }
        )
    return stubs


def _collect_stubs(max_articles: int = 200) -> list[dict]:
    session = requests.Session()
    try:
        resp = session.get(NEWS_INDEX, headers=_REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
        all_stubs = _parse_listing_items(resp.text)
    except Exception as exc:
        logger.warning("TheInvestor page 1 fetch failed: %s", exc)
        return []

    seen_urls = {s["url"] for s in all_stubs}
    if not all_stubs:
        return []

    last_news_id = all_stubs[-1].get("news_id")
    if not isinstance(last_news_id, int):
        return all_stubs[:max_articles]

    while len(all_stubs) < max_articles:
        time.sleep(REQUEST_DELAY)
        params = {
            "mod": "iframe",
            "act": "load_more_home",
            "id_last_news": last_news_id,
            "cate_id": CATE_ID,
        }
        try:
            resp = session.get(
                LOAD_MORE_URL, headers=_LOAD_MORE_HEADERS, params=params, timeout=20
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(
                "TheInvestor load-more failed at cursor %s: %s", last_news_id, exc
            )
            break

        if not resp.text.strip():
            break

        page_stubs = _parse_listing_items(resp.text)
        if not page_stubs:
            break

        new_count = 0
        for stub in page_stubs:
            if stub["url"] in seen_urls:
                continue
            seen_urls.add(stub["url"])
            all_stubs.append(stub)
            new_count += 1

        next_cursor = page_stubs[-1].get("news_id")
        if (
            not isinstance(next_cursor, int)
            or next_cursor >= last_news_id
            or new_count == 0
        ):
            break

        last_news_id = next_cursor

    return all_stubs[:max_articles]


def _fetch_article(stub: dict) -> dict | None:
    raw = _webclaw_scrape(stub["url"])
    if not raw:
        return None

    plain = raw.get("content", {}).get("plain_text", "")
    meta = _extract_metadata(raw)
    title = meta["title"] or stub["title"]
    body = _clean_body(plain, title=title)

    news_id = stub.get("news_id") or _extract_news_id_from_url(stub["url"])
    if news_id is None:
        return None

    published_date = meta["published_date"]
    if not published_date:
        published_date = datetime.now(timezone.utc).isoformat()

    return {
        "news_id": int(news_id),
        "title": title,
        "news_content": body,
        "publish_date": published_date,
        "source_url": stub["url"],
        "tags": meta.get("tags", []),
    }


class TheInvestorExtractor(NewsExtractor):
    def extract(self) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stubs = _collect_stubs(max_articles=200)
        logger.info("TheInvestorExtractor: collected %s article stubs", len(stubs))

        articles: list[dict] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_fetch_article, stub): stub for stub in stubs}
            for future in as_completed(futures):
                try:
                    article = future.result()
                except Exception as exc:
                    logger.warning("TheInvestorExtractor article fetch error: %s", exc)
                    continue
                if article is None:
                    continue

                haystack_extra = " ".join(article.get("tags", []))
                matched = self._match_tickers(
                    article["title"],
                    f"{article['news_content'] or ''} {haystack_extra}",
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
                            "source": "theinvestor",
                            "source_url": article["source_url"],
                        }
                    )

        logger.info(
            "TheInvestorExtractor: produced %s ticker-matched records", len(articles)
        )
        return articles
