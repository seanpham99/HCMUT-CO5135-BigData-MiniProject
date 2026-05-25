"""
Extractor factory — runs all registered extractors concurrently.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dags.etl_modules.extractors.bizhub import BizhubExtractor
from dags.etl_modules.extractors.theinvestor import TheInvestorExtractor
from dags.etl_modules.fetcher import get_active_vn_stock_tickers

logger = logging.getLogger(__name__)

EXTRACTOR_CLASSES = [BizhubExtractor, TheInvestorExtractor]
EXTRACTOR_TIMEOUT_SECS = 120


def run_all_extractors() -> list[dict]:
    """
    Fetch news from all registered sources in parallel.

    - One extractor failing does not block the others.
    - Raises RuntimeError only when every extractor fails or returns empty.
    """
    tickers = get_active_vn_stock_tickers(raise_on_fallback=True)
    logger.info("Extractor factory: %s active tickers loaded", len(tickers))

    results: list[dict] = []

    def safe_extract(cls):
        try:
            return cls(tickers=tickers).extract()
        except Exception as exc:
            logger.warning("%s failed: %s", cls.__name__, exc)
            return []

    with ThreadPoolExecutor(max_workers=len(EXTRACTOR_CLASSES)) as pool:
        futures = {pool.submit(safe_extract, cls): cls for cls in EXTRACTOR_CLASSES}
        for future, cls in futures.items():
            try:
                data = future.result(timeout=EXTRACTOR_TIMEOUT_SECS)
                results.extend(data)
                logger.info("%s: %s records returned", cls.__name__, len(data))
            except FuturesTimeoutError:
                logger.warning(
                    "%s timed out after %ss — skipping",
                    cls.__name__,
                    EXTRACTOR_TIMEOUT_SECS,
                )

    if not results:
        raise RuntimeError(
            "All news extractors failed or returned empty. "
            "Check extractor logs for details."
        )

    return results
