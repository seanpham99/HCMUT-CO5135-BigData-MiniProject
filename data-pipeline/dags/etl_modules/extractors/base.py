from abc import ABC, abstractmethod


class NewsExtractor(ABC):
    """
    Abstract base class for all news source extractors.

    Each concrete extractor is responsible for fetching articles from one source
    and returning them in the canonical normalized record format.

    Ticker matching is done via simple substring search of the article title and
    body text.  An article that matches N tickers produces N records — one per
    matched ticker.  Articles that match no tickers are skipped.
    """

    def __init__(self, tickers: list[dict]) -> None:
        """
        Parameters
        ----------
        tickers:
            list[dict] with keys ``symbol`` (str) and ``asset_id`` (str).
            Obtained from ``get_active_vn_stock_tickers()``.
        """
        self.tickers = tickers

    @abstractmethod
    def extract(self) -> list[dict]:
        """
        Fetch and return normalized news records.

        Each dict **must** contain:
            symbol       : str
            asset_id     : str (UUID)
            news_id      : int   — bigint-castable; cast inside the extractor
            title        : str
            news_content : str
            publish_date : str   — ISO 8601
            source       : str   — "bizhub" | "theinvestor"
            source_url   : str
        """

    def _match_tickers(self, title: str, body: str) -> list[dict]:
        """
        Return all tickers whose symbol appears in ``title`` or ``body``.

        Matching is case-insensitive whole-word to avoid "FPT" matching inside
        "FPTS".  Returns an empty list when nothing matches.
        """
        import re

        haystack = f"{title} {body}".upper()
        matched = []
        seen: set[str] = set()
        for ticker in self.tickers:
            symbol = ticker["symbol"].upper()
            if symbol in seen:
                continue
            pattern = rf"\b{re.escape(symbol)}\b"
            if re.search(pattern, haystack):
                matched.append(ticker)
                seen.add(symbol)
        return matched
