from __future__ import annotations

import logging
from typing import Any, Callable

import requests

from dags.etl_modules.settings import get_env


class TelegramNotificationAdapter:
    def __init__(
        self,
        *,
        token: str | None = None,
        chat_id: str | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._timeout_seconds = timeout_seconds

    def _resolve_credentials(self) -> tuple[str | None, str | None]:
        token = self._token or get_env("TELEGRAM_BOT_TOKEN")
        chat_id = self._chat_id or get_env("TELEGRAM_CHAT_ID")
        return token, chat_id

    def is_configured(self) -> bool:
        token, chat_id = self._resolve_credentials()
        return bool(token and chat_id)

    def send_message(self, *, text: str, parse_mode: str = "Markdown") -> bool:
        token, chat_id = self._resolve_credentials()
        if not token or not chat_id:
            logging.warning("Telegram credentials not found. Skipping notification.")
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        try:
            response = requests.post(url, json=payload, timeout=self._timeout_seconds)
            response.raise_for_status()
            return True
        except Exception as exc:
            logging.error(f"Failed to send Telegram notification: {exc}")
            return False


class CallableNewsSummaryProvider:
    def __init__(
        self,
        *,
        summarize_fn: Callable[[list[dict[str, Any]]], str | None],
    ) -> None:
        self._summarize_fn = summarize_fn

    def summarize(self, news_data: list[dict[str, Any]]) -> str | None:
        return self._summarize_fn(news_data)
