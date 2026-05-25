from unittest.mock import patch

import pytest
import responses

from dags.etl_modules.adapters.notification_adapter import (
    CallableNewsSummaryProvider,
    TelegramNotificationAdapter,
)
from tests.mocks.api_responses import get_telegram_url, mock_telegram_success


@pytest.mark.unit
def test_callable_news_summary_provider_delegates_to_callable():
    provider = CallableNewsSummaryProvider(
        summarize_fn=lambda data: f"summary:{len(data)}"
    )

    assert provider.summarize([{"ticker": "HPG"}]) == "summary:1"


@pytest.mark.unit
def test_telegram_notification_adapter_skips_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    adapter = TelegramNotificationAdapter(token=None, chat_id=None)

    assert adapter.is_configured() is False
    sent = adapter.send_message(text="hello", parse_mode="Markdown")

    assert sent is False


@pytest.mark.unit
@responses.activate
def test_telegram_notification_adapter_posts_to_telegram_api():
    token = "telegram-token"
    chat_id = "123456"
    responses.add(
        responses.POST,
        get_telegram_url(token),
        json=mock_telegram_success(),
        status=200,
    )

    adapter = TelegramNotificationAdapter(token=token, chat_id=chat_id)
    sent = adapter.send_message(text="hello", parse_mode="Markdown")

    assert sent is True
    assert len(responses.calls) == 1
    assert b"hello" in responses.calls[0].request.body


@pytest.mark.unit
def test_telegram_notification_adapter_resolves_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-from-env")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-from-env")

    with patch(
        "dags.etl_modules.adapters.notification_adapter.requests.post"
    ) as mock_post:
        mock_response = mock_post.return_value
        mock_response.raise_for_status.return_value = None

        adapter = TelegramNotificationAdapter()
        sent = adapter.send_message(text="hello")

    assert sent is True
