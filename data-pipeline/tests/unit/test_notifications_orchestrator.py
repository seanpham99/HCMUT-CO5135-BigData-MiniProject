from datetime import datetime
from typing import get_type_hints

import pytest

from dags.etl_modules.orchestrators import notifications_orchestrator


class _SenderStub:
    def __init__(self, *, configured: bool = True):
        self.messages: list[tuple[str, str]] = []
        self.configured = configured

    def is_configured(self) -> bool:
        return self.configured

    def send_message(self, *, text: str, parse_mode: str = "Markdown") -> bool:
        self.messages.append((text, parse_mode))
        return True


class _SummaryStub:
    def __init__(self, summary: str | None):
        self.summary = summary

    def summarize(self, news_data):
        return self.summary


class _SummaryMustNotBeCalled:
    def summarize(self, news_data):
        raise AssertionError("summarize should not be called")


@pytest.mark.unit
def test_send_dag_status_notification_formats_failure_message():
    sender = _SenderStub()
    context = {
        "dag": type("Dag", (), {"dag_id": "market_data_prices_daily"})(),
        "task_instance": type("Ti", (), {"task_id": "process_price_chunk"})(),
        "run_id": "scheduled__2026-01-01",
        "logical_date": datetime(2026, 1, 1, 19, 0),
        "exception": RuntimeError("provider timeout"),
    }

    notifications_orchestrator.send_dag_status_notification(
        context,
        "FAILED",
        sender=sender,
    )

    assert len(sender.messages) == 1
    text, parse_mode = sender.messages[0]
    assert "DAG Failed" in text
    assert "process_price_chunk" in text
    assert "provider timeout" in text
    assert parse_mode == "Markdown"


@pytest.mark.unit
def test_send_news_summary_notification_prefers_ai_summary():
    sender = _SenderStub()
    summarizer = _SummaryStub("AI summary content")
    news_data = [{"ticker": "HPG", "title": "Article", "price_change_ratio": 1.5}]

    notifications_orchestrator.send_news_summary_notification(
        news_data,
        sender=sender,
        summarizer=summarizer,
        as_of=datetime(2026, 1, 1),
    )

    text, _ = sender.messages[0]
    assert "AI Market News Summary - 2026-01-01" in text
    assert "AI summary content" in text


@pytest.mark.unit
def test_send_news_summary_notification_falls_back_to_basic_summary():
    sender = _SenderStub()
    summarizer = _SummaryStub(None)
    news_data = [{"ticker": "HPG", "title": "Article", "price_change_ratio": -0.5}]

    notifications_orchestrator.send_news_summary_notification(
        news_data,
        sender=sender,
        summarizer=summarizer,
        as_of=datetime(2026, 1, 1),
    )

    text, _ = sender.messages[0]
    assert "Market News Summary - 2026-01-01" in text
    assert "HPG" in text


@pytest.mark.unit
def test_send_news_summary_notification_skips_when_sender_not_configured():
    sender = _SenderStub(configured=False)
    summarizer = _SummaryMustNotBeCalled()
    news_data = [{"ticker": "HPG", "title": "Article", "price_change_ratio": -0.5}]

    notifications_orchestrator.send_news_summary_notification(
        news_data,
        sender=sender,
        summarizer=summarizer,
        as_of=datetime(2026, 1, 1),
    )

    assert sender.messages == []


@pytest.mark.unit
def test_public_notifications_orchestrator_interfaces_are_typed():
    status_hints = get_type_hints(
        notifications_orchestrator.send_dag_status_notification
    )
    success_hints = get_type_hints(notifications_orchestrator.send_success_notification)
    failure_hints = get_type_hints(notifications_orchestrator.send_failure_notification)
    news_hints = get_type_hints(
        notifications_orchestrator.send_news_summary_notification
    )

    assert {"context", "status", "sender", "return"} <= set(status_hints.keys())
    assert {"context", "sender", "return"} <= set(success_hints.keys())
    assert {"context", "sender", "return"} <= set(failure_hints.keys())
    assert {"news_data", "sender", "summarizer", "return"} <= set(news_hints.keys())
