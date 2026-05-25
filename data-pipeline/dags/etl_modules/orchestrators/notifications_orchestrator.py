from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TypeAlias

from dags.etl_modules.contracts.providers import NewsSummaryProvider, NotificationSender

NewsRecord: TypeAlias = dict[str, Any]


def _format_dag_status_message(context: dict[str, Any], status: str) -> str:
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", "unknown_dag")
    run_id = context.get("run_id")
    execution_date = context.get("logical_date") or context.get("execution_date")

    if status == "FAILED":
        task_instance = context.get("task_instance")
        task_id = getattr(task_instance, "task_id", "Unknown")
        exception = context.get("exception")
        return (
            "🔴 *DAG Failed*\n"
            f"DAG: `{dag_id}`\n"
            f"Task: `{task_id}`\n"
            f"Run ID: `{run_id}`\n"
            f"Time: `{execution_date}`\n"
            f"Error: `{exception}`"
        )

    return (
        "🟢 *DAG Success*\n"
        f"DAG: `{dag_id}`\n"
        f"Run ID: `{run_id}`\n"
        f"Time: `{execution_date}`"
    )


def _format_basic_news_summary(
    news_data: list[NewsRecord],
    *,
    as_of: datetime,
) -> str:
    header = f"📰 *Market News Summary - {as_of.strftime('%Y-%m-%d')}*\n\n"
    news_by_ticker: dict[str, list[NewsRecord]] = {}
    for item in news_data:
        ticker = str(item.get("ticker") or "UNKNOWN")
        news_by_ticker.setdefault(ticker, []).append(item)

    message_parts = [header]
    for ticker, articles in sorted(news_by_ticker.items()):
        message_parts.append(f"*{ticker}* ({len(articles)} articles)")
        for index, article in enumerate(articles[:3], 1):
            title = str(article.get("title") or "No title")[:100]
            price_change = float(article.get("price_change_ratio") or 0)
            price_emoji = (
                "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            )
            message_parts.append(
                f"{index}. {title}... {price_emoji} {price_change:.2f}%"
            )

        if len(articles) > 3:
            message_parts.append(f"   _...and {len(articles) - 3} more_")
        message_parts.append("")

    return "\n".join(message_parts)


def send_dag_status_notification(
    context: dict[str, Any],
    status: str,
    *,
    sender: NotificationSender,
) -> None:
    if not sender.is_configured():
        logging.warning("Telegram credentials not found. Skipping notification.")
        return

    text = _format_dag_status_message(context, status)
    sender.send_message(text=text, parse_mode="Markdown")


def send_success_notification(
    context: dict[str, Any],
    *,
    sender: NotificationSender,
) -> None:
    send_dag_status_notification(context, "SUCCESS", sender=sender)


def send_failure_notification(
    context: dict[str, Any],
    *,
    sender: NotificationSender,
) -> None:
    send_dag_status_notification(context, "FAILED", sender=sender)


def send_news_summary_notification(
    news_data: list[NewsRecord] | None,
    *,
    sender: NotificationSender,
    summarizer: NewsSummaryProvider,
    as_of: datetime | None = None,
) -> None:
    if not sender.is_configured():
        logging.warning("Telegram credentials not found. Skipping news notification.")
        return

    if not news_data:
        logging.info("No news data to send.")
        return

    as_of = as_of or datetime.now()
    generated_summary = summarizer.summarize(news_data)

    if generated_summary:
        header = (
            f"📰 *AI Market News Summary - {as_of.strftime('%Y-%m-%d')}*\n"
            "🤖 _Powered by Gemini AI_\n\n"
        )
        text = header + generated_summary
    else:
        text = _format_basic_news_summary(news_data, as_of=as_of)

    if len(text) > 4000:
        text = text[:3900] + "\n\n_...message truncated_"

    sender.send_message(text=text, parse_mode="Markdown")
