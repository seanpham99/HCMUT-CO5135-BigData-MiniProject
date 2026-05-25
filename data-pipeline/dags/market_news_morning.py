import json
import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras
from airflow import DAG
from airflow.sdk import task
from pendulum import timezone

from dags.etl_modules.extractors import run_all_extractors
from dags.etl_modules.notifications import (
    send_failure_notification,
    send_success_notification,
)

logger = logging.getLogger(__name__)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

default_args = {
    "owner": "data_engineer",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

local_tz = timezone("Asia/Bangkok")

with DAG(
    dag_id="market_news_morning",
    default_args=default_args,
    schedule="0 7 * * 1-5",  # 7 AM Vietnam Time Mon-Fri
    start_date=datetime(2024, 1, 1, tzinfo=local_tz),
    catchup=False,
    tags=["news", "supabase", "morning-brief"],
    on_success_callback=send_success_notification,
    on_failure_callback=send_failure_notification,
) as dag:

    @task
    def extract_news() -> list[dict]:
        data = run_all_extractors()
        for row in data:
            if row.get("publish_date") and not isinstance(row["publish_date"], str):
                row["publish_date"] = str(row["publish_date"])
        logger.info("extract_news: %s records from all sources", len(data))
        return data

    @task
    def score_news(data: list[dict]) -> list[dict]:
        """Add sentiment score to each news record using Gemini with batching."""
        if not data:
            return []

        from google import genai
        from google.genai import errors, types

        task_logger = logging.getLogger("airflow.task")

        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT public.fetch_secret_by_name('GEMINI_API_KEY')")
                row = cur.fetchone()
                google_api_key = row[0] if row else None
        finally:
            conn.close()

        if not google_api_key:
            task_logger.warning(
                "GEMINI_API_KEY not found in Vault. Skipping sentiment scoring."
            )
            for row in data:
                row["sentiment_score"] = 0.0
            return data

        try:
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            task_logger.error(f"Failed to initialize Gemini client: {e}")
            raise

        batch_size = 15
        total_items = len(data)
        task_logger.info(
            "Scoring sentiment for %s news items in batches of %s...",
            total_items,
            batch_size,
        )

        for i in range(0, total_items, batch_size):
            batch = data[i : i + batch_size]

            items_text = ""
            for idx, item in enumerate(batch):
                title = item.get("title", "No Title")
                desc = item.get("description", "")
                items_text += f"ID: {idx}\nTitle: {title}\nContent: {desc}\n---\n"

            prompt = f"""
            Analyze sentiment for the following {len(batch)} financial news items
            from the Vietnamese stock market.
            For each item, provide a sentiment score between -1.0
            (extremely negative) and 1.0 (extremely positive).
            0.0 is neutral.

            Return results as JSON with key "scores" containing an array.
            Each item must include "id" (the ID in this prompt)
            and "score" (numeric sentiment score).

            Items:
            {items_text}
            """

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                try:
                    raw_text = (response.text or "").strip()
                    if raw_text.startswith("```"):
                        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                        if match:
                            raw_text = match.group(0)

                    result = json.loads(raw_text)
                    scores_list = result.get("scores", [])
                    scores_map = {
                        item["id"]: item["score"]
                        for item in scores_list
                        if "id" in item and "score" in item
                    }

                    for idx, item in enumerate(batch):
                        score = scores_map.get(idx, scores_map.get(str(idx), 0.0))
                        item["sentiment_score"] = max(-1.0, min(1.0, float(score)))
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    task_logger.error(
                        "Failed to parse Gemini response for batch starting at %s: %s",
                        i,
                        e,
                    )
                    task_logger.debug("Raw response: %s", response.text)
                    for item in batch:
                        item["sentiment_score"] = 0.0

            except errors.APIError as e:
                task_logger.error("Gemini API Error (batch starting at %s): %s", i, e)
                raise
            except Exception as e:
                task_logger.error(
                    "Unexpected error scoring batch starting at %s: %s", i, e
                )
                raise

        return data

    @task
    def load_news(data: list[dict]) -> None:
        if not data:
            logger.info("No news to load.")
            return

        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL environment variable is not set")

        cols = [
            "asset_id",
            "news_id",
            "publish_date",
            "title",
            "news_content",
            "source",
            "source_url",
            "sentiment_score",
        ]
        tuples = []
        for row in data:
            if row.get("publish_date"):
                row["publish_date"] = pd.to_datetime(row["publish_date"])
            tuples.append([row.get(c) for c in cols])

        logger.info("load_news: inserting %s rows into market_data.news", len(tuples))
        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO market_data.news
                            (asset_id, news_id, publish_date, title,
                             news_content, source, source_url, sentiment_score)
                        VALUES %s
                        ON CONFLICT (asset_id, news_id) DO UPDATE SET
                            sentiment_score = EXCLUDED.sentiment_score,
                            ingested_at     = NOW()
                        """,
                        tuples,
                    )
        finally:
            conn.close()

        logger.info("load_news: complete")

    @task
    def embed_news() -> None:
        """
        Embed today's new articles with gemini-embedding-001 @ 768d.
        Reads from the DB directly — no XCom dependency.
        API key is retrieved from Supabase Vault via fetch_secret_by_name().
        """
        import time
        from datetime import date

        import numpy as np
        from google import genai
        from google.genai import errors, types

        task_logger = logging.getLogger("airflow.task")

        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL environment variable is not set")

        # ── Retrieve GEMINI_API_KEY from Supabase Vault ───────────────────────
        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT public.fetch_secret_by_name('GEMINI_API_KEY')")
                row = cur.fetchone()
                gemini_api_key = row[0] if row else None
        finally:
            conn.close()

        if not gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in Supabase Vault — "
                "add the secret via the Vault UI before running this task."
            )

        client = genai.Client(api_key=gemini_api_key)
        MODEL = "gemini-embedding-001"
        DIM = 768
        BATCH = 100

        # ── Fetch today's unembedded articles ─────────────────────────────────
        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT n.asset_id, n.news_id, n.title, n.news_content
                    FROM market_data.news n
                    LEFT JOIN market_data.news_embeddings e
                        ON n.asset_id = e.asset_id
                        AND n.news_id = e.news_id
                    WHERE n.publish_date::date = %s
                      AND e.news_id IS NULL
                    """,
                    (date.today(),),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            task_logger.info("embed_news: no new articles to embed today")
            return

        texts = [
            f"{r['title']}. {r['news_content'] or ''}".strip()
            for r in rows
        ]
        task_logger.info(
            "embed_news: embedding %s articles with %s @ %dd", len(texts), MODEL, DIM
        )

        # ── Batch embedding with rate-limit retry ─────────────────────────────
        def _embed_batch(batch: list[str]) -> list:
            try:
                result = client.models.embed_content(
                    model=MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=DIM,
                    ),
                )
                return result.embeddings
            except errors.APIError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    task_logger.warning("Rate limit hit, sleeping 60s then retrying")
                    time.sleep(60)
                    result = client.models.embed_content(
                        model=MODEL,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=DIM,
                        ),
                    )
                    return result.embeddings
                raise

        all_embeddings = []
        for i in range(0, len(texts), BATCH):
            all_embeddings.extend(_embed_batch(texts[i : i + BATCH]))

        # ── Normalize (required for MRL-truncated dimensions < 3072) ──────────
        def _normalize(vec: list[float]) -> list[float]:
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return (arr / norm if norm > 0 else arr).tolist()

        tuples = [
            (
                str(rows[i]["asset_id"]),
                int(rows[i]["news_id"]),
                _normalize(emb.values),
                "gemini-embedding-001-768d",
            )
            for i, emb in enumerate(all_embeddings)
        ]

        # ── Upsert into news_embeddings ───────────────────────────────────────
        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO market_data.news_embeddings
                            (asset_id, news_id, embedding, model_ver)
                        VALUES %s
                        ON CONFLICT (asset_id, news_id) DO UPDATE SET
                            embedding   = EXCLUDED.embedding,
                            model_ver   = EXCLUDED.model_ver,
                            embedded_at = NOW()
                        """,
                        tuples,
                    )
        finally:
            conn.close()

        task_logger.info(
            "embed_news: inserted %s embeddings with model %s", len(tuples), MODEL
        )

    # ── DAG orchestration ─────────────────────────────────────────────────────
    # embed_news reads from the DB directly, but news rows must exist first
    # so we declare an explicit dependency on the load_news task instance.
    raw_news = extract_news()
    scored_news = score_news(raw_news)
    loaded = load_news(scored_news)
    embed = embed_news()
    loaded >> embed
