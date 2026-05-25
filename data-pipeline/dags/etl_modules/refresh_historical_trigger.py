from __future__ import annotations

from typing import Any


def ensure_corporate_events_table_exists(cur: Any) -> None:
    cur.execute(
        """
        SELECT
            to_regclass('market_data.corporate_events') IS NOT NULL AS has_events
        """
    )
    (has_events,) = cur.fetchone()

    if not has_events:
        raise RuntimeError("Required table market_data.corporate_events does not exist")


def fetch_tickers_for_refresh(
    cur: Any, events_lookback_days: int
) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT DISTINCT a.symbol, ce.asset_id
        FROM market_data.corporate_events ce
        JOIN market_data.assets a
            ON a.id = ce.asset_id
        LEFT JOIN (
            SELECT asset_id, MAX(ingested_at) AS last_price_ingested_at
            FROM market_data.prices
            GROUP BY asset_id
        ) mp
            ON mp.asset_id = ce.asset_id
        WHERE a.symbol IS NOT NULL
          AND (
              mp.last_price_ingested_at IS NULL
              OR ce.ingested_at > mp.last_price_ingested_at
              OR (
                  ce.ingested_at IS NULL
                  AND COALESCE(ce.exright_date, ce.event_date, ce.public_date)
                      >= CURRENT_DATE - (%s * INTERVAL '1 day')
              )
          )
        """,
        (events_lookback_days,),
    )
    rows = cur.fetchall()
    return [{"symbol": row[0], "asset_id": row[1]} for row in rows]
