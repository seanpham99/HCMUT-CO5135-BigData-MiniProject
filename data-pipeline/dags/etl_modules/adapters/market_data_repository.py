from itertools import islice

import psycopg2
import psycopg2.extras

from dags.etl_modules.errors import ConfigurationError


class MarketDataRepository:
    def _chunked_rows(self, rows, chunk_size):
        iterator = iter(rows)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                return
            yield chunk

    def upsert_rows_in_batches(
        self,
        conn,
        query: str,
        rows,
        *,
        table_name: str,
        batch_size: int,
    ) -> list[dict[str, object]]:
        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be > 0, got {batch_size}")

        if not rows:
            print(f"No rows to upsert for {table_name}.")
            return []

        failed_batches = []
        upserted_rows = 0
        total_batches = (len(rows) + batch_size - 1) // batch_size
        for batch_index, batch_rows in enumerate(
            self._chunked_rows(rows, batch_size),
            start=1,
        ):
            try:
                with conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(cur, query, batch_rows)
                upserted_rows += len(batch_rows)
                print(
                    f"{table_name}: upserted batch {batch_index}/{total_batches} "
                    f"({len(batch_rows)} rows)"
                )
            except Exception as exc:
                conn.rollback()
                failed_batches.append(
                    {
                        "batch_index": batch_index,
                        "size": len(batch_rows),
                        "error": str(exc),
                    }
                )
                print(
                    f"{table_name}: batch {batch_index}/{total_batches} failed "
                    f"({len(batch_rows)} rows): {exc}"
                )

        print(
            f"{table_name}: upsert summary {upserted_rows}/{len(rows)} rows, "
            f"failed_batches={len(failed_batches)}"
        )
        return failed_batches

    def upsert_rows(
        self,
        *,
        db_url: str | None,
        query: str,
        rows,
        table_name: str,
        batch_size: int,
    ) -> tuple[list[dict[str, object]], str | None]:
        if not rows:
            return [], None
        if not db_url:
            return [], "SUPABASE_DB_URL environment variable is not set"

        conn = None
        try:
            conn = psycopg2.connect(db_url)
            failed_batches = self.upsert_rows_in_batches(
                conn,
                query,
                rows,
                table_name=table_name,
                batch_size=batch_size,
            )
            return failed_batches, None
        except Exception as exc:
            return [], str(exc)
        finally:
            if conn:
                conn.close()
