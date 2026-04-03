"""Analytics repository helpers for repository."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.types.json import Json

from src.config.settings import get_settings


class AnalyticsRepository:
    """Persistence adapter for pipeline metadata and evaluations."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def create_stt_run(
        self,
        provider: str,
        model_name: str,
        run_scope: str,
        run_parameters: dict[str, Any],
        run_timestamp: datetime | None = None,
    ) -> tuple[str, datetime]:
        run_id = str(uuid4())
        timestamp = run_timestamp or datetime.now(timezone.utc)
        if run_scope not in {"full", "sample"}:
            raise ValueError("run_scope must be 'full' or 'sample'")
        query = """
            INSERT INTO clinical_ai.stt_runs (
                run_id, provider, model_name, run_scope, run_timestamp, run_parameters
            ) VALUES (%(run_id)s, %(provider)s, %(model_name)s, %(run_scope)s, %(run_timestamp)s, %(run_parameters)s)
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "run_id": run_id,
                        "provider": provider,
                        "model_name": model_name,
                        "run_scope": run_scope,
                        "run_timestamp": timestamp,
                        "run_parameters": Json(run_parameters),
                    },
                )
            conn.commit()
        return run_id, timestamp

    def insert_stt_output(
        self,
        run_id: str,
        run_timestamp: datetime,
        model_name: str,
        sample_id: str,
        provider: str,
        payload: dict[str, Any],
    ) -> None:
        query = """
            INSERT INTO clinical_ai.stt_outputs (
                sample_id, provider, transcript_text, language, audio_duration_s,
                transcription_time_s, metadata, run_id, model_name, run_timestamp
            ) VALUES (%(sample_id)s, %(provider)s, %(transcript_text)s, %(language)s,
                      %(audio_duration_s)s, %(transcription_time_s)s, %(metadata)s,
                      %(run_id)s, %(model_name)s, %(run_timestamp)s)
        """
        params = {
            "sample_id": sample_id,
            "provider": provider,
            "transcript_text": payload.get("text", ""),
            "language": payload.get("language", "unknown"),
            "audio_duration_s": payload.get("duration_s", 0.0),
            "transcription_time_s": payload.get("elapsed_s", 0.0),
            "metadata": Json(payload),
            "run_id": run_id,
            "model_name": model_name,
            "run_timestamp": run_timestamp,
        }
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()

    def insert_eval_metric(self, sample_id: str, metric_name: str, metric_value: float, details: dict[str, Any]) -> None:
        query = """
            INSERT INTO clinical_ai.evaluation_metrics (
                sample_id, metric_name, metric_value, details, created_at
            ) VALUES (%(sample_id)s, %(metric_name)s, %(metric_value)s, %(details)s, %(created_at)s)
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "sample_id": sample_id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "details": Json(details),
                        "created_at": datetime.now(timezone.utc),
                    },
                )
            conn.commit()

    def insert_eval_metrics_batch(
        self,
        rows: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Bulk insert evaluation rows in chunks to reduce DB round-trips."""
        if not rows:
            return
        query = """
            INSERT INTO clinical_ai.evaluation_metrics (
                sample_id, metric_name, metric_value, details, created_at
            ) VALUES (%(sample_id)s, %(metric_name)s, %(metric_value)s, %(details)s, %(created_at)s)
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i : i + batch_size]
                    cur.executemany(
                        query,
                        [
                            {
                                "sample_id": str(r["sample_id"]),
                                "metric_name": str(r["metric_name"]),
                                "metric_value": float(r["metric_value"]),
                                "details": Json(r["details"]),
                                "created_at": datetime.now(timezone.utc),
                            }
                            for r in chunk
                        ],
                    )
            conn.commit()

    def get_latest_stt_output(self, sample_id: str) -> str | None:
        query = """
            SELECT transcript_text
            FROM clinical_ai.stt_outputs
            WHERE sample_id = %(sample_id)s
            ORDER BY run_timestamp DESC NULLS LAST, id DESC
            LIMIT 1
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"sample_id": sample_id})
                row = cur.fetchone()
        if not row:
            return None
        return str(row[0])

    def get_latest_stt_outputs(self, sample_ids: list[str]) -> dict[str, str]:
        """Fetch latest transcript text for multiple sample_ids in one query."""
        if not sample_ids:
            return {}
        query = """
            SELECT DISTINCT ON (sample_id) sample_id, transcript_text
            FROM clinical_ai.stt_outputs
            WHERE sample_id = ANY(%(sample_ids)s)
            ORDER BY sample_id, run_timestamp DESC NULLS LAST, id DESC
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"sample_ids": sample_ids})
                rows = cur.fetchall()
        return {str(sample_id): str(transcript_text) for sample_id, transcript_text in rows}

    def get_stt_outputs_for_run(self, run_id: str) -> dict[str, str]:
        query = """
            SELECT DISTINCT ON (sample_id) sample_id, transcript_text
            FROM clinical_ai.stt_outputs
            WHERE run_id = %(run_id)s
            ORDER BY sample_id, run_timestamp DESC NULLS LAST, id DESC
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"run_id": run_id})
                rows = cur.fetchall()
        return {str(sample_id): str(transcript_text) for sample_id, transcript_text in rows}

    def get_stt_run_info(self, run_id: str) -> dict[str, Any] | None:
        query = """
            SELECT run_id, provider, model_name, run_scope, run_timestamp, run_parameters
            FROM clinical_ai.stt_runs
            WHERE run_id = %(run_id)s
            LIMIT 1
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"run_id": run_id})
                row = cur.fetchone()
        if not row:
            return None
        return {
            "run_id": str(row[0]),
            "provider": str(row[1]),
            "model_name": str(row[2]),
            "run_scope": str(row[3]),
            "run_timestamp": row[4],
            "run_parameters": row[5] if isinstance(row[5], dict) else {},
        }

    def upsert_transcript_insight(
        self,
        run_id: str,
        sample_id: str,
        insight_model: str,
        prompt_version: str,
        insights: dict[str, Any],
        raw_output: str,
    ) -> None:
        query = """
            INSERT INTO clinical_ai.transcript_insights (
                run_id, sample_id, insight_model, prompt_version, insights, raw_output, created_at
            ) VALUES (
                %(run_id)s, %(sample_id)s, %(insight_model)s, %(prompt_version)s, %(insights)s, %(raw_output)s, %(created_at)s
            )
            ON CONFLICT (run_id, sample_id, insight_model, prompt_version)
            DO UPDATE SET
                insights = EXCLUDED.insights,
                raw_output = EXCLUDED.raw_output,
                created_at = EXCLUDED.created_at
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "insight_model": insight_model,
                        "prompt_version": prompt_version,
                        "insights": Json(insights),
                        "raw_output": raw_output,
                        "created_at": datetime.now(timezone.utc),
                    },
                )
            conn.commit()

    def get_existing_transcript_insight_sample_ids(
        self,
        *,
        run_id: str,
        insight_model: str,
        prompt_version: str,
        sample_ids: list[str] | None = None,
    ) -> set[str]:
        """Return sample_ids that already have transcript_insights rows for this run/model/prompt."""
        base_query = """
            SELECT sample_id
            FROM clinical_ai.transcript_insights
            WHERE run_id = %(run_id)s
              AND insight_model = %(insight_model)s
              AND prompt_version = %(prompt_version)s
        """
        params: dict[str, Any] = {
            "run_id": run_id,
            "insight_model": insight_model,
            "prompt_version": prompt_version,
        }
        if sample_ids:
            query = base_query + " AND sample_id = ANY(%(sample_ids)s)"
            params["sample_ids"] = sample_ids
        else:
            query = base_query

        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return {str(r[0]) for r in rows}
