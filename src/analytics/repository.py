from datetime import datetime
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
        timestamp = run_timestamp or datetime.utcnow()
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
                        "created_at": datetime.utcnow(),
                    },
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
