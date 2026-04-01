from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Json

from src.config.settings import get_settings


class AnalyticsRepository:
    """Persistence adapter for pipeline metadata and evaluations."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def insert_stt_run(self, sample_id: str, provider: str, payload: dict[str, Any]) -> None:
        query = """
            INSERT INTO clinical_ai.stt_outputs (
                sample_id, provider, transcript_text, language, audio_duration_s,
                transcription_time_s, created_at, metadata
            ) VALUES (%(sample_id)s, %(provider)s, %(transcript_text)s, %(language)s,
                      %(audio_duration_s)s, %(transcription_time_s)s, %(created_at)s, %(metadata)s)
        """
        params = {
            "sample_id": sample_id,
            "provider": provider,
            "transcript_text": payload.get("text", ""),
            "language": payload.get("language", "unknown"),
            "audio_duration_s": payload.get("duration_s", 0.0),
            "transcription_time_s": payload.get("elapsed_s", 0.0),
            "created_at": datetime.utcnow(),
            "metadata": Json(payload),
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
            ORDER BY created_at DESC
            LIMIT 1
        """
        with psycopg.connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"sample_id": sample_id})
                row = cur.fetchone()
        if not row:
            return None
        return str(row[0])
