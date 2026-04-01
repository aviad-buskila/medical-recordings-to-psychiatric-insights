CREATE SCHEMA IF NOT EXISTS clinical_ai;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS clinical_ai.stt_outputs (
    id BIGSERIAL PRIMARY KEY,
    sample_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    transcript_text TEXT NOT NULL,
    language TEXT,
    audio_duration_s DOUBLE PRECISION DEFAULT 0,
    transcription_time_s DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_stt_outputs_sample_id
    ON clinical_ai.stt_outputs(sample_id);

CREATE TABLE IF NOT EXISTS clinical_ai.evaluation_metrics (
    id BIGSERIAL PRIMARY KEY,
    sample_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_sample_metric
    ON clinical_ai.evaluation_metrics(sample_id, metric_name);
