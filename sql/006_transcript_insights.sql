CREATE TABLE IF NOT EXISTS clinical_ai.transcript_insights (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    insight_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL DEFAULT 'v1',
    insights JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_output TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_transcript_insights_run_sample_model_prompt
    ON clinical_ai.transcript_insights(run_id, sample_id, insight_model, prompt_version);

CREATE INDEX IF NOT EXISTS idx_transcript_insights_run_id
    ON clinical_ai.transcript_insights(run_id);

CREATE INDEX IF NOT EXISTS idx_transcript_insights_sample_id
    ON clinical_ai.transcript_insights(sample_id);
