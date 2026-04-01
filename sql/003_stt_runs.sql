CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS clinical_ai.stt_runs (
    run_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    run_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE clinical_ai.stt_outputs
    ADD COLUMN IF NOT EXISTS run_id TEXT,
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD COLUMN IF NOT EXISTS run_timestamp TIMESTAMPTZ;

DO $$
DECLARE
    legacy_run_id TEXT := 'legacy-backfill-001';
    first_ts TIMESTAMPTZ;
BEGIN
    SELECT MIN(created_at) INTO first_ts FROM clinical_ai.stt_outputs;

    IF first_ts IS NOT NULL THEN
        INSERT INTO clinical_ai.stt_runs (run_id, provider, model_name, run_timestamp, run_parameters)
        VALUES (
            legacy_run_id,
            'legacy',
            'unknown',
            first_ts,
            '{"backfill": true, "source": "stt_outputs.created_at"}'::jsonb
        )
        ON CONFLICT (run_id) DO NOTHING;

        UPDATE clinical_ai.stt_outputs
        SET
            run_id = COALESCE(run_id, legacy_run_id),
            model_name = COALESCE(model_name, 'unknown'),
            run_timestamp = COALESCE(run_timestamp, created_at)
        WHERE run_id IS NULL OR model_name IS NULL OR run_timestamp IS NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_stt_outputs_run_id
    ON clinical_ai.stt_outputs(run_id);

CREATE INDEX IF NOT EXISTS idx_stt_runs_run_timestamp
    ON clinical_ai.stt_runs(run_timestamp);
