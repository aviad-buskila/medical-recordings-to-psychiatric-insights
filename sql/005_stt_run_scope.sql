ALTER TABLE clinical_ai.stt_runs
    ADD COLUMN IF NOT EXISTS run_scope TEXT;

UPDATE clinical_ai.stt_runs
SET run_scope = CASE
    WHEN COALESCE(run_parameters ->> 'limit', '') = '' THEN 'full'
    WHEN run_parameters ->> 'limit' = 'null' THEN 'full'
    ELSE 'sample'
END
WHERE run_scope IS NULL;

ALTER TABLE clinical_ai.stt_runs
    ALTER COLUMN run_scope SET NOT NULL;

ALTER TABLE clinical_ai.stt_runs
    DROP CONSTRAINT IF EXISTS stt_runs_run_scope_check;

ALTER TABLE clinical_ai.stt_runs
    ADD CONSTRAINT stt_runs_run_scope_check
    CHECK (run_scope IN ('full', 'sample'));
