-- Backfill model_name for legacy rows where unknown.
UPDATE clinical_ai.stt_outputs
SET model_name = CASE
    WHEN provider = 'mlx-whisper' THEN 'mlx-community/whisper-large-v3-turbo'
    WHEN provider IS NOT NULL AND provider <> '' THEN provider
    ELSE 'unknown'
END
WHERE model_name IS NULL OR model_name = 'unknown';

UPDATE clinical_ai.stt_runs r
SET model_name = COALESCE(
    (
        SELECT o.model_name
        FROM clinical_ai.stt_outputs o
        WHERE o.run_id = r.run_id AND o.model_name IS NOT NULL AND o.model_name <> 'unknown'
        LIMIT 1
    ),
    CASE
        WHEN r.provider = 'mlx-whisper' THEN 'mlx-community/whisper-large-v3-turbo'
        WHEN r.provider IS NOT NULL AND r.provider <> '' THEN r.provider
        ELSE r.model_name
    END
)
WHERE r.model_name IS NULL OR r.model_name = 'unknown';

-- Drop redundant timestamps kept from earlier schema versions.
ALTER TABLE clinical_ai.stt_outputs
    DROP COLUMN IF EXISTS created_at;

ALTER TABLE clinical_ai.stt_runs
    DROP COLUMN IF EXISTS created_at;
