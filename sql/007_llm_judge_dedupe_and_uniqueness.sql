-- Deduplicate historical LLM-judge metric rows and prevent future duplicates.
-- Keep the newest row by (created_at, id) for each logical key:
--   sample_id + metric_name + details.run_id + details.ref_run_id

WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY
                sample_id,
                metric_name,
                details->>'run_id',
                COALESCE(details->>'ref_run_id', '')
            ORDER BY created_at DESC, id DESC
        ) AS rn
    FROM clinical_ai.evaluation_metrics
    WHERE metric_name IN ('llm_judge_compare', 'llm_judge_score')
)
DELETE FROM clinical_ai.evaluation_metrics em
USING ranked r
WHERE em.id = r.id
  AND r.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_metrics_llm_judge_logical_key
    ON clinical_ai.evaluation_metrics (
        sample_id,
        metric_name,
        (details->>'run_id'),
        (COALESCE(details->>'ref_run_id', ''))
    )
    WHERE metric_name IN ('llm_judge_compare', 'llm_judge_score');
