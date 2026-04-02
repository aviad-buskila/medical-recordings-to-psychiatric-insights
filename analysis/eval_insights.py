from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg

from src.config.settings import get_settings


@dataclass
class RunMeta:
    run_id: str
    provider: str | None
    model_name: str | None
    run_scope: str | None
    run_timestamp: Any | None
    run_parameters: dict[str, Any]


def _connect() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.postgres_dsn)


def fetch_eval_rows(
    candidate_run_id: str,
    baseline_run_id: str,
    sample_id: str | None = None,
) -> pd.DataFrame:
    """
    Load evaluation rows for a candidate-vs-baseline pair.

    The query includes rows where:
    - details.run_id == candidate and details.ref_run_id == baseline
    - or details.run_id == baseline and details.ref_run_id == candidate
    - or details.run_id is either run when ref_run_id is absent
    """
    query = """
        SELECT
            em.id,
            em.sample_id,
            em.metric_name,
            em.metric_value,
            em.details,
            em.created_at,
            em.details->>'run_id' AS details_run_id,
            em.details->>'ref_run_id' AS details_ref_run_id
        FROM clinical_ai.evaluation_metrics em
        WHERE (
            (em.details->>'run_id' = %(candidate)s AND COALESCE(em.details->>'ref_run_id', '') = %(baseline)s)
            OR
            (em.details->>'run_id' = %(baseline)s AND COALESCE(em.details->>'ref_run_id', '') = %(candidate)s)
            OR
            (em.details->>'run_id' IN (%(candidate)s, %(baseline)s))
        )
        AND (%(sample_id)s IS NULL OR em.sample_id = %(sample_id)s)
        ORDER BY em.created_at ASC, em.id ASC
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "candidate": candidate_run_id,
                "baseline": baseline_run_id,
                "sample_id": sample_id,
            },
        )
    if df.empty:
        return df

    details_df = pd.json_normalize(df["details"]).add_prefix("details.")
    out = pd.concat([df.drop(columns=["details"]), details_df], axis=1)
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    return out


def fetch_run_meta(*run_ids: str) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame(
            columns=["run_id", "provider", "model_name", "run_scope", "run_timestamp", "run_parameters"]
        )
    query = """
        SELECT
            run_id,
            provider,
            model_name,
            run_scope,
            run_timestamp,
            run_parameters
        FROM clinical_ai.stt_runs
        WHERE run_id = ANY(%(run_ids)s)
        ORDER BY run_timestamp ASC
    """
    with _connect() as conn:
        df = pd.read_sql_query(query, conn, params={"run_ids": list(run_ids)})
    if df.empty:
        return df
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], errors="coerce", utc=True)
    return df


def build_metric_summary(
    df: pd.DataFrame,
    candidate_run_id: str,
    baseline_run_id: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["metric_name", "run_id", "samples", "mean", "median", "std"])

    # Prefer details.run_id when available; fallback to ref fields for compare rows.
    run_id_series = df.get("details.run_id")
    if run_id_series is None:
        run_id_series = df.get("details_run_id")
    normalized = df.copy()
    normalized["run_id"] = run_id_series.fillna("")
    normalized.loc[normalized["run_id"] == "", "run_id"] = normalized["details_run_id"].fillna("")

    normalized = normalized[normalized["run_id"].isin([candidate_run_id, baseline_run_id])]
    if normalized.empty:
        return pd.DataFrame(columns=["metric_name", "run_id", "samples", "mean", "median", "std"])

    agg = (
        normalized.groupby(["metric_name", "run_id"], dropna=False)["metric_value"]
        .agg(samples="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values(["metric_name", "run_id"])
    )
    return agg


def recommend_winner_table(
    summary_df: pd.DataFrame,
    candidate_run_id: str,
    baseline_run_id: str,
) -> pd.DataFrame:
    """
    Compare candidate vs baseline means per metric.

    Lower is better for error metrics, higher is better for score-like metrics.
    """
    if summary_df.empty:
        return pd.DataFrame(
            columns=["metric_name", "candidate_mean", "baseline_mean", "direction", "winner", "relative_delta_pct"]
        )

    lower_better = {
        "wer",
        "cer",
        "mer",
        "wil",
        "cp_wer",
        "cp-wer",
    }
    higher_better = {
        "llm_judge_score",
        "llm_judge_compare",
        "bertscore_f1",
        "bertscore_ref_f1",
        "bertscore_delta_f1_vs_ref",
    }

    rows: list[dict[str, Any]] = []
    for metric_name, metric_df in summary_df.groupby("metric_name"):
        cand = metric_df.loc[metric_df["run_id"] == candidate_run_id, "mean"]
        base = metric_df.loc[metric_df["run_id"] == baseline_run_id, "mean"]
        if cand.empty or base.empty:
            continue
        cand_mean = float(cand.iloc[0])
        base_mean = float(base.iloc[0])
        metric_key = metric_name.replace("-", "_").lower()
        if metric_key in lower_better:
            direction = "lower_is_better"
            winner = "candidate" if cand_mean < base_mean else "baseline"
            rel = ((base_mean - cand_mean) / base_mean * 100.0) if base_mean else 0.0
        elif metric_key in higher_better:
            direction = "higher_is_better"
            winner = "candidate" if cand_mean > base_mean else "baseline"
            rel = ((cand_mean - base_mean) / abs(base_mean) * 100.0) if base_mean else 0.0
        else:
            # Unknown direction; keep visible but unranked.
            direction = "unknown"
            winner = "undetermined"
            rel = 0.0
        rows.append(
            {
                "metric_name": metric_name,
                "candidate_mean": cand_mean,
                "baseline_mean": base_mean,
                "direction": direction,
                "winner": winner,
                "relative_delta_pct": rel,
            }
        )

    return pd.DataFrame(rows).sort_values(["winner", "metric_name"])

