#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from src.config.settings import get_settings


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
PIPELINE_OUT_DIR = PROCESSED_DIR / "full_pipeline"
ANALYSIS_OUT_DIR = PROCESSED_DIR / "analysis_notebooks"


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class CmdResult:
    command: list[str]
    stdout: str
    stderr: str


def run_cmd(command: list[str], env: dict[str, str] | None = None) -> CmdResult:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return CmdResult(command=command, stdout=proc.stdout, stderr=proc.stderr)


def select_recordings(limit: int) -> list[str]:
    recordings_dir = ROOT / "data" / "raw" / "recordings"
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
    files = sorted([p for p in recordings_dir.glob("*") if p.is_file() and p.suffix.lower() in allowed], key=lambda p: p.name)
    return [p.stem for p in files[:limit]]


def parse_stt_run_ids(stdout: str) -> tuple[str, str]:
    m = re.search(r"default_run_id=([a-f0-9-]+)\s+quality_run_id=([a-f0-9-]+)", stdout)
    if not m:
        raise RuntimeError(f"Could not parse run IDs from run-stt output:\n{stdout}")
    return m.group(1), m.group(2)


def parse_eval_report_path(stdout: str) -> str | None:
    m = re.search(r"Eval report written to\s+(.+)$", stdout, re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_insights_artifact_path(stdout: str) -> str | None:
    m = re.search(r"Artifact:\s+(.+)$", stdout, re.MULTILINE)
    return m.group(1).strip() if m else None


def run_notebook(input_nb: Path, output_nb: Path, env: dict[str, str] | None = None) -> None:
    output_nb.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "python",
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(input_nb),
            "--output",
            output_nb.name,
            "--output-dir",
            str(output_nb.parent),
        ],
        env=env,
    )


def fetch_db_summary(candidate_run_id: str, baseline_run_id: str) -> dict[str, Any]:
    settings = get_settings()
    out: dict[str, Any] = {}
    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, provider, model_name, run_scope, run_timestamp, run_parameters
                FROM clinical_ai.stt_runs
                WHERE run_id IN (%s, %s)
                ORDER BY run_timestamp
                """,
                (candidate_run_id, baseline_run_id),
            )
            runs = cur.fetchall()
            out["stt_runs"] = [
                {
                    "run_id": str(r[0]),
                    "provider": str(r[1]),
                    "model_name": str(r[2]),
                    "run_scope": str(r[3]),
                    "run_timestamp": r[4].isoformat() if r[4] else None,
                    "run_parameters": r[5] if isinstance(r[5], dict) else {},
                }
                for r in runs
            ]

            cur.execute("SELECT run_id, count(*) FROM clinical_ai.stt_outputs WHERE run_id IN (%s, %s) GROUP BY run_id", (candidate_run_id, baseline_run_id))
            out["stt_outputs_count_by_run"] = {str(rid): int(c) for rid, c in cur.fetchall()}

            cur.execute(
                """
                SELECT details->>'run_id' AS rid, metric_name, count(*), avg(metric_value)
                FROM clinical_ai.evaluation_metrics
                WHERE details->>'run_id' IN (%s, %s)
                GROUP BY details->>'run_id', metric_name
                ORDER BY rid, metric_name
                """,
                (candidate_run_id, baseline_run_id),
            )
            out["evaluation_metrics"] = [
                {
                    "run_id": str(r[0]),
                    "metric_name": str(r[1]),
                    "rows": int(r[2]),
                    "avg_metric_value": float(r[3]) if r[3] is not None else None,
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT run_id, insight_model, count(*)
                FROM clinical_ai.transcript_insights
                WHERE run_id IN (%s, %s)
                GROUP BY run_id, insight_model
                ORDER BY run_id, insight_model
                """,
                (candidate_run_id, baseline_run_id),
            )
            out["insights_count_by_run_model"] = [
                {"run_id": str(r[0]), "insight_model": str(r[1]), "rows": int(r[2])} for r in cur.fetchall()
            ]
    return out


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Full Pipeline Run Summary")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{payload['timestamp_utc']}`")
    lines.append(f"- Limit: `{payload['limit']}`")
    lines.append(f"- Selected recordings: `{', '.join(payload['selected_recordings'])}`")
    lines.append(f"- Baseline run_id: `{payload['baseline_run_id']}`")
    lines.append(f"- Candidate run_id: `{payload['candidate_run_id']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for k, v in payload["artifacts"].items():
        if isinstance(v, list):
            lines.append(f"- **{k}**:")
            for item in v:
                lines.append(f"  - `{item}`")
        elif v:
            lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## DB Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload["db_summary"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Commands Executed")
    lines.append("")
    for c in payload["commands"]:
        lines.append(f"- `{c}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `run-eval` used `--workers auto` for conservative machine-friendly parallelism.")
    lines.append("- Analysis notebooks were executed to generated notebook outputs under `data/processed/analysis_notebooks/`.")
    lines.append("- Full raw command outputs are available in generated eval/artifact files listed above.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full local STT eval+analysis pipeline on a small sample.")
    parser.add_argument("--limit", type=int, default=5, help="Number of recordings to process (default: 5).")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    stamp = ts_utc()
    PIPELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_recordings(limit=args.limit)
    if not selected:
        raise SystemExit("No recordings found under data/raw/recordings")
    sample_for_single = selected[0]

    commands_run: list[str] = []

    # 1) run both STT models on first N recordings
    stt_cmd = ["python", "-m", "src.cli.main", "run-stt", "--flavor", "both", "--limit", str(args.limit)]
    stt_res = run_cmd(stt_cmd)
    commands_run.append(" ".join(stt_cmd))
    baseline_run_id, candidate_run_id = parse_stt_run_ids(stt_res.stdout)

    # 2) full eval compare
    eval_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "run-eval",
        "--run-id",
        candidate_run_id,
        "--ref-run-id",
        baseline_run_id,
        "--limit",
        str(args.limit),
        "--workers",
        "auto",
    ]
    eval_res = run_cmd(eval_cmd)
    commands_run.append(" ".join(eval_cmd))
    eval_report = parse_eval_report_path(eval_res.stdout)

    # 3) llm judge compare
    llm_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "run-llm-judge",
        "--run-id",
        candidate_run_id,
        "--ref-run-id",
        baseline_run_id,
        "--limit",
        str(args.limit),
    ]
    llm_res = run_cmd(llm_cmd)
    commands_run.append(" ".join(llm_cmd))
    llm_report = parse_eval_report_path(llm_res.stdout)

    # 4) show alignment compare
    align_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "show-alignment",
        "--run-id",
        candidate_run_id,
        "--ref-run-id",
        baseline_run_id,
        "--limit",
        str(args.limit),
    ]
    align_res = run_cmd(align_cmd)
    commands_run.append(" ".join(align_cmd))
    align_report = parse_eval_report_path(align_res.stdout)

    # 5) bertscore compare
    bert_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "run-bertscore",
        "--run-id",
        candidate_run_id,
        "--ref-run-id",
        baseline_run_id,
        "--limit",
        str(args.limit),
    ]
    bert_res = run_cmd(bert_cmd)
    commands_run.append(" ".join(bert_cmd))
    bert_report = parse_eval_report_path(bert_res.stdout)

    # 6) insights for both runs
    ins_candidate_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "insights-extract",
        "--run-id",
        candidate_run_id,
        "--limit",
        str(args.limit),
    ]
    ins_candidate_res = run_cmd(ins_candidate_cmd)
    commands_run.append(" ".join(ins_candidate_cmd))
    insights_candidate_artifact = parse_insights_artifact_path(ins_candidate_res.stdout)

    ins_baseline_cmd = [
        "python",
        "-m",
        "src.cli.main",
        "insights-extract",
        "--run-id",
        baseline_run_id,
        "--limit",
        str(args.limit),
    ]
    ins_baseline_res = run_cmd(ins_baseline_cmd)
    commands_run.append(" ".join(ins_baseline_cmd))
    insights_baseline_artifact = parse_insights_artifact_path(ins_baseline_res.stdout)

    # 7) execute all analysis notebooks
    nb_model_out = ANALYSIS_OUT_DIR / f"model_eval_insights_{stamp}.ipynb"
    env_model = {
        **os.environ,
        "CANDIDATE_RUN_ID": candidate_run_id,
        "BASELINE_RUN_ID": baseline_run_id,
        "SAMPLE_ID": sample_for_single,
    }
    run_notebook(ROOT / "analysis" / "model_eval_insights.ipynb", nb_model_out, env=env_model)

    nb_align_out = ANALYSIS_OUT_DIR / f"show_alignment_visualizer_{stamp}.ipynb"
    env_align = {
        **os.environ,
        "ALIGNMENT_REPORT_PATH": align_report or "",
        "SAMPLE_ID": sample_for_single,
        "MAX_CHUNKS_PER_RUN": "6",
    }
    run_notebook(ROOT / "analysis" / "show_alignment_visualizer.ipynb", nb_align_out, env=env_align)

    nb_timeline_out = ANALYSIS_OUT_DIR / f"gold_speaker_timeline_{stamp}.ipynb"
    run_notebook(ROOT / "analysis" / "gold_speaker_timeline.ipynb", nb_timeline_out)

    db_summary = fetch_db_summary(candidate_run_id=candidate_run_id, baseline_run_id=baseline_run_id)

    summary_payload = {
        "timestamp_utc": stamp,
        "limit": args.limit,
        "selected_recordings": selected,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "artifacts": {
            "run_eval_report": eval_report,
            "run_llm_judge_report": llm_report,
            "show_alignment_report": align_report,
            "run_bertscore_report": bert_report,
            "insights_extract_artifacts": [insights_candidate_artifact, insights_baseline_artifact],
            "analysis_notebooks": [str(nb_model_out), str(nb_align_out), str(nb_timeline_out)],
        },
        "db_summary": db_summary,
        "commands": commands_run,
    }
    summary_md = PIPELINE_OUT_DIR / f"full_pipeline_{stamp}.md"
    write_markdown_summary(summary_md, summary_payload)
    print(f"Full pipeline completed. Summary: {summary_md}")


if __name__ == "__main__":
    main()
