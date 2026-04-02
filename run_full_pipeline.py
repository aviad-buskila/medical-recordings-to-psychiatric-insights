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
STATE_LATEST = PIPELINE_OUT_DIR / "full_pipeline_state_latest.json"


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_log_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class CmdResult:
    command: list[str]
    stdout: str
    stderr: str


def log(msg: str) -> None:
    print(f"[{now_log_ts()}] [full-pipeline] {msg}", flush=True)


def step(name: str) -> None:
    print(f"\n[{now_log_ts()}] === {name} ===", flush=True)


def run_cmd(command: list[str], env: dict[str, str] | None = None) -> CmdResult:
    log(f"Running: {' '.join(command)}")
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
    log(f"Done: {' '.join(command)}")
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


def latest_processed_artifact(prefix: str) -> str | None:
    files = sorted(PROCESSED_DIR.glob(f"{prefix}_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def save_state(state: dict[str, Any]) -> None:
    PIPELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_LATEST.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"State saved: {STATE_LATEST}")


def load_state(path: Path | None) -> dict[str, Any]:
    p = path or STATE_LATEST
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def resolve_insights_model(requested: str | None) -> str | None:
    if requested:
        return requested
    # Prefer explicit env override.
    env_model = os.environ.get("OLLAMA_INSIGHTS_MODEL", "").strip()
    if env_model:
        return env_model
    # Auto-fallback for common model naming mismatch.
    try:
        out = run_cmd(["ollama", "list"]).stdout.lower()
        if "medaibase/medgemma1.5:4b" in out:
            return "medaibase/medgemma1.5:4b"
        if "med-gemma1.5:4b" in out:
            return "med-gemma1.5:4b"
    except Exception:
        pass
    return None


def run_notebook(input_nb: Path, output_nb: Path, env: dict[str, str] | None = None) -> None:
    output_nb.parent.mkdir(parents=True, exist_ok=True)
    log(f"Executing notebook: {input_nb.name}")
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
    parser.add_argument("--resume-state", type=str, default=None, help="Resume from a saved pipeline state JSON.")
    parser.add_argument("--candidate-run-id", type=str, default=None, help="Reuse existing candidate run_id (skip STT).")
    parser.add_argument("--baseline-run-id", type=str, default=None, help="Reuse existing baseline run_id (skip STT).")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT step and use provided run IDs/state.")
    parser.add_argument("--skip-evals", action="store_true", help="Skip eval commands and reuse latest artifacts.")
    parser.add_argument("--insights-model", type=str, default=None, help="Model override for insights-extract.")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    step("Initialize")

    state = load_state(Path(args.resume_state) if args.resume_state else None)
    stamp = state.get("timestamp_utc") or ts_utc()
    PIPELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = state.get("selected_recordings") or select_recordings(limit=args.limit)
    if not selected:
        raise SystemExit("No recordings found under data/raw/recordings")
    sample_for_single = selected[0]
    log(f"Selected recordings ({len(selected)}): {', '.join(selected)}")

    commands_run: list[str] = state.get("commands", [])
    baseline_run_id = args.baseline_run_id or state.get("baseline_run_id")
    candidate_run_id = args.candidate_run_id or state.get("candidate_run_id")
    artifacts = state.get("artifacts", {})

    step("STT Runs")
    # 1) run both STT models on first N recordings (or reuse provided IDs)
    if not (args.skip_stt or (baseline_run_id and candidate_run_id)):
        stt_cmd = ["python", "-m", "src.cli.main", "run-stt", "--flavor", "both", "--limit", str(args.limit)]
        stt_res = run_cmd(stt_cmd)
        commands_run.append(" ".join(stt_cmd))
        baseline_run_id, candidate_run_id = parse_stt_run_ids(stt_res.stdout)
        log(f"Parsed run IDs: baseline={baseline_run_id} candidate={candidate_run_id}")
        state.update(
            {
                "timestamp_utc": stamp,
                "limit": args.limit,
                "selected_recordings": selected,
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "commands": commands_run,
                "artifacts": artifacts,
            }
        )
        save_state(state)
    else:
        log(f"Skipping STT. baseline={baseline_run_id} candidate={candidate_run_id}")

    if not (baseline_run_id and candidate_run_id):
        raise SystemExit("Missing run IDs. Provide --baseline-run-id and --candidate-run-id, or run without --skip-stt.")

    step("Evaluations")
    # 2-5) evals
    if not args.skip_evals:
        if not artifacts.get("run_eval_report"):
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
            artifacts["run_eval_report"] = parse_eval_report_path(eval_res.stdout)
            log(f"run-eval artifact: {artifacts['run_eval_report']}")
            state.update({"commands": commands_run, "artifacts": artifacts})
            save_state(state)

        if not artifacts.get("run_llm_judge_report"):
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
            artifacts["run_llm_judge_report"] = parse_eval_report_path(llm_res.stdout)
            log(f"run-llm-judge artifact: {artifacts['run_llm_judge_report']}")
            state.update({"commands": commands_run, "artifacts": artifacts})
            save_state(state)

        if not artifacts.get("show_alignment_report"):
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
            artifacts["show_alignment_report"] = parse_eval_report_path(align_res.stdout)
            log(f"show-alignment artifact: {artifacts['show_alignment_report']}")
            state.update({"commands": commands_run, "artifacts": artifacts})
            save_state(state)

        if not artifacts.get("run_bertscore_report"):
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
            artifacts["run_bertscore_report"] = parse_eval_report_path(bert_res.stdout)
            log(f"run-bertscore artifact: {artifacts['run_bertscore_report']}")
            state.update({"commands": commands_run, "artifacts": artifacts})
            save_state(state)
    else:
        artifacts.setdefault("run_eval_report", latest_processed_artifact("run-eval"))
        artifacts.setdefault("run_llm_judge_report", latest_processed_artifact("run-llm-judge"))
        artifacts.setdefault("show_alignment_report", latest_processed_artifact("show-alignment"))
        artifacts.setdefault("run_bertscore_report", latest_processed_artifact("run-bertscore"))
        log("Skipped evals; reused latest eval artifacts from data/processed")

    step("Insights Extraction")
    # 6) insights for both runs (resume-friendly + model fallback)
    insights_model = resolve_insights_model(args.insights_model)
    log(f"Insights model: {insights_model or 'default from CLI/env'}")
    insights_list = artifacts.get("insights_extract_artifacts", [])
    if len(insights_list) < 2:
        insights_list = [None, None]

    if not insights_list[0]:
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
        if insights_model:
            ins_candidate_cmd += ["--model", insights_model]
        ins_candidate_res = run_cmd(ins_candidate_cmd)
        commands_run.append(" ".join(ins_candidate_cmd))
        insights_list[0] = parse_insights_artifact_path(ins_candidate_res.stdout)
        log(f"Candidate insights artifact: {insights_list[0]}")
        artifacts["insights_extract_artifacts"] = insights_list
        state.update({"commands": commands_run, "artifacts": artifacts})
        save_state(state)
    else:
        log(f"Reusing candidate insights artifact: {insights_list[0]}")

    if not insights_list[1]:
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
        if insights_model:
            ins_baseline_cmd += ["--model", insights_model]
        ins_baseline_res = run_cmd(ins_baseline_cmd)
        commands_run.append(" ".join(ins_baseline_cmd))
        insights_list[1] = parse_insights_artifact_path(ins_baseline_res.stdout)
        log(f"Baseline insights artifact: {insights_list[1]}")
        artifacts["insights_extract_artifacts"] = insights_list
        state.update({"commands": commands_run, "artifacts": artifacts})
        save_state(state)
    else:
        log(f"Reusing baseline insights artifact: {insights_list[1]}")

    step("Analysis Notebooks")
    # 7) execute all analysis notebooks
    nb_model_out = ANALYSIS_OUT_DIR / f"model_eval_insights_{stamp}.ipynb"
    env_model = {
        **os.environ,
        "CANDIDATE_RUN_ID": candidate_run_id,
        "BASELINE_RUN_ID": baseline_run_id,
        "SAMPLE_ID": sample_for_single,
    }
    if not nb_model_out.exists():
        run_notebook(ROOT / "analysis" / "model_eval_insights.ipynb", nb_model_out, env=env_model)
    else:
        log(f"Reusing notebook output: {nb_model_out}")

    nb_align_out = ANALYSIS_OUT_DIR / f"show_alignment_visualizer_{stamp}.ipynb"
    env_align = {
        **os.environ,
        "ALIGNMENT_REPORT_PATH": artifacts.get("show_alignment_report") or "",
        "SAMPLE_ID": sample_for_single,
        "MAX_CHUNKS_PER_RUN": "6",
    }
    if not nb_align_out.exists():
        run_notebook(ROOT / "analysis" / "show_alignment_visualizer.ipynb", nb_align_out, env=env_align)
    else:
        log(f"Reusing notebook output: {nb_align_out}")

    nb_timeline_out = ANALYSIS_OUT_DIR / f"gold_speaker_timeline_{stamp}.ipynb"
    if not nb_timeline_out.exists():
        run_notebook(ROOT / "analysis" / "gold_speaker_timeline.ipynb", nb_timeline_out)
    else:
        log(f"Reusing notebook output: {nb_timeline_out}")

    step("DB Summary + Final Report")
    db_summary = fetch_db_summary(candidate_run_id=candidate_run_id, baseline_run_id=baseline_run_id)

    summary_payload = {
        "timestamp_utc": stamp,
        "limit": args.limit,
        "selected_recordings": selected,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "artifacts": {
            "run_eval_report": artifacts.get("run_eval_report"),
            "run_llm_judge_report": artifacts.get("run_llm_judge_report"),
            "show_alignment_report": artifacts.get("show_alignment_report"),
            "run_bertscore_report": artifacts.get("run_bertscore_report"),
            "insights_extract_artifacts": artifacts.get("insights_extract_artifacts", []),
            "analysis_notebooks": [str(nb_model_out), str(nb_align_out), str(nb_timeline_out)],
        },
        "db_summary": db_summary,
        "commands": commands_run,
    }
    summary_md = PIPELINE_OUT_DIR / f"full_pipeline_{stamp}.md"
    write_markdown_summary(summary_md, summary_payload)
    state.update(summary_payload)
    save_state(state)
    log(f"Full pipeline completed. Summary: {summary_md}")


if __name__ == "__main__":
    main()
