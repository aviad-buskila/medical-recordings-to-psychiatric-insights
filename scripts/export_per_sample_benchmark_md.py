#!/usr/bin/env python3
"""Build a per-sample markdown table from run-eval, run-bertscore, and run-llm-judge report .txt files.

Parses the JSON block after ``===== EVAL REPORT RESULTS`` in each artifact. Used to refresh
``data/processed/reports/per_sample_benchmark.md`` for README linking.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_eval_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "===== EVAL REPORT RESULTS" not in text:
        raise ValueError(f"No eval report JSON marker in {path}")
    start = text.index("{")
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"Could not parse JSON object in {path}")
    return json.loads(text[start:end])


def _load_run_eval_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = _extract_eval_json(path)
    by_sample: dict[str, dict[str, Any]] = {}
    for row in data.get("metrics", []):
        if row.get("metric_name") != "wer":
            continue
        sid = str(row.get("sample_id", ""))
        d = row.get("details") or {}
        wb = d.get("wer_breakdown") or {}
        rwb = d.get("ref_wer_breakdown") or {}
        cer = (d.get("cer_breakdown") or {}).get("cer")
        ref_cer = d.get("ref_cer")
        mw = d.get("mer_wil_breakdown") or {}
        rmw = d.get("ref_mer_wil_breakdown") or {}
        cp = d.get("cp_wer") or {}
        rcp = d.get("ref_cp_wer") or {}
        sp = d.get("speaker_wer") or {}
        spr = d.get("speaker_wer_ref_run") or {}

        def _f(x: Any) -> float | None:
            if x is None:
                return None
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        by_sample[sid] = {
            "wer_c": _f(row.get("metric_value")),
            "wer_b": _f(d.get("ref_wer")),
            "wer_S_c": _f(wb.get("substitutions")),
            "wer_I_c": _f(wb.get("insertions")),
            "wer_D_c": _f(wb.get("deletions")),
            "wer_S_b": _f(rwb.get("substitutions")),
            "wer_I_b": _f(rwb.get("insertions")),
            "wer_D_b": _f(rwb.get("deletions")),
            "cer_c": _f(cer),
            "cer_b": _f(ref_cer),
            "mer_c": _f(mw.get("mer")),
            "mer_b": _f(rmw.get("mer")),
            "wil_c": _f(mw.get("wil")),
            "wil_b": _f(rmw.get("wil")),
            "cpwer_c": _f(cp.get("cp_wer")),
            "cpwer_b": _f(rcp.get("cp_wer")),
            "spk1_wer_c": _f((sp.get("speaker_1") or {}).get("wer")),
            "spk2_wer_c": _f((sp.get("speaker_2") or {}).get("wer")),
            "spk1_wer_b": _f((spr.get("speaker_1") or {}).get("wer")),
            "spk2_wer_b": _f((spr.get("speaker_2") or {}).get("wer")),
        }
    return by_sample


def _load_bertscore(path: Path) -> dict[str, dict[str, float | None]]:
    data = _extract_eval_json(path)
    summary = data.get("result_summary") or {}
    per = summary.get("per_sample") or []
    out: dict[str, dict[str, float | None]] = {}
    for row in per:
        sid = str(row.get("sample_id", ""))
        out[sid] = {
            "bert_f1_c": _maybe_float(row.get("f1")),
            "bert_f1_b": _maybe_float(row.get("ref_f1")),
            "bert_df1": _maybe_float(row.get("delta_f1_vs_ref")),
        }
    return out


def _load_llm_judge(path: Path) -> dict[str, dict[str, Any]]:
    data = _extract_eval_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("metrics", []):
        if row.get("metric_name") != "llm_judge_compare":
            continue
        sid = str(row.get("sample_id", ""))
        jr = (row.get("details") or {}).get("judge_result") or {}
        out[sid] = {
            "llm_delta": _maybe_float(row.get("metric_value")),
            "llm_winner": str(jr.get("winner", "")).lower() or None,
            "llm_cand": _maybe_float(jr.get("candidate_overall_score")),
            "llm_base": _maybe_float(jr.get("baseline_overall_score")),
        }
    return out


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def per_sample_column_legend_lines() -> list[str]:
    """Markdown table: column header + separator + one row per column (kept in sync with README)."""
    rows = [
        ("sample_id", "Recording id (matches `data/raw/recordings/` stem)."),
        ("WER_c", "Word error rate: **candidate** (quality STT) vs **gold** reference. Lower is better."),
        ("WER_b", "Word error rate: **baseline** (turbo STT) vs gold. Lower is better."),
        ("WER_S_c", "WER **substitutions** (word count): candidate vs gold."),
        ("WER_I_c", "WER **insertions** (word count): candidate vs gold."),
        ("WER_D_c", "WER **deletions** (word count): candidate vs gold."),
        ("WER_S_b", "WER substitutions: baseline vs gold."),
        ("WER_I_b", "WER insertions: baseline vs gold."),
        ("WER_D_b", "WER deletions: baseline vs gold."),
        ("CER_c", "Character error rate: candidate vs gold. Lower is better."),
        ("CER_b", "Character error rate: baseline vs gold."),
        ("MER_c", "Match error rate: candidate vs gold. Lower is better."),
        ("MER_b", "Match error rate: baseline vs gold."),
        ("WIL_c", "Word information lost: candidate vs gold. Lower is better."),
        ("WIL_b", "Word information lost: baseline vs gold."),
        ("cpWER_c", "cpWER (speaker-block permutation WER): candidate vs gold when `transcribed/*.json` exists. Lower is better."),
        ("cpWER_b", "cpWER: baseline vs gold."),
        ("spk1_WER_c", "Per-speaker WER for **speaker 1** (candidate); empty if no usable diarization JSON."),
        ("spk2_WER_c", "Per-speaker WER for **speaker 2** (candidate)."),
        ("spk1_WER_b", "Per-speaker WER speaker 1 (baseline)."),
        ("spk2_WER_b", "Per-speaker WER speaker 2 (baseline)."),
        ("BERT_F1_c", "BERTScore F1: semantic overlap of **candidate** transcript vs gold (encoder in bertscore artifact). Higher is better."),
        ("BERT_F1_b", "BERTScore F1: baseline transcript vs gold."),
        ("BERT_ΔF1", "**Candidate F1 − baseline F1** for this sample. Positive favors candidate in embedding space."),
        ("LLM_Δ", "LLM judge: **candidate overall score − baseline overall score** (each 0–10 vs reference). Positive favors candidate."),
        ("LLM_winner", "Judge label: `candidate`, `baseline`, or `tie`."),
        ("LLM_score_c", "Judge overall score (0–10) for candidate vs reference."),
        ("LLM_score_b", "Judge overall score (0–10) for baseline vs reference."),
    ]
    out = [
        "## Column legend",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
    ]
    for col, meaning in rows:
        out.append(f"| {col} | {meaning} |")
    out.extend(
        [
            "",
            "_**Note:** WER/CER/MER/WIL/cpWER — lower is better. BERT F1 — higher is better. "
            "`LLM_score_*` = 0 with `tie` may indicate a failed judge call; see `judge_error` in DB or the llm-judge report._",
        ]
    )
    return out


def build_markdown(
    *,
    run_eval: Path,
    run_bertscore: Path,
    run_llm_judge: Path,
    candidate_label: str,
    baseline_label: str,
) -> str:
    ev = _load_run_eval_rows(run_eval)
    bs = _load_bertscore(run_bertscore)
    lj = _load_llm_judge(run_llm_judge)
    samples = sorted(ev.keys())

    lines: list[str] = []
    lines.append("# Per-sample benchmark (all recordings in run)")
    lines.append("")
    lines.append(f"- **Candidate (quality):** {candidate_label}")
    lines.append(f"- **Baseline (turbo):** {baseline_label}")
    lines.append(f"- **Sources:** `{run_eval.name}`, `{run_bertscore.name}`, `{run_llm_judge.name}`")
    lines.append("")
    lines.append(
        "Lexical metrics are vs gold reference. "
        "**WER S/I/D** = substitutions / insertions / deletions (word-level). "
        "**cpWER** uses speaker blocks from transcribed JSON when available. "
        "**BERTScore F1** uses the encoder recorded in the bertscore artifact. "
        "**LLM** = comparative judge: Δ = candidate−baseline overall score (0–10); winner from judge."
    )
    lines.append("")

    headers = [
        "sample_id",
        "WER_c",
        "WER_b",
        "WER_S_c",
        "WER_I_c",
        "WER_D_c",
        "WER_S_b",
        "WER_I_b",
        "WER_D_b",
        "CER_c",
        "CER_b",
        "MER_c",
        "MER_b",
        "WIL_c",
        "WIL_b",
        "cpWER_c",
        "cpWER_b",
        "spk1_WER_c",
        "spk2_WER_c",
        "spk1_WER_b",
        "spk2_WER_b",
        "BERT_F1_c",
        "BERT_F1_b",
        "BERT_ΔF1",
        "LLM_Δ",
        "LLM_winner",
        "LLM_score_c",
        "LLM_score_b",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for sid in samples:
        e = ev.get(sid, {})
        b = bs.get(sid, {})
        l = lj.get(sid, {})
        row = [
            sid,
            _fmt(e.get("wer_c")),
            _fmt(e.get("wer_b")),
            _fmt(e.get("wer_S_c"), 0),
            _fmt(e.get("wer_I_c"), 0),
            _fmt(e.get("wer_D_c"), 0),
            _fmt(e.get("wer_S_b"), 0),
            _fmt(e.get("wer_I_b"), 0),
            _fmt(e.get("wer_D_b"), 0),
            _fmt(e.get("cer_c")),
            _fmt(e.get("cer_b")),
            _fmt(e.get("mer_c")),
            _fmt(e.get("mer_b")),
            _fmt(e.get("wil_c")),
            _fmt(e.get("wil_b")),
            _fmt(e.get("cpwer_c")),
            _fmt(e.get("cpwer_b")),
            _fmt(e.get("spk1_wer_c")),
            _fmt(e.get("spk2_wer_c")),
            _fmt(e.get("spk1_wer_b")),
            _fmt(e.get("spk2_wer_b")),
            _fmt(b.get("bert_f1_c")),
            _fmt(b.get("bert_f1_b")),
            _fmt(b.get("bert_df1")),
            _fmt(l.get("llm_delta"), 2),
            (l.get("llm_winner") or ""),
            _fmt(l.get("llm_cand"), 1),
            _fmt(l.get("llm_base"), 1),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.extend(per_sample_column_legend_lines())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Export per-sample benchmark markdown from eval artifacts.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--run-eval",
        type=Path,
        default=root / "data/processed/run-eval_20260402T222829Z.txt",
    )
    p.add_argument(
        "--run-bertscore",
        type=Path,
        default=root / "data/processed/run-bertscore_20260403T110608Z.txt",
    )
    p.add_argument(
        "--run-llm-judge",
        type=Path,
        default=root / "data/processed/run-llm-judge_20260403T093504Z.txt",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "data/processed/reports/per_sample_benchmark_N43.md",
    )
    p.add_argument("--candidate-label", default="mlx-community/whisper-large-v3-mlx (quality)")
    p.add_argument("--baseline-label", default="mlx-community/whisper-large-v3-turbo (default)")
    args = p.parse_args()

    md = build_markdown(
        run_eval=args.run_eval.resolve(),
        run_bertscore=args.run_bertscore.resolve(),
        run_llm_judge=args.run_llm_judge.resolve(),
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
