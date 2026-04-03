"""Analysis helpers for alignment viz outputs."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_RE = re.compile(r"\s{2,}")
SAMPLE_RE = re.compile(r"^sample_id:\s*(?P<sid>\S+)\s*$")
WER_RE = re.compile(
    r"^WER vs gold \(run\s+(?P<run>[^)]+)\):\s*(?P<wer>[0-9.]+)\s+S=(?P<S>\d+)\s+I=(?P<I>\d+)\s+D=(?P<D>\d+)\s+\(ref words=(?P<ref>\d+)\)"
)
MODEL_RE = re.compile(r"^\s*model:\s*(?P<model>.+?)\s+scope:\s*(?P<scope>\S+)\s*$")


@dataclass
class AlignmentChunk:
    gold_tokens: list[str]
    hyp_tokens: list[str]
    op_tokens: list[str]


@dataclass
class RunAlignment:
    run_id_short: str
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int
    chunks: list[AlignmentChunk] = field(default_factory=list)


@dataclass
class SampleAlignment:
    sample_id: str
    runs: list[RunAlignment] = field(default_factory=list)


def _split_payload(line: str, prefix: str) -> list[str]:
    payload = line[len(prefix) :].strip()
    if not payload:
        return []
    return [tok for tok in SPLIT_RE.split(payload) if tok != ""]


def _safe_align_triplet(gold: list[str], hyp: list[str], op: list[str]) -> AlignmentChunk:
    n = max(len(gold), len(hyp), len(op))
    gold = gold + [""] * (n - len(gold))
    hyp = hyp + [""] * (n - len(hyp))
    op = op + [""] * (n - len(op))
    return AlignmentChunk(gold_tokens=gold, hyp_tokens=hyp, op_tokens=op)


def parse_show_alignment_report(path: str | Path) -> dict[str, Any]:
    """
    Parse `show-alignment_*.txt` artifact into structured sample/run/chunk objects.
    """
    resolved = _resolve_report_path(path)
    text = resolved.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    meta: dict[str, Any] = {"candidate_run_id": None, "baseline_run_id": None, "candidate_model": None, "baseline_model": None}
    samples: list[SampleAlignment] = []

    current_sample: SampleAlignment | None = None
    current_run: RunAlignment | None = None
    last_gold: list[str] | None = None
    last_hyp: list[str] | None = None

    for line in lines:
        if line.startswith("Candidate run_id:"):
            meta["candidate_run_id"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Baseline run_id:"):
            meta["baseline_run_id"] = line.split(":", 1)[1].strip()
            continue
        m_model = MODEL_RE.match(line)
        if m_model and meta["candidate_model"] is None:
            meta["candidate_model"] = m_model.group("model").strip()
            continue
        if m_model and meta["baseline_model"] is None and meta["candidate_model"] is not None:
            model = m_model.group("model").strip()
            if model != meta["candidate_model"]:
                meta["baseline_model"] = model
            continue

        m_sample = SAMPLE_RE.match(line)
        if m_sample:
            current_sample = SampleAlignment(sample_id=m_sample.group("sid"))
            samples.append(current_sample)
            current_run = None
            last_gold = None
            last_hyp = None
            continue

        m_wer = WER_RE.match(line)
        if m_wer and current_sample is not None:
            current_run = RunAlignment(
                run_id_short=m_wer.group("run"),
                wer=float(m_wer.group("wer")),
                substitutions=int(m_wer.group("S")),
                insertions=int(m_wer.group("I")),
                deletions=int(m_wer.group("D")),
                ref_words=int(m_wer.group("ref")),
            )
            current_sample.runs.append(current_run)
            last_gold = None
            last_hyp = None
            continue

        if current_run is None:
            continue

        if line.startswith("GOLD "):
            last_gold = _split_payload(line, "GOLD ")
            continue
        if line.startswith("HYP  "):
            last_hyp = _split_payload(line, "HYP  ")
            continue
        if line.startswith("OP   "):
            op = _split_payload(line, "OP   ")
            if last_gold is not None and last_hyp is not None:
                current_run.chunks.append(_safe_align_triplet(last_gold, last_hyp, op))
            last_gold = None
            last_hyp = None

    return {"meta": meta, "samples": samples}


def _resolve_report_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append((cwd / p).resolve())

    # If notebook runs from analysis/, also check project root.
    if cwd.name == "analysis":
        candidates.append((cwd.parent / p).resolve())

    # If notebook runs from repo root, also check analysis/ sibling relation.
    candidates.append((cwd / "analysis" / p).resolve())

    for c in candidates:
        if c.exists():
            return c

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find alignment report: {path}\nChecked:\n{checked}\n"
        "Tip: use an absolute path or set ALIGNMENT_REPORT_PATH relative to repo root."
    )


def build_run_summary_df(parsed: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in parsed["samples"]:
        for r in s.runs:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "run_id_short": r.run_id_short,
                    "wer": r.wer,
                    "substitutions": r.substitutions,
                    "insertions": r.insertions,
                    "deletions": r.deletions,
                    "ref_words": r.ref_words,
                    "chunks": len(r.chunks),
                }
            )
    return pd.DataFrame(rows)


def render_alignment_html(run: RunAlignment, max_chunks: int = 8) -> str:
    """
    Render one run alignment as a compact color-coded HTML table.
    """
    color = {
        "=": "#e8f5e9",
        "S": "#fff3e0",
        "I": "#e3f2fd",
        "D": "#ffebee",
    }
    legend = (
        "<div style='margin:6px 0;font-family:monospace'>"
        "<span style='background:#e8f5e9;padding:2px 6px;margin-right:6px'>= match</span>"
        "<span style='background:#fff3e0;padding:2px 6px;margin-right:6px'>S substitution</span>"
        "<span style='background:#e3f2fd;padding:2px 6px;margin-right:6px'>I insertion</span>"
        "<span style='background:#ffebee;padding:2px 6px;margin-right:6px'>D deletion</span>"
        "</div>"
    )

    parts = [
        "<div style='font-family:monospace'>",
        f"<h4>Run {html.escape(run.run_id_short)} | WER={run.wer:.4f} "
        f"(S={run.substitutions}, I={run.insertions}, D={run.deletions}, ref={run.ref_words})</h4>",
        legend,
    ]
    for idx, chunk in enumerate(run.chunks[:max_chunks], start=1):
        parts.append(f"<div style='margin:10px 0'><b>Chunk {idx}</b></div>")
        parts.append("<table style='border-collapse:collapse;font-family:monospace;font-size:12px'>")
        parts.append("<tr><th style='text-align:left;padding:4px'>OP</th><th style='text-align:left;padding:4px'>GOLD</th><th style='text-align:left;padding:4px'>HYP</th></tr>")
        for op, g, h in zip(chunk.op_tokens, chunk.gold_tokens, chunk.hyp_tokens):
            bg = color.get(op, "#f5f5f5")
            parts.append(
                "<tr>"
                f"<td style='background:{bg};padding:2px 6px;border:1px solid #ddd'>{html.escape(op or ' ')}</td>"
                f"<td style='padding:2px 6px;border:1px solid #ddd'>{html.escape(g or ' ')}</td>"
                f"<td style='padding:2px 6px;border:1px solid #ddd'>{html.escape(h or ' ')}</td>"
                "</tr>"
            )
        parts.append("</table>")
    hidden = max(0, len(run.chunks) - max_chunks)
    if hidden:
        parts.append(f"<div style='margin-top:8px;color:#666'>... {hidden} more chunks not shown</div>")
    parts.append("</div>")
    return "".join(parts)

