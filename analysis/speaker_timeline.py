"""Analysis helpers for speaker timeline outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    cwd = Path.cwd().resolve()
    candidates = [(cwd / p).resolve()]
    if cwd.name == "analysis":
        candidates.append((cwd.parent / p).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_speaker(v: Any) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip()
    if not s:
        return "unknown"
    return s


def _load_pickle(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return {}
    return obj


def _extract_turns_from_entry(sample_id: str, entry: Any) -> list[dict[str, Any]]:
    """
    Flexible parser for possible dataset.pickle structures.
    Expected output rows: sample_id, speaker, start_s, end_s, duration_s, source.
    """
    out: list[dict[str, Any]] = []

    if isinstance(entry, dict):
        # Common possible fields for a turn list.
        for key in ("segments", "turns", "utterances", "speaker_segments", "dialogue_segments", "words"):
            segs = entry.get(key)
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                start = _as_float(seg.get("start") or seg.get("start_s") or seg.get("start_time"))
                end = _as_float(seg.get("end") or seg.get("end_s") or seg.get("end_time"))
                sp = _norm_speaker(seg.get("speaker") or seg.get("speaker_id") or seg.get("speaker_label"))
                if start is None or end is None or end < start:
                    continue
                out.append(
                    {
                        "sample_id": sample_id,
                        "speaker": sp,
                        "start_s": start,
                        "end_s": end,
                        "duration_s": end - start,
                        "source": "dataset.pickle",
                    }
                )
            if out:
                return out

    # Some pickles may map sample_id directly to a list of segments.
    if isinstance(entry, list):
        for seg in entry:
            if not isinstance(seg, dict):
                continue
            start = _as_float(seg.get("start") or seg.get("start_s") or seg.get("start_time"))
            end = _as_float(seg.get("end") or seg.get("end_s") or seg.get("end_time"))
            sp = _norm_speaker(seg.get("speaker") or seg.get("speaker_id") or seg.get("speaker_label"))
            if start is None or end is None or end < start:
                continue
            out.append(
                {
                    "sample_id": sample_id,
                    "speaker": sp,
                    "start_s": start,
                    "end_s": end,
                    "duration_s": end - start,
                    "source": "dataset.pickle",
                }
            )
    return out


def _extract_turns_from_transcribed_json(sample_id: str, json_path: Path) -> list[dict[str, Any]]:
    """
    Fallback extractor for `transcripts/transcribed/<sample_id>.json`.
    Supports nested utterance/segment style formats with start/end times.
    """
    out: list[dict[str, Any]] = []
    if not json_path.exists():
        return out
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return out

    blocks: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for key in ("segments", "turns", "utterances", "dialogue", "items"):
            v = data.get(key)
            if isinstance(v, list):
                blocks = [x for x in v if isinstance(x, dict)]
                if blocks:
                    break
    elif isinstance(data, list):
        blocks = [x for x in data if isinstance(x, dict)]

    for b in blocks:
        speaker = _norm_speaker(b.get("speaker") or b.get("speaker_id") or b.get("speaker_label"))
        # Prefer block-level times if present.
        b_start = _as_float(b.get("start") or b.get("start_s") or b.get("start_time"))
        b_end = _as_float(b.get("end") or b.get("end_s") or b.get("end_time"))
        if b_start is not None and b_end is not None and b_end >= b_start:
            out.append(
                {
                    "sample_id": sample_id,
                    "speaker": speaker,
                    "start_s": b_start,
                    "end_s": b_end,
                    "duration_s": b_end - b_start,
                    "source": "transcribed_json",
                }
            )
            continue

        # Fallback to nested words/tokens with timing.
        nested_lists: list[list[dict[str, Any]]] = []
        for key in ("words", "tokens", "items"):
            v = b.get(key)
            if isinstance(v, list):
                nested_lists.append([x for x in v if isinstance(x, dict)])
        for seq in nested_lists:
            starts = [_as_float(x.get("start") or x.get("start_s") or x.get("start_time")) for x in seq]
            ends = [_as_float(x.get("end") or x.get("end_s") or x.get("end_time")) for x in seq]
            starts = [x for x in starts if x is not None]
            ends = [x for x in ends if x is not None]
            if starts and ends:
                s = min(starts)
                e = max(ends)
                if e >= s:
                    out.append(
                        {
                            "sample_id": sample_id,
                            "speaker": speaker,
                            "start_s": s,
                            "end_s": e,
                            "duration_s": e - s,
                            "source": "transcribed_json",
                        }
                    )
                    break
    # If there are no timestamps in the JSON (common in this repo), fall back to a
    # relative "timeline" computed from turn order + word-count progress.
    if out:
        return out

    # Build relative segments from each (speaker, dialogue) block in order.
    rel: list[dict[str, Any]] = []
    # Use word counts as a proxy for time spent; normalize to [0, 1].
    units: list[tuple[str, int, str]] = []
    for b in blocks:
        speaker = _norm_speaker(b.get("speaker") or b.get("speaker_id") or b.get("speaker_label"))
        dialogue = b.get("dialogue")
        dialogue_text: str | None = None
        if isinstance(dialogue, str):
            dialogue_text = dialogue
        elif isinstance(dialogue, list):
            # In this repo, `dialogue` is commonly a 1-element list of strings.
            parts = [str(x) for x in dialogue if isinstance(x, str) and x.strip()]
            dialogue_text = " ".join(parts) if parts else None
        if not dialogue_text:
            continue
        # Basic tokenization consistent enough for relative progress.
        words = [w for w in dialogue_text.strip().split() if w]
        n = len(words)
        if n <= 0:
            continue
        units.append((speaker, n, dialogue_text))

    if not units:
        return []

    total_units = sum(n for _, n, _ in units) or 1
    cursor = 0.0
    for speaker, n, _dialogue in units:
        dur = n / total_units
        start = cursor
        end = min(1.0, cursor + dur)
        cursor = end
        rel.append(
            {
                "sample_id": sample_id,
                "speaker": speaker,
                "start_s": start,
                "end_s": end,
                "duration_s": max(0.0, end - start),
                "source": "transcribed_json_relative",
            }
        )

    return rel


def load_gold_speaker_timeline(
    dataset_pickle_path: str | Path,
    transcripts_dir: str | Path | None = None,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    payload = _load_pickle(dataset_pickle_path)
    sample_filter = set(sample_ids or [])
    rows: list[dict[str, Any]] = []

    # Try common containers in dataset.pickle.
    candidate_maps: list[dict[str, Any]] = []
    if payload:
        for key in ("transcribed", "speaker_turns", "segments", "turns", "alignment", "metadata"):
            v = payload.get(key)
            if isinstance(v, dict):
                candidate_maps.append(v)
        if not candidate_maps:
            candidate_maps = [payload]

    seen_samples: set[str] = set()
    for mapping in candidate_maps:
        for sid_raw, entry in mapping.items():
            sid = str(sid_raw)
            if sample_filter and sid not in sample_filter:
                continue
            if limit is not None and len(seen_samples) >= limit and sid not in seen_samples:
                continue
            turns = _extract_turns_from_entry(sid, entry)
            if turns:
                rows.extend(turns)
                seen_samples.add(sid)

    # Fallback to transcribed JSON files for missing samples.
    if transcripts_dir:
        base = _resolve_path(transcripts_dir) / "transcribed"
        target_ids: list[str]
        if sample_filter:
            target_ids = sorted(sample_filter)
        else:
            # include known transcript IDs from pickle when available
            t = payload.get("transcripts", {})
            target_ids = sorted(str(k) for k in t.keys()) if isinstance(t, dict) else []
            if not target_ids and base.exists():
                target_ids = sorted(p.stem for p in base.glob("*.json"))
        for sid in target_ids:
            if limit is not None and len(seen_samples) >= limit and sid not in seen_samples:
                continue
            if sid in seen_samples:
                continue
            turns = _extract_turns_from_transcribed_json(sid, base / f"{sid}.json")
            if turns:
                rows.extend(turns)
                seen_samples.add(sid)

    if not rows:
        return pd.DataFrame(columns=["sample_id", "speaker", "start_s", "end_s", "duration_s", "source"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["sample_id", "start_s", "end_s"]).reset_index(drop=True)
    return df


def build_speaker_share_df(timeline_df: pd.DataFrame) -> pd.DataFrame:
    if timeline_df.empty:
        return pd.DataFrame(columns=["sample_id", "speaker", "speaker_duration_s", "sample_duration_s", "speaker_share"])
    g = timeline_df.groupby(["sample_id", "speaker"], as_index=False)["duration_s"].sum()
    g = g.rename(columns={"duration_s": "speaker_duration_s"})
    total = g.groupby("sample_id", as_index=False)["speaker_duration_s"].sum().rename(
        columns={"speaker_duration_s": "sample_duration_s"}
    )
    out = g.merge(total, on="sample_id", how="left")
    out["speaker_share"] = out["speaker_duration_s"] / out["sample_duration_s"].where(out["sample_duration_s"] != 0, 1.0)
    return out


def plot_speaker_timeline_gantt(
    timeline_df: pd.DataFrame,
    sample_order: list[str] | None = None,
    max_samples: int | None = 25,
    figsize: tuple[float, float] = (16, 9),
) -> tuple[plt.Figure, plt.Axes]:
    if timeline_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No speaker timeline data found", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    df = timeline_df.copy()
    if sample_order:
        ordered = [s for s in sample_order if s in set(df["sample_id"])]
    else:
        ordered = list(df["sample_id"].drop_duplicates())
    if max_samples is not None:
        ordered = ordered[:max_samples]
    df = df[df["sample_id"].isin(ordered)].copy()

    speakers = sorted(df["speaker"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {sp: cmap(i % 20) for i, sp in enumerate(speakers)}
    y_map = {sid: i for i, sid in enumerate(ordered)}

    fig, ax = plt.subplots(figsize=figsize)
    for row in df.itertuples(index=False):
        y = y_map[str(row.sample_id)]
        start = float(row.start_s)
        dur = max(0.0, float(row.duration_s))
        ax.broken_barh([(start, dur)], (y - 0.35, 0.7), facecolors=color_map.get(str(row.speaker), "#999999"))

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(ordered, fontsize=8)
    # If we only have relative progress (no timestamps), show a clearer axis label.
    try:
        end_max = float(df["end_s"].max()) if not df.empty else 0.0
    except Exception:
        end_max = 0.0
    if end_max <= 1.05:
        ax.set_xlabel("Timeline position (relative, 0-1)")
    else:
        ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Talk / sample_id")
    ax.set_title("Gold Speaker Timeline (Gantt-style)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    handles = [plt.Line2D([0], [0], color=color_map[sp], lw=6, label=sp) for sp in speakers]
    if handles:
        ax.legend(handles=handles, title="Speaker", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax

