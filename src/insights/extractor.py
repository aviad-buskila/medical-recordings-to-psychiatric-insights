"""Insight extraction logic for psychiatric outputs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Some model responses wrap JSON with extra text; recover first valid object.
    candidates = re.findall(r"\{[\s\S]*\}", raw)
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return {
        "clinical_presentation": "",
        "risk_flags": [],
        "symptoms": [],
        "diagnostic_hypotheses": [],
        "recommended_followup": [],
        "confidence": None,
        "parse_error": "Model output was not valid JSON.",
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat() if obj.tzinfo else obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _quote_supported(quote: str, transcript: str) -> bool:
    q = _norm_text(quote)
    t = _norm_text(transcript)
    if len(q) < 8:
        return False
    return q in t


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def _sanitize_with_evidence(payload: dict[str, Any], transcript: str) -> dict[str, Any]:
    """
    Content guardrail:
    Keep only claims that include transcript-grounded evidence quotes.
    """
    out = dict(payload)
    dropped = 0
    evidence: dict[str, list[dict[str, str]]] = {
        "risk_flags": [],
        "diagnostic_hypotheses": [],
        "recommended_followup": [],
    }
    for key in ("risk_flags", "diagnostic_hypotheses", "recommended_followup"):
        kept_claims: list[str] = []
        raw_items = payload.get(key, [])
        if not isinstance(raw_items, list):
            out[key] = []
            continue
        for raw in raw_items:
            claim = ""
            quote = ""
            if isinstance(raw, dict):
                claim = str(raw.get("claim", "") or raw.get("item", "") or raw.get("text", "")).strip()
                quote = str(raw.get("evidence_quote", "")).strip()
            elif isinstance(raw, str):
                # Backward compatibility: plain strings are unsupported under this guardrail.
                claim = raw.strip()
                quote = ""
            if not claim:
                continue
            # Only keep claims backed by a quote present in transcript text.
            if _quote_supported(quote, transcript):
                kept_claims.append(claim)
                evidence[key].append({"claim": claim, "evidence_quote": quote})
            else:
                dropped += 1
        out[key] = kept_claims

    out["symptoms"] = _coerce_str_list(payload.get("symptoms"))
    out["clinical_presentation"] = str(payload.get("clinical_presentation", "") or "").strip()
    try:
        conf = payload.get("confidence", None)
        out["confidence"] = None if conf is None else float(conf)
    except (TypeError, ValueError):
        out["confidence"] = None
    if isinstance(out.get("confidence"), float):
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    out["evidence"] = evidence
    out["guardrail"] = {
        "name": "evidence_quote_required_for_claims",
        "dropped_unsupported_claims": dropped,
    }
    return out


@dataclass
class TranscriptInsightsExtractor:
    model_name: str
    prompt_version: str = "v2"

    def __post_init__(self) -> None:
        self.client = OllamaClient()
        self.settings = get_settings()

    def _prompt(self, transcript: str) -> str:
        return (
            "You are a clinical psychiatry assistant.\n"
            "Read the transcript and extract psychiatry-focused insights.\n"
            "Return ONLY valid JSON with keys:\n"
            "- clinical_presentation: string\n"
            "- risk_flags: list[object] where each object has: claim, evidence_quote\n"
            "- symptoms: list[string]\n"
            "- diagnostic_hypotheses: list[object] where each object has: claim, evidence_quote\n"
            "- recommended_followup: list[object] where each object has: claim, evidence_quote\n"
            "- confidence: number (0.0 to 1.0)\n"
            "Critical guardrail: for risk_flags, diagnostic_hypotheses, and recommended_followup,\n"
            "ONLY include claims supported by an exact quote from TRANSCRIPT.\n"
            "If no supporting quote exists, omit the claim.\n"
            "Do not include markdown. Keep items concise and clinically relevant.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

    def extract(self, transcript: str) -> tuple[dict[str, Any], str]:
        options = {
            # Bound generation length for deterministic latency.
            "num_predict": int(self.settings.ollama_insights_max_tokens),
            "temperature": 0.1,
        }
        try:
            raw = self.client.generate(
                prompt=self._prompt(transcript),
                model=self.model_name,
                options=options,
                response_format="json",
            )
        except Exception:
            # One retry for transient Ollama stalls/timeouts.
            raw = self.client.generate(
                prompt=self._prompt(transcript),
                model=self.model_name,
                options=options,
                response_format="json",
            )
        parsed = _safe_json_loads(raw)
        return _sanitize_with_evidence(parsed, transcript=transcript), raw


def run_insights_extract(
    run_id: str,
    sample_id: str | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    output_json: Path | None = None,
    skip_existing: bool = False,
) -> Path:
    settings = get_settings()
    analytics = AnalyticsRepository()
    run_info = analytics.get_stt_run_info(run_id)
    if not run_info:
        raise ValueError(f"run_id not found in stt_runs: {run_id}")

    resolved_model = model_name or settings.ollama_insights_model
    extractor = TranscriptInsightsExtractor(model_name=resolved_model)

    run_outputs = analytics.get_stt_outputs_for_run(run_id)
    sample_ids = sorted(run_outputs.keys())
    if sample_id:
        sample_ids = [sid for sid in sample_ids if sid == sample_id]
    if limit is not None:
        sample_ids = sample_ids[:limit]

    skipped_existing = 0
    if skip_existing and sample_ids:
        # Resume helper: avoid recomputing rows already stored for same run/model/prompt.
        existing_ids = analytics.get_existing_transcript_insight_sample_ids(
            run_id=run_id,
            insight_model=resolved_model,
            prompt_version=extractor.prompt_version,
            sample_ids=sample_ids,
        )
        if existing_ids:
            skipped_existing = len(existing_ids)
            sample_ids = [sid for sid in sample_ids if sid not in existing_ids]
            logger.info(
                "Skipping existing insights rows run_id=%s model=%s prompt_version=%s skipped=%s remaining=%s",
                run_id,
                resolved_model,
                extractor.prompt_version,
                skipped_existing,
                len(sample_ids),
            )

    results: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for sid in sample_ids:
        transcript = run_outputs.get(sid, "").strip()
        if not transcript:
            continue
        transcript_char_count = len(transcript)
        transcript_word_count = len([w for w in transcript.split() if w])
        sample_started = time.perf_counter()
        parsed, raw = extractor.extract(transcript)
        # Keep both parsed structure and raw model output for audit/debugging.
        insights_payload = {**parsed, "prompt_version": extractor.prompt_version}
        elapsed_s = time.perf_counter() - sample_started
        row = {
            "run_id": run_id,
            "sample_id": sid,
            "insight_model": resolved_model,
            "prompt_version": extractor.prompt_version,
            "insights": insights_payload,
            "raw_output": raw,
            "transcript_preview": transcript[:500],
            "timing": {
                "elapsed_s": round(elapsed_s, 3),
                "transcript_char_count": transcript_char_count,
                "transcript_word_count": transcript_word_count,
            },
        }
        results.append(row)
        analytics.upsert_transcript_insight(
            run_id=run_id,
            sample_id=sid,
            insight_model=resolved_model,
            prompt_version=extractor.prompt_version,
            insights=insights_payload,
            raw_output=raw,
        )
        logger.info(
            "Extracted psychiatry insights sample_id=%s model=%s elapsed_s=%.3f words=%s chars=%s",
            sid,
            resolved_model,
            elapsed_s,
            transcript_word_count,
            transcript_char_count,
        )

    ts = _ts_utc()
    out_dir = Path(settings.insights_extract_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_json or out_dir / f"insights_extract_{run_id}_{ts}.json"
    artifact = {
        "command": "insights_extract",
        "run_id": run_id,
        "timestamp_utc": ts,
        "insight_model": resolved_model,
        "prompt_version": extractor.prompt_version,
        "stt_run_info": run_info,
        "sample_filter": sample_id,
        "limit": limit,
        "skip_existing": skip_existing,
        "samples_skipped_existing": skipped_existing,
        "samples_processed": len(results),
        "timing": {
            "total_elapsed_s": round(time.perf_counter() - run_started, 3),
            "avg_elapsed_s_per_sample": round(
                (sum(float(r.get("timing", {}).get("elapsed_s", 0.0)) for r in results) / len(results)) if results else 0.0,
                3,
            ),
        },
        "results": results,
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Wrote insights extract artifact: %s", output_path)
    return output_path

