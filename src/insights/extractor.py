from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


@dataclass
class TranscriptInsightsExtractor:
    model_name: str
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        self.client = OllamaClient()

    def _prompt(self, transcript: str) -> str:
        return (
            "You are a clinical psychiatry assistant.\n"
            "Read the transcript and extract psychiatry-focused insights.\n"
            "Return ONLY valid JSON with keys:\n"
            "- clinical_presentation: string\n"
            "- risk_flags: list[string]\n"
            "- symptoms: list[string]\n"
            "- diagnostic_hypotheses: list[string]\n"
            "- recommended_followup: list[string]\n"
            "- confidence: number (0.0 to 1.0)\n"
            "Do not include markdown. Keep items concise and clinically relevant.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

    def extract(self, transcript: str) -> tuple[dict[str, Any], str]:
        raw = self.client.generate(prompt=self._prompt(transcript), model=self.model_name)
        parsed = _safe_json_loads(raw)
        return parsed, raw


def run_insights_extract(
    run_id: str,
    sample_id: str | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    output_json: Path | None = None,
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

    results: list[dict[str, Any]] = []
    for sid in sample_ids:
        transcript = run_outputs.get(sid, "").strip()
        if not transcript:
            continue
        parsed, raw = extractor.extract(transcript)
        row = {
            "run_id": run_id,
            "sample_id": sid,
            "insight_model": resolved_model,
            "prompt_version": extractor.prompt_version,
            "insights": parsed,
            "raw_output": raw,
            "transcript_preview": transcript[:500],
        }
        results.append(row)
        analytics.upsert_transcript_insight(
            run_id=run_id,
            sample_id=sid,
            insight_model=resolved_model,
            prompt_version=extractor.prompt_version,
            insights=parsed,
            raw_output=raw,
        )
        logger.info("Extracted psychiatry insights sample_id=%s model=%s", sid, resolved_model)

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
        "samples_processed": len(results),
        "results": results,
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Wrote insights extract artifact: %s", output_path)
    return output_path

