"""Speech-to-text pipeline utilities for pipeline."""

import logging
from datetime import datetime
from pathlib import Path

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.ingestion.dataset_loader import DatasetLoader
from src.stt.mlx_whisper_service import MLXWhisperService

logger = logging.getLogger(__name__)


def run_stt_pipeline(
    limit: int | None = None,
    stt_profile: str = "default",
    selected_sample_ids: list[str] | None = None,
    allow_fallback: bool = True,
) -> str:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()
    if selected_sample_ids:
        selected_set = set(selected_sample_ids)
        samples = [s for s in samples if s.sample_id in selected_set]
    total_candidates = sum(1 for s in samples if s.recording_path is not None)
    stt, provider_name = _resolve_stt_engine(stt_profile, allow_fallback=allow_fallback)
    analytics = AnalyticsRepository()
    fallback_model = getattr(stt, "fallback_model_name", None)
    compute_type = getattr(stt, "compute_type", None)
    run_parameters = {
        "limit": limit,
        "total_candidates_with_audio": total_candidates,
        "fallback_model": fallback_model,
        "compute_type": compute_type,
        "profile": stt_profile,
        "allow_fallback": allow_fallback,
    }
    run_scope = "sample" if limit is not None else "full"
    run_id, run_timestamp = analytics.create_stt_run(
        provider=provider_name,
        model_name=stt.model_name,
        run_scope=run_scope,
        run_parameters=run_parameters,
        run_timestamp=datetime.utcnow(),
    )
    run_dir = _prepare_run_output_dir(
        base_dir=Path(settings.generated_transcripts_dir),
        run_id=run_id,
        run_timestamp=run_timestamp,
    )
    processed = 0
    skipped_invalid_audio = 0
    failed_other = 0
    seen_with_audio = 0
    logger.info(
        "Starting STT pipeline. run_id=%s model=%s scope=%s candidates_with_audio=%s target_limit=%s",
        run_id,
        stt.model_name,
        run_scope,
        total_candidates,
        "all" if limit is None else limit,
    )
    for idx, sample in enumerate(samples, start=1):
        if limit is not None and processed >= limit:
            break
        if sample.recording_path is None:
            logger.warning("Skipping sample with no recording: %s", sample.sample_id)
            continue
        seen_with_audio += 1
        logger.info(
            "STT progress: sample=%s index=%s/%s successful=%s skipped_invalid=%s failed=%s",
            sample.sample_id,
            seen_with_audio,
            total_candidates,
            processed,
            skipped_invalid_audio,
            failed_other,
        )
        try:
            payload = stt.transcribe(sample.recording_path)
            model_used = str(payload.get("model_name", stt.model_name))
            analytics.insert_stt_output(
                run_id=run_id,
                run_timestamp=run_timestamp,
                model_name=model_used,
                sample_id=sample.sample_id,
                provider=provider_name,
                payload=payload,
            )
            _write_transcript_file(run_dir=run_dir, sample_id=sample.sample_id, transcript=payload.get("text", ""))
            processed += 1
            logger.info(
                "STT success: sample=%s successful=%s/%s",
                sample.sample_id,
                processed,
                "all" if limit is None else limit,
            )
        except Exception as exc:
            if stt.is_invalid_audio_error(exc):
                skipped_invalid_audio += 1
                logger.warning(
                    "Skipping invalid/corrupt audio for sample %s: %s",
                    sample.sample_id,
                    sample.recording_path,
                )
                continue
            failed_other += 1
            logger.exception("STT failed for sample %s", sample.sample_id)
            if limit is not None:
                # In sampled mode, continue scanning so limit still means successful transcriptions.
                continue
            raise
    logger.info(
        "STT pipeline finished. processed=%s skipped_invalid_audio=%s failed_other=%s",
        processed,
        skipped_invalid_audio,
        failed_other,
    )
    return run_id


def run_stt_both_profiles(limit: int | None = None, allow_fallback: bool = True) -> dict[str, str]:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    all_samples = loader.load_samples()
    with_audio = [s for s in all_samples if s.recording_path is not None]
    target_samples = with_audio[:limit] if limit is not None else with_audio
    selected_sample_ids = [s.sample_id for s in target_samples]
    logger.info(
        "Running both STT profiles on the same files. selected_samples=%s",
        len(selected_sample_ids),
    )
    default_run_id = run_stt_pipeline(
        limit=limit,
        stt_profile="default",
        selected_sample_ids=selected_sample_ids,
        allow_fallback=allow_fallback,
    )
    quality_run_id = run_stt_pipeline(
        limit=limit,
        stt_profile="quality",
        selected_sample_ids=selected_sample_ids,
        allow_fallback=allow_fallback,
    )
    return {"default": default_run_id, "quality": quality_run_id}


def _resolve_stt_engine(stt_profile: str, allow_fallback: bool = True):
    settings = get_settings()
    profile = stt_profile.lower().strip()
    if profile == "quality":
        return (
            MLXWhisperService(
                model_name=settings.stt_mlx_quality_model,
                fallback_model_name=settings.stt_mlx_quality_fallback_model,
                enable_fallback=allow_fallback,
            ),
            settings.stt_provider,
        )
    return MLXWhisperService(enable_fallback=allow_fallback), settings.stt_provider


def _prepare_run_output_dir(base_dir: Path, run_id: str, run_timestamp: datetime) -> Path:
    timestamp_str = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"{run_id}_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_transcript_file(run_dir: Path, sample_id: str, transcript: str | object) -> None:
    safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
    out_path = run_dir / f"{safe_sample_id}.txt"
    out_path.write_text(str(transcript or ""), encoding="utf-8")
