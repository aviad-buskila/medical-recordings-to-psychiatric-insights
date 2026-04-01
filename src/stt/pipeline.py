import logging

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.ingestion.dataset_loader import DatasetLoader
from src.stt.mlx_whisper_service import MLXWhisperService

logger = logging.getLogger(__name__)


def run_stt_pipeline(limit: int | None = None) -> None:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()
    stt = MLXWhisperService()
    analytics = AnalyticsRepository()
    processed = 0
    for sample in samples:
        if limit is not None and processed >= limit:
            break
        if sample.recording_path is None:
            logger.warning("Skipping sample with no recording: %s", sample.sample_id)
            continue
        payload = stt.transcribe(sample.recording_path)
        analytics.insert_stt_run(sample_id=sample.sample_id, provider="mlx-whisper", payload=payload)
        processed += 1
    logger.info("STT pipeline finished. Processed recordings: %s", processed)
