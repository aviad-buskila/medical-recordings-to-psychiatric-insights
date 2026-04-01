import logging

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.ingestion.dataset_loader import DatasetLoader
from src.stt.faster_whisper_service import FasterWhisperService

logger = logging.getLogger(__name__)


def run_stt_pipeline() -> None:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()
    stt = FasterWhisperService()
    analytics = AnalyticsRepository()
    for sample in samples:
        if sample.recording_path is None:
            logger.warning("Skipping sample with no recording: %s", sample.sample_id)
            continue
        payload = stt.transcribe(sample.recording_path)
        analytics.insert_stt_run(sample_id=sample.sample_id, provider="faster-whisper", payload=payload)
