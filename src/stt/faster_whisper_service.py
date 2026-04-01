import time
from pathlib import Path

from faster_whisper import WhisperModel

from src.config.settings import get_settings


class FasterWhisperService:
    """Local STT wrapper around faster-whisper."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = WhisperModel(
            settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )

    def transcribe(self, recording_path: Path) -> dict[str, str | float]:
        started_at = time.perf_counter()
        segments, info = self.model.transcribe(str(recording_path), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_s = time.perf_counter() - started_at
        return {
            "text": text,
            "language": getattr(info, "language", "unknown"),
            "duration_s": float(getattr(info, "duration", 0.0)),
            "elapsed_s": elapsed_s,
        }
