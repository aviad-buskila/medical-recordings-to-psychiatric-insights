import time
from pathlib import Path

from faster_whisper import WhisperModel

from src.config.settings import get_settings


class FasterWhisperService:
    """Higher-accuracy local STT wrapper using faster-whisper."""

    def __init__(self, model_name: str | None = None, compute_type: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.stt_quality_model
        requested_compute_type = compute_type or settings.stt_quality_compute_type
        self.compute_type = requested_compute_type
        self.model = self._init_model_with_fallbacks(requested_compute_type)

    def transcribe(self, recording_path: Path) -> dict[str, str | float]:
        started_at = time.perf_counter()
        segments, info = self.model.transcribe(str(recording_path), beam_size=5)
        elapsed_s = time.perf_counter() - started_at
        text = " ".join(s.text.strip() for s in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", "unknown"),
            "duration_s": float(getattr(info, "duration", 0.0)),
            "elapsed_s": elapsed_s,
            "model_name": self.model_name,
        }

    def _init_model_with_fallbacks(self, requested_compute_type: str) -> WhisperModel:
        compute_type_candidates = [requested_compute_type]
        for candidate in ("float16", "int8", "float32"):
            if candidate not in compute_type_candidates:
                compute_type_candidates.append(candidate)

        last_exc: Exception | None = None
        for candidate in compute_type_candidates:
            try:
                model = WhisperModel(self.model_name, device="auto", compute_type=candidate)
                self.compute_type = candidate
                return model
            except ValueError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unable to initialize faster-whisper model.")

    @staticmethod
    def is_invalid_audio_error(exc: Exception) -> bool:
        current: BaseException | None = exc
        while current is not None:
            message = str(current).lower()
            if "invalid data found when processing input" in message:
                return True
            if "failed to load audio" in message:
                return True
            if "unsupported format" in message:
                return True
            current = current.__cause__
        return False
