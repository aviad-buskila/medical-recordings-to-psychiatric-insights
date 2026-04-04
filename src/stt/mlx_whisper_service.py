"""Speech-to-text pipeline utilities for mlx whisper service."""

import os
import shutil
import time
import logging
from pathlib import Path
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# CR: Code is coupled to apple sillicon, no abstraction if wanted to support other platforms
class MLXWhisperService:
    """Local STT wrapper using Apple's MLX Whisper backend."""

    def __init__(
        self,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
        enable_fallback: bool = True,
    ) -> None:
        settings = get_settings()
        self.model_name = self._resolve_model_alias(model_name or settings.stt_model)
        fallback = self._resolve_model_alias(fallback_model_name or settings.stt_model_fallback)
        self.fallback_model_name = fallback if enable_fallback else None
        self.enable_fallback = enable_fallback
        if settings.hf_token:
            # Ensure huggingface_hub can use token even when only .env is configured.
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)

    # CR: this blocks caller thread for all duration. Not an issue for the current implementation, but becomes a bottleneck if you want to overlap transscrive with llm call/evaluation
    # CR: also, you load the model each time, maybe we can cache it? depends on the usecase
    def transcribe(self, recording_path: Path) -> dict[str, str | float]:
        started_at = time.perf_counter()
        result, model_used = self._run_transcription(recording_path)
        elapsed_s = time.perf_counter() - started_at

        text = str(result.get("text", "")).strip()
        language = str(result.get("language", "unknown"))
        duration_s = float(result.get("duration_s", 0.0))

        return {
            "text": text,
            "language": language,
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "model_name": model_used,
        }

    def _run_transcription(self, recording_path: Path) -> tuple[dict[str, Any], str]:
        # Import lazily to avoid import-time failure when dependency is missing.
        import mlx_whisper
        self._ensure_ffmpeg_available()

        try:
            raw = self._transcribe_with_model(mlx_whisper, recording_path, self.model_name)
            model_used = self.model_name
        except Exception as exc:
            if self.is_invalid_audio_error(exc):
                raise RuntimeError(
                    f"Invalid or unreadable audio file: {recording_path}"
                ) from exc
            if self.fallback_model_name and self.fallback_model_name != self.model_name:
                try:
                    raw = self._transcribe_with_model(
                        mlx_whisper,
                        recording_path,
                        self.fallback_model_name,
                    )
                    model_used = self.fallback_model_name
                    logger.warning(
                        "Primary MLX model failed (%s). Using fallback model (%s).",
                        self.model_name,
                        self.fallback_model_name,
                    )
                except Exception as fallback_exc:
                    if self.is_invalid_audio_error(fallback_exc):
                        raise RuntimeError(
                            f"Invalid or unreadable audio file: {recording_path}"
                        ) from fallback_exc
                    if self._is_ffmpeg_error(fallback_exc):
                        raise RuntimeError(
                            "ffmpeg is required by mlx-whisper but was not found. "
                            "Install it (e.g. `brew install ffmpeg`) and retry."
                        ) from fallback_exc
                    raise RuntimeError(
                        "MLX Whisper failed for primary and fallback models. "
                        "Check STT_MODEL / STT_MODEL_FALLBACK and HF token auth."
                    ) from fallback_exc
            else:
                if self._is_ffmpeg_error(exc):
                    raise RuntimeError(
                        "ffmpeg is required by mlx-whisper but was not found. "
                        "Install it (e.g. `brew install ffmpeg`) and retry."
                    ) from exc
                raise RuntimeError(
                    "MLX Whisper failed to load/transcribe. Check STT_MODEL and HF auth."
                ) from exc
        if isinstance(raw, dict):
            return raw, model_used
        return {"text": str(raw)}, model_used

    @staticmethod
    def _resolve_model_alias(model_name: str) -> str:
        aliases = {
            "mlx-community/whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
        }
        return aliases.get(model_name, model_name)

    @staticmethod
    def _transcribe_with_model(mlx_whisper: Any, recording_path: Path, model_name: str) -> Any:
        return mlx_whisper.transcribe(
            str(recording_path),
            path_or_hf_repo=model_name,
        )

    @staticmethod
    def _ensure_ffmpeg_available() -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required by mlx-whisper but is not installed. "
                "Install with `brew install ffmpeg`."
            )

    @staticmethod
    def _is_ffmpeg_error(exc: Exception) -> bool:
        return isinstance(exc, FileNotFoundError) and "ffmpeg" in str(exc).lower()

    @staticmethod
    def is_invalid_audio_error(exc: Exception) -> bool:
        current: BaseException | None = exc
        while current is not None:
            message = str(current).lower()
            if "invalid or unreadable audio file" in message:
                return True
            if "failed to load audio" in message:
                return True
            if "invalid data found when processing input" in message:
                return True
            current = current.__cause__
        return False
