from src.stt.mlx_whisper_service import MLXWhisperService


class FasterWhisperService(MLXWhisperService):
    """Backward-compatible alias to keep existing imports working.

    This project now uses MLX Whisper on Apple Silicon by default.
    """
