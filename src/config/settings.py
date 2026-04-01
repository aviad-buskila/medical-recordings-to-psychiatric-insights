from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env with strict typing."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    postgres_host: str = Field(alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    postgres_schema: str = Field(default="clinical_ai", alias="POSTGRES_SCHEMA")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.2", alias="OPENAI_MODEL")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gemma3:12b", alias="OLLAMA_MODEL")
    ollama_judge_model: str = Field(default="gemma3:12b", alias="OLLAMA_JUDGE_MODEL")
    hf_token: str = Field(default="", alias="HF_TOKEN")

    stt_provider: str = Field(default="mlx-whisper", alias="STT_PROVIDER")
    stt_model: str = Field(default="mlx-community/whisper-large-v3-turbo", alias="STT_MODEL")
    stt_model_fallback: str = Field(
        default="mlx-community/whisper-large-v3-turbo",
        alias="STT_MODEL_FALLBACK",
    )
    stt_mlx_quality_model: str = Field(
        default="mlx-community/whisper-large-v3-mlx",
        alias="STT_MLX_QUALITY_MODEL",
    )
    stt_mlx_quality_fallback_model: str = Field(
        default="mlx-community/whisper-large-v3-turbo",
        alias="STT_MLX_QUALITY_FALLBACK_MODEL",
    )
    stt_quality_model: str = Field(default="large-v3", alias="STT_QUALITY_MODEL")
    stt_quality_compute_type: str = Field(default="int8_float16", alias="STT_QUALITY_COMPUTE_TYPE")

    data_root: str = Field(default="./data/raw", alias="DATA_ROOT")
    transcripts_dir: str = Field(default="./data/raw/transcripts", alias="TRANSCRIPTS_DIR")
    casenotes_dir: str = Field(default="./data/raw/casenotes", alias="CASENOTES_DIR")
    recordings_dir: str = Field(default="./data/raw/recordings", alias="RECORDINGS_DIR")
    dataset_pickle_path: str = Field(default="./data/raw/dataset.pickle", alias="DATASET_PICKLE_PATH")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
