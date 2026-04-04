"""LLM client integrations used by the pipeline."""

from typing import Any, Literal

from ollama import Client

from src.config.settings import get_settings


class OllamaClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.client = Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_s)
    # CR: A better design is to have generate accept system_prompt and then user_content and not make the callers concatenate it themselves,
    # CR: Also it will allow you to send the system prompt only once and not for every call
    # CR: You can also send them with different user roles so that they have different meaning for the model
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        *,
        response_format: Literal["json"] | None = None,
    ) -> str:
        fmt: str | None = "json" if response_format == "json" else None
        response = self.client.chat(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options or {},
            format=fmt,
        )
        return response["message"]["content"]
