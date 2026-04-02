from typing import Any

from ollama import Client

from src.config.settings import get_settings


class OllamaClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.client = Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_s)

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        response = self.client.chat(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options or {},
        )
        return response["message"]["content"]
