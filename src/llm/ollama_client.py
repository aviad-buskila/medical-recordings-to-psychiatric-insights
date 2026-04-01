from ollama import Client

from src.config.settings import get_settings


class OllamaClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.client = Client(host=settings.ollama_base_url)

    def generate(self, prompt: str, model: str | None = None) -> str:
        response = self.client.chat(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
