from anthropic import Anthropic

from src.config.settings import get_settings


class AnthropicClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.anthropic_model
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content:
            return response.content[0].text
        return ""
