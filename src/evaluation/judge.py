import json

from src.config.settings import get_settings
from src.llm.ollama_client import OllamaClient


class OllamaJudge:
    """Local LLM-as-a-judge rubric for medical grounding and hallucination risk."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_judge_model
        self.client = OllamaClient()

    def evaluate(self, generated_summary: str, retrieved_context: str) -> dict:
        prompt = (
            "You are a strict medical QA judge.\n"
            "Return JSON with keys: accuracy_score, grounding_score, hallucination_risk, rationale.\n"
            "Scores are 1-5 where higher is better except hallucination_risk where lower is better.\n\n"
            f"Summary:\n{generated_summary}\n\nContext:\n{retrieved_context}"
        )
        response = self.client.generate(prompt=prompt, model=self.model)
        # TODO: add robust schema validation and retry parsing.
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "accuracy_score": 1,
                "grounding_score": 1,
                "hallucination_risk": 5,
                "rationale": "Judge output was not valid JSON.",
            }
