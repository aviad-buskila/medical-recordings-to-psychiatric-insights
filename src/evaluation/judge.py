import json
import re

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

    def evaluate_transcript(self, reference_text: str, predicted_text: str) -> dict:
        prompt = (
            "You are a strict clinical transcription judge.\n"
            "Compare PREDICTED transcript against REFERENCE transcript.\n"
            "Return only JSON with keys:\n"
            "- overall_score (1-10)\n"
            "- deletion_error_severity (1-10)\n"
            "- insertion_error_severity (1-10)\n"
            "- substitution_error_severity (1-10)\n"
            "- medical_safety_risk (1-10)\n"
            "- rationale (string)\n"
            "Scoring: higher overall_score is better; higher severities/risks are worse.\n\n"
            f"REFERENCE:\n{reference_text}\n\nPREDICTED:\n{predicted_text}"
        )
        response = self.client.generate(prompt=prompt, model=self.model)
        return self._safe_parse(response)

    def compare_transcripts(self, reference_text: str, candidate_text: str, baseline_text: str) -> dict:
        prompt = (
            "You are a strict clinical transcription judge.\n"
            "Compare two candidate transcripts against the REFERENCE.\n"
            "Return only JSON with keys:\n"
            "- winner (\"candidate\" or \"baseline\" or \"tie\")\n"
            "- candidate_overall_score (1-10)\n"
            "- baseline_overall_score (1-10)\n"
            "- score_delta (candidate_minus_baseline)\n"
            "- rationale (string)\n\n"
            f"REFERENCE:\n{reference_text}\n\n"
            f"CANDIDATE:\n{candidate_text}\n\n"
            f"BASELINE:\n{baseline_text}"
        )
        response = self.client.generate(prompt=prompt, model=self.model)
        return self._safe_parse(response)

    @staticmethod
    def _safe_parse(response: str) -> dict:
        def _coerce_compare_shape(payload: dict) -> dict:
            payload = dict(payload)
            candidate_score = float(payload.get("candidate_overall_score", 0) or 0)
            baseline_score = float(payload.get("baseline_overall_score", 0) or 0)
            delta = candidate_score - baseline_score
            payload["candidate_overall_score"] = candidate_score
            payload["baseline_overall_score"] = baseline_score
            payload["score_delta"] = delta

            winner = str(payload.get("winner", "")).strip().lower()
            if winner not in {"candidate", "baseline", "tie"}:
                if delta > 0:
                    winner = "candidate"
                elif delta < 0:
                    winner = "baseline"
                else:
                    winner = "tie"
            payload["winner"] = winner
            payload.setdefault("rationale", "")
            return payload

        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return _coerce_compare_shape(parsed)
        except json.JSONDecodeError:
            parsed = None

        # Try extracting JSON from markdown/code-fenced or mixed output.
        json_candidates = re.findall(r"\{[\s\S]*\}", response)
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return _coerce_compare_shape(parsed)
            except json.JSONDecodeError:
                continue

        return {
            "winner": "unknown",
            "candidate_overall_score": 0,
            "baseline_overall_score": 0,
            "score_delta": 0.0,
            "rationale": "Judge output was not valid JSON.",
            "raw_output": response,
        }
