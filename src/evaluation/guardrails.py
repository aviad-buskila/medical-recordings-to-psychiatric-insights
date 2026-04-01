from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    reasons: list[str]


def apply_guardrails(judge_result: dict) -> GuardrailResult:
    reasons: list[str] = []
    if judge_result.get("accuracy_score", 0) < 3:
        reasons.append("accuracy_score below threshold")
    if judge_result.get("grounding_score", 0) < 3:
        reasons.append("grounding_score below threshold")
    if judge_result.get("hallucination_risk", 5) > 2:
        reasons.append("hallucination_risk above threshold")
    return GuardrailResult(passed=len(reasons) == 0, reasons=reasons)
