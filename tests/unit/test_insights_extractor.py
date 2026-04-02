from __future__ import annotations

from src.insights.extractor import _sanitize_with_evidence


def test_sanitize_with_evidence_drops_unsupported_claims() -> None:
    transcript = "Patient reports feeling low mood and passive suicidal thoughts over two weeks."
    payload = {
        "clinical_presentation": "depressive symptoms",
        "risk_flags": [
            {"claim": "possible suicidality", "evidence_quote": "passive suicidal thoughts"},
            {"claim": "homicidal intent", "evidence_quote": "plans to harm others"},
        ],
        "symptoms": ["low mood"],
        "diagnostic_hypotheses": [
            {"claim": "major depressive episode", "evidence_quote": "feeling low mood"},
        ],
        "recommended_followup": [
            {"claim": "urgent safety assessment", "evidence_quote": "passive suicidal thoughts"},
        ],
        "confidence": 0.9,
    }

    out = _sanitize_with_evidence(payload, transcript=transcript)
    assert out["risk_flags"] == ["possible suicidality"]
    assert out["diagnostic_hypotheses"] == ["major depressive episode"]
    assert out["recommended_followup"] == ["urgent safety assessment"]
    assert out["guardrail"]["dropped_unsupported_claims"] == 1
    assert out["evidence"]["risk_flags"][0]["evidence_quote"] == "passive suicidal thoughts"

