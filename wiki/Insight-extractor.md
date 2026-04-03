# Insight extractor

[Wiki home](Home) · [Gemma and MedGemma](Gemma-and-MedGemma) · [Pipeline and other components](Pipeline-and-other-components)

---

## Role

Turns a **single STT transcript** (from a chosen `run_id`) into **structured psychiatry-oriented fields**: presentation, symptoms, risk flags, hypotheses, follow-up, plus a **confidence** scalar. Runs **locally** through Ollama using the insights model (default **MedGemma 1.5 4B** — see [Gemma and MedGemma](Gemma-and-MedGemma)).

---

## Configuration

| Env / setting | Meaning |
|---------------|---------|
| `OLLAMA_INSIGHTS_MODEL` | Model id (default `medaibase/medgemma1.5:4b`) |
| `OLLAMA_INSIGHTS_MAX_TOKENS` | Bound on generated tokens (`num_predict` in Ollama options); `.env.example` uses `4000`; Pydantic default in code is `384` unless overridden |
| `OLLAMA_TIMEOUT_S` | Client timeout |
| `INSIGHTS_EXTRACT_DIR` | Where JSON artifacts are written |

CLI: `insights-extract --run-id <UUID> [--limit N] [--model ...]`.

---

## Prompt contract (`prompt_version` v2)

The model is instructed to return **only JSON** with:

- `clinical_presentation` — string  
- `risk_flags` — list of `{ claim, evidence_quote }`  
- `symptoms` — list of strings  
- `diagnostic_hypotheses` — list of `{ claim, evidence_quote }`  
- `recommended_followup` — list of `{ claim, evidence_quote }`  
- `confidence` — number in \([0,1]\)

Temperature is kept low (~0.1) for stability; generation length is capped via settings.

Source: `TranscriptInsightsExtractor._prompt` in `src/insights/extractor.py`.

---

## Evidence guardrail (post-processing)

`_sanitize_with_evidence` **drops** list items that do not include an **evidence quote** that appears (normalized) in the transcript text for:

- `risk_flags`
- `diagnostic_hypotheses`
- `recommended_followup`

Plain-string list items without quotes are not accepted for those keys. The output includes `guardrail.dropped_unsupported_claims` and structured `evidence` for what survived.

This reduces **hallucinated clinical claims** but is not a formal safety guarantee.

---

## Persistence

- Rows go to PostgreSQL `transcript_insights` (schema `clinical_ai`), including raw model output and metadata for audit.
- Optional JSON artifacts under `data/processed/insights_extract/`.

---

## See also

- Notebook `analysis/insights_quality.ipynb` — guardrail rates, schema checks, distributions.
- [LLM as judge](LLM-as-judge) — same Ollama stack, different task.
