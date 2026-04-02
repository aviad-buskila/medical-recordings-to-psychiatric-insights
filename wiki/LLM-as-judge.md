# LLM as judge

[Wiki home](Home) · [Evaluation metrics](Evaluation-metrics) · [Gemma and MedGemma](Gemma-and-MedGemma)

---

## Purpose

The judge is a **local** LLM (via **Ollama**) that scores or **compares** transcripts when a gold reference exists. It is **not** the same as BERTScore: it uses natural-language rubrics and JSON outputs, and should be treated as **directional** — useful for spotting clinically risky omissions or insertions, not as a calibrated production metric.

---

## Configuration

| Env variable | Role |
|--------------|------|
| `OLLAMA_BASE_URL` | Ollama HTTP API (default `http://localhost:11434`) |
| `OLLAMA_JUDGE_MODEL` | Model id for judge calls (default **`gemma3:12b`**) |
| `OLLAMA_TIMEOUT_S` | Request timeout |

Pull the model in Ollama before running, e.g. `ollama pull gemma3:12b`.

---

## Code paths

| Class / module | Behavior |
|----------------|----------|
| `src/evaluation/judge.py` — `OllamaJudge` | Builds prompts, calls `OllamaClient`, parses JSON (with fallbacks if the model returns non-JSON). |
| `src/evaluation/llm_judge_eval.py` — `run_llm_judge_eval` | Loads gold + STT outputs from DB, runs **pairwise** comparison when `ref_run_id` is set (candidate vs baseline vs reference). |
| `src/llm/ollama_client.py` | Thin wrapper: `ollama` Python client, `chat` with optional `format: json`. |

---

## Prompt shapes (summary)

1. **Summary vs context** (`evaluate`) — legacy medical QA style: `accuracy_score`, `grounding_score`, `hallucination_risk`, `rationale` (1–5 scales in prompt).
2. **Single transcript vs reference** (`evaluate_transcript`) — `overall_score`, deletion/insertion/substitution severities, `medical_safety_risk`, `rationale`.
3. **Head-to-head** (`compare_transcripts`) — used for benchmarking two STT runs: `winner` (`candidate` | `baseline` | `tie`), per-arm scores, `score_delta` (candidate minus baseline), `rationale`.

The comparative path aligns with `run-llm-judge --run-id <candidate> --ref-run-id <baseline>`.

---

## Interpreting outputs

- **Score scale:** Prompts ask for roughly **1–10** overall quality where higher is better for “good transcript”; severity fields are “higher = worse” where applicable.
- **Winner / delta:** `score_delta` is **candidate minus baseline**; negative values favor the baseline in the default coercion logic.
- **Failures:** If JSON parsing fails, the code returns safe defaults and may increment parse-failure counts in logs/reports.

---

## References

- Model family: see [Gemma and MedGemma](Gemma-and-MedGemma) for Gemma 3 12B pointers.
- Ollama: [https://ollama.com](https://ollama.com)

---

## See also

- [Insight extractor](Insight-extractor) — different prompts, different task (structured psychiatry insights, not transcript scoring).
