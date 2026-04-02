# Gemma and MedGemma

[Wiki home](Home) · [LLM as judge](LLM-as-judge) · [Insight extractor](Insight-extractor)

---

## Roles in this repository

| Model (Ollama tag) | Primary use |
|--------------------|-------------|
| **`gemma3:12b`** | LLM-as-**judge** for transcript scoring / comparison (`OLLAMA_JUDGE_MODEL`) |
| **`medaibase/medgemma1.5:4b`** | **Insight extraction** — structured psychiatry fields from dialogue (`OLLAMA_INSIGHTS_MODEL`) |

Both are invoked through the **Ollama** HTTP API (`OLLAMA_BASE_URL`). The default chat model in `OllamaClient` is `OLLAMA_MODEL` (also `gemma3:12b` in `.env.example`).

---

## Gemma 3 — 12B (`gemma3:12b`)

**What it is:** A mid-size model in the **Gemma 3** family from Google DeepMind, released in 2025, with multimodal and long-context variants across sizes (community packages expose a **12B** instruct checkpoint as `gemma3:12b` in Ollama).

**Why here:** Strong instruction-following and JSON formatting for judge prompts; runs locally when VRAM/RAM permits.

**References:**

- Gemma 3 overview — [Google DeepMind: Gemma 3](https://deepmind.google/models/gemma/gemma-3/)
- Technical report — *Gemma 3 Technical Report*, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- Product announcement — [Google Developers Blog, Mar 2025](https://blog.google/technology/developers/gemma-3/)

---

## MedGemma 1.5 — **4B** (not 3B)

> **Note:** You asked about “MedGemma 1.5 **3B**.” Public **MedGemma 1.5** releases from Google are documented as **4B** parameter models. This repo’s default is the **4B** Ollama artifact `medaibase/medgemma1.5:4b`. There is no separate “1.5 3B” variant in the same documented family; if you need a smaller model, use another Ollama tag or model entirely and set `OLLAMA_INSIGHTS_MODEL` accordingly.

**What it is:** MedGemma 1.5 is a **medical** instruction-tuned model built on **Gemma 3**, aimed at clinical text (and multimodal medical tasks in Google’s full release). The Hugging Face id used in Google’s catalog is typically **`google/medgemma-1.5-4b-it`**.

**Why here:** Domain alignment for psychiatry-flavored extraction from transcripts, under the evidence guardrail in [[Insight extractor]].

**Ollama:** Community builds such as [MedAIBase/MedGemma1.5:4b on Ollama](https://ollama.com/) mirror quantized weights — verify the exact tag you pull matches `OLLAMA_INSIGHTS_MODEL`.

**References:**

- Model card (Google) — [MedGemma 1.5 — Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- Overview — [MedGemma — Google for Developers](https://developers.google.com/health-ai-developer-foundations/medgemma)
- Hugging Face — [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it)
- Research blog — [Next-generation medical image interpretation with MedGemma 1.5…](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr) (also mentions MedASR for speech — separate from this repo’s Whisper path)

**Licensing / use:** Google’s MedGemma terms apply (Health AI Developer Foundations). Review the model card before production or clinical use.

---

## If you need a smaller model than 4B

MedGemma 1.5 in Google’s published lineup is **4B**. For lighter footprints, pick another Ollama model (general or medical) and point `OLLAMA_INSIGHTS_MODEL` at it; re-validate extraction quality and the evidence guardrail.

---

## Environment quick reference

```text
OLLAMA_JUDGE_MODEL=gemma3:12b
OLLAMA_INSIGHTS_MODEL=medaibase/medgemma1.5:4b
```

---

## See also

- [LLM as judge](LLM-as-judge) — prompts and JSON fields for Gemma 3 as judge  
- [Insight extractor](Insight-extractor) — JSON schema and guardrails for MedGemma outputs
