# Wiki home

This wiki documents how **medical-recordings-to-psychiatric-insights** is put together: models, metrics, LLM usage, and supporting systems. It complements the repository README with deeper, topic-focused pages.

**Scope:** Local-first pipeline on Apple Silicon (STT via MLX), PostgreSQL storage, Ollama-hosted LLMs.

---

## Contents

| Topic | Page |
|--------|------|
| Whisper / MLX STT profiles and fallbacks | [Speech-to-text models](Speech-to-text-models) |
| WER, CER, MER, WIL, cpWER, BERTScore | [Evaluation metrics](Evaluation-metrics) |
| Comparative transcript judging | [LLM as judge](LLM-as-judge) |
| Psychiatry-oriented extraction + evidence guardrail | [Insight extractor](Insight-extractor) |
| Gemma 3 12B and MedGemma 1.5 (4B) — refs and config | [Gemma and MedGemma](Gemma-and-MedGemma) |
| DB, ingestion, notebooks, tooling | [Pipeline and other components](Pipeline-and-other-components) |

---

## Publishing to GitHub Wiki

The wiki is a **separate git repository** from the main code repo.

1. On GitHub: enable **Wiki** for the repository (Settings → Features → Wikis).
2. Clone the wiki repo (replace `OWNER` / `REPO`):

   ```bash
   git clone https://github.com/OWNER/REPO.wiki.git
   cd REPO.wiki
   ```

3. Copy the Markdown files from this repository’s `wiki/` folder into the wiki clone, commit, and push:

   ```bash
   cp /path/to/medical-recordings-to-psychiatric-insights/wiki/*.md .
   git add .
   git commit -m "Import wiki pages from main repository"
   git push origin master
   ```

GitHub may use `master` or `main` as the default branch for the wiki; use whatever `git branch` shows after clone.

---

## Conventions

- **Env vars** refer to `.env` (see `.env.example` in the main repo).
- **Code paths** are relative to the repository root unless stated otherwise.
