# Pipeline and other components

[Wiki home](Home) · [Speech-to-text models](Speech-to-text-models) · [Evaluation metrics](Evaluation-metrics)

---

## End-to-end flow

1. **Ingestion** — `data/raw/`: `dataset.pickle` (gold), `recordings/*.wav`, optional `transcripts/transcribed/*.json` (speaker-aware metrics), optional casenotes paths (`DatasetLoader`).
2. **STT** — MLX Whisper; outputs to PostgreSQL and `data/generated_transcripts/`.
3. **Evaluation** — Lexical + BERTScore + optional LLM judge; metrics in `evaluation_metrics`.
4. **Insights** — Ollama + MedGemma; rows in `transcript_insights`.
5. **Analysis** — Jupyter notebooks under `analysis/` (pandas, matplotlib, seaborn; `statistical_significance.ipynb` uses **SciPy**).
6. **Reporting** — `run_full_pipeline.py` consolidates commands and writes Markdown under `data/processed/full_pipeline/` (ignored by git by default).

---

## Database (PostgreSQL)

| Piece | Detail |
|-------|--------|
| Schema | `clinical_ai` (configurable via `POSTGRES_SCHEMA`) |
| Core tables | `stt_runs`, `stt_outputs`, `evaluation_metrics`, `transcript_insights` |
| Migrations | SQL files `sql/001_init.sql` … `006_transcript_insights.sql`; applied via `make db-init` |

JSONB on metrics and insights keeps full payloads for later recomputation or auditing.

---

## Orchestration and CLI

| Component | Role |
|-----------|------|
| `src/cli/main.py` | Click CLI: `validate-dataset`, `run-stt`, `run-eval`, `run-bertscore`, `run-llm-judge`, `show-alignment`, `insights-extract`, `restore-stt-from-generated`, etc. |
| `run_full_pipeline.py` | Scripted benchmark: both STT profiles → evals → insights → notebook execution → report |
| `Makefile` | `install`, `db-init`, `test`, `up` / `down` for Docker |

---

## External services (not pip packages)

| Dependency | Purpose |
|------------|---------|
| **Docker** | Runs PostgreSQL (`docker compose`) |
| **ffmpeg** | Audio handling where required by the STT stack |
| **Ollama** | Local LLM server for judge + insights |

---

## Python libraries (high level)

Pinned in `requirements.txt`: e.g. `mlx`, `mlx-whisper`, `torch`, `bert-score`, `psycopg`, `pydantic`, `rapidfuzz`, `ollama`, `jupyter` / notebook stack for analysis. Transitive deps include **SciPy** (used by notebooks), **NumPy**, **transformers**-related stacks for BERTScore, etc.

---

## Documentation in-repo

| Asset | Content |
|-------|---------|
| `README.md` | Quick start, commands, architecture diagram |
| `docs/cli.md` | CLI reference (if present; keep in sync with `src/cli/main.py`) |

---

## See also

- [Speech-to-text models](Speech-to-text-models) — MLX / Whisper ids  
- [Gemma and MedGemma](Gemma-and-MedGemma) — Ollama model ids  
- [Insight extractor](Insight-extractor) — DB + artifacts for insights  
