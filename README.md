# Clinical STT evaluation (local MLX + PostgreSQL + Ollama)

Local pipeline for medical speech-to-text on Apple Silicon (**mlx-whisper**), persisting runs in **PostgreSQL**, and evaluating with **WER** (normalized text + substitution/insertion/deletion counts) plus optional **LLM-as-a-judge** via **Ollama**.

## What is included

- **STT:** **mlx-whisper**; default `mlx-community/whisper-large-v3-turbo`, quality profile `mlx-community/whisper-large-v3-mlx`
- **Storage:** PostgreSQL tables `stt_runs`, `stt_outputs`, `evaluation_metrics`
- **Files:** Per-run transcripts under `data/generated_transcripts/<run_id>_<timestamp>/`
- **Gold eval:** `dataset.pickle` from the [automated medical transcription dataset](https://github.com/nazmulkazi/dataset_automated_medical_transcription)
- **LLM judge:** Ollama (default model `gemma3:12b`, configurable)

## Prerequisites

- Python 3.11+
- `ffmpeg` on PATH (e.g. `brew install ffmpeg`)
- Docker (for PostgreSQL)
- Ollama with your judge model pulled (e.g. `ollama pull gemma3:12b`)
- Hugging Face token in `.env` if your chosen MLX model requires it (`HF_TOKEN`)

## Setup

```bash
cp .env.example .env
# Edit .env: POSTGRES_*, HF_TOKEN, OLLAMA_*, STT_* as needed

make install
docker compose --env-file .env up -d
make db-init
```

`make db-init` applies SQL in order: `001_init.sql`, `003_stt_runs.sql`, `004_stt_remove_created_at_and_backfill_model.sql`, `005_stt_run_scope.sql`.

## Dataset layout

Under `data/raw/`:

- `recordings/` — audio files (often gitignored; large files stay local)
- `dataset.pickle` — gold transcripts for evaluation
- Optional: `transcripts/`, `casenotes/` for layout validation

## CLI

```bash
source .venv/bin/activate

python -m src.cli.main validate-dataset

# STT (writes DB + generated_transcripts/)
python -m src.cli.main run-stt
python -m src.cli.main run-stt --limit 5
python -m src.cli.main run-stt --profile quality
python -m src.cli.main run-stt --flavor both --limit 3
python -m src.cli.main run-stt --flavor both --limit 3 --no-fallback

# WER vs gold (dataset.pickle)
python -m src.cli.main run-eval
python -m src.cli.main run-eval --limit 10
python -m src.cli.main run-eval --run-id <uuid>
python -m src.cli.main run-eval --run-id <candidate_uuid> --ref-run-id <baseline_uuid>

# LLM-as-a-judge (Ollama)
python -m src.cli.main run-llm-judge --run-id <uuid>
python -m src.cli.main run-llm-judge --run-id <uuid> --ref-run-id <other_uuid> --limit 3

# Full pass: validate → STT (default profile) → WER eval
python -m src.cli.main run-all
```

## Makefile shortcuts

- `make venv` / `make install` — create venv and install dependencies
- `make up` / `make down` — start or stop Postgres
- `make db-init` — apply SQL migrations
- `make test` — run pytest
- `make run-pipeline` — same as `run-all`

## Run IDs

Each STT run logs `run_id=...` at start. After `--flavor both`, the CLI prints `default_run_id` and `quality_run_id`. You can also query:

```bash
docker compose --env-file .env exec -T postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT run_id, model_name, run_scope, run_timestamp FROM clinical_ai.stt_runs ORDER BY run_timestamp DESC LIMIT 20;"'
```

## Configuration reference

See `.env.example` for `POSTGRES_*`, `OLLAMA_*`, `STT_*`, `HF_TOKEN`, paths, and `GENERATED_TRANSCRIPTS_DIR`.
