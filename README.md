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
- `dataset.pickle` — gold transcripts for evaluation (primary reference for WER / BERTScore)
- Optional: `transcripts/`, `casenotes/` for layout validation
- Optional: `transcripts/transcribed/<sample_id>.json` — speaker-tagged dialogue for **per-speaker WER and CER** (speakers 1 and 2) and **cpWER** on `run-eval`. Each file is a JSON array of `{ "speaker": 1, "dialogue": ["...", ...] }` objects. Gold text still comes from `dataset.pickle`; the JSON is used to label words/characters by speaker and for concatenated-minimum-permutation WER (see `src/evaluation/cp_wer.py`).

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

# WER vs gold (dataset.pickle): wer, cer, mer, wil, cp_wer when transcribed JSON exists
python -m src.cli.main run-eval
python -m src.cli.main run-eval --limit 10
python -m src.cli.main run-eval --run-id <uuid>
python -m src.cli.main run-eval --run-id <candidate_uuid> --ref-run-id <baseline_uuid>

# BERTScore (semantic P/R/F1; not written to evaluation_metrics)
python -m src.cli.main run-bertscore --run-id <uuid>
python -m src.cli.main run-bertscore --run-id <uuid> --ref-run-id <baseline_uuid> --limit 10
python -m src.cli.main run-bertscore --run-id <uuid> -o bertscore_summary.json

# LLM-as-a-judge (Ollama)
python -m src.cli.main run-llm-judge --run-id <uuid>
python -m src.cli.main run-llm-judge --run-id <uuid> --ref-run-id <other_uuid> --limit 3

# Full pass: validate → STT (default profile) → WER eval
python -m src.cli.main run-all
```

Note: `run-eval`, `run-bertscore`, `run-llm-judge`, and `show-alignment` also write an artifact to `data/processed/<evalname>_<timestamp>.txt` (command line + full output; plus JSON results for the DB-writing evals).

### Word alignment (gold vs hypothesis, S / I / D)

Gold text comes from `dataset.pickle` (same normalization as WER). For each sample, prints `GOLD` / `HYP` / `OP` rows where `=` is match, `S` substitution, `D` deletion, `I` insertion. Long transcripts wrap in chunks of `--chunk-columns` words.

```bash
python -m src.cli.main show-alignment --run-id <uuid>
python -m src.cli.main show-alignment --run-id <candidate_uuid> --ref-run-id <baseline_uuid>
python -m src.cli.main show-alignment --run-id <uuid> --sample-id D0420-S1-T01 --limit 3
python -m src.cli.main show-alignment --run-id <uuid> -o alignment_report.txt
```

### BERTScore (`run-bertscore`)

Embedding-based **precision, recall, and F1** between gold and STT text from PostgreSQL. Optional semantic overlap beyond word errors. Logs to the console; `-o` / `--output-json` writes a full summary. Does **not** insert rows into `evaluation_metrics`.

- Pass `--run-id` to score a specific STT run; omit to use the latest STT output per sample.
- Encoder defaults to `roberta-large`; override with `--model-type` or `BERTSCORE_MODEL` in `.env`.
- First run downloads encoder weights; `--no-rescale` is faster (scores not comparable to rescaling baselines).

Dependencies: `torch` and `bert-score` are in `requirements.txt`.

## Makefile shortcuts

- `make venv` / `make install` — create venv and install dependencies
- `make up` / `make down` — start or stop Postgres
- `make db-init` — apply SQL migrations
- `make test` — run pytest (`tests/unit` — metrics and mocks; `tests/integration` — subprocess `python -m src.cli.main`, temp-dir `validate-dataset`, filesystem pickle/loader, evaluation chain; optional Postgres smoke test skips if DB is down)
- `make run-pipeline` — same as `run-all`

## Run IDs

Each STT run logs `run_id=...` at start. After `--flavor both`, the CLI prints `default_run_id` and `quality_run_id`. You can also query:

```bash
docker compose --env-file .env exec -T postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT run_id, model_name, run_scope, run_timestamp FROM clinical_ai.stt_runs ORDER BY run_timestamp DESC LIMIT 20;"'
```

## Configuration reference

See `.env.example` for `POSTGRES_*`, `OLLAMA_*`, `STT_*`, `HF_TOKEN`, `BERTSCORE_MODEL`, paths, and `GENERATED_TRANSCRIPTS_DIR`.
