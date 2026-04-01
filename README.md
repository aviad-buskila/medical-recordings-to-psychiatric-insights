# Eleos Health Clinical AI Pipeline

Production-ready scaffold for a clinical AI workflow:

- local speech-to-text from clinical recordings (Apple MLX Whisper)
- hybrid clinical retrieval grounding (pgvector + BM25)
- multi-provider generation (OpenAI, Anthropic, Ollama fallback)
- evaluation and guardrails with analytics persisted in PostgreSQL

## Quick Start

1. Copy env vars:
   - `cp .env.example .env`
2. Install dependencies:
   - `make install`
3. Start local PostgreSQL:
   - `docker compose --env-file .env up -d`
4. Run pipeline scaffold:
   - `source .venv/bin/activate && python -m src.cli.main run-all`
5. Execute tests:
   - `make test`

## STT Backend (Apple Silicon)

- Provider: `mlx-whisper`
- Default model: `mlx-community/whisper-large-v3-turbo`
- Fallback model: `mlx-community/whisper-large-v3-turbo`
- Quality profile model: `mlx-community/whisper-large-v3-mlx`
- Optimized for Apple M-series via MLX/Metal unified memory

## Expected Dataset Layout

Place dataset folders under `data/raw`:

- `data/raw/transcripts`
- `data/raw/casenotes`
- `data/raw/recordings`
- `data/raw/dataset.pickle` (used as primary gold source for evaluation)

## CLI Commands

- `python -m src.cli.main validate-dataset`
- `python -m src.cli.main run-stt`
- `python -m src.cli.main run-stt --limit 5`
- `python -m src.cli.main run-stt --profile quality`
- `python -m src.cli.main run-stt --flavor both --limit 3`
- `python -m src.cli.main run-stt --flavor both --limit 3 --no-fallback`
- `python -m src.cli.main build-rag-index`
- `python -m src.cli.main run-eval`
- `python -m src.cli.main run-eval --limit 5`
- `python -m src.cli.main run-eval --run-id <stt_run_id>`
- `python -m src.cli.main run-eval --run-id <stt_run_id> --ref-run-id <baseline_run_id>`
- `python -m src.cli.main run-llm-judge --run-id <stt_run_id>`
- `python -m src.cli.main run-llm-judge --run-id <stt_run_id> --ref-run-id <baseline_run_id> --limit 3`
- `python -m src.cli.main run-all`

### Word alignment (gold vs hypothesis, S / I / D)

Gold text comes from `dataset.pickle` (same normalization as WER). For each sample, prints `GOLD` / `HYP` / `OP` rows where `=` is match, `S` substitution, `D` deletion (word in gold missing in hypothesis), `I` insertion (extra word in hypothesis). Long transcripts wrap in chunks of `--chunk-columns` words.

- `python -m src.cli.main show-alignment --run-id <stt_run_id>`
- `python -m src.cli.main show-alignment --run-id <candidate_run_id> --ref-run-id <baseline_run_id>`
- `python -m src.cli.main show-alignment --run-id <stt_run_id> --sample-id D0420-S1-T01 --limit 3`
- `python -m src.cli.main show-alignment --run-id <stt_run_id> -o alignment_report.txt`

## Generated Transcripts

Each STT run also writes local transcript files to:

- `data/generated_transcripts/<run_id>_<timestamp>/<sample_id>.txt`
