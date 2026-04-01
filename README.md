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
- `python -m src.cli.main build-rag-index`
- `python -m src.cli.main run-eval`
- `python -m src.cli.main run-all`
