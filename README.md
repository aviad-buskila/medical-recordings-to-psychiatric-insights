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
- `data/raw/transcripts/transcribed/<sample_id>.json` (optional) — speaker-tagged dialogue for **per-speaker WER and CER** (speakers 1 and 2) on `run-eval`. Each file is a JSON array of `{ "speaker": 1, "dialogue": ["...", ...] }` objects. Gold text for evaluation still comes from `dataset.pickle`; the JSON is used only to label words/characters by speaker.

## CLI Commands

- `python -m src.cli.main validate-dataset`
- `python -m src.cli.main run-stt`
- `python -m src.cli.main run-stt --limit 5`
- `python -m src.cli.main run-stt --profile quality`
- `python -m src.cli.main run-stt --flavor both --limit 3`
- `python -m src.cli.main run-stt --flavor both --limit 3 --no-fallback`
- `python -m src.cli.main build-rag-index`
- `python -m src.cli.main run-eval` (stores `wer`, `cer`, `mer`, `wil`, and `cp_wer` when `transcripts/transcribed/<id>.json` exists; **MER** = \(E/(R+H)\), **WIL** = \(2E/(R+H)\); **cpWER** = minimum WER over chronological JSON text and all permutations of speaker word blocks — see `src/evaluation/cp_wer.py`)
- `python -m src.cli.main run-eval --limit 5`
- `python -m src.cli.main run-eval --run-id <stt_run_id>`
- `python -m src.cli.main run-eval --run-id <stt_run_id> --ref-run-id <baseline_run_id>`
- `python -m src.cli.main run-bertscore --run-id <stt_run_id>` (BERTScore P/R/F1 vs gold; separate from `run-eval`, not written to `evaluation_metrics`)
- `python -m src.cli.main run-bertscore --run-id <stt_run_id> --ref-run-id <baseline_run_id> --limit 10`
- `python -m src.cli.main run-bertscore --run-id <stt_run_id> -o bertscore_summary.json`
- `python -m src.cli.main run-llm-judge --run-id <stt_run_id>`
- `python -m src.cli.main run-llm-judge --run-id <stt_run_id> --ref-run-id <baseline_run_id> --limit 3`
- `python -m src.cli.main run-all`

### Word alignment (gold vs hypothesis, S / I / D)

Gold text comes from `dataset.pickle` (same normalization as WER). For each sample, prints `GOLD` / `HYP` / `OP` rows where `=` is match, `S` substitution, `D` deletion (word in gold missing in hypothesis), `I` insertion (extra word in hypothesis). Long transcripts wrap in chunks of `--chunk-columns` words.

- `python -m src.cli.main show-alignment --run-id <stt_run_id>`
- `python -m src.cli.main show-alignment --run-id <candidate_run_id> --ref-run-id <baseline_run_id>`
- `python -m src.cli.main show-alignment --run-id <stt_run_id> --sample-id D0420-S1-T01 --limit 3`
- `python -m src.cli.main show-alignment --run-id <stt_run_id> -o alignment_report.txt`

### BERTScore (`run-bertscore`)

Embedding-based **precision, recall, and F1** between gold (from `dataset.pickle` / transcript fallback, same as WER eval) and STT text from PostgreSQL. Use when you care about semantic overlap, not only word errors. Results log to the console; optional `-o` / `--output-json` writes a full summary file. Does **not** insert rows into `evaluation_metrics` (unlike `run-eval`).

- Pass `--run-id` to score transcripts from a specific STT run; omit it to use the latest STT output per sample from the DB.
- Encoder defaults to `roberta-large`; override with `--model-type` or `BERTSCORE_MODEL` in `.env`.
- First run downloads the encoder weights; use `--no-rescale` for faster scoring (scores are not comparable to rescaling baselines).

Dependencies: `torch` and `bert-score` are listed in `requirements.txt`.

## Generated Transcripts

Each STT run also writes local transcript files to:

- `data/generated_transcripts/<run_id>_<timestamp>/<sample_id>.txt`
