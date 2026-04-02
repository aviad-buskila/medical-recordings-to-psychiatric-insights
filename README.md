# Psychiatry Conversation Intelligence Pipeline

Product-oriented local platform to:

- transcribe psychiatrist-patient conversations
- evaluate transcript quality with multiple metrics
- extract psychiatry-focused structured insights
- analyze model behavior and outcomes with reproducible artifacts

Built for Apple Silicon with local-first tooling (`mlx-whisper`, PostgreSQL, Ollama, Jupyter).

## What this repo does

### Product capabilities

- **Clinical transcription:** generate transcripts from audio using two STT profiles (speed vs quality).
- **Quality benchmarking:** compare transcript runs against gold references.
- **Clinical insights extraction:** derive structured psychiatry insights from transcript content.
- **Decision support analytics:** notebook-based visuals for run comparison, alignment inspection, and speaker-flow exploration.

### Technical capabilities

- **STT engine:** `mlx-whisper`
- **Evaluation:** `WER`, `CER`, `MER`, `WIL`, `cpWER`, `BERTScore`, `LLM-as-a-judge`
- **Storage:** PostgreSQL (`stt_runs`, `stt_outputs`, `evaluation_metrics`, `transcript_insights`)
- **Artifacts:** deterministic outputs under `data/processed/*` for auditability and reproducibility

## Data source

Gold transcripts come from:

- [automated medical transcription dataset](https://github.com/nazmulkazi/dataset_automated_medical_transcription)

Expected local dataset file:

- `data/raw/dataset.pickle`

## Architecture

High-level flow:

1. **Ingestion**
   - reads local audio/transcript assets in `data/raw/`
2. **Transcription**
   - runs STT and writes outputs to DB + `data/generated_transcripts/`
3. **Evaluation**
   - computes lexical/semantic/LLM metrics and persists results
4. **Insights extraction**
   - uses Ollama model to extract psychiatry-focused structured claims
5. **Analysis**
   - notebooks in `analysis/` generate visuals and comparison summaries
6. **Reporting**
   - command artifacts + full pipeline markdown summary for traceability

Core folders:

- `src/cli/` - command entrypoints
- `src/stt/` - transcription pipeline/services
- `src/evaluation/` - metrics, judge, alignment
- `src/insights/` - psychiatry insight extraction
- `src/analytics/` - DB repository layer
- `analysis/` - visual notebooks + helpers
- `sql/` - schema/migrations

## Machine requirements

### Minimum (functional)

- macOS (Apple Silicon recommended)
- Python `3.11+`
- `8+` CPU cores
- `16GB` RAM
- `30GB+` free disk (models + artifacts + DB)

### Recommended (smooth local benchmarking)

- Apple Silicon (M-series)
- `32GB-48GB` RAM
- `10+` CPU cores
- fast SSD storage

### Required software

- Docker Desktop (PostgreSQL)
- Ollama
- `ffmpeg` on PATH (`brew install ffmpeg`)

## Quick start (from zero)

```bash
cp .env.example .env
# Edit .env for POSTGRES_*, STT_*, OLLAMA_* and paths

make install
docker compose --env-file .env up -d
make db-init
```

Pull required local LLMs:

```bash
ollama pull gemma3:12b
ollama pull medaibase/medgemma1.5:4b
```

## Dataset layout

Under `data/raw/`:

- `recordings/` - audio files
- `dataset.pickle` - gold references (primary eval source)
- `transcripts/transcribed/<sample_id>.json` - optional speaker-tagged dialogue for speaker metrics and cpWER
- `casenotes/` - optional clinical notes metadata

## Main commands

Activate env first:

```bash
source .venv/bin/activate
```

### 1) Validate dataset

```bash
python -m src.cli.main validate-dataset
```

### 2) Run STT

```bash
python -m src.cli.main run-stt
python -m src.cli.main run-stt --limit 5
python -m src.cli.main run-stt --profile quality
python -m src.cli.main run-stt --flavor both --limit 5
```

### 3) Evaluate transcript quality

```bash
python -m src.cli.main run-eval --run-id <RUN_UUID>
python -m src.cli.main run-eval --run-id <CANDIDATE_UUID> --ref-run-id <BASELINE_UUID> --workers auto

python -m src.cli.main run-bertscore --run-id <RUN_UUID>
python -m src.cli.main run-llm-judge --run-id <RUN_UUID>
python -m src.cli.main show-alignment --run-id <RUN_UUID>
```

### 4) Extract psychiatry insights

```bash
python -m src.cli.main insights-extract --run-id <RUN_UUID>
python -m src.cli.main insights-extract --run-id <RUN_UUID> --sample-id D0420-S1-T01
python -m src.cli.main insights-extract --run-id <RUN_UUID> --model medaibase/medgemma1.5:4b
```

### 5) Open analysis notebooks

```bash
jupyter notebook analysis/model_eval_insights.ipynb
jupyter notebook analysis/show_alignment_visualizer.ipynb
jupyter notebook analysis/gold_speaker_timeline.ipynb
```

### 6) Run full sampled benchmark pipeline

```bash
python run_full_pipeline.py
python run_full_pipeline.py --limit 5
```

This script runs STT-both + all evals + insights + notebook execution and writes a consolidated markdown report to `data/processed/full_pipeline/`.

## Artifacts and outputs

- STT transcripts: `data/generated_transcripts/<run_id>_<timestamp>/`
- Eval artifacts: `data/processed/<evalname>_<timestamp>.txt`
- Insights artifacts: `data/processed/insights_extract/insights_extract_<run_id>_<timestamp>.json`
- Executed analysis notebooks: `data/processed/analysis_notebooks/`
- Full benchmark summaries: `data/processed/full_pipeline/full_pipeline_<timestamp>.md`

## Database schema (operational)

- `clinical_ai.stt_runs` - run-level metadata (model, scope, parameters, timestamp)
- `clinical_ai.stt_outputs` - transcript outputs per sample/run
- `clinical_ai.evaluation_metrics` - metric rows with detailed payloads
- `clinical_ai.transcript_insights` - extracted psychiatry insight payloads

Run migrations via:

```bash
make db-init
```

Applied SQL files:

- `001_init.sql`
- `003_stt_runs.sql`
- `004_stt_remove_created_at_and_backfill_model.sql`
- `005_stt_run_scope.sql`
- `006_transcript_insights.sql`

## Configuration

See `.env.example`. Most important keys:

- `POSTGRES_*`
- `STT_MODEL`, `STT_MLX_QUALITY_MODEL`
- `OLLAMA_BASE_URL`, `OLLAMA_JUDGE_MODEL`, `OLLAMA_INSIGHTS_MODEL`
- `DATASET_PICKLE_PATH`, `TRANSCRIPTS_DIR`, `RECORDINGS_DIR`
- `EVAL_REPORTS_DIR`, `INSIGHTS_EXTRACT_DIR`

## Reproducibility and testing

- `make test` runs unit + integration tests.
- Tests are configured to write temporary artifacts (not `data/processed`) to avoid repo noise.
- Every major command writes explicit artifacts to support audit trails and debugging.
