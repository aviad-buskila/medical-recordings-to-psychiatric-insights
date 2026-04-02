# Medical Conversation Intelligence Pipeline

> **Wiki — go deeper:** STT models, metrics, LLM judge, insights, Gemma & MedGemma, pipeline — all in [the repo's Wiki](https://github.com/aviad-buskila/medical-recordings-to-psychiatric-insights/wiki)

An on-prem/local platform for benchmarking speech-to-text models for clinical use by extracting structured psychiatric insights from patient-doctor conversations. Covers the full behavioral health AI stack: STT evaluation, clinical NLP, reproducible pipelines, and multi-metric model comparison analytics.

Built for Apple Silicon. Local-first by design — no audio leaves the machine.

---

## What it does

- **Benchmarks two STT models head-to-head** against gold-standard transcripts using 7 complementary metrics — from character-level edit distance to LLM-as-judge comparative scoring
- **Extracts structured psychiatric insights** from transcripts using MedGemma, with an evidence guardrail that drops any claim not grounded in the actual dialogue
- **Persists everything to PostgreSQL** with full JSONB payloads for retroactive analysis and metric recomputation without re-running inference
- **Runs as a resumable pipeline** — because local model inference is slow enough that mid-run failures are a real operational concern

---

## Latest benchmark

Most recent end-to-end run: **`data/processed/full_pipeline/full_pipeline_20260402T172054Z.md`**

| | |
| --- | --- |
| Command | `python run_full_pipeline.py --limit 5` |
| Samples | `D0420-S1-T01` … `D0420-S1-T05` (N=5) |
| Baseline STT | `mlx-community/whisper-large-v3-turbo` · run `22f69385` |
| Candidate STT | `mlx-community/whisper-large-v3-mlx` · run `507e804d` |
| Insights | MedGemma `medaibase/medgemma1.5:4b` · 5 rows per run |

Mean metrics vs gold:

| Metric | Baseline | Candidate | Δ (c−b) |
| --- | ---: | ---: | ---: |
| WER | 0.180 | 0.198 | +0.018 |
| CER | 0.182 | 0.202 | +0.020 |
| MER | 0.095 | 0.103 | +0.008 |
| WIL | 0.190 | 0.206 | +0.017 |
| cpWER | 0.082 | 0.105 | +0.023 |
| BERTScore F1 (`roberta-large`, rescale on) | 0.147 | 0.144 | −0.003 |
| LLM judge (mean comparative delta, N=5) | — | — | −0.4 |

Lower is better for WER/CER/MER/WIL/cpWER. LLM judge delta negative means baseline was preferred. These are point estimates over five files — see `analysis/statistical_significance.ipynb` before drawing conclusions.

BERTScore figures above used the code default encoder (`roberta-large` via `BERTSCORE_MODEL` unset). If you copy `.env.example` to `.env`, `BERTSCORE_MODEL` is set to a biomedical encoder instead — rerun evals to compare under that setting.

---

## Architecture

```
Audio files (WAV)
       │
       ▼
  [Ingestion]  ──── dataset.pickle (gold transcripts)
       │
       ▼
  [STT Engine]  ──── mlx-whisper
  two profiles: turbo (speed) vs large-v3 (quality)
       │
       ▼                              PostgreSQL
  [Evaluation] ─────────────────────► stt_runs
  WER · CER · MER · WIL · cpWER      stt_outputs
  BERTScore · LLM-as-judge(Gemma3:12b)            evaluation_metrics
       │
       ▼
  [Insights Extraction]  ──── Ollama (MedGemma 4B)
  Evidence guardrail                 transcript_insights
       │
       ▼
  [Analysis Notebooks]
  model comparison · alignment viz · speaker timeline
       │
       ▼
  [Pipeline Report]  ──── data/processed/full_pipeline/
```

**Pipeline steps:**

1. **Ingestion** — discovers audio, gold transcripts, optional speaker-tagged JSON, and casenotes from `data/raw/`
2. **Transcription** — runs STT on both profiles; writes outputs to postgres DB and `data/generated_transcripts/`
3. **Evaluation** — computes all metrics against gold references; persists full JSONB payloads for auditability
4. **Insights extraction** — uses Ollama + MedGemma1.5:4b to derive structured clinical claims; evidence guardrail drops unsupported/unmentioned claims
5. **Analysis** — executes Jupyter notebooks generating model comparison charts, word alignment views, and speaker flow plots
6. **Reporting** — writes a consolidated Markdown summary with all artifact paths and DB metric aggregates

---

## Evaluation metrics

| Metric | What it measures | Range | Notes |
| --- | --- | --- | --- |
| WER | Word-level edit distance from reference | 0.0 → ∞ | Primary STT quality metric |
| CER | Character-level edit distance | 0.0 → ∞ | Catches character-level noise |
| MER | % of reference words matched incorrectly | 0.0 → 1.0 | Complementary to WER |
| WIL | Semantic coverage loss | 0.0 → 1.0 | Penalizes insertions differently from WER |
| cpWER | WER minimized over speaker block permutations | 0.0 → ∞ | Preferred metric when speaker diarization labels are available |
| BERTScore F1 | Semantic overlap between STT hypothesis and gold reference | 0.0 → 1.0 | Captures paraphrase and domain synonyms that WER misses |
| LLM-judge | Comparative quality score from a language model | Varies | Exploratory; treat as directional only |

**On WER thresholds:** General ASR achieves 5–15% WER on clean speech. Medical dialogue is harder — drug names, dosages, and clinical acronyms (SSRI, GAD, DSM-5) push error rates up. WER under 20% on medical dialogue without domain fine-tuning is a reasonable baseline. cpWER is preferred over WER when diarization labels are available.

---

## Quick start

### Requirements

- macOS with Apple Silicon (required for `mlx-whisper`; evaluation and insights run on any platform)
- Python 3.11+
- 32–48 GB RAM recommended (16 GB minimum)
- 30 GB+ free disk

```bash
brew install ffmpeg
brew install --cask docker
# Install Ollama from https://ollama.com
```

### Setup

```bash
cp .env.example .env
# Edit .env: POSTGRES_*, STT_*, OLLAMA_*, and path variables

make install
docker compose --env-file .env up -d
make db-init
```

Pull models:

```bash
ollama pull gemma3:12b                    # LLM judge
ollama pull medaibase/medgemma1.5:4b      # Insights extraction
```

Place data under `data/raw/`:
- `dataset.pickle` — gold transcripts (see [Dataset](#dataset))
- `recordings/*.wav` — audio files
- `transcripts/transcribed/<sample_id>.json` — optional speaker-tagged dialogue (enables cpWER and per-speaker WER)

---

## Commands

Activate environment first:

```bash
source .venv/bin/activate
```

### STT

```bash
python -m src.cli.main run-stt --limit 5                       # Speed profile (turbo)
python -m src.cli.main run-stt --profile quality --limit 5     # Quality profile (large-v3)
python -m src.cli.main run-stt --flavor both --limit 5         # Both profiles on same files
```

### Evaluation

```bash
# Lexical metrics (WER/CER/MER/WIL/cpWER)
python -m src.cli.main run-eval --run-id <UUID> --workers auto
python -m src.cli.main run-eval --run-id <CANDIDATE> --ref-run-id <BASELINE> --workers auto

# Semantic metric (BERTScore; see BERTSCORE_MODEL in Configuration)
python -m src.cli.main run-bertscore --run-id <UUID>
python -m src.cli.main run-bertscore --run-id <CANDIDATE> --ref-run-id <BASELINE>

# LLM comparative judgment
python -m src.cli.main run-llm-judge --run-id <CANDIDATE> --ref-run-id <BASELINE>

# Word alignment visualization
python -m src.cli.main show-alignment --run-id <UUID> --limit 5
```

### Insights extraction

```bash
python -m src.cli.main insights-extract --run-id <UUID> --limit 5
python -m src.cli.main insights-extract --run-id <UUID> --model medaibase/medgemma1.5:4b
python -m src.cli.main insights-extract --run-id <UUID> --sample-id D0420-S1-T01
```

### Full benchmark pipeline

```bash
python run_full_pipeline.py --limit 5

# Resume after partial failure (run IDs preserved in state file)
python run_full_pipeline.py --skip-stt --baseline-run-id <UUID> --candidate-run-id <UUID>

# Skip evals, re-run notebooks and report only
python run_full_pipeline.py --skip-stt --skip-evals --baseline-run-id <UUID> --candidate-run-id <UUID>
```

Runs both STT profiles → all evaluations → insights extraction → notebook execution → consolidated Markdown report at `data/processed/full_pipeline/full_pipeline_<timestamp>.md`.

---

## Analysis notebooks

| Notebook | What it shows |
| --- | --- |
| `analysis/model_eval_insights.ipynb` | Side-by-side metric comparison; per-sample delta plots |
| `analysis/show_alignment_visualizer.ipynb` | Word-level alignment blocks showing substitution/insertion/deletion patterns |
| `analysis/gold_speaker_timeline.ipynb` | Therapist vs. patient speech patterns and turn lengths |
| `analysis/statistical_significance.ipynb` | Paired Wilcoxon signed-rank tests and bootstrap 95% CIs on metric deltas |
| `analysis/insights_quality.ipynb` | Guardrail drop-rate, schema adherence, claim category distribution, cross-run consistency |

```bash
jupyter notebook analysis/model_eval_insights.ipynb
```

---

## Artifacts and outputs

| Artifact | Location |
| --- | --- |
| STT transcripts | `data/generated_transcripts/<run_id>_<timestamp>/` |
| Eval reports | `data/processed/<evalname>_<timestamp>.txt` |
| Insights JSON | `data/processed/insights_extract/insights_extract_<run_id>_<timestamp>.json` |
| Executed notebooks | `data/processed/analysis_notebooks/` |
| Pipeline summary | `data/processed/full_pipeline/full_pipeline_<timestamp>.md` |

---

## Database schema

All tables live under the `clinical_ai` schema:

| Table | Contents |
| --- | --- |
| `stt_runs` | Run-level metadata: model, provider, scope, parameters, timestamp |
| `stt_outputs` | Transcript text per sample per run |
| `evaluation_metrics` | Metric rows with full JSONB detail payloads |
| `transcript_insights` | Extracted insight JSONB, raw model output, `prompt_version`, `insight_model` |

Apply migrations:

```bash
make db-init
```

Migration order: `001_init.sql` → `003_stt_runs.sql` → `004_stt_remove_created_at_and_backfill_model.sql` → `005_stt_run_scope.sql` → `006_transcript_insights.sql`

---

## Configuration

See `.env.example`. Key variables:

| Variable | Purpose |
| --- | --- |
| `POSTGRES_*` | Database connection |
| `STT_MODEL` | Default (speed) Whisper model |
| `STT_MLX_QUALITY_MODEL` | Quality Whisper model |
| `STT_MODEL_FALLBACK` | Fallback if primary model fails to load |
| `OLLAMA_BASE_URL` | Ollama server address |
| `OLLAMA_JUDGE_MODEL` | Model for LLM-as-judge evaluation |
| `OLLAMA_INSIGHTS_MODEL` | Model for clinical insight extraction |
| `DATASET_PICKLE_PATH` | Gold transcript source |
| `BERTSCORE_MODEL` | Encoder for BERTScore. **Code default** (no env var): `roberta-large`. **`.env.example`** sets `dmis-lab/biobert-base-cased-v1.2` so a copied `.env` uses BioBERT unless you change it. |

---

## Testing

```bash
make test
```

Unit tests cover each metric module (WER, CER, MER, WIL, cpWER, BERTScore, alignment, text normalization). Integration tests cover CLI commands and filesystem interactions. PostgreSQL-dependent tests are skipped automatically if the DB is unreachable (`@pytest.mark.postgres`).

---

## Dataset

**Source:** [nazmulkazi / dataset_automated_medical_transcription](https://github.com/nazmulkazi/dataset_automated_medical_transcription) — automated medical transcription benchmark.

Expected local path: `data/raw/dataset.pickle`

**Note:** This dataset contains scripted medical dialogue, not naturalistic psychiatric sessions. The pipeline is designed to work with real clinical recordings — this dataset is a development and benchmarking proxy.

---

## Design decisions

| Decision | Rationale |
| --- | --- |
| Local-first (no cloud APIs) | Clinical audio should not leave the machine during development |
| Two STT profiles | Controlled speed/quality tradeoff comparison against a shared gold reference |
| Evidence guardrail on insights | The system should only surface claims grounded in what was actually said |
| PostgreSQL with JSONB details | Full payload storage enables retroactive analysis without re-running STT |
| Resumable pipeline state | Local model inference is slow enough that mid-run failures are a real operational concern |
| ProcessPoolExecutor for eval | Each sample evaluation is CPU-bound and independent — process parallelism bypasses the GIL |

---

## Known limitations

- **Synthetic dataset**: Gold transcripts are scripted dialogue, not naturalistic psychiatric sessions. Real clinical audio would produce different benchmarks.
- **Whisper struggles with medical terminology**: Drug names, dosages, and acronyms (SSRI, DSM-5, GAD, PTSD) have elevated error rates without domain fine-tuning — precisely the highest-value terms in psychiatric transcription.
- **BERTScore encoder choice**: With the code default (`roberta-large`), scores reflect a general encoder not tuned for clinical text. The sample `.env` uses BioBERT (`dmis-lab/biobert-base-cased-v1.2`) for more domain-appropriate similarity; other biomedical encoders (e.g. SciBERT) are also reasonable overrides via `BERTSCORE_MODEL`.
- **Self-reported confidence**: The `confidence` field in insights comes from the LLM's own output — it is not a calibrated probability.
- **Apple Silicon required for STT**: `mlx-whisper` requires Apple Silicon. Evaluation and insights extraction run anywhere.
- **Sequential pipeline**: Steps run sequentially. Production deployment would require async execution, batching, and a job queue.
- **Small-sample benchmarks**: Point estimates over five files are directional only. Run `analysis/statistical_significance.ipynb` (or increase `--limit`) before drawing conclusions about model differences.
