# Psychiatry Conversation Intelligence Pipeline

End-to-end local platform for benchmarking clinical speech-to-text models and extracting psychiatry-focused structured insights from patient-therapist conversations. Built to demonstrate the full data science and AI engineering stack relevant to behavioral health CareOps: STT evaluation, clinical NLP, reproducible pipelines, and model comparison analytics.

Gold transcripts come from the [automated medical transcription dataset](https://github.com/nazmulkazi/dataset_automated_medical_transcription?tab=readme-ov-file) (see [Dataset](#dataset)).

Built for Apple Silicon using local-first tooling (`mlx-whisper`, Ollama, PostgreSQL, Jupyter).

---

## Latest full pipeline benchmark

Most recent end-to-end run: **`data/processed/full_pipeline/full_pipeline_20260402T172054Z.md`** (commands, artifact paths, and raw DB JSON). Current pointer: **`data/processed/full_pipeline/full_pipeline_state_latest.json`**.

| | |
| --- | --- |
| Command | `python run_full_pipeline.py --limit 5` |
| Samples | `D0420-S1-T01` … `D0420-S1-T05` (N=5) |
| Baseline STT | `mlx-community/whisper-large-v3-turbo` · run `22f69385-e088-4e6c-8e63-b148b36f0849` |
| Candidate STT | `mlx-community/whisper-large-v3-mlx` · run `507e804d-4260-406a-b7b5-a97e395ad625` |
| Insights | MedGemma `medaibase/medgemma1.5:4b` · 5 rows per run |

Mean metrics vs **gold** (`dataset.pickle`), unless noted:

| Metric | Baseline | Candidate | Δ (c−b) |
| --- | ---: | ---: | ---: |
| WER | 0.180 | 0.198 | +0.018 |
| CER | 0.182 | 0.202 | +0.020 |
| MER | 0.095 | 0.103 | +0.008 |
| WIL | 0.190 | 0.206 | +0.017 |
| cpWER | 0.082 | 0.105 | +0.023 |
| BERTScore F1 (each STT vs gold; `roberta-large`, rescale on) | 0.147 | 0.144 | −0.003 |
| LLM judge (mean comparative score delta, N=5) | — | — | −0.4 |

Lower is better for WER, CER, MER, WIL, and cpWER. BERTScore F1 is shown for **each model against gold** (not hypothesis-vs-hypothesis). LLM judge delta is exploratory (negative ⇒ baseline favored on average). These are **point estimates over five files**—see `analysis/statistical_significance.ipynb` before drawing firm conclusions.

---

## What This Demonstrates

- **STT evaluation rigor**: 7 evaluation metrics covering lexical accuracy (WER/CER/MER/WIL/cpWER), semantic fidelity (BERTScore), and comparative quality (LLM-as-judge)
- **Clinical NLP**: Evidence-grounded insight extraction with a hallucination guardrail, medical model selection (MedGemma), and speaker-aware metrics
- **AI engineering**: Resumable multi-step pipeline, two-profile model comparison, reproducible artifact system, local-first architecture for data privacy
- **Data science**: Multi-metric model benchmarking, run-level analytics in PostgreSQL, notebook-based analysis with alignment visualization and speaker flow exploration

---

## Architecture

```
Audio files (WAV)
       |
       v
  [Ingestion]  ──── dataset.pickle (gold transcripts)
       |
       v
  [STT Engine]  ──── mlx-whisper
  two profiles: turbo (speed) vs large-v3 (quality)
       |
       v                              PostgreSQL
  [Evaluation] ─────────────────────► stt_runs
  WER · CER · MER · WIL · cpWER      stt_outputs
  BERTScore · LLM-as-judge            evaluation_metrics
       |
       v
  [Insights Extraction]  ──── Ollama (MedGemma 4B)
  Evidence guardrail                 transcript_insights
       |
       v
  [Analysis Notebooks]
  model comparison · alignment viz · speaker timeline
       |
       v
  [Pipeline Report]  ──── data/processed/full_pipeline/
```

**Step-by-step:**
1. **Ingestion** — discovers audio, gold transcripts, optional speaker-tagged JSON, and casenotes from `data/raw/`
2. **Transcription** — runs STT on both profiles; writes outputs to DB and `data/generated_transcripts/`
3. **Evaluation** — computes all metrics against gold references; persists results with full JSONB payloads for auditability
4. **Insights extraction** — uses Ollama + MedGemma to derive structured clinical claims; evidence guardrail drops unsupported claims
5. **Analysis** — executes Jupyter notebooks that generate model comparison charts, word alignment views, and speaker flow plots
6. **Reporting** — writes a consolidated Markdown summary with all artifact paths and DB metric aggregates

---

## Evaluation Metrics

| Metric | What it measures | Range | Notes |
| --- | --- | --- | --- |
| WER | Word-level edit distance from reference | 0.0 (perfect) → ∞ | Primary STT quality metric |
| CER | Character-level edit distance | 0.0 → ∞ | Useful for detecting character-level noise |
| MER | Match Error Rate: % of reference words matched incorrectly | 0.0 → 1.0 | Complementary to WER |
| WIL | Word Information Lost: semantic coverage loss | 0.0 → 1.0 | Penalizes insertions differently from WER |
| cpWER | Chronological Permutation WER | 0.0 → ∞ | WER minimized over speaker block permutations; appropriate for diarized conversations where speaker order may differ from reference |
| BERTScore F1 | Semantic overlap between each STT hypothesis and the **gold** reference from `dataset.pickle` (not hypothesis-vs-hypothesis). With `--ref-run-id`, both runs are scored vs gold and deltas are reported | 0.0 → 1.0 | Captures paraphrase and domain-appropriate synonyms that WER misses |
| LLM-judge | Comparative quality score from a language model judge | Varies | Exploratory; model not clinically fine-tuned, results should be interpreted as directional only |

**On WER thresholds for clinical transcription:** General-purpose ASR systems typically achieve WER 5–15% on clean speech. Medical dialogue is harder — drug names, dosages, and clinical acronyms (SSRI, GAD, DSM-5) increase error rates. A WER under 20% on medical dialogue without domain fine-tuning is a reasonable baseline. cpWER is preferred over WER when speaker diarization labels are available.

---

## Machine Requirements

### Minimum (functional)

- macOS (Apple Silicon required for `mlx-whisper`)
- Python 3.11+
- 16 GB RAM
- 30 GB+ free disk (models + artifacts + DB)
- 8+ CPU cores

### Recommended

- Apple M-series (M2 Pro or later)
- 32–48 GB RAM
- Fast SSD

### Required software

```bash
brew install ffmpeg
brew install --cask docker
# Install Ollama from https://ollama.com
```

---

## Quick Start

```bash
cp .env.example .env
# Edit .env: POSTGRES_*, STT_*, OLLAMA_*, and path variables

make install
docker compose --env-file .env up -d
make db-init
```

Pull required local models:

```bash
ollama pull gemma3:12b                    # LLM judge
ollama pull medaibase/medgemma1.5:4b      # Insights extraction
```

Place dataset under `data/raw/`:
- `dataset.pickle` — gold transcripts (see [Dataset](#dataset))
- `recordings/*.wav` — audio files
- `transcripts/transcribed/<sample_id>.json` — optional speaker-tagged dialogue (enables cpWER and per-speaker WER)

---

## Dataset

**Source:** [nazmulkazi / dataset_automated_medical_transcription](https://github.com/nazmulkazi/dataset_automated_medical_transcription?tab=readme-ov-file) — automated medical transcription benchmark (download and preparation steps are in that repository’s README).

Expected local file: `data/raw/dataset.pickle` (gold reference text for evaluation metrics).

**Dataset limitations:** This dataset contains scripted medical dialogue, not naturalistic psychiatric sessions. It does not include real diarized patient-therapist conversations. The pipeline is built to work with real clinical recordings — this dataset is a proxy for development and benchmarking.

---

## Commands Reference

Command cheat sheet (copy-paste examples): [docs/cli.md](docs/cli.md).

Activate environment first:

```bash
source .venv/bin/activate
```

### Validate dataset

```bash
python -m src.cli.main validate-dataset
```

### Run STT

```bash
python -m src.cli.main run-stt --limit 5                      # Speed profile (turbo)
python -m src.cli.main run-stt --profile quality --limit 5    # Quality profile (large-v3)
python -m src.cli.main run-stt --flavor both --limit 5        # Both profiles on same files
```

### Evaluate transcript quality

```bash
# Lexical metrics (WER/CER/MER/WIL/cpWER)
python -m src.cli.main run-eval --run-id <UUID> --workers auto
python -m src.cli.main run-eval --run-id <CANDIDATE> --ref-run-id <BASELINE> --workers auto

# Semantic metric (BERTScore)
python -m src.cli.main run-bertscore --run-id <UUID>
python -m src.cli.main run-bertscore --run-id <CANDIDATE> --ref-run-id <BASELINE>

# LLM comparative judgment
python -m src.cli.main run-llm-judge --run-id <CANDIDATE> --ref-run-id <BASELINE>

# Word alignment visualization
python -m src.cli.main show-alignment --run-id <UUID> --limit 5
```

### Extract psychiatry insights

```bash
python -m src.cli.main insights-extract --run-id <UUID> --limit 5
python -m src.cli.main insights-extract --run-id <UUID> --model medaibase/medgemma1.5:4b
python -m src.cli.main insights-extract --run-id <UUID> --sample-id D0420-S1-T01
```

### Analysis notebooks

| Notebook | What it shows |
| --- | --- |
| `analysis/model_eval_insights.ipynb` | Side-by-side metric comparison between candidate and baseline runs; per-sample delta plots |
| `analysis/show_alignment_visualizer.ipynb` | Word-level alignment blocks showing substitution/insertion/deletion patterns per sample |
| `analysis/gold_speaker_timeline.ipynb` | Speaker turn distribution from gold transcripts — therapist vs patient speech patterns and turn lengths |
| `analysis/statistical_significance.ipynb` | Paired Wilcoxon signed-rank tests and bootstrap 95% CIs on metric deltas; requires N ≥ 5 samples |
| `analysis/insights_quality.ipynb` | Guardrail drop-rate (hallucination proxy), schema adherence, claim category distribution, cross-run insight consistency |

```bash
jupyter notebook analysis/model_eval_insights.ipynb
jupyter notebook analysis/show_alignment_visualizer.ipynb
jupyter notebook analysis/gold_speaker_timeline.ipynb
jupyter notebook analysis/statistical_significance.ipynb
jupyter notebook analysis/insights_quality.ipynb
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

## Artifacts and Outputs

| Artifact | Location |
| --- | --- |
| STT transcripts | `data/generated_transcripts/<run_id>_<timestamp>/` |
| Eval reports | `data/processed/<evalname>_<timestamp>.txt` |
| Insights JSON | `data/processed/insights_extract/insights_extract_<run_id>_<timestamp>.json` |
| Executed notebooks | `data/processed/analysis_notebooks/` |
| Pipeline summary | `data/processed/full_pipeline/full_pipeline_<timestamp>.md` |

---

## Database Schema

Operational tables under the `clinical_ai` schema:

| Table | Contents |
| --- | --- |
| `stt_runs` | Run-level metadata: model, provider, scope, parameters, timestamp |
| `stt_outputs` | Transcript text per sample per run |
| `evaluation_metrics` | Metric rows with full JSONB detail payloads for auditability |
| `transcript_insights` | Extracted insight JSONB (`prompt_version` duplicated inside the payload for self-contained exports), raw model output, and `prompt_version` / `insight_model` columns |

Apply migrations:

```bash
make db-init
```

Migration files applied in order: `001_init.sql` → `003_stt_runs.sql` → `004_stt_remove_created_at_and_backfill_model.sql` → `005_stt_run_scope.sql` → `006_transcript_insights.sql`.

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
| `BERTSCORE_MODEL` | Encoder for BERTScore (default: `roberta-large`) |

---

## Testing

```bash
make test
```

Unit tests cover each metric module (WER, CER, MER, WIL, cpWER, BERTScore, alignment, text normalization). Integration tests cover CLI commands and filesystem interactions. PostgreSQL-dependent tests are skipped automatically if the DB is unreachable (`@pytest.mark.postgres`).

---

## Known Limitations

- **Dataset is synthetic**: Gold transcripts are scripted medical dialogue, not naturalistic psychiatric sessions. Clinical audio with real speech patterns, crosstalk, and domain-specific jargon would produce different benchmarks.
- **Whisper struggles with medical terminology**: Drug names, dosages, and acronyms (SSRI, DSM-5, GAD, PTSD) have elevated error rates without domain fine-tuning. These are the highest-value terms in psychiatric transcription.
- **BERTScore encoder is general-purpose**: `roberta-large` was not trained on clinical text. A biomedical encoder (BioBERT, SciBERT) would produce more meaningful semantic similarity scores for this domain.
- **Insights confidence is model self-reported**: The `confidence` field comes from the LLM's own output — it is not a calibrated probability.
- **Apple Silicon only for STT**: `mlx-whisper` requires Apple Silicon. Evaluation and insights extraction run on any platform.
- **Sequential pipeline**: Steps run sequentially. Production-scale deployment would require async execution, batching, and a job queue.
- **Small-sample benchmarks**: Point estimates without statistical significance tests are directional only. Use `analysis/statistical_significance.ipynb` (or larger `run_full_pipeline.py --limit`) before drawing conclusions about model differences.

---

## Design Decisions

| Decision | Rationale |
| --- | --- |
| Local-first (no cloud APIs) | Data privacy: clinical audio should not leave the machine during development |
| Two STT profiles | Controlled comparison of speed/quality tradeoff against a shared gold reference |
| Evidence guardrail on insights | Clinical risk: the system should only surface claims grounded in what was actually said |
| PostgreSQL with JSONB details | Full payload storage enables retroactive analysis and metric recomputation without re-running STT |
| Resumable pipeline state | Local model inference is slow enough that mid-run failures on small samples are a real operational concern |
| ProcessPoolExecutor for eval | Each sample evaluation is CPU-bound and independent — process-level parallelism bypasses the Python GIL |
