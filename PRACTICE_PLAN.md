# Clinical AI Practice Plan (8-10 Hours)

## Goal
Complete the scaffold into a working, interview-ready clinical AI pipeline with measurable quality and reproducibility.

## Hour 0.0-1.0: Environment & Baseline Validation
- Configure `.env` from `.env.example`.
- Run `make install` and `docker compose --env-file .env up -d`.
- Confirm DB init with `make db-init`.
- Validate import health with `pytest -q`.

Acceptance checks:
- CLI loads with `python -m src.cli.main --help`.
- PostgreSQL is healthy and schema exists.

## Hour 1.0-2.5: Dataset Mapping & STT Reliability
- Implement robust filename-to-sample matching in `src/ingestion/dataset_loader.py`:
  - `DatasetLoader.load_samples()`
- Improve STT decoding and segmentation controls in `src/stt/faster_whisper_service.py`:
  - `FasterWhisperService.transcribe()`
- Add transcript artifact write-out in `src/stt/pipeline.py`:
  - `run_stt_pipeline()`

Acceptance checks:
- At least 3 recordings transcribed end-to-end.
- STT outputs stored in `clinical_ai.stt_outputs`.

## Hour 2.5-4.0: Analytics Persistence & Retrieval-Ready Data
- Add read methods in `src/analytics/repository.py`:
  - `get_latest_stt_output(sample_id)`
  - `upsert_chunk(...)`
- Implement chunk persistence in `src/rag/indexing.py`:
  - `build_rag_index()`
- Normalize clinical text preprocessing before chunking in `src/rag/chunking.py`:
  - `chunk_clinical_text(...)`

Acceptance checks:
- `document_chunks` populated with source-aligned chunk metadata.

## Hour 4.0-6.0: Hybrid Retrieval & Grounded Generation
- Implement dense retrieval SQL + vector similarity in `src/rag/hybrid_retriever.py`:
  - `HybridRetriever.search()`
- Implement BM25 corpus build and lexical scoring in same file.
- Add weighted fusion strategy and top-k rerank.
- Improve prompting in `src/llm/insight_generator.py`:
  - `InsightGenerator.generate_grounded_summary(...)`

Acceptance checks:
- Returned context includes clinically relevant chunks.
- Summaries include evidence-backed claims only.

## Hour 6.0-7.5: Evaluation (WER) + Judge Guardrails
- Wire DB hypothesis loading in `src/evaluation/stt_eval.py`:
  - `evaluate_stt_against_gold()`
- Improve WER normalization/tokenization strategy in `src/evaluation/wer.py`:
  - `word_error_rate(...)`
- Add strict judge output schema checks in `src/evaluation/judge.py`:
  - `OllamaJudge.evaluate(...)`
- Enforce fail-closed policy in `src/evaluation/guardrails.py`:
  - `apply_guardrails(...)`

Acceptance checks:
- WER metrics logged per sample.
- Guardrails return pass/fail with actionable reasons.

## Hour 7.5-9.0: Tests, Reliability, and Demo Readiness
- Expand `tests/unit/test_chunking.py` and `tests/unit/test_wer.py` with edge cases.
- Add integration tests for DB writes and CLI flow in `tests/integration`.
- Add deterministic test fixtures under `tests/fixtures` (optional extension).
- Update `README.md` with final runbook and troubleshooting notes.

Acceptance checks:
- `pytest -q` passes locally.
- `python -m src.cli.main run-all` executes without crashes on sample subset.

## Stretch (Hour 9.0-10.0): Interview Polish
- Add latency/cost metrics per stage in `src/analytics/repository.py`.
- Add a concise architecture section in `README.md` with tradeoffs:
  - Why local STT
  - Why pgvector + BM25
  - Why local judge for privacy and cost
- Prepare a 5-minute demo script:
  - Ingest -> STT -> Retrieval -> Grounded Summary -> Metrics

Acceptance checks:
- You can explain tradeoffs and quality controls clearly to a DS team lead.
