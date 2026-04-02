# Evaluation metrics

[Wiki home](Home) · [Speech-to-text models](Speech-to-text-models) · [LLM as judge](LLM-as-judge)

---

## Overview

Metrics compare **STT hypotheses** to **gold** text (primarily from `dataset.pickle`, with optional per-file overrides). Optional **speaker-tagged JSON** under `data/raw/transcripts/transcribed/` unlocks speaker-aware and permutation-aware scores.

Implementation lives under `src/evaluation/`. Lexical metrics use the same text normalization as WER unless noted.

---

## Lexical / alignment metrics

| Metric | What it measures | Interpretation |
|--------|-------------------|----------------|
| **WER** (word error rate) | Word-level edits (substitutions, insertions, deletions) vs reference length | Classic ASR score. **Lower is better.** Unbounded in pathological cases; typically reported 0–1+ for clean setups. |
| **CER** (character error rate) | Character-level edits | Catches spelling and short-token errors words might miss. **Lower is better.** |
| **MER** (match error rate) | \( \mathrm{MER} = E / (R + H) \) with \(E = S+I+D\), \(R\) = ref word count, \(H\) = hyp word count | Penalizes length mismatch differently from raw WER. Range **0–1** in this codebase when \(R+H>0\). **Lower is better.** |
| **WIL** (word information lost) | \( \mathrm{WIL} = 2 \times \mathrm{MER} \) | Same alignment as MER; scaled view of information loss. **Lower is better.** |
| **cpWER** (concatenated minimum-permutation WER) | Minimum WER over reference orderings built from **speaker blocks** in transcribed JSON | When diarization/speaker order is ambiguous, optimizes over speaker-block permutations (capped speaker count). **Lower is better.** Requires speaker-aware reference; else skipped or degenerates. |

Definitions for MER/WIL match the module docstring in `src/evaluation/mer_wil.py` (jiwer-style reporting).

cpWER logic: `src/evaluation/cp_wer.py` (permutation over speaker blocks, hypothesis unsegmented).

---

## Semantic metric: BERTScore

| Aspect | Detail |
|--------|--------|
| Role | Soft semantic overlap between hypothesis and reference using contextual embeddings |
| Outputs | Precision / recall / F1 (implementation uses `bert-score` + PyTorch) |
| Encoder | `BERTSCORE_MODEL` — code default `roberta-large`; `.env.example` may set a biomedical encoder (e.g. BioBERT) for clinical text |

**Higher** F1 is generally better for semantic fidelity; it is **not** a substitute for WER on rare tokens.

**References:**

- Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)

---

## LLM-as-judge (separate from BERTScore)

A **second**, non-metric rubric: an LLM compares transcripts for clinical safety and preference. See [LLM as judge](LLM-as-judge) — scores are **exploratory**, not a replacement for WER/CER.

---

## Where results go

- Aggregates and details are stored in PostgreSQL (`evaluation_metrics` with JSONB payloads).
- CLI `run-eval` / `run-bertscore` / `run-llm-judge` write human-readable reports under `data/processed/` (paths in the main README).

---

## See also

- Statistical follow-up: notebook `analysis/statistical_significance.ipynb` (paired tests, bootstrap CIs).
- Alignment visualization: `show-alignment` CLI and `analysis/show_alignment_visualizer.ipynb`.
