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

Most recent end-to-end run: **`data/processed/full_pipeline/full_pipeline_20260403T110743Z.md`**

| | |
| --- | --- |
| Command | `python run_full_pipeline.py --limit 43 --eval-workers 1` |
| Samples | `D0420-S1-T01` … `D0423-S1-T03` (N=43) |
| Baseline STT | `mlx-community/whisper-large-v3-turbo` · run `50a52f67` |
| Candidate STT | `mlx-community/whisper-large-v3-mlx` · run `e7fc04ce` |
| Insights | MedGemma `medaibase/medgemma1.5:4b` · 43 rows per run |

Mean lexical metrics (candidate vs gold):

| Metric | Candidate |
| --- | ---: |
| WER | 0.256 |
| CER | 0.216 |
| MER | 0.128 |
| WIL | 0.256 |
| cpWER | 0.196 |

Comparative semantic/judge summary:

| Metric | Result |
| --- | ---: |
| BERTScore F1 (`roberta-large`, candidate mean, N=43) | 0.199 |
| BERTScore F1 (`roberta-large`, baseline mean, N=43) | 0.197 |
| BERTScore ΔF1 (candidate−baseline mean, N=43) | +0.002 |
| LLM judge wins (candidate / baseline / ties) | 27 / 13 / 3 |
| LLM judge mean comparative delta (candidate−baseline, N=43) | +0.372 |

Lower is better for WER/CER/MER/WIL/cpWER. For BERTScore and LLM judge delta, higher is better for the candidate. See `analysis/statistical_significance.ipynb` before drawing significance conclusions.

BERTScore figures above are taken from `data/processed/run-bertscore_20260403T110608Z.txt`, which used encoder `roberta-large` (`model_type` in the artifact and current `.env`).

### Per-recording performance (all N=43 samples)

The table below lists **every scalar metric we store per sample** for the quality candidate vs. the turbo baseline (same runs as in [Latest benchmark](#latest-benchmark)): lexical scores vs gold, WER edit counts, per-speaker WER when `transcribed/*.json` exists, BERTScore F1 vs gold, and LLM-as-judge comparative scores. Subscripts **`_c`** = candidate (large-v3-mlx), **`_b`** = baseline (large-v3-turbo). **WER S/I/D** = substitutions / insertions / deletions (word counts). **LLM_Δ** = candidate overall score minus baseline (0–10 scale). Empty speaker-WER cells mean that sample had no usable diarization JSON.

Canonical copy (for diffs and regeneration): [`data/processed/reports/per_sample_benchmark_N43.md`](data/processed/reports/per_sample_benchmark_N43.md). Refresh after new eval artifacts:

`python scripts/export_per_sample_benchmark_md.py`

| sample_id | WER_c | WER_b | WER_S_c | WER_I_c | WER_D_c | WER_S_b | WER_I_b | WER_D_b | CER_c | CER_b | MER_c | MER_b | WIL_c | WIL_b | cpWER_c | cpWER_b | spk1_WER_c | spk2_WER_c | spk1_WER_b | spk2_WER_b | BERT_F1_c | BERT_F1_b | BERT_ΔF1 | LLM_Δ | LLM_winner | LLM_score_c | LLM_score_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0420-S1-T01 | 0.239 | 0.190 | 203 | 104 | 420 | 115 | 26 | 436 | 0.244 | 0.207 | 0.126 | 0.102 | 0.253 | 0.204 | 0.122 | 0.061 | 0.302 | 0.210 | 0.253 | 0.160 | 0.009 | 0.011 | -0.002 | 3.00 | candidate | 9.0 | 6.0 |
| D0420-S1-T02 | 0.180 | 0.164 | 20 | 6 | 78 | 21 | 6 | 68 | 0.180 | 0.166 | 0.096 | 0.087 | 0.192 | 0.173 | 0.076 | 0.058 | 0.170 | 0.195 | 0.150 | 0.186 | 0.089 | 0.085 | 0.004 | -1.00 | tie | 9.0 | 10.0 |
| D0420-S1-T03 | 0.151 | 0.168 | 33 | 2 | 122 | 33 | 4 | 137 | 0.159 | 0.166 | 0.080 | 0.090 | 0.161 | 0.179 | 0.056 | 0.074 | 0.170 | 0.136 | 0.176 | 0.161 | 0.158 | 0.162 | -0.004 | -2.00 | baseline | 7.0 | 9.0 |
| D0420-S1-T04 | 0.281 | 0.144 | 109 | 200 | 164 | 58 | 10 | 175 | 0.259 | 0.134 | 0.139 | 0.076 | 0.278 | 0.152 | 0.222 | 0.064 | 0.203 | 0.318 | 0.171 | 0.132 | 0.361 | 0.369 | -0.008 | -2.00 | baseline | 7.0 | 9.0 |
| D0420-S1-T05 | 0.158 | 0.161 | 101 | 21 | 290 | 111 | 25 | 284 | 0.156 | 0.159 | 0.083 | 0.085 | 0.167 | 0.170 | 0.061 | 0.066 | 0.160 | 0.156 | 0.158 | 0.166 | 0.105 | 0.106 | -0.001 | 2.00 | candidate | 9.0 | 7.0 |
| D0420-S2-T01 | 0.209 | 0.131 | 351 | 451 | 680 | 266 | 54 | 609 | 0.183 | 0.122 | 0.106 | 0.068 | 0.212 | 0.136 | 0.150 | 0.064 | 0.238 | 0.192 | 0.135 | 0.128 | 0.172 | 0.175 | -0.003 | 2.00 | candidate | 9.0 | 7.0 |
| D0420-S3-T01 | 0.231 | 0.140 | 117 | 192 | 184 | 88 | 14 | 197 | 0.197 | 0.126 | 0.116 | 0.073 | 0.231 | 0.147 | 0.170 | 0.064 | 0.234 | 0.230 | 0.196 | 0.111 | 0.286 | 0.297 | -0.011 | -3.00 | baseline | 6.0 | 9.0 |
| D0420-S3-T02 | 0.281 | 0.180 | 311 | 424 | 903 | 209 | 50 | 792 | 0.257 | 0.187 | 0.146 | 0.096 | 0.293 | 0.192 | 0.177 | 0.057 | 0.424 | 0.229 | 0.318 | 0.130 | -0.017 | -0.006 | -0.011 | 2.00 | candidate | 8.0 | 6.0 |
| D0420-S3-T03 | 0.264 | 0.087 | 383 | 560 | 183 | 115 | 44 | 212 | 0.197 | 0.076 | 0.126 | 0.044 | 0.252 | 0.089 | 0.236 | 0.047 | 0.337 | 0.234 | 0.118 | 0.075 | 0.402 | 0.468 | -0.066 | 2.00 | candidate | 9.0 | 7.0 |
| D0420-S4-T01 | 0.285 | 0.182 | 206 | 274 | 274 | 137 | 42 | 303 | 0.245 | 0.173 | 0.143 | 0.096 | 0.285 | 0.192 | 0.213 | 0.093 | 0.419 | 0.233 | 0.205 | 0.174 | 0.146 | 0.135 | 0.011 | -3.00 | baseline | 6.0 | 9.0 |
| D0420-S4-T02 | 0.196 | 0.120 | 523 | 585 | 526 | 373 | 169 | 459 | 0.167 | 0.102 | 0.098 | 0.061 | 0.195 | 0.122 | 0.158 | 0.078 | 0.269 | 0.186 | 0.258 | 0.101 | 0.237 | 0.230 | 0.007 | 2.00 | candidate | 9.0 | 7.0 |
| D0421-S1-T01 | 0.210 | 0.136 | 371 | 282 | 689 | 227 | 57 | 584 | 0.186 | 0.132 | 0.108 | 0.071 | 0.216 | 0.141 | 0.139 | 0.056 | 0.223 | 0.196 | 0.143 | 0.128 | 0.108 | 0.143 | -0.035 | 2.00 | candidate | 8.0 | 6.0 |
| D0421-S1-T02 | 0.233 | 0.134 | 325 | 409 | 438 | 200 | 56 | 418 | 0.215 | 0.128 | 0.117 | 0.069 | 0.233 | 0.139 | 0.169 | 0.062 | 0.198 | 0.253 | 0.161 | 0.118 | 0.093 | 0.091 | 0.002 | 2.00 | candidate | 9.0 | 7.0 |
| D0421-S1-T03 | 0.288 | 0.116 | 197 | 407 | 188 | 81 | 36 | 202 | 0.224 | 0.106 | 0.139 | 0.060 | 0.277 | 0.120 | 0.241 | 0.053 | 0.196 | 0.338 | 0.159 | 0.093 | 0.228 | 0.234 | -0.006 | 2.00 | candidate | 8.0 | 6.0 |
| D0421-S1-T04 | 0.289 | 0.095 | 442 | 300 | 266 | 111 | 32 | 189 | 0.225 | 0.089 | 0.144 | 0.049 | 0.288 | 0.097 | 0.252 | 0.045 | 0.296 | 0.286 | 0.155 | 0.076 | 0.229 | 0.170 | 0.059 | 3.00 | candidate | 9.0 | 6.0 |
| D0421-S1-T05 | 0.251 | 0.081 | 233 | 426 | 116 | 90 | 37 | 123 | 0.183 | 0.066 | 0.120 | 0.041 | 0.239 | 0.082 | 0.227 | 0.047 | 0.428 | 0.198 | 0.120 | 0.070 | 0.312 | 0.293 | 0.019 | -3.00 | baseline | 6.0 | 9.0 |
| D0421-S2-T01 | 0.158 | 0.141 | 405 | 407 | 500 | 347 | 106 | 719 | 0.135 | 0.125 | 0.079 | 0.073 | 0.158 | 0.146 | 0.111 | 0.094 | 0.152 | 0.163 | 0.150 | 0.132 | 0.284 | 0.284 | -0.000 | 2.00 | candidate | 9.0 | 7.0 |
| D0421-S2-T02 | 0.214 | 0.148 | 506 | 415 | 622 | 362 | 113 | 594 | 0.187 | 0.138 | 0.109 | 0.077 | 0.217 | 0.154 | 0.154 | 0.080 | 0.251 | 0.196 | 0.149 | 0.148 | 0.223 | 0.226 | -0.003 | 2.00 | candidate | 8.0 | 6.0 |
| D0421-S3-T01 | 0.385 | 0.162 | 457 | 774 | 306 | 274 | 70 | 301 | 0.279 | 0.131 | 0.182 | 0.083 | 0.364 | 0.166 | 0.349 | 0.107 | 0.385 | 0.385 | 0.174 | 0.156 | 0.224 | 0.226 | -0.002 | -3.00 | baseline | 6.0 | 9.0 |
| D0421-S3-T02 | 0.312 | 0.160 | 447 | 543 | 168 | 255 | 110 | 230 | 0.237 | 0.126 | 0.149 | 0.082 | 0.297 | 0.163 | 0.287 | 0.125 | 0.220 | 0.350 | 0.138 | 0.170 | 0.146 | 0.146 | -0.000 | 2.00 | candidate | 8.0 | 6.0 |
| D0421-S3-T03 | 0.325 | 0.167 | 471 | 706 | 221 | 241 | 259 | 221 | 0.233 | 0.124 | 0.154 | 0.083 | 0.307 | 0.167 | 0.298 | 0.130 | 0.350 | 0.313 | 0.137 | 0.182 | 0.146 | 0.182 | -0.037 | 2.00 | candidate | 8.0 | 6.0 |
| D0421-S3-T04 | 0.269 | 0.118 | 245 | 545 | 269 | 159 | 40 | 264 | 0.220 | 0.103 | 0.130 | 0.061 | 0.260 | 0.121 | 0.225 | 0.063 | 0.248 | 0.286 | 0.127 | 0.110 | 0.153 | 0.167 | -0.014 | -3.00 | baseline | 6.0 | 9.0 |
| D0421-S3-T05 | 0.234 | 0.108 | 360 | 486 | 174 | 203 | 75 | 193 | 0.172 | 0.087 | 0.113 | 0.055 | 0.226 | 0.110 | 0.212 | 0.077 | 0.366 | 0.185 | 0.152 | 0.092 | 0.470 | 0.437 | 0.033 | 2.00 | candidate | 9.0 | 7.0 |
| D0422-S1-T01 | 0.262 | 0.176 | 571 | 530 | 629 | 493 | 106 | 563 | 0.218 | 0.148 | 0.132 | 0.091 | 0.264 | 0.182 | 0.207 | 0.114 | 0.270 | 0.255 | 0.210 | 0.148 | 0.186 | 0.181 | 0.005 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S1-T02 | 0.285 | 0.135 | 478 | 481 | 322 | 251 | 70 | 285 | 0.218 | 0.109 | 0.140 | 0.069 | 0.280 | 0.138 | 0.247 | 0.087 | 0.378 | 0.241 | 0.157 | 0.125 | 0.314 | 0.309 | 0.005 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S1-T03 | 0.248 | 0.165 | 353 | 226 | 423 | 218 | 48 | 402 | 0.239 | 0.152 | 0.127 | 0.087 | 0.254 | 0.173 | 0.175 | 0.083 | 0.298 | 0.227 | 0.247 | 0.131 | 0.112 | 0.114 | -0.002 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S1-T04 | 0.314 | 0.142 | 446 | 363 | 206 | 190 | 51 | 216 | 0.240 | 0.121 | 0.153 | 0.073 | 0.307 | 0.145 | 0.283 | 0.090 | 0.495 | 0.222 | 0.228 | 0.097 | 0.090 | 0.080 | 0.010 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S2-T01 | 0.275 | 0.151 | 370 | 295 | 338 | 174 | 108 | 267 | 0.228 | 0.134 | 0.139 | 0.077 | 0.277 | 0.154 | 0.226 | 0.092 | 0.368 | 0.233 | 0.159 | 0.147 | 0.249 | 0.245 | 0.003 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S2-T02 | 0.363 | 0.222 | 1528 | 1155 | 1913 | 711 | 204 | 1893 | 0.318 | 0.214 | 0.187 | 0.119 | 0.374 | 0.238 | 0.272 | 0.100 | 0.387 | 0.344 | 0.249 | 0.200 | 0.199 | 0.208 | -0.009 | 0.00 | tie | 0.0 | 0.0 |
| D0422-S3-T01 | 0.266 | 0.153 | 670 | 622 | 985 | 365 | 192 | 749 | 0.230 | 0.142 | 0.136 | 0.079 | 0.272 | 0.158 | 0.203 | 0.076 | 0.257 | 0.273 | 0.173 | 0.137 | -0.145 | -0.183 | 0.038 | 2.00 | candidate | 9.0 | 7.0 |
| D0422-S3-T02 | 0.356 | 0.186 | 398 | 385 | 456 | 124 | 53 | 470 | 0.310 | 0.195 | 0.180 | 0.099 | 0.360 | 0.198 | 0.277 | 0.067 | 0.359 | 0.354 | 0.204 | 0.171 | -0.023 | -0.033 | 0.009 | -4.00 | baseline | 4.0 | 8.0 |
| D0422-S3-T03 | 0.267 | 0.122 | 527 | 479 | 311 | 227 | 57 | 319 | 0.201 | 0.102 | 0.131 | 0.063 | 0.263 | 0.126 | 0.230 | 0.072 | 0.276 | 0.265 | 0.207 | 0.099 | 0.103 | 0.099 | 0.004 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S3-T04 | 0.258 | 0.147 | 174 | 334 | 276 | 131 | 25 | 289 | 0.227 | 0.142 | 0.128 | 0.077 | 0.256 | 0.153 | 0.189 | 0.062 | 0.266 | 0.252 | 0.166 | 0.131 | 0.146 | 0.175 | -0.029 | -4.00 | baseline | 4.0 | 8.0 |
| D0422-S3-T05 | 0.350 | 0.119 | 155 | 217 | 301 | 72 | 35 | 121 | 0.289 | 0.110 | 0.179 | 0.061 | 0.358 | 0.121 | 0.312 | 0.062 | 0.543 | 0.210 | 0.137 | 0.105 | 0.384 | 0.364 | 0.020 | -3.00 | baseline | 6.0 | 9.0 |
| D0422-S3-T06 | 0.269 | 0.254 | 40 | 13 | 198 | 42 | 10 | 185 | 0.284 | 0.270 | 0.149 | 0.140 | 0.298 | 0.280 | 0.098 | 0.079 | 0.268 | 0.270 | 0.249 | 0.260 | -0.018 | -0.078 | 0.060 | -3.00 | baseline | 6.0 | 9.0 |
| D0422-S4-T01 | 0.228 | 0.140 | 560 | 539 | 819 | 278 | 73 | 828 | 0.202 | 0.142 | 0.116 | 0.073 | 0.232 | 0.147 | 0.153 | 0.051 | 0.220 | 0.232 | 0.162 | 0.126 | 0.270 | 0.269 | 0.001 | 2.00 | candidate | 8.0 | 6.0 |
| D0422-S4-T02 | 0.222 | 0.136 | 548 | 452 | 611 | 291 | 70 | 628 | 0.180 | 0.127 | 0.112 | 0.071 | 0.224 | 0.142 | 0.158 | 0.061 | 0.261 | 0.189 | 0.144 | 0.129 | 0.292 | 0.312 | -0.020 | 2.00 | candidate | 9.0 | 7.0 |
| D0422-S4-T03 | 0.257 | 0.153 | 270 | 405 | 361 | 210 | 72 | 332 | 0.219 | 0.138 | 0.128 | 0.079 | 0.256 | 0.158 | 0.195 | 0.082 | 0.268 | 0.254 | 0.204 | 0.133 | 0.254 | 0.255 | -0.001 | 2.00 | candidate | 9.0 | 7.0 |
| D0422-S4-T04 | 0.262 | 0.124 | 406 | 405 | 289 | 173 | 64 | 285 | 0.216 | 0.116 | 0.129 | 0.064 | 0.258 | 0.128 | 0.218 | 0.067 | 0.327 | 0.252 | 0.290 | 0.099 | 0.280 | 0.270 | 0.010 | -3.00 | baseline | 6.0 | 9.0 |
| D0422-S4-T05 | 0.229 | 0.134 | 310 | 310 | 181 | 199 | 82 | 189 | 0.178 | 0.107 | 0.112 | 0.068 | 0.225 | 0.136 | 0.192 | 0.093 | 0.170 | 0.246 | 0.161 | 0.126 | 0.486 | 0.451 | 0.036 | -3.00 | baseline | 6.0 | 9.0 |
| D0423-S1-T01 | 0.213 | 0.135 | 600 | 529 | 910 | 308 | 92 | 893 | 0.188 | 0.135 | 0.109 | 0.071 | 0.218 | 0.141 | 0.141 | 0.053 | 0.199 | 0.223 | 0.142 | 0.131 | 0.186 | 0.198 | -0.012 | 0.00 | tie | 0.0 | 0.0 |
| D0423-S1-T02 | 0.193 | 0.123 | 302 | 332 | 392 | 207 | 86 | 362 | 0.168 | 0.109 | 0.097 | 0.063 | 0.194 | 0.127 | 0.142 | 0.066 | 0.194 | 0.193 | 0.119 | 0.127 | 0.266 | 0.229 | 0.037 | 2.00 | candidate | 9.0 | 7.0 |
| D0423-S1-T03 | 0.254 | 0.122 | 264 | 259 | 135 | 139 | 41 | 136 | 0.203 | 0.098 | 0.124 | 0.062 | 0.248 | 0.124 | 0.220 | 0.082 | 0.193 | 0.266 | 0.189 | 0.109 | 0.366 | 0.368 | -0.001 | 2.00 | candidate | 9.0 | 7.0 |

#### Column legend (metrics in the table above)

| Column | Meaning |
| --- | --- |
| sample_id | Recording id (matches `data/raw/recordings/` stem). |
| WER_c | Word error rate: **candidate** (quality STT) vs **gold** reference. Lower is better. |
| WER_b | Word error rate: **baseline** (turbo STT) vs gold. Lower is better. |
| WER_S_c | WER **substitutions** (word count): candidate vs gold. |
| WER_I_c | WER **insertions** (word count): candidate vs gold. |
| WER_D_c | WER **deletions** (word count): candidate vs gold. |
| WER_S_b | WER substitutions: baseline vs gold. |
| WER_I_b | WER insertions: baseline vs gold. |
| WER_D_b | WER deletions: baseline vs gold. |
| CER_c | Character error rate: candidate vs gold. Lower is better. |
| CER_b | Character error rate: baseline vs gold. |
| MER_c | Match error rate: candidate vs gold. Lower is better. |
| MER_b | Match error rate: baseline vs gold. |
| WIL_c | Word information lost: candidate vs gold. Lower is better. |
| WIL_b | Word information lost: baseline vs gold. |
| cpWER_c | cpWER (speaker-block permutation WER): candidate vs gold when `transcribed/*.json` exists. Lower is better. |
| cpWER_b | cpWER: baseline vs gold. |
| spk1_WER_c | Per-speaker WER for **speaker 1** (candidate); empty if no usable diarization JSON. |
| spk2_WER_c | Per-speaker WER for **speaker 2** (candidate). |
| spk1_WER_b | Per-speaker WER speaker 1 (baseline). |
| spk2_WER_b | Per-speaker WER speaker 2 (baseline). |
| BERT_F1_c | BERTScore F1: semantic overlap of **candidate** transcript vs gold (encoder in bertscore artifact). Higher is better. |
| BERT_F1_b | BERTScore F1: baseline transcript vs gold. |
| BERT_ΔF1 | **Candidate F1 − baseline F1** for this sample. Positive favors candidate in embedding space. |
| LLM_Δ | LLM judge: **candidate overall score − baseline overall score** (each 0–10 vs reference). Positive favors candidate. |
| LLM_winner | Judge label: `candidate`, `baseline`, or `tie`. |
| LLM_score_c | Judge overall score (0–10) for candidate vs reference. |
| LLM_score_b | Judge overall score (0–10) for baseline vs reference. |

**Note:** For WER/CER/MER/WIL/cpWER, lower is better. For BERT F1, higher is better. Rows with `LLM_score_*` = `0.0` and winner `tie` often mean a failed or placeholder judge call—check `judge_error` in `evaluation_metrics` (JSONB) or the llm-judge report artifact.

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

# Low RAM (sequential lexical eval + small BERTScore batches)
python run_full_pipeline.py --limit 43 --eval-workers 1 --bertscore-batch-size 1
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
- **No native diarization in Whisper**: Whisper does not perform speaker diarization out of the box. End-to-end diarization (who spoke when) was **not** implemented or evaluated in this pipeline’s scope; any per-speaker metrics in reports depend on separate diarization JSON when present, not on Whisper itself.
- **BERTScore encoder choice**: With the code default (`roberta-large`), scores reflect a general encoder not tuned for clinical text. The sample `.env` uses BioBERT (`dmis-lab/biobert-base-cased-v1.2`) for more domain-appropriate similarity; other biomedical encoders (e.g. SciBERT) are also reasonable overrides via `BERTSCORE_MODEL`.
- **Self-reported confidence**: The `confidence` field in insights comes from the LLM's own output — it is not a calibrated probability.
- **Apple Silicon required for STT**: `mlx-whisper` requires Apple Silicon. Evaluation and insights extraction run anywhere.
- **Sequential pipeline**: Steps run sequentially. Production deployment would require async execution, batching, and a job queue.
- **Small-sample benchmarks**: Point estimates over five files are directional only. Run `analysis/statistical_significance.ipynb` (or increase `--limit`) before drawing conclusions about model differences.

