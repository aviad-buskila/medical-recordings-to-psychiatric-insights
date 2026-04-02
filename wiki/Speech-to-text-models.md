# Speech-to-text models

[Wiki home](Home) · [Evaluation metrics](Evaluation-metrics) · [Pipeline and other components](Pipeline-and-other-components)

---

## Provider and stack

| Item | Value |
|------|--------|
| Provider | `mlx-whisper` (`STT_PROVIDER`) |
| Runtime | [MLX](https://github.com/ml-explore/mlx) on **Apple Silicon** |
| Python integration | `mlx-whisper` package; models identified as Hugging Face–style repo ids |

Audio never leaves the machine; inference is local.

---

## Two benchmark profiles

The pipeline compares a **speed-oriented** profile against a **quality-oriented** profile. Defaults come from `src/config/settings.py` and `.env.example`.

| Role | Env / setting | Typical model id |
|------|----------------|------------------|
| **Baseline (default / “turbo”)** | `STT_MODEL`, `STT_MODEL_FALLBACK` | `mlx-community/whisper-large-v3-turbo` |
| **Candidate (quality)** | `STT_MLX_QUALITY_MODEL` | `mlx-community/whisper-large-v3-mlx` |
| **Quality fallback** | `STT_MLX_QUALITY_FALLBACK_MODEL` | Same as default turbo if the quality model fails to load |

CLI: `run-stt` with `--profile default` vs `--profile quality`, or `--flavor both` to run both on the same sample set (used by `run_full_pipeline.py`).

---

## Whisper family (what the names imply)

- **whisper-large-v3** — OpenAI Whisper “large-v3”–class weights, adapted for MLX.
- **Turbo** variant — Distilled / faster checkpoint (`whisper-large-v3-turbo`), trading some accuracy for throughput.
- **MLX** variant — Full large-v3–style model in the MLX ecosystem (`whisper-large-v3-mlx`), used as the “quality” arm in this project.

Exact hub cards: see the [MLX Community organization on Hugging Face](https://huggingface.co/mlx-community) and the specific repos named in settings.

---

## References

- [OpenAI Whisper paper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356) — baseline ASR architecture.
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — MLX Whisper inference.
- [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) and related MLX Community repos — pinned versions in `requirements.txt` / your environment govern behavior.

---

## See also

- Gold references and audio layout: [Pipeline and other components](Pipeline-and-other-components)
- How STT outputs are scored: [Evaluation metrics](Evaluation-metrics)
