import logging
from pathlib import Path
from typing import Any

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.cer import character_error_breakdown
from src.evaluation.mer_wil import word_mer_wil_breakdown
from src.evaluation.speaker_wer import compute_speaker_wer_for_sample
from src.evaluation.transcribed_json import transcribed_json_path
from src.evaluation.wer import word_error_breakdown
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader

logger = logging.getLogger(__name__)


def evaluate_stt_against_gold(
    limit: int | None = None,
    run_id: str | None = None,
    ref_run_id: str | None = None,
) -> None:
    """Compare STT outputs against gold: WER, CER, MER, WIL; per-speaker when JSON exists.

    Gold references are sourced primarily from dataset.pickle. Per-speaker metrics use
    ``transcripts/transcribed/<sample_id>.json`` for speaker labels.
    """
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    pickle_loader = DatasetPickleLoader(settings.dataset_pickle_path)
    gold_transcripts = pickle_loader.load_transcripts()
    analytics = AnalyticsRepository()
    samples = loader.load_samples()
    samples_by_id = {s.sample_id: s for s in samples}
    run_outputs = analytics.get_stt_outputs_for_run(run_id) if run_id else {}
    ref_run_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    evaluated = 0
    if run_id:
        candidate_sample_ids = list(run_outputs.keys())
        if ref_run_id:
            candidate_sample_ids = [sid for sid in candidate_sample_ids if sid in ref_run_outputs]
    else:
        candidate_sample_ids = [s.sample_id for s in samples]

    for sample_id in candidate_sample_ids:
        if limit is not None and evaluated >= limit:
            break
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        reference = _resolve_gold_reference(sample.sample_id, gold_transcripts, sample.transcript_path)
        if not reference:
            continue
        hypothesis = run_outputs.get(sample.sample_id) if run_id else analytics.get_latest_stt_output(sample.sample_id)
        hypothesis = hypothesis or ""
        if not hypothesis:
            if not run_id:
                logger.warning("No STT output found in DB for sample_id=%s", sample.sample_id)
            continue
        breakdown = word_error_breakdown(reference=reference, hypothesis=hypothesis)
        wer = float(breakdown["wer"])
        cer_breakdown = character_error_breakdown(reference=reference, hypothesis=hypothesis)
        cer = float(cer_breakdown["cer"])
        mer_wil_breakdown = word_mer_wil_breakdown(reference=reference, hypothesis=hypothesis)
        mer = float(mer_wil_breakdown["mer"])
        wil = float(mer_wil_breakdown["wil"])
        details: dict = {
            "reference_source": "dataset.pickle",
            "run_id": run_id,
            "wer_breakdown": breakdown,
            "cer_breakdown": cer_breakdown,
            "mer_wil_breakdown": mer_wil_breakdown,
        }

        json_path = transcribed_json_path(settings.transcripts_dir, sample.sample_id)
        sp_wer = compute_speaker_wer_for_sample(
            gold_text=reference,
            hypothesis_text=hypothesis,
            transcribed_json_path=json_path,
        )
        if sp_wer:
            details["speaker_wer"] = sp_wer

        if ref_run_id:
            ref_hypothesis = ref_run_outputs.get(sample.sample_id, "")
            if not ref_hypothesis:
                continue
            ref_breakdown = word_error_breakdown(reference=reference, hypothesis=ref_hypothesis)
            ref_wer = float(ref_breakdown["wer"])
            delta = wer - ref_wer
            ref_cer_breakdown = character_error_breakdown(reference=reference, hypothesis=ref_hypothesis)
            ref_cer = float(ref_cer_breakdown["cer"])
            delta_cer = cer - ref_cer
            ref_mer_wil = word_mer_wil_breakdown(reference=reference, hypothesis=ref_hypothesis)
            ref_mer = float(ref_mer_wil["mer"])
            ref_wil = float(ref_mer_wil["wil"])
            delta_mer = mer - ref_mer
            delta_wil = wil - ref_wil
            details.update(
                {
                    "ref_run_id": ref_run_id,
                    "ref_wer": ref_wer,
                    "ref_wer_breakdown": ref_breakdown,
                    "delta_vs_ref": delta,
                    "ref_cer": ref_cer,
                    "ref_cer_breakdown": ref_cer_breakdown,
                    "delta_cer_vs_ref": delta_cer,
                    "ref_mer_wil_breakdown": ref_mer_wil,
                    "delta_mer_vs_ref": delta_mer,
                    "delta_wil_vs_ref": delta_wil,
                }
            )
            sp_ref = compute_speaker_wer_for_sample(
                gold_text=reference,
                hypothesis_text=ref_hypothesis,
                transcribed_json_path=json_path,
            )
            if sp_ref:
                details["speaker_wer_ref_run"] = sp_ref

            logger.info(
                "Sample %s WER run=%s: %.4f (S:%s I:%s D:%s) | CER: %.4f (S:%s I:%s D:%s chars) | "
                "MER=%.4f WIL=%.4f (R=%s H=%s) | "
                "ref=%s WER: %.4f (S:%s I:%s D:%s) | ref CER: %.4f (S:%s I:%s D:%s) | "
                "ref MER=%.4f WIL=%.4f | delta_WER=%.4f delta_CER=%.4f delta_MER=%.4f delta_WIL=%.4f",
                sample.sample_id,
                run_id,
                wer,
                breakdown["substitutions"],
                breakdown["insertions"],
                breakdown["deletions"],
                cer,
                cer_breakdown["substitutions"],
                cer_breakdown["insertions"],
                cer_breakdown["deletions"],
                mer,
                wil,
                mer_wil_breakdown["reference_word_count"],
                mer_wil_breakdown["hypothesis_word_count"],
                ref_run_id,
                ref_wer,
                ref_breakdown["substitutions"],
                ref_breakdown["insertions"],
                ref_breakdown["deletions"],
                ref_cer,
                ref_cer_breakdown["substitutions"],
                ref_cer_breakdown["insertions"],
                ref_cer_breakdown["deletions"],
                ref_mer,
                ref_wil,
                delta,
                delta_cer,
                delta_mer,
                delta_wil,
            )
            if sp_wer:
                _log_speaker_metrics_block(sample.sample_id, sp_wer, label="candidate")
            if sp_ref:
                _log_speaker_metrics_block(sample.sample_id, sp_ref, label="ref_run")
        else:
            logger.info(
                "Sample %s WER: %.4f (S:%s I:%s D:%s) | CER: %.4f (S:%s I:%s D:%s chars) | "
                "MER=%.4f WIL=%.4f (R=%s H=%s)",
                sample.sample_id,
                wer,
                breakdown["substitutions"],
                breakdown["insertions"],
                breakdown["deletions"],
                cer,
                cer_breakdown["substitutions"],
                cer_breakdown["insertions"],
                cer_breakdown["deletions"],
                mer,
                wil,
                mer_wil_breakdown["reference_word_count"],
                mer_wil_breakdown["hypothesis_word_count"],
            )
            if sp_wer:
                _log_speaker_metrics_block(sample.sample_id, sp_wer, label="candidate")

        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="wer",
            metric_value=wer,
            details=details,
        )
        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="cer",
            metric_value=cer,
            details=details,
        )
        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="mer",
            metric_value=mer,
            details=details,
        )
        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="wil",
            metric_value=wil,
            details=details,
        )
        evaluated += 1


def _fmt_sp_line(tag: str, sp: dict[str, Any]) -> str:
    w = sp.get("wer")
    wer_s = f"{float(w):.4f}" if w is not None else "n/a"
    return (
        f"{tag}: WER={wer_s} (S:{sp['substitutions']} I:{sp['insertions']} D:{sp['deletions']}, "
        f"n_ref={sp['reference_word_count']})"
    )


def _fmt_mer_wil_sp_line(tag: str, sp: dict[str, Any]) -> str:
    m = sp.get("mer")
    w = sp.get("wil")
    mer_s = f"{float(m):.4f}" if m is not None else "n/a"
    wil_s = f"{float(w):.4f}" if w is not None else "n/a"
    return (
        f"{tag}: MER={mer_s} WIL={wil_s} "
        f"(R={sp['reference_word_count']} H={sp['hypothesis_word_count']})"
    )


def _fmt_cer_sp_line(tag: str, sp: dict[str, Any]) -> str:
    c = sp.get("cer")
    cer_s = f"{float(c):.4f}" if c is not None else "n/a"
    return (
        f"{tag}: CER={cer_s} (S:{sp['substitutions']} I:{sp['insertions']} D:{sp['deletions']}, "
        f"n_ref_chars={sp['reference_character_count']})"
    )


def _log_speaker_metrics_block(sample_id: str, payload: dict[str, Any], label: str) -> None:
    s1 = payload["speaker_1"]
    s2 = payload["speaker_2"]
    logger.info(
        "Sample %s per-speaker [%s] %s | %s",
        sample_id,
        label,
        _fmt_sp_line("sp1", s1),
        _fmt_sp_line("sp2", s2),
    )
    logger.info(
        "Sample %s per-speaker [%s] %s | %s",
        sample_id,
        label,
        _fmt_mer_wil_sp_line("sp1", s1),
        _fmt_mer_wil_sp_line("sp2", s2),
    )
    cer_block = payload.get("cer")
    if cer_block:
        c1 = cer_block["speaker_1"]
        c2 = cer_block["speaker_2"]
        logger.info(
            "Sample %s per-speaker [%s] %s | %s",
            sample_id,
            label,
            _fmt_cer_sp_line("sp1", c1),
            _fmt_cer_sp_line("sp2", c2),
        )


def _resolve_gold_reference(sample_id: str, gold_transcripts: dict[str, str], transcript_path: Path | None) -> str | None:
    # Try direct sample ID match first, then normalized variants.
    candidates = [sample_id, sample_id.upper(), sample_id.replace("_", "-"), sample_id.replace("-", "_")]
    for candidate in candidates:
        if candidate in gold_transcripts:
            return gold_transcripts[candidate]

    # Fallback to transcript files if present.
    if transcript_path and transcript_path.exists():
        return Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    return None
