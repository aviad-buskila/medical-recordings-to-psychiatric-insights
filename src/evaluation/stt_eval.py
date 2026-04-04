"""Evaluation utilities for stt eval."""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.cp_wer import cp_wer_breakdown_from_json
from src.evaluation.cer import character_error_breakdown
from src.evaluation.mer_wil import word_mer_wil_breakdown
from src.evaluation.speaker_wer import compute_speaker_wer_for_sample
from src.evaluation.transcribed_json import transcribed_json_path
from src.evaluation.wer import word_error_breakdown
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader
from src.core.eval_run_report import EvalRunReporter

logger = logging.getLogger(__name__)


def evaluate_stt_against_gold(
    limit: int | None = None,
    run_id: str | None = None,
    ref_run_id: str | None = None,
    sample_id: str | None = None,
    reporter: EvalRunReporter | None = None,
    workers: int = 1,
    metrics: set[str] | None = None,
    skip_cp_wer: bool = False,
    skip_speaker_metrics: bool = False,
) -> None:
    """Compare STT outputs against gold: WER, CER, MER, WIL, cpWER; per-speaker when JSON exists.

    Gold references are sourced primarily from dataset.pickle. Speaker-aware metrics use
    ``transcripts/transcribed/<sample_id>.json``. **cpWER** minimizes WER over chronological
    reference and permutations of speaker-aggregated blocks (see ``cp_wer.py``).
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
    if reporter:
        reporter.set_run_metadata(
            run_id=run_id,
            ref_run_id=ref_run_id,
            sample_id=sample_id,
            candidate_run_info=analytics.get_stt_run_info(run_id) if run_id else None,
            ref_run_info=analytics.get_stt_run_info(ref_run_id) if ref_run_id else None,
            workers=workers,
            metrics=sorted(metrics) if metrics else None,
            skip_cp_wer=skip_cp_wer,
            skip_speaker_metrics=skip_speaker_metrics,
        )
    samples = loader.load_samples()
    samples_by_id = {s.sample_id: s for s in samples}
    run_outputs = analytics.get_stt_outputs_for_run(run_id) if run_id else {}
    ref_run_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}
    latest_outputs = analytics.get_latest_stt_outputs([s.sample_id for s in samples]) if not run_id else {}

    evaluated = 0
    any_cp_wer = False
    metric_rows_to_insert: list[dict[str, Any]] = []
    if run_id:
        candidate_sample_ids = list(run_outputs.keys())
        if ref_run_id:
            candidate_sample_ids = [sid for sid in candidate_sample_ids if sid in ref_run_outputs]
    else:
        candidate_sample_ids = [s.sample_id for s in samples]
    if sample_id:
        candidate_sample_ids = [sid for sid in candidate_sample_ids if sid == sample_id]

    jobs: list[dict[str, Any]] = []
    # CR: you override sample_id which is a method parameter , can cause bugs
    for sample_id in candidate_sample_ids:
        if limit is not None and len(jobs) >= limit:
            break
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        reference = _resolve_gold_reference(sample.sample_id, gold_transcripts, sample.transcript_path)
        if not reference:
            continue
        hypothesis = run_outputs.get(sample.sample_id) if run_id else latest_outputs.get(sample.sample_id)
        hypothesis = (hypothesis or "").strip()
        if not hypothesis:
            if not run_id:
                logger.warning("No STT output found in DB for sample_id=%s", sample.sample_id)
            continue
        ref_hypothesis = (ref_run_outputs.get(sample.sample_id, "") or "").strip() if ref_run_id else None
        if ref_run_id and not ref_hypothesis:
            continue
        jobs.append(
            {
                "sample_id": sample.sample_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "ref_hypothesis": ref_hypothesis,
                "run_id": run_id,
                "ref_run_id": ref_run_id,
                "transcripts_dir": settings.transcripts_dir,
                "metrics": sorted(metrics) if metrics else None,
                "skip_cp_wer": skip_cp_wer,
                "skip_speaker_metrics": skip_speaker_metrics,
            }
        )

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results_iter = ex.map(_compute_sample_eval_payload, jobs)
            results = [r for r in results_iter if r is not None]
    else:
        results = [r for r in (_compute_sample_eval_payload(job) for job in jobs) if r is not None]

    for result in results:
        sample_id = str(result["sample_id"])
        details = result["details"]
        wer = float(result["wer"])
        cer = float(result["cer"])
        mer = float(result["mer"])
        wil = float(result["wil"])
        cp_val = result.get("cp_val")
        enabled = result.get("enabled", {})

        _log_result_payload(
            sample_id=sample_id,
            run_id=run_id,
            ref_run_id=ref_run_id,
            payload=result,
        )

        def _insert_metric(metric_name: str, metric_value: float) -> None:
            metric_rows_to_insert.append(
                {
                    "sample_id": sample_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "details": details,
                }
            )
            if reporter:
                reporter.add_metric(
                    sample_id=sample_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    details=details,
                )

        if enabled.get("wer", True):
            _insert_metric(metric_name="wer", metric_value=wer)
        if result["has_cer"]:
            _insert_metric(metric_name="cer", metric_value=cer)
        if result["has_mer"]:
            _insert_metric(metric_name="mer", metric_value=mer)
        if result["has_wil"]:
            _insert_metric(metric_name="wil", metric_value=wil)
        if cp_val is not None:
            any_cp_wer = True
            _insert_metric(metric_name="cp_wer", metric_value=float(cp_val))
        evaluated += 1

    analytics.insert_eval_metrics_batch(metric_rows_to_insert)

    if reporter:
        metric_names: list[str] = []
        if any(r.get("enabled", {}).get("wer", True) for r in results):
            metric_names.append("wer")
        if any(r["has_cer"] for r in results):
            metric_names.append("cer")
        if any(r["has_mer"] for r in results):
            metric_names.append("mer")
        if any(r["has_wil"] for r in results):
            metric_names.append("wil")
        if any_cp_wer:
            metric_names.append("cp_wer")
        reporter.set_result_summary(
            evaluated=evaluated,
            metric_names=metric_names,
            skip_cp_wer=skip_cp_wer,
            skip_speaker_metrics=skip_speaker_metrics,
            workers=workers,
        )


def _log_cp_wer_lines(
    sample_id: str,
    candidate: dict[str, Any] | None,
    ref_payload: dict[str, Any] | None,
    delta_cp: float | None,
) -> None:
    if not candidate:
        return
    if ref_payload is not None:
        if delta_cp is not None:
            logger.info(
                "Sample %s cpWER=%.4f (best=%s, candidates=%s) | ref cpWER=%.4f | delta_cpWER=%.4f",
                sample_id,
                float(candidate["cp_wer"]),
                candidate["best_candidate_label"],
                candidate["candidates_evaluated"],
                float(ref_payload["cp_wer"]),
                delta_cp,
            )
        else:
            logger.info(
                "Sample %s cpWER=%.4f (best=%s, candidates=%s) | ref cpWER=%.4f",
                sample_id,
                float(candidate["cp_wer"]),
                candidate["best_candidate_label"],
                candidate["candidates_evaluated"],
                float(ref_payload["cp_wer"]),
            )
    else:
        logger.info(
            "Sample %s cpWER=%.4f (best=%s, candidates=%s)",
            sample_id,
            float(candidate["cp_wer"]),
            candidate["best_candidate_label"],
            candidate["candidates_evaluated"],
        )


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


# CR: notice code duplication with llm_judge_eval.py.
def _resolve_gold_reference(sample_id: str, gold_transcripts: dict[str, str], transcript_path: Path | None) -> str | None:
    # Try direct sample ID match first, then normalized variants.
    candidates = [sample_id, sample_id.upper(), sample_id.replace("_", "-"), sample_id.replace("-", "_")]
    for candidate in candidates:
        if candidate in gold_transcripts:
            return gold_transcripts[candidate]

    # Fallback to transcript files if present.
    if transcript_path and transcript_path.exists():
        # CR: errors="ignore" silently drops encoding issues, better to switch to warning
        return Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    return None


# CR: really long and complex function hard to read, worth seprating to high level funcationalitu- compute candidate, compute deltas, build details , compute samples eval
# CR: better have dataclass for job:dict: (not casting)
# @dataclass(frozen=True)
# class EvalJob:
#     sample_id: str
#     reference: str
#     hypothesis: str
#     run_id: str | None
def _compute_sample_eval_payload(job: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = str(job["sample_id"])
    reference = str(job["reference"])
    hypothesis = str(job["hypothesis"])
    run_id = job.get("run_id")
    ref_run_id = job.get("ref_run_id")
    ref_hypothesis = job.get("ref_hypothesis")
    transcripts_dir = str(job["transcripts_dir"])
    metrics_raw = job.get("metrics")
    metrics = set(metrics_raw) if isinstance(metrics_raw, list) else None
    skip_cp_wer = bool(job.get("skip_cp_wer", False))
    skip_speaker_metrics = bool(job.get("skip_speaker_metrics", False))
    # CR: each metric follows similiar logic, can we extract a common inteface and avoid all the if-else?
    metric_enabled = {
        "wer": metrics is None or "wer" in metrics,
        "cer": metrics is None or "cer" in metrics,
        "mer": metrics is None or "mer" in metrics,
        "wil": metrics is None or "wil" in metrics,
        "cp_wer": (metrics is None or "cp_wer" in metrics) and not skip_cp_wer,
    }
    # WER breakdown powers logs/details and may be needed by downstream fields.
    if not metric_enabled["wer"]:
        metric_enabled["wer"] = True

    # CR: normalize is called redundantly in word_error_breakdown? can we normalize once at beginning ? can improve performance maybe?
    breakdown = word_error_breakdown(reference=reference, hypothesis=hypothesis)
    wer = float(breakdown["wer"])
    cer_breakdown = character_error_breakdown(reference=reference, hypothesis=hypothesis) if metric_enabled["cer"] else None
    cer = float(cer_breakdown["cer"]) if cer_breakdown else 0.0
    mer_wil_breakdown = (
        word_mer_wil_breakdown(reference=reference, hypothesis=hypothesis)
        if metric_enabled["mer"] or metric_enabled["wil"]
        else None
    )
    mer = float(mer_wil_breakdown["mer"]) if mer_wil_breakdown else 0.0
    wil = float(mer_wil_breakdown["wil"]) if mer_wil_breakdown else 0.0
    details: dict[str, Any] = {
        "reference_source": "dataset.pickle",
        "run_id": run_id,
        "wer_breakdown": breakdown,
    }
    if cer_breakdown:
        details["cer_breakdown"] = cer_breakdown
    if mer_wil_breakdown:
        details["mer_wil_breakdown"] = mer_wil_breakdown

    json_path = transcribed_json_path(transcripts_dir, sample_id)
    sp_wer = (
        compute_speaker_wer_for_sample(
            gold_text=reference,
            hypothesis_text=hypothesis,
            transcribed_json_path=json_path,
        )
        if not skip_speaker_metrics
        else None
    )
    if sp_wer:
        details["speaker_wer"] = sp_wer

    cp_payload = cp_wer_breakdown_from_json(hypothesis, json_path) if metric_enabled["cp_wer"] else None
    cp_val: float | None = None
    if cp_payload:
        details["cp_wer"] = cp_payload
        cp_val = float(cp_payload["cp_wer"])

    ref_breakdown = None
    ref_cer_breakdown = None
    ref_mer_wil = None
    ref_cp_payload = None
    sp_ref = None
    delta_cp = None
    if ref_run_id and ref_hypothesis:
        ref_breakdown = word_error_breakdown(reference=reference, hypothesis=str(ref_hypothesis))
        ref_wer = float(ref_breakdown["wer"])
        delta = wer - ref_wer
        ref_cer_breakdown = (
            character_error_breakdown(reference=reference, hypothesis=str(ref_hypothesis))
            if metric_enabled["cer"]
            else None
        )
        ref_cer = float(ref_cer_breakdown["cer"]) if ref_cer_breakdown else 0.0
        delta_cer = cer - ref_cer if ref_cer_breakdown else None
        ref_mer_wil = (
            word_mer_wil_breakdown(reference=reference, hypothesis=str(ref_hypothesis))
            if metric_enabled["mer"] or metric_enabled["wil"]
            else None
        )
        ref_mer = float(ref_mer_wil["mer"]) if ref_mer_wil else 0.0
        ref_wil = float(ref_mer_wil["wil"]) if ref_mer_wil else 0.0
        delta_mer = mer - ref_mer if ref_mer_wil else None
        delta_wil = wil - ref_wil if ref_mer_wil else None
        ref_cp_payload = cp_wer_breakdown_from_json(str(ref_hypothesis), json_path) if metric_enabled["cp_wer"] else None
        if cp_payload and ref_cp_payload:
            delta_cp = float(cp_payload["cp_wer"]) - float(ref_cp_payload["cp_wer"])
        details.update(
            {
                "ref_run_id": ref_run_id,
                "ref_wer": ref_wer,
                "ref_wer_breakdown": ref_breakdown,
                "delta_vs_ref": delta,
            }
        )
        if ref_cer_breakdown:
            details["ref_cer"] = ref_cer
            details["ref_cer_breakdown"] = ref_cer_breakdown
            details["delta_cer_vs_ref"] = delta_cer
        if ref_mer_wil:
            details["ref_mer_wil_breakdown"] = ref_mer_wil
            details["delta_mer_vs_ref"] = delta_mer
            details["delta_wil_vs_ref"] = delta_wil
        if ref_cp_payload:
            details["ref_cp_wer"] = ref_cp_payload
        if delta_cp is not None:
            details["delta_cp_wer_vs_ref"] = delta_cp
        sp_ref = (
            compute_speaker_wer_for_sample(
                gold_text=reference,
                hypothesis_text=str(ref_hypothesis),
                transcribed_json_path=json_path,
            )
            if not skip_speaker_metrics
            else None
        )
        if sp_ref:
            details["speaker_wer_ref_run"] = sp_ref

    return {
        "sample_id": sample_id,
        "wer": wer,
        "cer": cer,
        "mer": mer,
        "wil": wil,
        "cp_val": cp_val,
        "details": details,
        "breakdown": breakdown,
        "cer_breakdown": cer_breakdown,
        "mer_wil_breakdown": mer_wil_breakdown,
        "cp_payload": cp_payload,
        "sp_wer": sp_wer,
        "sp_ref": sp_ref,
        "ref_breakdown": ref_breakdown,
        "ref_cer_breakdown": ref_cer_breakdown,
        "ref_mer_wil": ref_mer_wil,
        "ref_cp_payload": ref_cp_payload,
        "delta_cp": delta_cp,
        "enabled": metric_enabled,
        "has_cer": cer_breakdown is not None,
        "has_mer": mer_wil_breakdown is not None,
        "has_wil": mer_wil_breakdown is not None,
    }


def _log_result_payload(
    sample_id: str,
    run_id: str | None,
    ref_run_id: str | None,
    payload: dict[str, Any],
) -> None:
    breakdown = payload["breakdown"]
    cer_breakdown = payload.get("cer_breakdown")
    mer_wil_breakdown = payload.get("mer_wil_breakdown")
    wer = float(payload["wer"])
    cer = float(payload.get("cer", 0.0))
    mer = float(payload.get("mer", 0.0))
    wil = float(payload.get("wil", 0.0))
    cp_payload = payload.get("cp_payload")
    sp_wer = payload.get("sp_wer")
    sp_ref = payload.get("sp_ref")

    if ref_run_id:
        ref_breakdown = payload.get("ref_breakdown")
        ref_cer_breakdown = payload.get("ref_cer_breakdown")
        ref_mer_wil = payload.get("ref_mer_wil")
        if ref_breakdown and ref_cer_breakdown and ref_mer_wil:
            logger.info(
                "Sample %s WER run=%s: %.4f (S:%s I:%s D:%s) | CER: %.4f (S:%s I:%s D:%s chars) | "
                "MER=%.4f WIL=%.4f (R=%s H=%s) | "
                "ref=%s WER: %.4f (S:%s I:%s D:%s) | ref CER: %.4f (S:%s I:%s D:%s) | "
                "ref MER=%.4f WIL=%.4f | delta_WER=%.4f delta_CER=%.4f delta_MER=%.4f delta_WIL=%.4f",
                sample_id,
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
                float(ref_breakdown["wer"]),
                ref_breakdown["substitutions"],
                ref_breakdown["insertions"],
                ref_breakdown["deletions"],
                float(ref_cer_breakdown["cer"]),
                ref_cer_breakdown["substitutions"],
                ref_cer_breakdown["insertions"],
                ref_cer_breakdown["deletions"],
                float(ref_mer_wil["mer"]),
                float(ref_mer_wil["wil"]),
                wer - float(ref_breakdown["wer"]),
                cer - float(ref_cer_breakdown["cer"]),
                mer - float(ref_mer_wil["mer"]),
                wil - float(ref_mer_wil["wil"]),
            )
        elif ref_breakdown:
            logger.info(
                "Sample %s WER run=%s: %.4f (S:%s I:%s D:%s) | ref=%s WER: %.4f | delta_WER=%.4f",
                sample_id,
                run_id,
                wer,
                breakdown["substitutions"],
                breakdown["insertions"],
                breakdown["deletions"],
                ref_run_id,
                float(ref_breakdown["wer"]),
                wer - float(ref_breakdown["wer"]),
            )
    else:
        if cer_breakdown and mer_wil_breakdown:
            logger.info(
                "Sample %s WER: %.4f (S:%s I:%s D:%s) | CER: %.4f (S:%s I:%s D:%s chars) | "
                "MER=%.4f WIL=%.4f (R=%s H=%s)",
                sample_id,
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
        else:
            logger.info(
                "Sample %s WER: %.4f (S:%s I:%s D:%s)",
                sample_id,
                wer,
                breakdown["substitutions"],
                breakdown["insertions"],
                breakdown["deletions"],
            )

    _log_cp_wer_lines(
        sample_id,
        cp_payload,
        payload.get("ref_cp_payload") if ref_run_id else None,
        payload.get("delta_cp"),
    )
    if sp_wer:
        _log_speaker_metrics_block(sample_id, sp_wer, label="candidate")
    if sp_ref:
        _log_speaker_metrics_block(sample_id, sp_ref, label="ref_run")
