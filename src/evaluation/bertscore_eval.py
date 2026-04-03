"""Standalone BERTScore evaluation: STT hypothesis vs gold reference from dataset.pickle.

Each ``bert_score`` call uses (candidates=hypothesis, references=gold). When ``ref_run_id``
is set, the tool scores *both* runs against the same gold and adds comparative deltas;
it does **not** score one hypothesis against the other.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.stt_eval import _resolve_gold_reference
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader
from src.core.eval_run_report import EvalRunReporter

logger = logging.getLogger(__name__)


def _resolve_num_layers_for_model(model_type: str) -> int | None:
    """Resolve num_layers for model ids missing from bert-score's built-in map."""
    try:
        from bert_score.utils import model2layers

        if model_type in model2layers:
            return None
    except Exception:
        pass

    try:
        from transformers import AutoConfig

        # For custom HF models, bert-score can run when we pass encoder depth explicitly.
        cfg = AutoConfig.from_pretrained(model_type)
        num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int) and num_hidden_layers > 0:
            return num_hidden_layers
    except Exception:
        pass

    return None


def run_bertscore_eval(
    run_id: str | None,
    ref_run_id: str | None,
    sample_id: str | None,
    limit: int | None,
    model_type: str | None,
    batch_size: int,
    rescale_with_baseline: bool,
    output_json: Path | None,
    reporter: EvalRunReporter | None = None,
) -> dict[str, Any]:
    """Compute BERTScore (P/R/F1) for each aligned (reference, hypothesis) pair.

    Uses the same gold resolution and STT run selection as ``run-eval``, but does not
    write rows to ``evaluation_metrics``.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise RuntimeError(
            "bert-score import failed. Install dependencies: pip install -r requirements.txt"
        ) from e

    settings = get_settings()
    resolved_model = model_type or settings.bertscore_model
    # Needed for biomedical/custom encoders that are absent from bert-score's model map.
    resolved_num_layers = _resolve_num_layers_for_model(resolved_model)

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
        )
    samples = loader.load_samples()
    samples_by_id = {s.sample_id: s for s in samples}
    run_outputs = analytics.get_stt_outputs_for_run(run_id) if run_id else {}
    ref_run_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    if run_id:
        candidate_sample_ids = list(run_outputs.keys())
        if ref_run_id:
            candidate_sample_ids = [sid for sid in candidate_sample_ids if sid in ref_run_outputs]
    else:
        candidate_sample_ids = [s.sample_id for s in samples]
    if sample_id:
        candidate_sample_ids = [sid for sid in candidate_sample_ids if sid == sample_id]

    sample_ids: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []
    ref_hyps: list[str] = []

    evaluated = 0
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
        if ref_run_id:
            ref_hypothesis = ref_run_outputs.get(sample.sample_id, "")
            if not ref_hypothesis:
                continue
        sample_ids.append(sample.sample_id)
        refs.append(reference)
        hyps.append(hypothesis)
        if ref_run_id:
            ref_hyps.append(ref_hypothesis)
        evaluated += 1

    if not sample_ids:
        logger.warning("No aligned samples for BERTScore (check run-id / gold / DB).")
        out: dict[str, Any] = {"samples": 0, "model_type": resolved_model}
        if reporter:
            reporter.set_result_summary(**out)
        if output_json:
            output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    p1, r1, f1 = bert_score_fn(
        hyps,
        refs,
        model_type=resolved_model,
        num_layers=resolved_num_layers,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        lang="en",
        use_fast_tokenizer=False,
    )
    p1_list = [float(x) for x in p1]
    r1_list = [float(x) for x in r1]
    f1_list = [float(x) for x in f1]

    for i, sid in enumerate(sample_ids):
        if ref_run_id:
            logger.info(
                "Sample %s BERTScore F1=%.4f P=%.4f R=%.4f (primary run)",
                sid,
                f1_list[i],
                p1_list[i],
                r1_list[i],
            )
        else:
            logger.info(
                "Sample %s BERTScore F1=%.4f P=%.4f R=%.4f",
                sid,
                f1_list[i],
                p1_list[i],
                r1_list[i],
            )

    summary: dict[str, Any] = {
        "metric": "bertscore",
        "reference_source": "dataset.pickle",
        "primary_scores": "stt_hypothesis_vs_gold",
        "model_type": resolved_model,
        "num_layers": resolved_num_layers,
        "rescale_with_baseline": rescale_with_baseline,
        "run_id": run_id,
        "ref_run_id": ref_run_id,
        "samples": len(sample_ids),
        "mean_precision": sum(p1_list) / len(p1_list),
        "mean_recall": sum(r1_list) / len(r1_list),
        "mean_f1": sum(f1_list) / len(f1_list),
        "per_sample": [
            {"sample_id": sid, "precision": p1_list[i], "recall": r1_list[i], "f1": f1_list[i]}
            for i, sid in enumerate(sample_ids)
        ],
    }

    if reporter:
        for i, sid in enumerate(sample_ids):
            reporter.add_metric(
                sample_id=sid,
                metric_name="bertscore_f1",
                metric_value=f1_list[i],
                details={
                    "precision": p1_list[i],
                    "recall": r1_list[i],
                    "run_id": run_id,
                    "ref_run_id": ref_run_id,
                    "model_type": resolved_model,
                    "num_layers": resolved_num_layers,
                    "rescale_with_baseline": rescale_with_baseline,
                },
            )

    if ref_run_id and len(ref_hyps) == len(sample_ids):
        p2, r2, f2 = bert_score_fn(
            ref_hyps,
            refs,
            model_type=resolved_model,
            num_layers=resolved_num_layers,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
            lang="en",
            use_fast_tokenizer=False,
        )
        p2_list = [float(x) for x in p2]
        r2_list = [float(x) for x in r2]
        f2_list = [float(x) for x in f2]
        for i, sid in enumerate(sample_ids):
            logger.info(
                "Sample %s BERTScore F1=%.4f P=%.4f R=%.4f (ref run) | delta_F1=%.4f",
                sid,
                f2_list[i],
                p2_list[i],
                r2_list[i],
                f1_list[i] - f2_list[i],
            )
        summary["ref_mean_precision"] = sum(p2_list) / len(p2_list)
        summary["ref_mean_recall"] = sum(r2_list) / len(r2_list)
        summary["ref_mean_f1"] = sum(f2_list) / len(f2_list)
        summary["mean_delta_f1_vs_ref"] = sum(f1_list[i] - f2_list[i] for i in range(len(sample_ids))) / len(
            sample_ids
        )
        for i, sid in enumerate(sample_ids):
            summary["per_sample"][i]["ref_precision"] = p2_list[i]
            summary["per_sample"][i]["ref_recall"] = r2_list[i]
            summary["per_sample"][i]["ref_f1"] = f2_list[i]
            summary["per_sample"][i]["delta_f1_vs_ref"] = f1_list[i] - f2_list[i]

        if reporter:
            for i, sid in enumerate(sample_ids):
                delta = f1_list[i] - f2_list[i]
                reporter.add_metric(
                    sample_id=sid,
                    metric_name="bertscore_ref_f1",
                    metric_value=f2_list[i],
                    details={
                        "precision": p2_list[i],
                        "recall": r2_list[i],
                        "run_id": run_id,
                        "ref_run_id": ref_run_id,
                        "model_type": resolved_model,
                        "num_layers": resolved_num_layers,
                        "rescale_with_baseline": rescale_with_baseline,
                    },
                )
                reporter.add_metric(
                    sample_id=sid,
                    metric_name="bertscore_delta_f1_vs_ref",
                    metric_value=delta,
                    details={
                        "primary_f1": f1_list[i],
                        "ref_f1": f2_list[i],
                        "run_id": run_id,
                        "ref_run_id": ref_run_id,
                        "model_type": resolved_model,
                        "num_layers": resolved_num_layers,
                        "rescale_with_baseline": rescale_with_baseline,
                    },
                )

    logger.info(
        "BERTScore aggregate: n=%s mean_F1=%.4f mean_P=%.4f mean_R=%.4f model=%s num_layers=%s",
        len(sample_ids),
        summary["mean_f1"],
        summary["mean_precision"],
        summary["mean_recall"],
        resolved_model,
        resolved_num_layers,
    )
    if "ref_mean_f1" in summary:
        logger.info(
            "BERTScore ref run aggregate: mean_F1=%.4f mean_delta_F1=%.4f",
            summary["ref_mean_f1"],
            summary["mean_delta_f1_vs_ref"],
        )

    if output_json:
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote BERTScore summary to %s", output_json)

    if reporter:
        reporter.set_result_summary(**summary)

    return summary
