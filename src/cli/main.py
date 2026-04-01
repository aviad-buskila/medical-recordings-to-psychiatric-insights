from pathlib import Path

import click

from src.config.settings import get_settings
from src.core.logging import configure_logging
from src.evaluation.alignment_report import run_alignment_report
from src.evaluation.bertscore_eval import run_bertscore_eval
from src.evaluation.llm_judge_eval import run_llm_judge_eval
from src.evaluation.stt_eval import evaluate_stt_against_gold
from src.ingestion.dataset_loader import DatasetLoader
from src.rag.indexing import build_rag_index
from src.stt.pipeline import run_stt_both_profiles, run_stt_pipeline


@click.group()
def cli() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@cli.command("validate-dataset")
def validate_dataset() -> None:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    summary = loader.validate_layout()
    click.echo(f"Dataset validation complete: {summary}")


@cli.command("run-stt")
@click.option("--limit", "-n", type=int, default=None, help="Process only N recordings.")
@click.option(
    "--profile",
    type=click.Choice(["default", "quality"], case_sensitive=False),
    default="default",
    show_default=True,
    help="STT profile: default (mlx turbo) or quality (mlx large-v3).",
)
@click.option(
    "--flavor",
    type=click.Choice(["single", "both"], case_sensitive=False),
    default="single",
    show_default=True,
    help="single: run one profile. both: run default+quality on same files.",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    default=False,
    help="Disable STT model fallback. Fail if selected model cannot be loaded.",
)
def run_stt(limit: int | None, profile: str, flavor: str, no_fallback: bool) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if flavor.lower() == "both":
        run_ids = run_stt_both_profiles(limit=limit, allow_fallback=not no_fallback)
        click.echo(
            "STT both-profiles completed. "
            f"default_run_id={run_ids['default']} quality_run_id={run_ids['quality']}"
        )
        return
    run_stt_pipeline(limit=limit, stt_profile=profile, allow_fallback=not no_fallback)
    click.echo("STT pipeline completed.")


@cli.command("build-rag-index")
def build_index() -> None:
    build_rag_index()
    click.echo("RAG index build scaffold completed.")


@cli.command("run-bertscore")
@click.option("--limit", "-n", type=int, default=None, help="Score only N samples.")
@click.option("--run-id", type=str, default=None, help="STT run_id for hypotheses (recommended).")
@click.option("--ref-run-id", type=str, default=None, help="Second STT run_id; reports delta vs primary.")
@click.option(
    "--model-type",
    type=str,
    default=None,
    help="BERTScore encoder (default: BERTSCORE_MODEL or roberta-large).",
)
@click.option("--batch-size", type=int, default=8, show_default=True, help="Batch size for scoring.")
@click.option(
    "--no-rescale",
    is_flag=True,
    default=False,
    help="Disable baseline rescaling (faster, scores not comparable to published baselines).",
)
@click.option(
    "--output-json",
    "-o",
    type=click.Path(path_type=Path, writable=True),
    default=None,
    help="Write full summary JSON to this path.",
)
def run_bertscore(
    limit: int | None,
    run_id: str | None,
    ref_run_id: str | None,
    model_type: str | None,
    batch_size: int,
    no_rescale: bool,
    output_json: Path | None,
) -> None:
    """Semantic overlap (BERTScore) vs gold; not persisted to evaluation_metrics (see README)."""
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if ref_run_id and not run_id:
        raise click.BadParameter("--run-id is required when --ref-run-id is provided")
    run_bertscore_eval(
        run_id=run_id,
        ref_run_id=ref_run_id,
        limit=limit,
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=not no_rescale,
        output_json=output_json,
    )
    click.echo("BERTScore evaluation completed.")


@cli.command("run-eval")
@click.option("--limit", "-n", type=int, default=None, help="Evaluate only N samples.")
@click.option("--run-id", type=str, default=None, help="Evaluate only outputs from a specific STT run_id.")
@click.option("--ref-run-id", type=str, default=None, help="Reference STT run_id for side-by-side WER comparison.")
def run_eval(limit: int | None, run_id: str | None, ref_run_id: str | None) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if ref_run_id and not run_id:
        raise click.BadParameter("--run-id is required when --ref-run-id is provided")
    evaluate_stt_against_gold(limit=limit, run_id=run_id, ref_run_id=ref_run_id)
    click.echo("Evaluation scaffold completed.")


@cli.command("run-all")
def run_all() -> None:
    validate_dataset()
    run_stt_pipeline(limit=None, stt_profile="default")
    build_index()
    evaluate_stt_against_gold(limit=None)
    click.echo("Full scaffold pipeline run complete.")


@cli.command("show-alignment")
@click.option("--run-id", type=str, required=True, help="STT run_id whose transcripts align to gold.")
@click.option(
    "--ref-run-id",
    type=str,
    default=None,
    help="Optional second run_id; shows a second GOLD vs HYP block for the same samples.",
)
@click.option("--limit", "-n", type=int, default=None, help="Only first N aligned samples.")
@click.option(
    "--sample-id",
    multiple=True,
    default=None,
    help="Restrict to one or more sample_ids (repeat flag).",
)
@click.option(
    "--chunk-columns",
    type=int,
    default=48,
    show_default=True,
    help="Alignment table width in word columns before wrapping.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    default=None,
    help="Write report to this UTF-8 file (still prints to stdout).",
)
def show_alignment(
    run_id: str,
    ref_run_id: str | None,
    limit: int | None,
    sample_id: tuple[str, ...] | None,
    chunk_columns: int,
    output: str | None,
) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if chunk_columns < 4:
        raise click.BadParameter("--chunk-columns must be at least 4")
    text = run_alignment_report(
        run_id=run_id,
        ref_run_id=ref_run_id,
        limit=limit,
        sample_ids=sample_id if sample_id else None,
        chunk_columns=chunk_columns,
        output_path=output,
    )
    click.echo(text)


@cli.command("run-llm-judge")
@click.option("--run-id", type=str, required=True, help="Target STT run_id to evaluate.")
@click.option("--ref-run-id", type=str, default=None, help="Reference STT run_id for side-by-side comparison.")
@click.option("--limit", "-n", type=int, default=None, help="Evaluate only N samples.")
def run_llm_judge(run_id: str, ref_run_id: str | None, limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    run_llm_judge_eval(run_id=run_id, ref_run_id=ref_run_id, limit=limit)
    click.echo("LLM judge evaluation completed.")


if __name__ == "__main__":
    cli()
