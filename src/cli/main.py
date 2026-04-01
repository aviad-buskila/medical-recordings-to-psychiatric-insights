import click

from src.config.settings import get_settings
from src.core.logging import configure_logging
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
def run_stt(limit: int | None, profile: str, flavor: str) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if flavor.lower() == "both":
        run_ids = run_stt_both_profiles(limit=limit)
        click.echo(
            "STT both-profiles completed. "
            f"default_run_id={run_ids['default']} quality_run_id={run_ids['quality']}"
        )
        return
    run_stt_pipeline(limit=limit, stt_profile=profile)
    click.echo("STT pipeline completed.")


@cli.command("build-rag-index")
def build_index() -> None:
    build_rag_index()
    click.echo("RAG index build scaffold completed.")


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
