import click

from src.config.settings import get_settings
from src.core.logging import configure_logging
from src.evaluation.stt_eval import evaluate_stt_against_gold
from src.ingestion.dataset_loader import DatasetLoader
from src.rag.indexing import build_rag_index
from src.stt.pipeline import run_stt_pipeline


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
def run_stt(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    run_stt_pipeline(limit=limit)
    click.echo("STT pipeline completed.")


@cli.command("build-rag-index")
def build_index() -> None:
    build_rag_index()
    click.echo("RAG index build scaffold completed.")


@cli.command("run-eval")
@click.option("--limit", "-n", type=int, default=None, help="Evaluate only N samples.")
def run_eval(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    evaluate_stt_against_gold(limit=limit)
    click.echo("Evaluation scaffold completed.")


@cli.command("run-all")
def run_all() -> None:
    validate_dataset()
    run_stt_pipeline(limit=None)
    build_index()
    evaluate_stt_against_gold(limit=None)
    click.echo("Full scaffold pipeline run complete.")


if __name__ == "__main__":
    cli()
