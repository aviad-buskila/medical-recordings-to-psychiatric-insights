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
def run_stt() -> None:
    run_stt_pipeline()
    click.echo("STT pipeline completed.")


@cli.command("build-rag-index")
def build_index() -> None:
    build_rag_index()
    click.echo("RAG index build scaffold completed.")


@cli.command("run-eval")
def run_eval() -> None:
    evaluate_stt_against_gold()
    click.echo("Evaluation scaffold completed.")


@cli.command("run-all")
def run_all() -> None:
    validate_dataset()
    run_stt()
    build_index()
    run_eval()
    click.echo("Full scaffold pipeline run complete.")


if __name__ == "__main__":
    cli()
