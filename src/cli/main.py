from pathlib import Path
import sys
from datetime import datetime, timezone

import click
import psycopg
from psycopg.types.json import Json

from src.config.settings import get_settings
from src.core.logging import configure_logging
from src.core.eval_run_report import EvalRunReporter, capture_terminal_to_file, make_eval_report_path
from src.evaluation.alignment_report import run_alignment_report
from src.evaluation.bertscore_eval import run_bertscore_eval
from src.evaluation.llm_judge_eval import run_llm_judge_eval
from src.evaluation.stt_eval import evaluate_stt_against_gold
from src.ingestion.dataset_loader import DatasetLoader
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


def _parse_generated_run_dir_name(name: str) -> tuple[str, datetime] | None:
    if "_" not in name:
        return None
    run_id, ts = name.split("_", 1)
    try:
        run_ts = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return run_id, run_ts


@cli.command("restore-stt-from-generated")
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Restore only a specific run_id from data/generated_transcripts.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Scan and report what would be restored without writing to DB.",
)
def restore_stt_from_generated(run_id: str | None, dry_run: bool) -> None:
    """Restore stt_runs + stt_outputs rows from data/generated_transcripts/*.txt files."""
    settings = get_settings()
    base_dir = Path(settings.generated_transcripts_dir)
    if not base_dir.exists():
        raise click.ClickException(f"Generated transcripts directory not found: {base_dir}")

    run_dirs: list[tuple[Path, str, datetime]] = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_generated_run_dir_name(d.name)
        if not parsed:
            continue
        rid, run_ts = parsed
        if run_id and rid != run_id:
            continue
        run_dirs.append((d, rid, run_ts))

    if not run_dirs:
        click.echo("No matching generated transcript runs found to restore.")
        return

    total_files = sum(len(list(d.glob("*.txt"))) for d, _, _ in run_dirs)
    if dry_run:
        click.echo(
            f"Dry-run: would restore runs={len(run_dirs)} transcripts={total_files} "
            f"from {base_dir}"
        )
        for d, rid, run_ts in run_dirs:
            click.echo(f"  - run_id={rid} run_timestamp={run_ts.isoformat()} files={len(list(d.glob('*.txt')))}")
        return

    restored_runs = 0
    restored_rows = 0
    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.cursor() as cur:
            for d, rid, run_ts in run_dirs:
                cur.execute(
                    """
                    INSERT INTO clinical_ai.stt_runs
                      (run_id, provider, model_name, run_scope, run_timestamp, run_parameters)
                    VALUES
                      (%(run_id)s, %(provider)s, %(model_name)s, %(run_scope)s, %(run_timestamp)s, %(run_parameters)s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    {
                        "run_id": rid,
                        "provider": settings.stt_provider,
                        "model_name": "recovered-from-generated-transcripts",
                        "run_scope": "full",
                        "run_timestamp": run_ts,
                        "run_parameters": Json({"recovered_from": str(d)}),
                    },
                )
                if cur.rowcount > 0:
                    restored_runs += 1

                for txt in d.glob("*.txt"):
                    sample_id = txt.stem
                    transcript_text = txt.read_text(encoding="utf-8", errors="ignore")
                    cur.execute(
                        """
                        INSERT INTO clinical_ai.stt_outputs
                          (sample_id, provider, transcript_text, language, audio_duration_s, transcription_time_s,
                           metadata, run_id, model_name, run_timestamp)
                        VALUES
                          (%(sample_id)s, %(provider)s, %(transcript_text)s, %(language)s, %(audio_duration_s)s, %(transcription_time_s)s,
                           %(metadata)s, %(run_id)s, %(model_name)s, %(run_timestamp)s)
                        """,
                        {
                            "sample_id": sample_id,
                            "provider": settings.stt_provider,
                            "transcript_text": transcript_text,
                            "language": "unknown",
                            "audio_duration_s": 0.0,
                            "transcription_time_s": 0.0,
                            "metadata": Json({"recovered_from": str(txt)}),
                            "run_id": rid,
                            "model_name": "recovered-from-generated-transcripts",
                            "run_timestamp": run_ts,
                        },
                    )
                    restored_rows += 1
        conn.commit()

    click.echo(
        f"Restore completed. runs_inserted={restored_runs} transcript_rows_inserted={restored_rows}"
    )


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
    eval_name = "run-bertscore"
    command_line = " ".join(sys.argv)
    report_path = make_eval_report_path(eval_name)
    header = [
        f"Command line: {command_line}",
        f"Report path: {report_path}",
        "=== Terminal output captured below ===",
    ]
    with capture_terminal_to_file(report_path, header_lines=header) as f:
        reporter = EvalRunReporter(
            eval_name=eval_name,
            command_line=command_line,
            report_path=report_path,
        )
        run_bertscore_eval(
            run_id=run_id,
            ref_run_id=ref_run_id,
            limit=limit,
            model_type=model_type,
            batch_size=batch_size,
            rescale_with_baseline=not no_rescale,
            output_json=output_json,
            reporter=reporter,
        )
        click.echo("BERTScore evaluation completed.")
        reporter.write_results_section(file=f)
        click.echo(f"Eval report written to {report_path}")


@cli.command("run-eval")
@click.option("--limit", "-n", type=int, default=None, help="Evaluate only N samples.")
@click.option("--run-id", type=str, default=None, help="Evaluate only outputs from a specific STT run_id.")
@click.option("--ref-run-id", type=str, default=None, help="Reference STT run_id for side-by-side WER comparison.")
@click.option("--workers", type=int, default=1, show_default=True, help="Parallel workers for per-sample compute.")
@click.option("--skip-cp-wer", is_flag=True, default=False, help="Skip cpWER computation.")
@click.option("--skip-speaker-metrics", is_flag=True, default=False, help="Skip per-speaker metrics (WER/CER/MER/WIL).")
def run_eval(
    limit: int | None,
    run_id: str | None,
    ref_run_id: str | None,
    workers: int,
    skip_cp_wer: bool,
    skip_speaker_metrics: bool,
) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    if ref_run_id and not run_id:
        raise click.BadParameter("--run-id is required when --ref-run-id is provided")
    if workers <= 0:
        raise click.BadParameter("--workers must be a positive integer")
    eval_name = "run-eval"
    command_line = " ".join(sys.argv)
    report_path = make_eval_report_path(eval_name)
    header = [
        f"Command line: {command_line}",
        f"Report path: {report_path}",
        "=== Terminal output captured below ===",
    ]
    with capture_terminal_to_file(report_path, header_lines=header) as f:
        reporter = EvalRunReporter(
            eval_name=eval_name,
            command_line=command_line,
            report_path=report_path,
        )
        evaluate_stt_against_gold(
            limit=limit,
            run_id=run_id,
            ref_run_id=ref_run_id,
            reporter=reporter,
            workers=workers,
            skip_cp_wer=skip_cp_wer,
            skip_speaker_metrics=skip_speaker_metrics,
        )
        click.echo("Evaluation completed.")
        reporter.write_results_section(file=f)
        click.echo(f"Eval report written to {report_path}")


@cli.command("run-all")
def run_all() -> None:
    validate_dataset()
    run_stt_pipeline(limit=None, stt_profile="default")
    evaluate_stt_against_gold(limit=None)
    click.echo("Full pipeline run complete (STT + WER eval).")


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
    # `show-alignment` does not persist metrics to DB, but we still write an artifact file
    # so runs are reproducible/debuggable.
    eval_name = "show-alignment"
    command_line = " ".join(sys.argv)
    report_path = make_eval_report_path(eval_name)
    report_body = "\n".join(
        [
            f"Command line: {command_line}",
            f"Report path: {report_path}",
            "=== show-alignment output (fully captured) ===",
            text,
        ]
    )
    report_path.write_text(report_body, encoding="utf-8")
    click.echo(text)
    click.echo(f"Eval report written to {report_path}")


@cli.command("run-llm-judge")
@click.option("--run-id", type=str, required=True, help="Target STT run_id to evaluate.")
@click.option("--ref-run-id", type=str, default=None, help="Reference STT run_id for side-by-side comparison.")
@click.option("--limit", "-n", type=int, default=None, help="Evaluate only N samples.")
def run_llm_judge(run_id: str, ref_run_id: str | None, limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise click.BadParameter("--limit must be a positive integer")
    eval_name = "run-llm-judge"
    command_line = " ".join(sys.argv)
    report_path = make_eval_report_path(eval_name)
    header = [
        f"Command line: {command_line}",
        f"Report path: {report_path}",
        "=== Terminal output captured below ===",
    ]
    with capture_terminal_to_file(report_path, header_lines=header) as f:
        reporter = EvalRunReporter(
            eval_name=eval_name,
            command_line=command_line,
            report_path=report_path,
        )
        run_llm_judge_eval(run_id=run_id, ref_run_id=ref_run_id, limit=limit, reporter=reporter)
        click.echo("LLM judge evaluation completed.")
        reporter.write_results_section(file=f)
        click.echo(f"Eval report written to {report_path}")


if __name__ == "__main__":
    cli()
