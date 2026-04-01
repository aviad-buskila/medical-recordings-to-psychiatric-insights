"""CLI smoke tests and validation (Click CliRunner)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli.main import cli


def test_cli_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in (
        "validate-dataset",
        "run-stt",
        "run-bertscore",
        "run-eval",
        "run-all",
        "show-alignment",
        "run-llm-judge",
    ):
        assert name in result.output


def test_run_eval_rejects_ref_without_run_id() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--ref-run-id", "abc"])
    assert result.exit_code != 0
    assert "run-id" in result.output.lower()


def test_run_bertscore_rejects_ref_without_run_id() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-bertscore", "--ref-run-id", "abc"])
    assert result.exit_code != 0


def test_run_eval_rejects_non_positive_limit() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--limit", "0"])
    assert result.exit_code != 0


def test_show_alignment_rejects_low_chunk_columns() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["show-alignment", "--run-id", "x", "--chunk-columns", "2"],
    )
    assert result.exit_code != 0


@patch("src.cli.main.run_bertscore_eval")
def test_run_bertscore_cli_invokes_eval(mock_eval: MagicMock) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run-bertscore", "--run-id", "rid", "--limit", "5", "--no-rescale"],
    )
    assert result.exit_code == 0
    assert "BERTScore evaluation completed" in result.output
    mock_eval.assert_called_once()
    kwargs = mock_eval.call_args.kwargs
    assert kwargs["run_id"] == "rid"
    assert kwargs["limit"] == 5
    assert kwargs["rescale_with_baseline"] is False


@patch("src.cli.main.evaluate_stt_against_gold")
def test_run_eval_cli_invokes_stt_eval(mock_eval: MagicMock) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--limit", "3"])
    assert result.exit_code == 0
    assert "Evaluation completed" in result.output
    mock_eval.assert_called_once_with(limit=3, run_id=None, ref_run_id=None)
