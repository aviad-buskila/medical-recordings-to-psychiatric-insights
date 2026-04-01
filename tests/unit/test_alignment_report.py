"""Tests for alignment report (mocked DB / pickle)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.evaluation.alignment_report import run_alignment_report


@patch("src.evaluation.alignment_report.get_settings")
@patch("src.evaluation.alignment_report.AnalyticsRepository")
@patch("src.evaluation.alignment_report.DatasetPickleLoader")
def test_run_alignment_report_includes_header_and_sample(
    mock_pl_cls: MagicMock,
    mock_repo_cls: MagicMock,
    mock_gs: MagicMock,
) -> None:
    mock_gs.return_value.dataset_pickle_path = "/tmp/dataset.pickle"

    mock_pl = MagicMock()
    mock_pl.load_transcripts.return_value = {"S1": "hello world here"}
    mock_pl_cls.return_value = mock_pl

    mock_repo = MagicMock()
    mock_repo.get_stt_outputs_for_run.return_value = {"S1": "hello world here"}
    mock_repo.get_stt_run_info.return_value = {"model_name": "m", "run_scope": "full"}
    mock_repo_cls.return_value = mock_repo

    text = run_alignment_report(
        run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        ref_run_id=None,
        limit=1,
        sample_ids=None,
        chunk_columns=48,
        output_path=None,
    )

    assert "WORD ALIGNMENT" in text
    assert "sample_id: S1" in text
    assert "GOLD" in text
    assert "hello" in text.lower()


@patch("src.evaluation.alignment_report.get_settings")
@patch("src.evaluation.alignment_report.DatasetPickleLoader")
def test_run_alignment_report_empty_gold(mock_pl_cls: MagicMock, mock_gs: MagicMock) -> None:
    mock_gs.return_value.dataset_pickle_path = "/tmp/x.pickle"
    mock_pl = MagicMock()
    mock_pl.load_transcripts.return_value = {}
    mock_pl_cls.return_value = mock_pl

    text = run_alignment_report("run-id", limit=None)
    assert "No gold transcripts found" in text
