"""BERTScore eval unit tests (mocked bert_score and IO; no real torch/bert-score required)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.contracts import ClinicalSample

from src.evaluation.bertscore_eval import run_bertscore_eval


@pytest.fixture
def fake_bert_score_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """``run_bertscore_eval`` imports ``bert_score`` lazily; inject a stub before calling."""
    mod = ModuleType("bert_score")

    def score(cands: list, refs: list, **kwargs: object):
        n = len(cands)
        return [0.9] * n, [0.8] * n, [0.85] * n

    mod.score = MagicMock(side_effect=score)
    monkeypatch.setitem(sys.modules, "bert_score", mod)
    return mod


def test_run_bertscore_empty_samples_returns_zero(
    fake_bert_score_module: ModuleType,
    tmp_path: Path,
) -> None:
    mock_settings = MagicMock()
    mock_settings.bertscore_model = "roberta-large"
    mock_settings.recordings_dir = str(tmp_path)
    mock_settings.transcripts_dir = str(tmp_path)
    mock_settings.casenotes_dir = str(tmp_path)
    mock_settings.dataset_pickle_path = str(tmp_path / "empty.pickle")

    mock_loader = MagicMock()
    mock_loader.load_samples.return_value = []
    mock_pickle = MagicMock()
    mock_pickle.load_transcripts.return_value = {}
    mock_analytics = MagicMock()
    mock_analytics.get_stt_outputs_for_run.return_value = {}

    out_path = tmp_path / "out.json"
    with (
        patch("src.evaluation.bertscore_eval.get_settings", return_value=mock_settings),
        patch("src.evaluation.bertscore_eval.DatasetLoader", return_value=mock_loader),
        patch("src.evaluation.bertscore_eval.DatasetPickleLoader", return_value=mock_pickle),
        patch("src.evaluation.bertscore_eval.AnalyticsRepository", return_value=mock_analytics),
    ):
        result = run_bertscore_eval(
            run_id="r1",
            ref_run_id=None,
            sample_id=None,
            limit=None,
            model_type=None,
            batch_size=4,
            rescale_with_baseline=True,
            output_json=out_path,
        )

    assert result["samples"] == 0
    assert out_path.read_text(encoding="utf-8").startswith("{")
    fake_bert_score_module.score.assert_not_called()


def test_run_bertscore_scores_one_pair(fake_bert_score_module: ModuleType, tmp_path: Path) -> None:
    mock_settings = MagicMock()
    mock_settings.bertscore_model = "roberta-large"
    mock_settings.recordings_dir = str(tmp_path)
    mock_settings.transcripts_dir = str(tmp_path)
    mock_settings.casenotes_dir = str(tmp_path)
    mock_settings.dataset_pickle_path = str(tmp_path / "d.pickle")

    sample = ClinicalSample("S1", None, None, None)
    mock_loader = MagicMock()
    mock_loader.load_samples.return_value = [sample]

    mock_pickle = MagicMock()
    mock_pickle.load_transcripts.return_value = {"S1": "reference text here"}

    mock_analytics = MagicMock()
    mock_analytics.get_stt_outputs_for_run.return_value = {"S1": "hypothesis text here"}

    with (
        patch("src.evaluation.bertscore_eval.get_settings", return_value=mock_settings),
        patch("src.evaluation.bertscore_eval.DatasetLoader", return_value=mock_loader),
        patch("src.evaluation.bertscore_eval.DatasetPickleLoader", return_value=mock_pickle),
        patch("src.evaluation.bertscore_eval.AnalyticsRepository", return_value=mock_analytics),
    ):
        result = run_bertscore_eval(
            run_id="run-uuid",
            ref_run_id=None,
            sample_id=None,
            limit=None,
            model_type="distilbert-base-uncased",
            batch_size=8,
            rescale_with_baseline=False,
            output_json=None,
        )

    assert result["samples"] == 1
    assert result["mean_f1"] == pytest.approx(0.85)
    assert result["per_sample"][0]["sample_id"] == "S1"
    fake_bert_score_module.score.assert_called_once()


def test_run_bertscore_with_ref_run_calls_score_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = ModuleType("bert_score")

    def score(cands: list, refs: list, **kwargs: object):
        n = len(cands)
        return [0.5] * n, [0.5] * n, [0.5] * n

    mod.score = MagicMock(side_effect=score)
    monkeypatch.setitem(sys.modules, "bert_score", mod)

    mock_settings = MagicMock()
    mock_settings.bertscore_model = "roberta-large"
    mock_settings.recordings_dir = str(tmp_path)
    mock_settings.transcripts_dir = str(tmp_path)
    mock_settings.casenotes_dir = str(tmp_path)
    mock_settings.dataset_pickle_path = str(tmp_path / "d.pickle")

    sample = ClinicalSample("S1", None, None, None)
    mock_loader = MagicMock()
    mock_loader.load_samples.return_value = [sample]

    mock_pickle = MagicMock()
    mock_pickle.load_transcripts.return_value = {"S1": "gold text"}

    mock_analytics = MagicMock()

    def outputs(run_id: str) -> dict[str, str]:
        if run_id == "a":
            return {"S1": "hyp a"}
        if run_id == "b":
            return {"S1": "hyp b"}
        return {}

    mock_analytics.get_stt_outputs_for_run.side_effect = outputs

    with (
        patch("src.evaluation.bertscore_eval.get_settings", return_value=mock_settings),
        patch("src.evaluation.bertscore_eval.DatasetLoader", return_value=mock_loader),
        patch("src.evaluation.bertscore_eval.DatasetPickleLoader", return_value=mock_pickle),
        patch("src.evaluation.bertscore_eval.AnalyticsRepository", return_value=mock_analytics),
    ):
        result = run_bertscore_eval(
            run_id="a",
            ref_run_id="b",
            sample_id=None,
            limit=None,
            model_type=None,
            batch_size=4,
            rescale_with_baseline=True,
            output_json=None,
        )

    assert result["samples"] == 1
    assert "ref_mean_f1" in result
    assert mod.score.call_count == 2
