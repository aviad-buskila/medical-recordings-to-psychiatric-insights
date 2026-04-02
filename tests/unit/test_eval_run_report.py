from __future__ import annotations

import sys
from pathlib import Path

from src.core.eval_run_report import EvalRunReporter, capture_terminal_to_file, make_eval_report_path


def test_make_eval_report_path_respects_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path))
    p = make_eval_report_path("run-test", timestamp_utc="20200101T000000Z")
    assert str(tmp_path) in str(p)
    assert p.name == "run-test_20200101T000000Z.txt"


def test_capture_terminal_to_file_and_write_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path))
    p = make_eval_report_path("run-test", timestamp_utc="20200101T000000Z")

    header = ["Command line: fake", "=== Terminal output captured below ==="]
    with capture_terminal_to_file(p, header_lines=header) as f:
        reporter = EvalRunReporter(
            eval_name="run-test",
            command_line="fake",
            report_path=p,
        )
        reporter.set_run_metadata(run_id="r1")
        reporter.add_metric(
            sample_id="S1",
            metric_name="wer",
            metric_value=0.5,
            details={"foo": "bar"},
        )
        print("hello-stdout")
        print("hello-stderr", file=sys.stderr)
        reporter.set_result_summary(mean_wer=0.5)
        reporter.write_results_section(file=f)

    txt = p.read_text(encoding="utf-8")
    assert "Command line: fake" in txt
    assert "hello-stdout" in txt
    assert "hello-stderr" in txt
    assert "\"metric_name\": \"wer\"" in txt
    assert "\"mean_wer\": 0.5" in txt

