from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


def _timestamp_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_eval_report_path(eval_name: str, timestamp_utc: str | None = None) -> Path:
    """Create `data/processed/<evalname>_<timestamp>.txt` path."""
    safe_eval_name = eval_name.replace(" ", "_").replace("/", "_")
    ts = timestamp_utc or _timestamp_utc_compact()
    processed_dir = Path(os.environ.get("EVAL_REPORTS_DIR", str(Path("data") / "processed")))
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir / f"{safe_eval_name}_{ts}.txt"


class _TeeTextIO:
    """Write to both the original stream and a file."""

    def __init__(self, original: TextIO, file: TextIO):
        self._original = original
        self._file = file

    def write(self, s: str) -> int:
        self._original.write(s)
        return self._file.write(s)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()


@contextlib.contextmanager
def capture_terminal_to_file(report_path: Path, header_lines: list[str]) -> Iterator[TextIO]:
    """Capture stdout+stderr fully into a file while still printing to terminal."""
    with report_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.flush()

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeTextIO(original_stdout, f)  # type: ignore[assignment]
        sys.stderr = _TeeTextIO(original_stderr, f)  # type: ignore[assignment]
        try:
            yield f
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@dataclass
class EvalRunReporter:
    eval_name: str
    command_line: str
    report_path: Path
    metrics: list[dict[str, Any]] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)

    def set_run_metadata(self, **kwargs: Any) -> None:
        self.run_metadata.update(kwargs)

    def add_metric(
        self,
        *,
        sample_id: str,
        metric_name: str,
        metric_value: float,
        details: dict[str, Any],
    ) -> None:
        self.metrics.append(
            {
                "sample_id": sample_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "details": details,
            }
        )

    def set_result_summary(self, **kwargs: Any) -> None:
        self.result_summary.update(kwargs)

    def write_results_section(self, file: TextIO) -> None:
        file.write("\n\n===== EVAL REPORT RESULTS (mirrors DB rows) =====\n")
        file.write(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        file.write("\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "command_line": self.command_line,
            "report_path": str(self.report_path),
            "run_metadata": self.run_metadata,
            "metrics": self.metrics,
            "result_summary": self.result_summary,
        }

