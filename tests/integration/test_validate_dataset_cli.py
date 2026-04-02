"""``validate-dataset`` against a real temp directory layout (subprocess)."""

from __future__ import annotations

from pathlib import Path

from tests.integration.conftest import run_main_cli


def test_validate_dataset_reports_counts(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    casenotes = tmp_path / "casenotes"
    for d in (recordings, transcripts, casenotes):
        d.mkdir()
    (recordings / "sample_a.wav").write_bytes(b"")
    (recordings / "sample_b.mp3").write_bytes(b"")
    (transcripts / "sample_a.txt").write_text("x", encoding="utf-8")
    (casenotes / "note.md").write_text("# n", encoding="utf-8")

    proc = run_main_cli(
        "validate-dataset",
        extra_env={
            "RECORDINGS_DIR": str(recordings),
            "TRANSCRIPTS_DIR": str(transcripts),
            "CASENOTES_DIR": str(casenotes),
        },
    )
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "Dataset validation complete" in out
    assert "recordings" in out
    assert "transcripts" in out
    assert "casenotes" in out
