"""Integration: DatasetLoader + DatasetPickleLoader with real files (no DB)."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader


def test_dataset_loader_discovers_unified_sample_ids(tmp_path: Path) -> None:
    rec = tmp_path / "recordings"
    tr = tmp_path / "transcripts"
    cn = tmp_path / "casenotes"
    rec.mkdir()
    tr.mkdir()
    cn.mkdir()
    (rec / "S1.wav").write_bytes(b"")
    (tr / "S1.txt").write_text("gold", encoding="utf-8")
    (cn / "S1.md").write_text("note", encoding="utf-8")

    loader = DatasetLoader(str(rec), str(tr), str(cn))
    summary = loader.validate_layout()
    assert summary["recordings"] == 1
    assert summary["transcripts"] == 1
    assert summary["casenotes"] == 1

    samples = loader.load_samples()
    assert len(samples) == 1
    assert samples[0].sample_id == "S1"
    assert samples[0].recording_path is not None
    assert samples[0].transcript_path is not None


def test_pickle_loader_reads_transcripts_from_disk(tmp_path: Path) -> None:
    p = tmp_path / "dataset.pickle"
    payload = {"transcripts": {"A-1": "hello world", "B-2": "other"}}
    p.write_bytes(pickle.dumps(payload))

    loader = DatasetPickleLoader(str(p))
    tx = loader.load_transcripts()
    assert tx["A-1"] == "hello world"
    assert tx["B-2"] == "other"
