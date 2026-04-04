"""Data ingestion utilities for contracts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClinicalSample:
    # CR: better to have sample_id sanitation here, and path full sanitization here
    sample_id: str
    recording_path: Path | None
    transcript_path: Path | None
    casenote_path: Path | None
