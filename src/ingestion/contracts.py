from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClinicalSample:
    sample_id: str
    recording_path: Path | None
    transcript_path: Path | None
    casenote_path: Path | None
