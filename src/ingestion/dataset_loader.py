from pathlib import Path

from src.ingestion.contracts import ClinicalSample


class DatasetLoader:
    """Loads and validates local dataset folders for recordings/transcripts/casenotes."""

    def __init__(self, recordings_dir: str, transcripts_dir: str, casenotes_dir: str) -> None:
        self.recordings_dir = Path(recordings_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.casenotes_dir = Path(casenotes_dir)

    def validate_layout(self) -> dict[str, int]:
        return {
            "recordings": self._count_files(self.recordings_dir, (".wav", ".mp3", ".m4a")),
            "transcripts": self._count_files(self.transcripts_dir, (".txt",)),
            "casenotes": self._count_files(self.casenotes_dir, (".txt", ".md")),
        }

    def load_samples(self) -> list[ClinicalSample]:
        # TODO: match files by robust ID extraction from filenames.
        recording_files = {p.stem: p for p in self.recordings_dir.glob("*") if p.suffix.lower() in {".wav"}}
        transcript_files = {p.stem: p for p in self.transcripts_dir.glob("*") if p.suffix.lower() == ".txt"}
        casenote_files = {p.stem: p for p in self.casenotes_dir.glob("*") if p.suffix.lower() in {".txt", ".md"}}
        all_ids = sorted(set(recording_files) | set(transcript_files) | set(casenote_files))
        return [
            ClinicalSample(
                sample_id=sid,
                recording_path=recording_files.get(sid),
                transcript_path=transcript_files.get(sid),
                casenote_path=casenote_files.get(sid),
            )
            for sid in all_ids
        ]

    @staticmethod
    def _count_files(folder: Path, suffixes: tuple[str, ...]) -> int:
        if not folder.exists():
            return 0
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
