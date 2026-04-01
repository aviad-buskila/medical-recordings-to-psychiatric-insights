from pathlib import Path

from src.config.settings import get_settings
from src.ingestion.dataset_loader import DatasetLoader
from src.rag.chunking import chunk_clinical_text


def build_rag_index() -> None:
    """Scaffold index builder for dense and sparse retrieval data."""
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()

    for sample in samples:
        if sample.casenote_path is None:
            continue
        text = Path(sample.casenote_path).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_clinical_text(source_id=sample.sample_id, text=text)
        # TODO: persist chunks to DB with embeddings + lexical term stats.
        _ = chunks
