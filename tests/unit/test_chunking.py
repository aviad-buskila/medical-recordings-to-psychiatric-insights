from src.rag.chunking import chunk_clinical_text


def test_chunking_returns_chunks_for_non_empty_text() -> None:
    chunks = chunk_clinical_text("s1", "A" * 2000, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert chunks[0].source_id == "s1"
