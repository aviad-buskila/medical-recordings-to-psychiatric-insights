from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_id: str
    text: str
    start_char: int
    end_char: int


def chunk_clinical_text(source_id: str, text: str, chunk_size: int = 800, overlap: int = 120) -> list[TextChunk]:
    """Simple sliding-window chunking scaffold for clinical documents."""
    if not text.strip():
        return []
    chunks: list[TextChunk] = []
    cursor = 0
    i = 0
    while cursor < len(text):
        end = min(len(text), cursor + chunk_size)
        chunk_text = text[cursor:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_id}_chunk_{i}",
                    source_id=source_id,
                    text=chunk_text,
                    start_char=cursor,
                    end_char=end,
                )
            )
            i += 1
        if end == len(text):
            break
        cursor = max(0, end - overlap)
    return chunks
