from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    source: str


class HybridRetriever:
    """Hybrid retriever combining dense similarity and BM25 lexical signals."""

    def __init__(self) -> None:
        # TODO: initialize DB connection, embedding model, BM25 index artifacts.
        pass

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # TODO: implement full hybrid retrieval:
        # 1. dense candidate fetch (pgvector)
        # 2. sparse candidate fetch (BM25)
        # 3. weighted score fusion + reranking
        _ = query, top_k
        return []
