from src.llm.anthropic_client import AnthropicClient
from src.llm.openai_client import OpenAIClient
from src.llm.ollama_client import OllamaClient
from src.rag.hybrid_retriever import HybridRetriever


class InsightGenerator:
    """Generates grounded clinical summaries using retrieved evidence."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.openai_client = OpenAIClient()
        self.anthropic_client = AnthropicClient()
        self.ollama_client = OllamaClient()

    def generate_grounded_summary(self, clinical_insight_query: str, provider: str = "openai") -> str:
        context_items = self.retriever.search(clinical_insight_query, top_k=6)
        context = "\n\n".join(f"- {item.text}" for item in context_items)
        prompt = (
            "You are a clinical AI assistant. Create a concise, structured summary with:\n"
            "1) key clinical findings\n2) supporting evidence\n3) open risks or missing data.\n"
            "Only use the supplied context.\n\n"
            f"Query:\n{clinical_insight_query}\n\nContext:\n{context}"
        )
        if provider == "anthropic":
            return self.anthropic_client.generate(prompt)
        if provider == "ollama":
            return self.ollama_client.generate(prompt)
        return self.openai_client.generate(prompt)
