"""Retrieval-augmented generation orchestration."""
from typing import Any


SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы по предоставленным документам.
Используй только факты из контекста. Не выдумывай сведения и не делай выводов,
которые не подтверждаются контекстом. Если ответа в контексте нет, прямо скажи
об этом. Отвечай на языке вопроса. При возможности указывай номера источников
в квадратных скобках, например [1]."""


class RAGChain:
    """Hybrid retrieval -> optional reranking -> grounded LLM generation."""

    def __init__(self, retriever: Any, llm_service: Any, *,
                 reranker: Any | None = None, rerank_candidates: int = 10,
                 top_k: int = 5) -> None:
        if rerank_candidates <= 0 or top_k <= 0:
            raise ValueError("rerank_candidates and top_k must be greater than zero")
        self.retriever = retriever
        self.llm_service = llm_service
        self.reranker = reranker
        self.rerank_candidates = max(rerank_candidates, top_k)
        self.top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        result_count = self.top_k if top_k is None else top_k
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")
        candidates = self.retriever.retrieve(
            question, top_k=max(self.rerank_candidates, result_count)
        )
        if self.reranker is not None and candidates:
            selected = self.reranker.rerank(question, candidates, top_k=result_count)
        else:
            selected = candidates[:result_count]

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for index, result in enumerate(selected, start=1):
            chunk = result["chunk"]
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            citation = f"[{index}] Source: {source}"
            if page is not None:
                citation += f", page: {page}"
            context_parts.append(f"{citation}\n{chunk['text']}")
            sources.append({
                "text": chunk["text"], "source": source, "score": result.get("score"),
                "retrieval_score": result.get("retrieval_score", result.get("score")),
                "rerank_score": result.get("rerank_score"), "metadata": metadata,
            })
        context = "\n\n---\n\n".join(context_parts)
        user_prompt = (
            f"Вопрос пользователя:\n{question.strip()}\n\n"
            f"Контекст документов:\n{context or '[Контекст не найден]'}\n\n"
            "Дай точный ответ по контексту. Если данных недостаточно, сообщи об этом."
        )
        answer = self.llm_service.generate(
            system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt
        )
        return {"question": question, "answer": answer, "sources": sources}
