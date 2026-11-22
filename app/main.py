"""FastAPI application for document ingestion and grounded RAG queries."""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.core.chunker import SmartChunker
from app.core.rag_chain import RAGChain
from app.core.reranker import CrossEncoderReranker
from app.core.retriever import HybridRetriever
from app.models.schemas import DocumentResponse, QueryRequest, QueryResponse
from app.services.llm_service import LLMService
from app.utils.file_processor import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Smart RAG Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGState:
    def __init__(self) -> None:
        self.retriever = None
        self.rag_chain = None
        self.chunks: list[dict] = []
        self.documents_processed = False


state = RAGState()


def build_rag_chain(retriever: HybridRetriever) -> RAGChain:
    if settings.llm_provider == "openai":
        secret = settings.openai_api_key
        model = settings.openai_model
    else:
        secret = settings.mistral_api_key
        model = settings.mistral_model
    llm = LLMService(
        settings.llm_provider,
        api_key=secret.get_secret_value() if secret else None,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.reranker_enabled
        else None
    )
    return RAGChain(
        retriever,
        llm,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
        top_k=settings.top_k,
    )


@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, split it into chunks and rebuild local indexes."""
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        documents = process_pdf(tmp_path)
        chunker = SmartChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        new_chunks = chunker.chunk_documents(documents)
        if not new_chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from PDF")

        combined_chunks = [*state.chunks, *new_chunks]
        retriever = HybridRetriever(
            embedding_model_name=settings.embedding_model,
            weight_bm25=0.3,
            weight_vector=0.7,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k,
            candidate_k=settings.retrieval_candidates,
            index_dir=str(settings.index_dir),
        )
        retriever.initialize(combined_chunks)
        state.retriever = retriever
        state.rag_chain = build_rag_chain(retriever)
        state.chunks = combined_chunks
        state.documents_processed = True

        return DocumentResponse(
            status="success",
            chunks_count=len(new_chunks),
            documents=[doc.get("metadata", {}).get("source", filename) for doc in documents],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document upload/indexing failed")
        raise HTTPException(status_code=500, detail="Document processing failed") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        await file.close()


@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Retrieve, rerank and generate a grounded answer."""
    if not state.documents_processed or state.rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents processed. Please upload a document first.",
        )
    try:
        return QueryResponse(**state.rag_chain.ask(request.question, top_k=request.top_k))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail="RAG generation failed") from exc


@app.get("/status")
async def get_status():
    return {
        "documents_loaded": state.documents_processed,
        "chunks_count": len(state.chunks),
        "is_ready": state.documents_processed and state.rag_chain is not None,
        "llm_provider": settings.llm_provider,
        "reranker_enabled": settings.reranker_enabled,
    }


@app.delete("/clear")
async def clear_index():
    state.retriever = None
    state.rag_chain = None
    state.chunks = []
    state.documents_processed = False
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
