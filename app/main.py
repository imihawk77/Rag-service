"""Hardened FastAPI application for local, persistent RAG."""
import hmac
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.core.chunker import SmartChunker
from app.core.rag_chain import RAGChain
from app.core.reranker import CrossEncoderReranker
from app.core.retriever import HybridRetriever
from app.models.schemas import DocumentResponse, QueryRequest, QueryResponse
from app.services.llm_service import LLMService
from app.storage.document_store import DocumentStoreError, LocalDocumentStore
from app.utils.file_processor import process_pdf

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


class RAGState:
    def __init__(self) -> None:
        self.retriever: HybridRetriever | None = None
        self.rag_chain: RAGChain | None = None
        self.chunks: list[dict] = []

    @property
    def ready(self) -> bool:
        return self.retriever is not None and self.rag_chain is not None


state = RAGState()


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    configured = settings.api_key.get_secret_value() if settings.api_key else None
    if not configured and settings.app_env != "production":
        return
    if not configured or not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def make_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedding_model_name=settings.embedding_model,
        weight_bm25=0.3,
        weight_vector=0.7,
        top_k=settings.top_k,
        rrf_k=settings.rrf_k,
        candidate_k=settings.retrieval_candidates,
        index_dir=str(settings.index_dir),
    )


def build_rag_chain(retriever: HybridRetriever) -> RAGChain:
    secret, model = (
        (settings.openai_api_key, settings.openai_model)
        if settings.llm_provider == "openai"
        else (settings.mistral_api_key, settings.mistral_model)
    )
    llm = LLMService(
        settings.llm_provider,
        api_key=secret.get_secret_value() if secret else None,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    reranker = CrossEncoderReranker(model_name=settings.reranker_model) if settings.reranker_enabled else None
    return RAGChain(
        retriever, llm, reranker=reranker,
        rerank_candidates=settings.rerank_candidates, top_k=settings.top_k,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    try:
        retriever = make_retriever()
        retriever.load_persisted()
        state.retriever = retriever
        state.rag_chain = build_rag_chain(retriever)
        state.chunks = retriever.chunks
        logger.info("Restored persisted index: %s chunks", len(state.chunks))
    except FileNotFoundError:
        logger.info("No persisted index found; upload a PDF to initialize")
    yield


app = FastAPI(
    title="Smart RAG Service",
    version="1.1.0",
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None if settings.app_env == "production" else "/redoc",
    openapi_url=None if settings.app_env == "production" else "/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def limit_request_body(request, call_next):
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > settings.max_upload_size_bytes + 1024 * 1024:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
    return await call_next(request)


@app.get("/health/live", include_in_schema=False)
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
async def readiness():
    if not state.ready:
        raise HTTPException(status_code=503, detail="RAG index is not ready")
    return {"status": "ready", "chunks_count": len(state.chunks)}


@app.post("/upload", response_model=DocumentResponse, dependencies=[Depends(require_api_key)])
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    tmp_path = None
    stored_path = None
    try:
        content = await file.read(settings.max_upload_size_bytes + 1)
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="Uploaded file is not a valid PDF")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        documents = process_pdf(tmp_path)
        new_chunks = SmartChunker(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        ).chunk_documents(documents)
        if not new_chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from PDF")

        document_id = str(uuid4())
        store = LocalDocumentStore(
            settings.data_dir / "documents",
            max_file_size_bytes=settings.max_upload_size_bytes,
        )
        stored_path = store.save(content, original_filename=filename, document_id=document_id)
        for chunk in new_chunks:
            metadata = chunk.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["source"] = filename
                metadata["document_id"] = document_id

        combined = [*state.chunks, *new_chunks]
        retriever = make_retriever()
        retriever.initialize(combined, persist=True)
        chain = build_rag_chain(retriever)
        state.retriever, state.rag_chain, state.chunks = retriever, chain, combined
        return DocumentResponse(status="success", chunks_count=len(new_chunks), documents=[filename])
    except HTTPException:
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise
    except DocumentStoreError as exc:
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        logger.exception("Document upload/indexing failed")
        raise HTTPException(status_code=500, detail="Document processing failed") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        await file.close()


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def ask_question(request: QueryRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="No indexed documents are available")
    try:
        return QueryResponse(**state.rag_chain.ask(request.question, top_k=request.top_k))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail="RAG generation failed") from exc


@app.get("/status", dependencies=[Depends(require_api_key)])
async def get_status():
    return {
        "documents_loaded": state.ready,
        "chunks_count": len(state.chunks),
        "is_ready": state.ready,
        "llm_provider": settings.llm_provider,
        "reranker_enabled": settings.reranker_enabled,
        "app_env": settings.app_env,
    }


@app.delete("/clear", dependencies=[Depends(require_api_key)])
async def clear_index():
    state.retriever = None
    state.rag_chain = None
    state.chunks = []
    return {"status": "cleared"}
