# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import tempfile
import os

from app.core.chunker import SmartChunker
from app.core.retriever import HybridRetriever
from app.core.rag_chain import RAGChain
from app.utils.file_processor import process_pdf
from app.models.schemas import QueryRequest, QueryResponse, DocumentResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart RAG Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное состояние
class RAGState:
    retriever = None
    rag_chain = None
    chunks = []
    documents_processed = False

state = RAGState()

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Загрузка PDF документа.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        documents = process_pdf(tmp_path)
        os.unlink(tmp_path)
        
        chunker = SmartChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)
        
        if state.retriever is None:
            state.retriever = HybridRetriever(
                embedding_model_name="intfloat/multilingual-e5-large",
                weight_bm25=0.3,
                weight_vector=0.7,
                top_k=5
            )
            state.chunks = []
        
        state.chunks.extend(chunks)
        state.retriever.initialize(state.chunks)
        
        if state.rag_chain is None:
            state.rag_chain = RAGChain(retriever=state.retriever)
        
        state.documents_processed = True
        
        return DocumentResponse(
            status="success",
            chunks_count=len(chunks),
            documents=[doc['metadata'].get('source', file.filename) for doc in documents]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Задать вопрос.
    """
    if not state.documents_processed or state.rag_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents processed. Please upload a document first."
        )
    
    try:
        result = state.rag_chain.ask(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "documents_loaded": state.documents_processed,
        "chunks_count": len(state.chunks) if state.chunks else 0,
        "is_ready": state.documents_processed and state.rag_chain is not None
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