# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]

class DocumentResponse(BaseModel):
    status: str
    chunks_count: int
    documents: List[str]