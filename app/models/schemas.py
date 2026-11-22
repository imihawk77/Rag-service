"""Pydantic request and response schemas for the HTTP API."""
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]


class DocumentResponse(BaseModel):
    status: str
    chunks_count: int
    documents: list[str]
