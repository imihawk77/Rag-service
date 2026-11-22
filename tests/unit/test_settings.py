"""Tests for validated environment configuration."""

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_default_settings_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.llm_provider == "openai"
    assert settings.embedding_model == "intfloat/multilingual-e5-large"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5
    assert settings.retrieval_candidates >= settings.top_k


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_retrieval_candidates_must_cover_top_k() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_CANDIDATES"):
        Settings(_env_file=None, top_k=10, retrieval_candidates=5)


def test_production_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="API_KEY is required"):
        Settings(_env_file=None, app_env="production")


def test_production_api_key_must_be_long_enough() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(_env_file=None, app_env="production", api_key="too-short")


def test_production_accepts_strong_api_key() -> None:
    settings = Settings(_env_file=None, app_env="production", api_key="x" * 40)

    assert settings.api_key is not None
    assert len(settings.api_key.get_secret_value()) == 40


def test_reranker_defaults_and_validation() -> None:
    settings = Settings(_env_file=None)
    assert settings.reranker_enabled is True
    assert settings.rerank_candidates >= settings.top_k
    with pytest.raises(ValidationError, match="RERANK_CANDIDATES"):
        Settings(_env_file=None, top_k=5, rerank_candidates=4)


def test_llm_generation_settings_are_validated() -> None:
    with pytest.raises(ValidationError, match="LLM_MAX_TOKENS"):
        Settings(_env_file=None, llm_max_tokens=0)
    with pytest.raises(ValidationError, match="LLM_TEMPERATURE"):
        Settings(_env_file=None, llm_temperature=2.5)
