"""Typed application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RAG service.

    Environment variable names are case-insensitive and map directly to field names.
    For example, API_KEY maps to api_key.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:7860", "http://127.0.0.1:7860"]

    llm_provider: Literal["openai", "mistral"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"

    embedding_model: str = "intfloat/multilingual-e5-large"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_candidates: int = 20
    rrf_k: int = 60

    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/metadata.db"

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> "Settings":
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than zero")
        if self.retrieval_candidates < self.top_k:
            raise ValueError("RETRIEVAL_CANDIDATES must be greater than or equal to TOP_K")
        if self.rrf_k <= 0:
            raise ValueError("RRF_K must be greater than zero")
        if self.app_env == "production" and not self.api_key:
            raise ValueError("API_KEY is required when APP_ENV=production")
        if self.app_env == "production" and self.api_key and len(self.api_key.get_secret_value()) < 32:
            raise ValueError("API_KEY must contain at least 32 characters in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()
