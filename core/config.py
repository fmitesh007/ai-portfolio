"""Environment-driven configuration shared by all three services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def data_dir() -> Path:
    path = Path(getenv("DATA_DIR", "./data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Settings:
    """Read once at import time; override with environment variables."""

    llm_provider: str = getenv("LLM_PROVIDER", "mock").lower()
    embedding_provider: str = getenv("EMBEDDING_PROVIDER", "local").lower()

    gemini_api_key: str = getenv("GEMINI_API_KEY", "")
    gemini_model: str = getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_embedding_model: str = getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

    openai_api_key: str = getenv("OPENAI_API_KEY", "")
    openai_base_url: str = getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    ).rstrip("/")
    openai_model: str = getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    metrics_db: Path = data_dir() / "metrics.sqlite3"
    chroma_dir: Path = data_dir() / "chroma"

    request_timeout: float = float(getenv("REQUEST_TIMEOUT", "30"))


settings = Settings()
