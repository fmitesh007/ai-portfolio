"""Pluggable embeddings with a zero-dependency offline default.

LocalHashEmbeddings is a stable hashed bag-of-words + character-trigram
embedder. It is deliberately simple: it keeps the whole pipeline runnable
without model downloads or API keys. Swap in Gemini/OpenAI with one env var.
"""

from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

from .config import settings


class BaseEmbeddings:
    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalHashEmbeddings(BaseEmbeddings):
    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(_vector(t, self.dim)) for t in texts]


class GeminiEmbeddings(BaseEmbeddings):
    dim = 768

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_embedding_model
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        vectors = []
        with httpx.Client(timeout=settings.request_timeout) as client:
            for text in texts:
                response = client.post(
                    url,
                    json={"model": self.model, "content": {"parts": [{"text": text}]}},
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"]["values"])
        return vectors


class OpenAIEmbeddings(BaseEmbeddings):
    dim = 1536

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=settings.request_timeout) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            return [
                item["embedding"]
                for item in sorted(response.json()["data"], key=lambda x: x["index"])
            ]


def get_embeddings() -> BaseEmbeddings:
    provider = settings.embedding_provider
    if provider == "gemini":
        return GeminiEmbeddings()
    if provider == "openai":
        return OpenAIEmbeddings()
    if provider == "local":
        return LocalHashEmbeddings()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r} (use local, gemini, or openai)"
    )


def _vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"\w+", text.lower())
    compact = "".join(words)
    features = words + [compact[i : i + 3] for i in range(max(0, len(compact) - 2))]
    for feature in features:
        bucket = int(hashlib.md5(feature.encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[bucket] += 1.0
    return vec


def _normalize(vec: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()
