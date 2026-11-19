"""Four selectable document chunking strategies.

char      - fixed character window with overlap
token     - fixed token window using tiktoken (whitespace fallback)
recursive - split on paragraph, line, sentence, then word boundaries
semantic  - group adjacent sentences by embedding similarity
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from core.embeddings import BaseEmbeddings

STRATEGIES = ("char", "token", "recursive", "semantic")


@dataclass
class Chunk:
    index: int
    text: str
    strategy: str


def chunk(
    text: str,
    strategy: str,
    *,
    size: int = 500,
    overlap: int = 80,
    embeddings: BaseEmbeddings | None = None,
) -> list[Chunk]:
    text = _clean(text)
    if not text:
        return []
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}; choose from {', '.join(STRATEGIES)}"
        )

    if strategy == "char":
        parts = _by_char(text, size, overlap)
    elif strategy == "token":
        parts = _by_token(text, size, overlap)
    elif strategy == "recursive":
        parts = _by_recursive(text, size, overlap)
    else:
        parts = _by_semantic(text, size, embeddings)

    return [
        Chunk(index=i, text=part, strategy=strategy)
        for i, part in enumerate(parts)
        if part.strip()
    ]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _by_char(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def _encode(text: str) -> list[str]:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base").encode(text)
    except Exception:  # pragma: no cover - offline fallback
        return text.split()


def _decode(tokens: list[str]) -> str:
    if (
        all(isinstance(token, str) for token in tokens)
        and tokens
        and isinstance(tokens[0], str)
        and tokens[0].isalnum()
    ):
        return " ".join(tokens)
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base").decode(tokens)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover
        return " ".join(str(token) for token in tokens)


def _by_token(text: str, size: int, overlap: int) -> list[str]:
    tokens = _encode(text)
    step = max(1, size - overlap)
    parts = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + size]
        if not chunk:
            continue
        decoded = _decode(chunk)
        if decoded.strip():
            parts.append(decoded)
        if start + size >= len(tokens):
            break
    return parts


def _by_recursive(text: str, size: int, overlap: int) -> list[str]:
    blocks = [block for block in re.split(r"\n\n+", text) if block.strip()]
    parts: list[str] = []
    for block in blocks:
        if len(block) <= size:
            parts.append(block)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", block)
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip()
            if len(candidate) <= size:
                buffer = candidate
                continue
            if buffer:
                parts.append(buffer)
                tail = buffer[-overlap:] if overlap else ""
                buffer = f"{tail} {sentence}".strip()
            else:
                parts.extend(_by_char(sentence, size, overlap))
                buffer = ""
        if buffer:
            parts.append(buffer)
    return parts


def _by_semantic(text: str, size: int, embeddings: BaseEmbeddings | None) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return []

    if embeddings is None:
        return _pack(sentences, size)

    vectors = np.asarray(embeddings.embed(sentences), dtype=np.float32)
    min_size = max(120, size // 2)
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        current = " ".join(groups[-1])
        similarity = float(np.dot(vectors[i], vectors[i - 1]))
        fits = len(current) + 1 + len(sentences[i]) <= size
        if fits and (similarity >= 0.15 or len(current) < min_size):
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    merged: list[list[str]] = [groups[0]]
    for group in groups[1:]:
        previous = " ".join(merged[-1])
        if (
            len(previous) < min_size
            and len(previous) + len(" ".join(group)) + 1 <= size
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return [" ".join(group) for group in merged]


def _pack(sentences: list[str], size: int) -> list[str]:
    parts, buffer = [], ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) > size and buffer:
            parts.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        parts.append(buffer)
    return parts
