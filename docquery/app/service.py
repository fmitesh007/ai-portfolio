"""RAG orchestration: ingest -> chunk -> embed -> retrieve -> generate -> record."""

from __future__ import annotations

from dataclasses import dataclass

from core.embeddings import BaseEmbeddings, get_embeddings
from core.llm import BaseLLM, LLMResponse, get_llm
from core.metrics import MetricsStore, event_from, track

from .chunking import STRATEGIES, Chunk, chunk
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "Answer strictly from the provided context. Cite sources by their number."
)


@dataclass
class RAGDependencies:
    store: VectorStore
    embeddings: BaseEmbeddings
    llm: BaseLLM
    metrics: MetricsStore


def build_dependencies() -> RAGDependencies:
    return RAGDependencies(
        store=VectorStore(),
        embeddings=get_embeddings(),
        llm=get_llm(),
        metrics=MetricsStore(),
    )


def read_document(filename: str, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix in {"txt", "md", "csv", "json", "log"}:
        return content.decode("utf-8", errors="replace")
    if suffix == "pdf":
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(
        f"Unsupported file type .{suffix}; use txt, md, pdf, csv, json or log"
    )


def ingest(
    deps: RAGDependencies,
    *,
    doc_id: str,
    filename: str,
    content: bytes,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 80,
) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}"
        )
    text = read_document(filename, content)
    chunks: list[Chunk] = chunk(
        text,
        strategy,
        size=chunk_size,
        overlap=overlap,
        embeddings=deps.embeddings if strategy == "semantic" else None,
    )
    if not chunks:
        raise ValueError("Document contains no usable text")
    vectors = deps.embeddings.embed([item.text for item in chunks])
    count = deps.store.upsert(doc_id, chunks, vectors)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "strategy": strategy,
        "chunks": count,
        "characters": len(text),
    }


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n".join(f"[{hit['rank']}] {hit['text']}" for hit in hits)
    return f"Answer the question using only the numbered context passages. Cite passages by number.\n\nQuestion: {question}\n\nContext:\n{context}"


def query(deps: RAGDependencies, question: str, top_k: int = 4) -> dict:
    if not question.strip():
        raise ValueError("question must not be empty")

    query_vector = deps.embeddings.embed_one(question)
    hits = deps.store.search(query_vector, top_k=top_k)
    prompt = build_prompt(question, hits)

    with track(deps.metrics, app="docquery", operation="query") as captured:
        response: LLMResponse = deps.llm.complete(prompt, system=SYSTEM_PROMPT)
        captured.append(event_from(response))

    return {
        "question": question,
        "answer": response.text,
        "sources": [
            {
                "doc_id": hit["doc_id"],
                "chunk_index": hit["chunk_index"],
                "text": hit["text"],
                "distance": hit["distance"],
            }
            for hit in hits
        ],
        "metrics": {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        },
    }
