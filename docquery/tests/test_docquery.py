from pathlib import Path

import pytest

from core.embeddings import LocalHashEmbeddings
from core.llm import MockLLM
from core.metrics import MetricsStore
from docquery.app.chunking import STRATEGIES, chunk
from docquery.app.service import RAGDependencies, ingest, query, read_document
from docquery.app.vectorstore import VectorStore

TEXT = """
Retrieval augmented generation grounds a language model in external documents.
The document is split into chunks before embedding. Chunking strategies include
character windows, token windows, recursive splitting and semantic grouping.
A vector database stores embeddings and returns the nearest passages for a query.
LangChain orchestrates the retrieval and generation steps into one pipeline.
"""


@pytest.fixture
def deps(tmp_path: Path) -> RAGDependencies:
    return RAGDependencies(
        store=VectorStore(path=tmp_path / "chroma"),
        embeddings=LocalHashEmbeddings(),
        llm=MockLLM(),
        metrics=MetricsStore(tmp_path / "metrics.sqlite3"),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_chunking_strategy_returns_chunks(strategy):
    chunks = chunk(
        TEXT, strategy, size=120, overlap=20, embeddings=LocalHashEmbeddings()
    )
    assert chunks
    assert all(item.text.strip() for item in chunks)
    assert [item.index for item in chunks] == list(range(len(chunks)))


def test_semantic_chunks_are_not_tiny():
    chunks = chunk(TEXT, "semantic", size=500, embeddings=LocalHashEmbeddings())
    assert len(chunks) <= 3
    assert all(len(item.text) >= 100 for item in chunks)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk(TEXT, "nope")


def test_ingest_then_query_is_grounded_and_metered(deps):
    result = ingest(
        deps,
        doc_id="rag-notes",
        filename="notes.txt",
        content=TEXT.encode(),
        strategy="recursive",
        chunk_size=120,
    )
    assert result["chunks"] > 0

    answer = query(deps, "How are documents split before embedding?")
    assert answer["answer"]
    assert answer["sources"]
    assert answer["sources"][0]["doc_id"] == "rag-notes"
    assert answer["metrics"]["input_tokens"] > 0
    assert answer["metrics"]["latency_ms"] >= 0

    summary = deps.metrics.summary(app="docquery")
    assert summary["calls"] == 1
    assert summary["errors"] == 0


def test_search_on_empty_index_returns_no_hits(deps):
    assert deps.store.search([0.0] * deps.embeddings.dim) == []


def test_delete_removes_document(deps):
    ingest(deps, doc_id="temp", filename="notes.txt", content=TEXT.encode())
    assert "temp" in deps.store.documents()
    assert deps.store.delete("temp") > 0
    assert "temp" not in deps.store.documents()


def test_unsupported_file_type(deps):
    with pytest.raises(ValueError):
        read_document("image.png", b"not a document")
