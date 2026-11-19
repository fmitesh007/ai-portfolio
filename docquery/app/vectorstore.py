"""Thin ChromaDB wrapper: persistent, cosine similarity, doc-level deletes."""

from __future__ import annotations

import chromadb

from core.config import settings


class VectorStore:
    def __init__(self, path=None, collection_name: str = "docquery"):
        client = chromadb.PersistentClient(path=str(path or settings.chroma_dir))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, doc_id: str, chunks, embeddings: list[list[float]]) -> int:
        self.delete(doc_id)
        self.collection.upsert(
            ids=[f"{doc_id}::{chunk.index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {"doc_id": doc_id, "index": chunk.index, "strategy": chunk.strategy}
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(self, embedding: list[float], top_k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rank, (text, meta, distance) in enumerate(
            zip(
                result["documents"][0] if result["documents"] else [],
                result["metadatas"][0] if result["metadatas"] else [],
                result["distances"][0] if result["distances"] else [],
            ),
            start=1,
        ):
            hits.append(
                {
                    "rank": rank,
                    "text": text,
                    "doc_id": meta["doc_id"],
                    "chunk_index": meta["index"],
                    "strategy": meta["strategy"],
                    "distance": round(float(distance), 4),
                }
            )
        return hits

    def delete(self, doc_id: str) -> int:
        existing = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def documents(self) -> list[str]:
        result = self.collection.get(include=["metadatas"])
        return sorted({meta["doc_id"] for meta in result["metadatas"]})

    def count(self) -> int:
        return self.collection.count()
