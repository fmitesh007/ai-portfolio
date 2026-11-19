"""DocQuery API -- RAG over your own documents."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import settings

from .chunking import STRATEGIES
from .service import build_dependencies, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.deps = build_dependencies()
    yield


app = FastAPI(title="DocQuery", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health():
    deps = app.state.deps
    return {
        "status": "ok",
        "llm": deps.llm.name,
        "embeddings": type(deps.embeddings).__name__,
        "indexed_chunks": deps.store.count(),
    }


@app.get("/chunk-strategies")
def chunk_strategies():
    return {"strategies": list(STRATEGIES)}


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    overlap: int = Form(80),
):
    content = await file.read()
    try:
        return ingest(
            app.state.deps,
            doc_id=doc_id,
            filename=file.filename or "document.txt",
            content=content,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query")
def ask(payload: QueryRequest):
    try:
        return query(app.state.deps, payload.question, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents")
def documents():
    return {"documents": app.state.deps.store.documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = app.state.deps.store.delete(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id, "chunks_removed": removed}


@app.get("/metrics")
def metrics():
    return app.state.deps.metrics.summary(app="docquery")


@app.get("/config")
def config():
    """Safe, non-secret view of the active providers."""
    return {
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "data_dir": str(settings.metrics_db.parent),
    }
