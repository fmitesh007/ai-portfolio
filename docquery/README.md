# DocQuery — RAG over your own documents

A FastAPI service that ingests PDF/Markdown/text, chunks it with four selectable
strategies, embeds it into ChromaDB, and answers questions with source citations
and per-call telemetry.

## Features
- Document ingestion: `.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.log`
- Chunking: **character**, **token** (tiktoken), **recursive** (paragraph → sentence → word), **semantic** (embedding similarity)
- Vector search with persistent ChromaDB (cosine similarity), doc-level delete
- Grounded answers with numbered source citations
- LLM meters: latency, input/output tokens, estimated cost
- Pluggable providers: offline mock, Gemini, any OpenAI-compatible endpoint
- Docker + GitHub Actions CI

## Run

Easiest (from the repo root, works in fish/zsh/bash):

```fish
bash scripts/run.sh docquery     # http://localhost:8001/docs
```

Or manually with a venv activated:

```bash
uvicorn docquery.app.main:app --reload --port 8001
```

Ingest a sample document, then ask a question:

```bash
curl -F doc_id=rag-primer -F strategy=semantic -F file=@docquery/samples/rag-primer.txt \
  http://localhost:8001/ingest

curl -X POST http://localhost:8001/query \
  -H 'content-type: application/json' \
  -d '{"question":"Why does chunking matter?","top_k":3}'
```

Useful endpoints: `/health`, `/chunk-strategies`, `/documents`, `/metrics`, `/config`.

## Resume-ready bullets
- Built a FastAPI RAG service that ingests PDF/Markdown/text and returns cited answers grounded in a persistent ChromaDB index.
- Implemented four chunking strategies (character, token, recursive and semantic) and compared retrieval quality across them.
- Added provider-agnostic Gemini/OpenAI integration with token counts, latency percentiles and cost estimation per request.
- Containerised the service with Docker and added GitHub Actions CI with tests and a smoke check.
