# LLMMeter — LLM observability

A tiny FastAPI dashboard that records token counts, latency percentiles,
estimated spend and errors for every LLM call made by the other services.

## Features
- `@metered(app, operation)` decorator — one line to instrument any function returning an `LLMResponse`
- SQLite event log with p50 / p95 latency, total tokens, cost and error rate
- Live HTML dashboard with recent-call table and latency bars
- Failure telemetry: exceptions are recorded before being re-raised
- Shared by DocQuery and AgentFlow through `core/metrics.py`

## Run

Easiest (from the repo root, works in fish/zsh/bash):

```fish
bash scripts/run.sh llmmeter     # http://localhost:8002
```

Or manually with a venv activated:

```bash
uvicorn llmmeter.app.main:app --reload --port 8002
curl -X POST http://localhost:8002/api/demo-call
open http://localhost:8002
```

`GET /api/summary?app=docquery` and `GET /api/events?limit=20` expose raw data.

## Resume-ready bullets
- Built an LLM observability service that tracks tokens, p50/p95 latency, estimated cost and error rates across multiple apps.
- Added a single decorator to instrument LLM calls and persist telemetry to SQLite with a live dashboard.
