# ai-portfolio

Three small production-style AI services built to learn and demonstrate the
full stack around LLMs: retrieval, orchestration and observability. Everything
runs offline with deterministic providers by default; switch to Gemini or any
OpenAI-compatible endpoint with one environment variable.

## Services

| Service | What it does | Port (compose) |
|---|---|---|
| [DocQuery](docquery/README.md) | RAG over PDF/TXT: 4 chunking strategies, Chroma vector store, cited answers, per-call metrics | 8001 |
| [LLMMeter](llmmeter/README.md) | Token / latency / cost / error telemetry with a live dashboard | 8002 |
| [AgentFlow](agentflow/README.md) | LangGraph tool-using agent with calculator, vector lookup and HTTP fetch | 8003 |

## Quick start

One script runs everything. It works in **any shell — fish included** — because
it uses the venv's Python directly instead of relying on shell activation:

```fish
bash scripts/run.sh help        # list all commands
bash scripts/run.sh setup       # create .venv + install requirements (first time)
bash scripts/run.sh test        # pytest, 17 tests, no API key needed
bash scripts/run.sh docquery    # http://localhost:8001/docs
bash scripts/run.sh llmmeter    # http://localhost:8002
bash scripts/run.sh agentflow   # http://localhost:8003/docs
bash scripts/run.sh all         # all three at once
```

Manual setup (fish: `source .venv/bin/activate.fish`, **not** `activate`):

```fish
python -m venv .venv && source .venv/bin/activate.fish
python -m pip install -r requirements.txt
python -m pytest -q             # 17 passed

python -m uvicorn docquery.app.main:app --reload --port 8001
python -m uvicorn llmmeter.app.main:app --reload --port 8002
python -m uvicorn agentflow.app.main:app --reload --port 8003
```

Or with Docker:

```bash
cp .env.example .env
docker compose up --build
```

## Using real models

```bash
export LLM_PROVIDER=gemini EMBEDDING_PROVIDER=gemini GEMINI_API_KEY=...
# or
export LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai \
       OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.openai.com/v1
# Ollama also works:
export LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai \
       OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 \
       OPENAI_MODEL=llama3.2 OPENAI_EMBEDDING_MODEL=nomic-embed-text
```

The `core/` package holds the provider factories, the metrics store and the
meter decorator. No application code changes when the provider changes.

## Architecture

```
  React / curl / any client
            |
   +--------+---------+---------+
   | DocQuery          | AgentFlow
   | ingest -> chunk   | decide -> tool -> loop
   | -> embed -> Chroma| (LangGraph)
   | -> retrieve -> LLM|
   +--------+---------+---------+
            |  LLMResponse (tokens, latency, cost)
      LLMMeter (SQLite + dashboard)
```

## Layout

```
core/        llm.py, embeddings.py, metrics.py, config.py
docquery/    app/{main,service,chunking,vectorstore}.py + tests
llmmeter/    app/{main,decorator}.py + tests
agentflow/   app/{main,graph,tools}.py + tests
```
