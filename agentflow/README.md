# AgentFlow — tool-using agent

A LangGraph ReAct-style agent that plans a step, calls a tool, observes the
result and repeats until it can answer. Every reasoning step is traceable.

## Tools
- `calculator` — safe arithmetic via AST (no `eval`)
- `vector_lookup` — searches the documents indexed by DocQuery
- `http_fetch` — fetches an http(s) URL with a timeout

## Run

Easiest (from the repo root, works in fish/zsh/bash):

```fish
bash scripts/run.sh agentflow    # http://localhost:8003/docs
```

Or manually with a venv activated:

```bash
uvicorn agentflow.app.main:app --reload --port 8003
curl -X POST http://localhost:8003/run \
  -H 'content-type: application/json' \
  -d '{"question":"What is 12 * 7 + 3?"}'
```

The response includes the final answer, the full step trace and LLM metrics.

## Resume-ready bullets
- Built a LangGraph agent with tool use (calculator, vector search, HTTP fetch) and bounded iterative reasoning.
- Exposed step-level traces and LLM telemetry through a FastAPI endpoint, with a deterministic offline policy for testing.
