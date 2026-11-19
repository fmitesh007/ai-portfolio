"""AgentFlow API -- run a tool-using agent and inspect its step traces."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from core.config import settings

from .graph import MockAgentLLM, build_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = MockAgentLLM() if settings.llm_provider == "mock" else None
    agent, metrics = build_agent(llm=llm)
    app.state.agent = agent
    app.state.metrics = metrics
    yield


app = FastAPI(title="AgentFlow", version="1.0.0", lifespan=lifespan)


class RunRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "max_steps": 4}


@app.post("/run")
def run(payload: RunRequest):
    result = app.state.agent.invoke(
        {
            "question": payload.question,
            "history": [],
            "steps": 0,
            "pending": "",
            "answer": "",
        }
    )
    return {
        "question": payload.question,
        "answer": result.get("answer", ""),
        "trace": result.get("history", []),
        "steps": result.get("steps", 0),
        "metrics": app.state.metrics.summary(app="agentflow"),
    }
