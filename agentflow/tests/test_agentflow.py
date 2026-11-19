from pathlib import Path

from core.metrics import MetricsStore
from agentflow.app.graph import MockAgentLLM, build_agent
from agentflow.app.tools import calculator, http_fetch, run_tool


def test_calculator_is_safe_and_correct():
    assert calculator("12 * 7 + 3") == "87.0"
    assert "error" in calculator("__import__('os')")


def test_unknown_tool_is_reported():
    assert "unknown tool" in run_tool("nope", "x")


def test_http_fetch_rejects_non_http_schemes():
    assert "error" in http_fetch("file:///etc/passwd")


def test_agent_uses_a_tool_then_finishes(tmp_path: Path):
    agent, metrics = build_agent(
        llm=MockAgentLLM(), metrics=MetricsStore(tmp_path / "metrics.sqlite3")
    )
    result = agent.invoke(
        {
            "question": "What is 12 * 7 + 3?",
            "history": [],
            "steps": 0,
            "pending": "",
            "answer": "",
        }
    )
    assert result["answer"]
    assert any("12 * 7" in step and "87" in step for step in result["history"])
    assert result["steps"] >= 1
    assert metrics.summary(app="agentflow")["calls"] >= 1
