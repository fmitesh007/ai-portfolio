"""LangGraph ReAct-style loop with streaming step traces and LLM metering."""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from core.llm import BaseLLM, get_llm
from core.metrics import MetricsStore, event_from, track

from .tools import run_tool

MAX_STEPS = 4

POLICY = """You are a tool-using agent. Choose ONE action per turn.

Available tools:
- calculator | a basic arithmetic expression, e.g. 12 * 7
- vector_lookup | a search query over the indexed document collection
- http_fetch | an http(s) URL to read
- FINISH | the final answer to the user's question

Reply with exactly one line in this format:
TOOL: <name> | <argument>
or, when done:
FINISH: <answer>

Question: {question}
History:
{history}
"""


class AgentState(TypedDict):
    question: str
    history: list[str]
    steps: int
    pending: str
    answer: str


class MockAgentLLM(BaseLLM):
    """Deterministic offline policy so the demo works without an API key."""

    name = "mock-agent"

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0):  # type: ignore[override]
        from core.llm import LLMResponse

        question = prompt.split("Question:", 1)[-1].split("History:", 1)[0].strip()
        history = prompt.split("History:", 1)[-1] if "History:" in prompt else ""
        observation = re.search(
            r"(?:calculator|vector_lookup|http_fetch)\([^\n]*?\) -> ([^\n]+)", history
        )
        if observation:
            action = f"FINISH: {observation.group(1).strip()[:300]}"
        elif "http://" in question or "https://" in question:
            url = re.search(r"https?://\S+", question)
            action = f"TOOL: http_fetch | {url.group(0) if url else ''}"
        elif match := re.search(r"\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*", question):
            action = f"TOOL: calculator | {match.group(0).strip()}"
        elif re.search(r"document|index|search|rag|vector", question, re.I):
            action = f"TOOL: vector_lookup | {question}"
        else:
            action = (
                f"FINISH: I can answer '{question}' directly from the conversation."
            )
        return LLMResponse(
            text=action,
            model=self.name,
            input_tokens=20,
            output_tokens=12,
            latency_ms=1.0,
        )


def build_agent(llm: BaseLLM | None = None, metrics: MetricsStore | None = None):
    llm = llm or get_llm()
    metrics = metrics or MetricsStore()

    def decide(state: AgentState) -> dict:
        if state["steps"] >= MAX_STEPS:
            return {
                "pending": "FINISH: maximum reasoning steps reached",
                "answer": "Maximum reasoning steps reached.",
            }
        prompt = POLICY.format(
            question=state["question"], history="\n".join(state["history"]) or "-"
        )
        with track(metrics, app="agentflow", operation="decide") as events:
            response = llm.complete(prompt, system="Be concise. Choose one action.")
            events.append(event_from(response))
        pending = response.text.strip()
        result = {
            "pending": pending,
            "history": state["history"] + [f"think: {pending}"],
        }
        if pending.upper().startswith("FINISH:") or not pending.upper().startswith(
            "TOOL:"
        ):
            result["answer"] = pending.replace("FINISH:", "", 1).strip()
        return result

    def execute(state: AgentState) -> dict:
        match = re.match(r"TOOL:\s*(\w+)\s*\|\s*(.+)", state["pending"], re.I | re.S)
        if not match:
            return {
                "answer": state["pending"].replace("FINISH:", "", 1).strip(),
                "steps": state["steps"] + 1,
            }
        name, argument = match.group(1), match.group(2).strip()
        result = run_tool(name, argument)
        return {
            "history": state["history"] + [f"{name}({argument}) -> {result}"],
            "steps": state["steps"] + 1,
        }

    def route_after_decide(state: AgentState) -> str:
        if state["pending"].upper().startswith("FINISH:") or not state[
            "pending"
        ].upper().startswith("TOOL:"):
            return "finish"
        return "execute"

    def route_after_execute(state: AgentState) -> str:
        return "finish" if state["steps"] >= MAX_STEPS else "decide"

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", route_after_decide, {"execute": "execute", "finish": END}
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"decide": "decide", "finish": END}
    )
    return graph.compile(), metrics
