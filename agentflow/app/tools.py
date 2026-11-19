"""Three real tools the agent can call: math, vector search, and HTTP fetch."""

from __future__ import annotations

import ast
import operator
import re

import httpx

from core.config import settings
from docquery.app.vectorstore import VectorStore

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression without using eval()."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001 - tool errors are returned to the agent
        return f"calculator error: {exc}"


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def vector_lookup(query: str) -> str:
    """Search the documents indexed by DocQuery's shared Chroma collection."""
    try:
        store = VectorStore()
        if store.count() == 0:
            return "vector store is empty; ingest documents with DocQuery first"
        from core.embeddings import get_embeddings

        hits = store.search(get_embeddings().embed_one(query), top_k=2)
        return " | ".join(f"[{h['doc_id']}] {h['text'][:240]}" for h in hits)
    except Exception as exc:  # noqa: BLE001
        return f"vector_lookup error: {exc}"


def http_fetch(url: str) -> str:
    """Fetch a URL and return the first 600 characters of text."""
    if not re.match(r"^https?://", url):
        return "http_fetch error: only http(s) URLs are allowed"
    try:
        with httpx.Client(
            timeout=settings.request_timeout, follow_redirects=True
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return re.sub(r"\s+", " ", response.text)[:600]
    except Exception as exc:  # noqa: BLE001
        return f"http_fetch error: {exc}"


TOOLS = {
    "calculator": calculator,
    "vector_lookup": vector_lookup,
    "http_fetch": http_fetch,
}


def run_tool(name: str, argument: str) -> str:
    func = TOOLS.get(name)
    if func is None:
        return f"unknown tool: {name}"
    return func(argument)
