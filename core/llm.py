"""Pluggable LLM layer: offline mock, Gemini, or any OpenAI-compatible endpoint.

All providers return the same :class:`LLMResponse`, so callers get token counts,
latency and a cost estimate for free -- that is what makes LLMMeter possible.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from .config import settings


# USD per 1M tokens. Override with PRICE_* env vars if your account differs.
PRICING = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLM:
    name = "base"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic offline stand-in.

    It is not a language model: it selects the context sentences that best
    overlap with the question and cites them. Enough to exercise the whole RAG
    pipeline (retrieval -> prompt -> generation -> telemetry) with no API key.
    """

    name = "mock-local"

    def __init__(self, model: str = "mock-local"):
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        question = _between(prompt, "Question:", "\n") or "the question"
        sentences = _context_sentences(prompt)
        scored = sorted(
            ((_overlap(question, s), i + 1, s) for i, s in enumerate(sentences)),
            key=lambda item: item[0],
            reverse=True,
        )
        best = [item for item in scored if item[0] > 0][:3]

        if best:
            body = " ".join(s for _, _, s in best)
            cites = ", ".join(f"[{i}]" for _, i, _ in best)
            text = f"Based on the retrieved context: {body} ({cites})"
        else:
            text = "I could not find a direct answer in the provided context."

        input_tokens = _tokens(prompt)
        output_tokens = _tokens(text)
        return LLMResponse(
            text=text[: max_tokens * 4],
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            cost_usd=0.0,
        )


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        with httpx.Client(timeout=settings.request_timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        text = data["candidates"][0]["content"]["parts"][0].get("text", "")
        usage = data.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", 0))
        out_tok = int(usage.get("candidatesTokenCount", 0))
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
        )


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=settings.request_timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
        )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


def get_llm() -> BaseLLM:
    provider = settings.llm_provider
    if provider == "gemini":
        return GeminiLLM()
    if provider == "openai":
        return OpenAILLM()
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r} (use mock, gemini, or openai)")


# --- small helpers used by the mock provider ---------------------------------


def _tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|\S", text)))


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    rest = text.split(start, 1)[1]
    return rest.split(end, 1)[0].strip() if end in rest else rest.strip()


def _context_sentences(prompt: str) -> list[str]:
    found = re.findall(r"^\[(\d+)\]\s*(.+)$", prompt, flags=re.MULTILINE)
    return [sentence for _, sentence in sorted(found, key=lambda pair: int(pair[0]))]


def _overlap(question: str, sentence: str) -> int:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "to",
        "and",
        "in",
        "for",
        "on",
        "what",
        "how",
        "does",
    }
    q = {w for w in re.findall(r"\w+", question.lower()) if w not in stop}
    s = {w for w in re.findall(r"\w+", sentence.lower()) if w not in stop}
    return len(q & s)
