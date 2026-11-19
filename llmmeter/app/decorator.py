"""Drop-in decorator for metering any function that returns an LLMResponse."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from core.llm import LLMResponse
from core.metrics import MetricsStore, event_from, track

F = TypeVar("F", bound=Callable[..., LLMResponse])


def metered(store: MetricsStore, *, app: str, operation: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> LLMResponse:
            with track(store, app=app, operation=operation) as events:
                response = func(*args, **kwargs)
                events.append(event_from(response))
            return response

        return wrapper  # type: ignore[return-value]

    return decorator
