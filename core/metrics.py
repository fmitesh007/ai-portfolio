"""SQLite-backed LLM telemetry: token counts, latency percentiles, cost, errors."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import settings
from .llm import LLMResponse


@dataclass
class MetricEvent:
    app: str
    operation: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    ok: bool = True
    error: str = ""
    ts: str = ""

    def __post_init__(self) -> None:
        # Space-separated UTC so string comparison works with SQLite datetime('now').
        self.ts = self.ts or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class MetricsStore:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or settings.metrics_db)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    app TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    ok INTEGER NOT NULL DEFAULT 1,
                    error TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, event: MetricEvent) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO llm_events
                   (ts, app, operation, model, input_tokens, output_tokens,
                    latency_ms, cost_usd, ok, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.ts,
                    event.app,
                    event.operation,
                    event.model,
                    event.input_tokens,
                    event.output_tokens,
                    event.latency_ms,
                    event.cost_usd,
                    int(event.ok),
                    event.error,
                ),
            )

    def summary(
        self, *, app: str | None = None, since_minutes: int | None = None
    ) -> dict:
        clauses, params = [], []
        if app:
            clauses.append("app = ?")
            params.append(app)
        if since_minutes:
            clauses.append("ts >= datetime('now', ?)")
            params.append(f"-{since_minutes} minutes")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT latency_ms, input_tokens, output_tokens, cost_usd, ok FROM llm_events {where}",
                params,
            ).fetchall()

        if not rows:
            return {
                "calls": 0,
                "errors": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }

        latencies = sorted(row["latency_ms"] for row in rows)
        return {
            "calls": len(rows),
            "errors": sum(1 for row in rows if not row["ok"]),
            "total_tokens": sum(
                row["input_tokens"] + row["output_tokens"] for row in rows
            ),
            "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 6),
            "p50_latency_ms": round(_percentile(latencies, 0.50), 2),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        }

    def recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM llm_events")


@contextmanager
def track(
    store: MetricsStore, *, app: str, operation: str, model: str = "unknown"
) -> Iterator[list[MetricEvent]]:
    """Record one LLM call, including failures.

    Usage::

        with track(store, app="docquery", operation="query") as events:
            response = llm.complete(prompt)
            events.append(event_from(response))
    """
    started = time.perf_counter()
    events: list[MetricEvent] = []
    try:
        yield events
    except Exception as exc:  # noqa: BLE001 - telemetry must not swallow the original error
        store.record(
            MetricEvent(
                app=app,
                operation=operation,
                model=model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    else:
        event = (
            events[0]
            if events
            else MetricEvent(app=app, operation=operation, model=model)
        )
        event.app, event.operation = app, operation
        if model != "unknown":
            event.model = model
        store.record(event)


def event_from(response: LLMResponse) -> MetricEvent:
    return MetricEvent(
        app="",
        operation="",
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]
