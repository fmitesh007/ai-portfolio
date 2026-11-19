from pathlib import Path

from datetime import datetime, timedelta, timezone

from core.llm import MockLLM
from core.metrics import MetricEvent, MetricsStore
from llmmeter.app.decorator import metered


def test_decorator_records_tokens_latency_and_cost(tmp_path: Path):
    store = MetricsStore(tmp_path / "metrics.sqlite3")

    @metered(store, app="unit", operation="complete")
    def call() -> object:
        return MockLLM().complete("What is RAG?")

    response = call()
    assert response.text

    summary = store.summary(app="unit")
    assert summary["calls"] == 1
    assert summary["errors"] == 0
    assert summary["total_tokens"] > 0
    assert summary["p95_latency_ms"] >= summary["p50_latency_ms"]


def test_since_minutes_filter_works(tmp_path: Path):
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    old = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    store.record(MetricEvent(app="old", operation="call", model="m", ts=old))
    store.record(MetricEvent(app="new", operation="call", model="m"))

    assert store.summary(since_minutes=60)["calls"] == 1


def test_failed_call_is_recorded_as_error(tmp_path: Path):
    store = MetricsStore(tmp_path / "metrics.sqlite3")

    @metered(store, app="unit", operation="broken")
    def call() -> object:
        raise RuntimeError("boom")

    try:
        call()
    except RuntimeError:
        pass

    summary = store.summary(app="unit")
    assert summary["calls"] == 1
    assert summary["errors"] == 1
