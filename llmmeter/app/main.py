"""LLMMeter dashboard -- latency, tokens, cost and errors across all services."""

from __future__ import annotations

from html import escape

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from core.llm import MockLLM
from core.metrics import MetricsStore

from .decorator import metered

app = FastAPI(title="LLMMeter", version="1.0.0")
metrics = MetricsStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary(app_name: str | None = None, since_minutes: int | None = None):
    return metrics.summary(app=app_name, since_minutes=since_minutes)


@app.get("/api/events")
def events(limit: int = 20):
    return {"events": metrics.recent(limit)}


@app.post("/api/demo-call")
def demo_call():
    """Make one metered call so the dashboard has data without an API key."""

    @metered(metrics, app="llmmeter", operation="demo")
    def call() -> object:
        return MockLLM().complete("Summarise LLM observability in one sentence.")

    return call()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    rows = metrics.recent(12)
    data = metrics.summary()
    bars = "".join(
        f'<div class="bar" style="width:{min(100, item["latency_ms"] / 5):.0f}%"></div>'
        for item in reversed(rows)
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LLMMeter</title>
<style>
 body{{font:14px system-ui,sans-serif;background:#11151c;color:#e7ebf0;margin:0;padding:32px}}
 h1{{font-weight:500;letter-spacing:1px}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
 .card{{background:#1b212b;border:1px solid #2b3442;border-radius:10px;padding:18px 22px;min-width:150px}}
 .card b{{display:block;font-size:28px;font-weight:600}} .card span{{color:#8b98a9;font-size:12px;text-transform:uppercase}}
 .bar{{height:10px;background:#5b9cff;border-radius:5px;margin:4px 0}}
 table{{border-collapse:collapse;width:100%;margin-top:24px;font-size:12px}}
 th,td{{text-align:left;padding:8px;border-bottom:1px solid #2b3442}}
 th{{color:#8b98a9;text-transform:uppercase}} .ok{{color:#5fd38d}} .err{{color:#ff6b6b}}
</style></head><body>
<h1>LLMMeter</h1>
<div class="cards">
 <div class="card"><b>{data["calls"]}</b><span>calls</span></div>
 <div class="card"><b>{data["errors"]}</b><span>errors</span></div>
 <div class="card"><b>{data["total_tokens"]}</b><span>tokens</span></div>
 <div class="card"><b>${data["total_cost_usd"]:.6f}</b><span>est. cost</span></div>
 <div class="card"><b>{data["p50_latency_ms"]:.1f}ms</b><span>p50 latency</span></div>
 <div class="card"><b>{data["p95_latency_ms"]:.1f}ms</b><span>p95 latency</span></div>
</div>
<h3>Recent latency</h3>{bars or "<p>No calls yet.</p>"}
<table><thead><tr><th>Time</th><th>App</th><th>Operation</th><th>Model</th><th>Tokens</th><th>Latency</th><th>Status</th></tr></thead><tbody>
{"".join(f'<tr><td>{escape(str(r["ts"]))}</td><td>{escape(r["app"])}</td><td>{escape(r["operation"])}</td><td>{escape(r["model"])}</td><td>{r["input_tokens"] + r["output_tokens"]}</td><td>{r["latency_ms"]:.1f}ms</td><td class="{"ok" if r["ok"] else "err"}">{"ok" if r["ok"] else "error"}</td></tr>' for r in rows)}
</tbody></table></body></html>"""
