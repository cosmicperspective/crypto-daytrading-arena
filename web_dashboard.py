"""Live web dashboard for the crypto daytrading arena.

Reads from arena.db (SQLite) and pushes updates to the browser via WebSocket.
Open http://localhost:8050 in your browser.

Usage:
    uv run python web_dashboard.py
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

load_dotenv()

DB_PATH = Path(__file__).parent / "arena.db"
INITIAL_CASH = 200.0
START_TIME = time.time()

logger = logging.getLogger(__name__)

# ── Live OpenRouter cost tracking ────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_or_spend_cache: dict = {"total": 0.0, "last_poll": 0.0, "limit": 0.0, "remaining": 0.0}
_or_model_prices: dict = {}  # model_id -> {input: $/token, output: $/token}
_OR_POLL_INTERVAL = 10.0  # seconds between API polls


def _poll_openrouter_spend() -> dict:
    """Poll OpenRouter /auth/key for real cumulative spend."""
    now = time.time()
    if now - _or_spend_cache["last_poll"] < _OR_POLL_INTERVAL:
        return _or_spend_cache
    if not OPENROUTER_API_KEY:
        return _or_spend_cache
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=5,
        )
        data = r.json().get("data", {})
        _or_spend_cache["total"] = float(data.get("usage", 0))
        _or_spend_cache["limit"] = float(data.get("limit", 0))
        _or_spend_cache["remaining"] = float(data.get("limit_remaining", 0))
        _or_spend_cache["last_poll"] = now
    except Exception as e:
        logger.debug("OpenRouter poll failed: %s", e)
    return _or_spend_cache


def _load_model_prices() -> None:
    """Fetch live per-model pricing from OpenRouter at startup."""
    if _or_model_prices or not OPENROUTER_API_KEY:
        return
    try:
        r = httpx.get("https://openrouter.ai/api/v1/models", timeout=15)
        for m in r.json().get("data", []):
            p = m.get("pricing", {})
            _or_model_prices[m["id"]] = {
                "input": float(p.get("prompt", 0)),
                "output": float(p.get("completion", 0)),
            }
    except Exception as e:
        logger.debug("OpenRouter models fetch failed: %s", e)

app = FastAPI()


def query_db() -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    agents = {}
    for row in conn.execute("SELECT a.agent_id, a.cash, s.trade_count, s.realized_pnl, "
                            "s.fees_paid, s.wins, s.losses, s.equity_peak, s.max_drawdown, "
                            "s.memory_calls, "
                            "m.strategy, m.model_id, m.provider "
                            "FROM accounts a "
                            "LEFT JOIN stats s ON a.agent_id = s.agent_id "
                            "LEFT JOIN agent_meta m ON a.agent_id = m.agent_id"):
        agents[row["agent_id"]] = {
            "cash": row["cash"] or 0,
            "trade_count": row["trade_count"] or 0,
            "realized_pnl": row["realized_pnl"] or 0,
            "fees_paid": row["fees_paid"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "equity_peak": row["equity_peak"] or INITIAL_CASH,
            "max_drawdown": row["max_drawdown"] or 0,
            "memory_calls": row["memory_calls"] or 0,
            "strategy": row["strategy"] or "unknown",
            "model_id": row["model_id"] or "unknown",
            "provider": row["provider"] or "unknown",
            "positions": {},
            "total_value": row["cash"] or 0,
        }

    for row in conn.execute("SELECT agent_id, product_id, qty, cost_basis FROM positions"):
        aid = row["agent_id"]
        if aid in agents:
            agents[aid]["positions"][row["product_id"]] = {
                "qty": row["qty"],
                "cost_basis": row["cost_basis"],
            }

    trades = []
    for row in conn.execute("SELECT ts, agent_id, action, product_id, qty, price, fee "
                            "FROM trades ORDER BY id DESC LIMIT 50"):
        trades.append({
            "ts": row["ts"],
            "agent": row["agent_id"],
            "action": row["action"],
            "product": row["product_id"],
            "qty": row["qty"],
            "price": row["price"],
            "fee": row["fee"],
            "total": round(row["qty"] * row["price"], 2),
        })

    # Get live prices and % changes from price_history table
    live_prices = {}
    now = time.time()
    timeframes = {
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "2h": 7200,
    }

    # Check if price_history table exists
    has_price_history = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'"
    ).fetchone() is not None

    if has_price_history:
        # Get latest price per product
        for row in conn.execute(
            "SELECT product_id, price, ts FROM price_history "
            "WHERE id IN (SELECT MAX(id) FROM price_history GROUP BY product_id) "
            "ORDER BY product_id"
        ):
            live_prices[row["product_id"]] = {
                "price": row["price"],
                "changes": {},
            }

        # Compute % change for each timeframe per product
        for pid in list(live_prices.keys()):
            current = live_prices[pid]["price"]
            for label, seconds in timeframes.items():
                target_ts = now - seconds
                row = conn.execute(
                    "SELECT price FROM price_history "
                    "WHERE product_id = ? AND ts <= ? "
                    "ORDER BY ts DESC LIMIT 1",
                    (pid, target_ts),
                ).fetchone()
                if row and row["price"] and row["price"] != 0:
                    pct = (current - row["price"]) / row["price"] * 100
                    live_prices[pid]["changes"][label] = round(pct, 4)
    else:
        # Fallback: get prices from most recent trades
        for row in conn.execute(
            "SELECT product_id, price, ts FROM trades "
            "WHERE id IN (SELECT MAX(id) FROM trades GROUP BY product_id) "
            "ORDER BY product_id"
        ):
            live_prices[row["product_id"]] = {
                "price": row["price"],
                "changes": {},
            }

    # Compute total values using live prices where available
    for aid, a in agents.items():
        pos_value = 0.0
        for pid, pos in a["positions"].items():
            if pid in live_prices:
                pos_value += pos["qty"] * live_prices[pid]["price"]
            else:
                pos_value += pos["cost_basis"]
        a["total_value"] = a["cash"] + pos_value

    # Trade counts by agent over time for sparkline
    trade_history = {}
    for row in conn.execute("SELECT agent_id, ts, action, product_id, qty, price "
                            "FROM trades ORDER BY id ASC"):
        aid = row["agent_id"]
        if aid not in trade_history:
            trade_history[aid] = []
        # Running total value after each trade
        trade_history[aid].append({
            "ts": row["ts"],
            "action": row["action"],
            "product": row["product_id"],
        })

    # Balance snapshots: compute running cash after each trade
    balance_series = {}
    for row in conn.execute("SELECT agent_id, action, qty, price, fee FROM trades ORDER BY id ASC"):
        aid = row["agent_id"]
        if aid not in balance_series:
            balance_series[aid] = [INITIAL_CASH]
        last = balance_series[aid][-1]
        if row["action"] == "buy":
            last = last - (row["qty"] * row["price"]) - row["fee"]
        else:
            last = last + (row["qty"] * row["price"]) - row["fee"]
        balance_series[aid].append(round(last, 2))

    # Live LLM spend from OpenRouter API
    total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    num_agents = max(len(agents), 1)
    elapsed_min = (time.time() - START_TIME) / 60

    or_data = _poll_openrouter_spend()
    real_spend = or_data["total"]
    spend_per_hour = (real_spend / max(elapsed_min, 0.1)) * 60 if real_spend > 0 else 0.0
    credit_remaining = or_data["remaining"]

    # Detect competition mode from agent data
    strategies = set(a["strategy"] for a in agents.values())
    models_set = set(a["model_id"] for a in agents.values())
    if strategies == {"neutral", "smart"} and len(models_set) > 1:
        mode = "v2"
        mode_label = "V2 A/B Test (Dumb vs Smart)"
    elif len(strategies) == 1 and len(models_set) > 1:
        mode = "llm"
        mode_label = "LLM Competition"
    elif len(strategies) > 1:
        mode = "strategy"
        mode_label = "Strategy Competition"
    else:
        mode = "unknown"
        mode_label = "Arena"

    conn.close()

    return {
        "agents": agents,
        "trades": trades,
        "balance_series": balance_series,
        "live_prices": live_prices,
        "mode": mode,
        "mode_label": mode_label,
        "llm_spend": {
            "total": round(real_spend, 4),
            "per_hour": round(spend_per_hour, 4),
            "credit_remaining": round(credit_remaining, 2),
            "total_trades": total_trades,
            "elapsed_min": round(elapsed_min, 1),
            "model": "multiple" if len(models_set) > 1 else next(iter(models_set), "unknown"),
            "model_prices": {
                mid: _or_model_prices.get(mid, {})
                for mid in models_set
                if mid in _or_model_prices
            },
        },
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Darwin</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0e17;
    color: #e0e6f0;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #0d1321 0%, #1a1f35 100%);
    border-bottom: 1px solid #2a3050;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .header h1 {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .spend-ticker {
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(135deg, #1a1230 0%, #1e1535 100%);
    border: 1px solid #2d2250;
    border-radius: 8px;
    padding: 8px 20px;
    font-family: 'Consolas', 'SF Mono', monospace;
  }
  .spend-amount {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6b6b, #ffa502);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .spend-detail {
    font-size: 11px;
    color: #8892b0;
    line-height: 1.5;
  }
  .spend-rate {
    color: #ffa502;
    font-weight: 600;
  }
  .live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #8892b0;
  }
  .live-dot {
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,255,136,0); }
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px 24px;
  }
  .full-width { grid-column: 1 / -1; }
  .card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: #334155; }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 16px;
  }

  /* Leaderboard */
  .leaderboard { grid-column: 1 / -1; }
  .lb-table { width: 100%; border-collapse: collapse; }
  .lb-table th {
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
  }
  .lb-table td {
    padding: 12px;
    border-bottom: 1px solid #1e293b;
    font-size: 14px;
  }
  .lb-table tr:hover { background: #1a2332; }
  .rank {
    font-weight: 700;
    font-size: 18px;
    width: 40px;
  }
  .rank-1 { color: #ffd700; }
  .rank-2 { color: #c0c0c0; }
  .rank-3 { color: #cd7f32; }
  .agent-name { font-weight: 600; color: #e0e6f0; }
  .strategy-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .strategy-default { background: #1e3a5f; color: #60a5fa; }
  .strategy-momentum { background: #1e3b3a; color: #34d399; }
  .strategy-brainrot { background: #3b1e3a; color: #f472b6; }
  .strategy-scalper { background: #3b351e; color: #fbbf24; }
  .strategy-neutral { background: #2a2a3a; color: #a78bfa; }
  .strategy-smart { background: #1a0d2e; color: #c084fc; border: 1px solid #7c3aed; }
  .positive { color: #00ff88; }
  .negative { color: #ff4757; }
  .neutral { color: #8892b0; }

  /* Agent cards */
  .agent-cards {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
  }
  .agent-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
  }
  .agent-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .agent-card-name { font-weight: 700; font-size: 15px; }
  .agent-card .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
  }
  .stat-label { color: #64748b; }
  .stat-value { font-weight: 600; }

  /* Trade log */
  .trade-log { grid-column: 1 / -1; max-height: 400px; overflow-y: auto; }
  .trade-table { width: 100%; border-collapse: collapse; }
  .trade-table th {
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    padding: 8px 10px;
    border-bottom: 1px solid #1e293b;
    position: sticky;
    top: 0;
    background: #111827;
  }
  .trade-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #0f172a;
    font-size: 13px;
    font-family: 'Consolas', 'SF Mono', monospace;
  }
  .trade-table tr:hover { background: #1a2332; }
  .buy-badge {
    background: #064e3b;
    color: #34d399;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 11px;
  }
  .sell-badge {
    background: #4c0519;
    color: #fb7185;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 11px;
  }

  /* Chart */
  .chart-container {
    grid-column: 1 / -1;
    height: 300px;
    position: relative;
  }

  /* Positions */
  .pos-table { width: 100%; border-collapse: collapse; }
  .pos-table th {
    text-align: left; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
    color: #64748b; padding: 6px 10px;
    border-bottom: 1px solid #1e293b;
  }
  .pos-table td {
    padding: 6px 10px; border-bottom: 1px solid #0f172a;
    font-size: 13px; font-family: 'Consolas', monospace;
  }

  .update-flash { animation: flash 0.5s ease; }
  @keyframes flash {
    0% { background: rgba(0,212,255,0.1); }
    100% { background: transparent; }
  }

  /* Price Ticker Bar */
  .price-bar {
    background: #0d1117;
    border-bottom: 1px solid #1e293b;
    padding: 10px 24px;
    display: flex;
    gap: 20px;
    overflow-x: auto;
    flex-wrap: wrap;
  }
  .price-chip {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 200px;
    flex: 1;
    transition: border-color 0.3s;
  }
  .price-chip.changed { border-color: #7b61ff; }
  .price-chip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .price-chip .coin-name {
    font-weight: 700;
    font-size: 14px;
    color: #e0e6f0;
  }
  .price-chip .coin-price {
    font-family: 'Consolas', monospace;
    font-size: 16px;
    font-weight: 700;
    color: #00d4ff;
  }
  .price-changes {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .price-change {
    font-family: 'Consolas', monospace;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    line-height: 1.4;
  }
  .price-change .tf-label {
    color: #64748b;
    font-weight: 400;
    margin-right: 2px;
  }
  .price-change.up { background: #064e3b; color: #34d399; }
  .price-change.down { background: #4c0519; color: #fb7185; }
  .price-change.flat { background: #1e293b; color: #64748b; }

  /* Strategy Guide */
  .strategy-guide {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }
  .strat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    border-top: 3px solid;
  }
  .strat-card.s-default { border-top-color: #60a5fa; }
  .strat-card.s-momentum { border-top-color: #34d399; }
  .strat-card.s-brainrot { border-top-color: #f472b6; }
  .strat-card.s-scalper { border-top-color: #fbbf24; }
  .strat-card h3 {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .strat-card .strat-subtitle {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
  }
  .strat-card.s-default .strat-subtitle { color: #60a5fa; }
  .strat-card.s-momentum .strat-subtitle { color: #34d399; }
  .strat-card.s-brainrot .strat-subtitle { color: #f472b6; }
  .strat-card.s-scalper .strat-subtitle { color: #fbbf24; }
  .strat-card.s-neutral { border-top-color: #a78bfa; }
  .strat-card.s-neutral .strat-subtitle { color: #a78bfa; }
  .strat-card.s-neutral li::before { background: #a78bfa; }
  .strat-card ul {
    list-style: none;
    padding: 0;
  }
  .strat-card li {
    font-size: 12px;
    color: #8892b0;
    padding: 3px 0;
    padding-left: 14px;
    position: relative;
    line-height: 1.4;
  }
  .strat-card li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 9px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }
  .strat-card.s-default li::before { background: #60a5fa; }
  .strat-card.s-momentum li::before { background: #34d399; }
  .strat-card.s-brainrot li::before { background: #f472b6; }
  .strat-card.s-scalper li::before { background: #fbbf24; }
  .strat-note {
    font-size: 11px;
    color: #475569;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
    font-style: italic;
  }

  /* System Info */
  .system-info {
    grid-column: 1 / -1;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }
  .sys-chip {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sys-chip .sys-label {
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 10px;
  }
  .sys-chip .sys-value {
    color: #e0e6f0;
    font-family: 'Consolas', monospace;
    font-weight: 600;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Darwin</h1>
  <div id="mode-badge" style="padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;border:1px solid #334155;color:#8892b0;"></div>
  <div class="spend-ticker" id="spend-ticker">
    <div>
      <div class="spend-amount" id="spend-total">$0.0000</div>
      <div class="spend-detail">LLM Spend (live)</div>
    </div>
    <div class="spend-detail">
      <div><span class="spend-rate" id="spend-rate">$0.00/hr</span></div>
      <div>Credit: $<span id="spend-remaining">0</span></div>
      <div><span id="spend-model">openrouter</span></div>
    </div>
  </div>
  <div class="live-badge">
    <div class="live-dot"></div>
    <span>LIVE</span>
    <span id="clock" style="margin-left:12px; font-family:monospace;"></span>
    <span id="update-count" style="margin-left:12px; color:#64748b;"></span>
  </div>
</div>

<div class="price-bar" id="price-bar">
  <span style="color:#64748b; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; align-self:center;">Live Prices</span>
</div>

<div class="grid">

  <!-- Leaderboard -->
  <div class="card leaderboard">
    <div class="card-title">Leaderboard</div>
    <table class="lb-table">
      <thead>
        <tr>
          <th>#</th><th>Agent</th><th>Strategy</th><th>Model</th>
          <th style="text-align:right">Total Value</th>
          <th style="text-align:right">ROI</th>
          <th style="text-align:right">Realized P&L</th>
          <th style="text-align:right">Trades</th>
          <th style="text-align:right">Win%</th>
          <th style="text-align:right">Memory</th>
          <th style="text-align:right">Max DD</th>
        </tr>
      </thead>
      <tbody id="leaderboard-body"></tbody>
    </table>
  </div>

  <!-- Agent Cards -->
  <div class="agent-cards" id="agent-cards"></div>

  <!-- Strategy / LLM Guide (dynamic) -->
  <div class="strategy-guide" id="guide-section"></div>

  <!-- System Info -->
  <div class="system-info">
    <div class="sys-chip">
      <span class="sys-label">Context</span>
      <span class="sys-value">Fresh per tick (no chat history)</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">Per Tick</span>
      <span class="sys-value">Market data + trade history tools</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">Data Interval</span>
      <span class="sys-value">2s</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">LLM Rate</span>
      <span class="sys-value" id="sys-llm-rate">~2/s</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">Agents</span>
      <span class="sys-value" id="sys-agent-count">5</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">Sim Preset</span>
      <span class="sys-value">Realistic (6bps fees)</span>
    </div>
    <div class="sys-chip">
      <span class="sys-label">Memory</span>
      <span class="sys-value" style="color:#34d399;">Trade history &amp; performance stats</span>
    </div>
  </div>

  <!-- Chart -->
  <div class="card chart-container">
    <div class="card-title">Cash Balance History</div>
    <canvas id="balanceChart"></canvas>
  </div>

  <!-- Positions -->
  <div class="card full-width">
    <div class="card-title">Open Positions</div>
    <table class="pos-table">
      <thead>
        <tr>
          <th>Agent</th><th>Ticker</th><th style="text-align:right">Qty</th>
          <th style="text-align:right">Cost Basis</th>
        </tr>
      </thead>
      <tbody id="positions-body"></tbody>
    </table>
  </div>

  <!-- Trade Log -->
  <div class="card trade-log">
    <div class="card-title">Trade Log (latest 50)</div>
    <table class="trade-table">
      <thead>
        <tr>
          <th>Time</th><th>Agent</th><th>Action</th><th>Product</th>
          <th style="text-align:right">Qty</th>
          <th style="text-align:right">Price</th>
          <th style="text-align:right">Total</th>
          <th style="text-align:right">Fee</th>
        </tr>
      </thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
</div>

<script>
const COLORS = {
  // Strategy mode agents
  'agent-default': '#60a5fa',
  'agent-momentum': '#34d399',
  'agent-brainrot': '#f472b6',
  'agent-scalper': '#fbbf24',
  // LLM mode agents
  'agent-gemini': '#60a5fa',
  'agent-gpt5nano': '#34d399',
  'agent-minimax': '#f472b6',
  'agent-grok': '#fbbf24',
  'agent-haiku': '#a78bfa',
  // V2 A/B mode agents (v1=muted, v2=vivid)
  'gemini-v1': '#3b6ea0', 'gemini-v2': '#60a5fa',
  'gpt5nano-v1': '#1f8a66', 'gpt5nano-v2': '#34d399',
  'minimax-v1': '#a04878', 'minimax-v2': '#f472b6',
  'grok-v1': '#a0871a', 'grok-v2': '#fbbf24',
  'haiku-v1': '#6b5a9e', 'haiku-v2': '#a78bfa',
};
const STRATEGY_CLASS = {
  'default': 'strategy-default',
  'momentum': 'strategy-momentum',
  'brainrot': 'strategy-brainrot',
  'scalper': 'strategy-scalper',
  'neutral': 'strategy-neutral',
  'smart': 'strategy-smart',
};

const STRATEGY_GUIDE = {
  'default': {cls:'s-default', title:'Default', sub:'Balanced Trader', points:['Analyzes trends, momentum, support/resistance','Considers risk management before every trade','Will hold if no clear opportunity exists','Balanced position sizing across assets'], note:'The benchmark \u2014 rational, measured decisions'},
  'momentum': {cls:'s-momentum', title:'Momentum', sub:'Trend Follower', points:['Buys assets with strong upward price action','Lets winners run, cuts losers fast','Avoids sideways markets \u2014 stays in cash','Concentrates capital on strongest trends'], note:'"The trend is your friend" \u2014 never fights the tape'},
  'brainrot': {cls:'s-brainrot', title:'Brainrot', sub:'YOLO WSB Energy', points:['Goes all-in on every opportunity','Max size positions \u2014 "diversification is for cowards"','Buys pumps, buys dips \u2014 always buying','Never sells at a loss \u2014 diamond hands'], note:'Pure vibes-based trading. Send it.'},
  'scalper': {cls:'s-scalper', title:'Scalper', sub:'High Frequency', points:['Many small, quick trades for tiny gains','Takes profits quickly \u2014 never holds long','Spreads across all products for opportunities','Manageable position sizes, minimal exposure'], note:'Edge comes from volume \u2014 many small wins compound'},
};

const LLM_GUIDE = {
  'google/gemini-2.5-flash': {color:'#60a5fa', title:'Gemini 2.5 Flash', provider:'Google', desc:'Fast multimodal model with strong reasoning. $0.30/$2.50 per 1M tokens.'},
  'openai/gpt-5-nano': {color:'#34d399', title:'GPT-5 Nano', provider:'OpenAI', desc:'Smallest GPT-5 variant. Low reasoning mode. $0.05/$0.40 per 1M tokens.'},
  'minimax/minimax-m2.5': {color:'#f472b6', title:'MiniMax M2.5', provider:'MiniMax', desc:'Competitive Chinese frontier model. $0.30/$1.10 per 1M tokens.'},
  'x-ai/grok-4.1-fast': {color:'#fbbf24', title:'Grok 4.1 Fast', provider:'xAI', desc:'Speed-optimized Grok. No reasoning mode. $0.20/$0.50 per 1M tokens.'},
  'anthropic/claude-haiku-4.5': {color:'#a78bfa', title:'Claude Haiku 4.5', provider:'Anthropic', desc:'Fastest Claude model. Matches Sonnet 4 reasoning. $1.00/$5.00 per 1M tokens.'},
};

let chart = null;
let updateCount = 0;

function initChart() {
  const ctx = document.getElementById('balanceChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 12 } } },
        tooltip: {
          backgroundColor: '#1e293b',
          titleColor: '#e0e6f0',
          bodyColor: '#e0e6f0',
          borderColor: '#334155',
          borderWidth: 1,
          callbacks: {
            label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: { display: false },
        y: {
          grid: { color: '#1e293b' },
          ticks: {
            color: '#64748b',
            callback: v => '$' + v.toLocaleString()
          }
        }
      }
    }
  });
}

function fmt(n, decimals=2) {
  return '$' + Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function pctClass(v) { return v >= 0 ? 'positive' : v < 0 ? 'negative' : 'neutral'; }

function render(data) {
  updateCount++;
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  document.getElementById('update-count').textContent = `#${updateCount}`;

  // Live Prices
  if (data.live_prices) {
    const bar = document.getElementById('price-bar');
    const label = bar.querySelector('span');
    const coins = Object.entries(data.live_prices).sort((a,b) => a[0].localeCompare(b[0]));
    bar.innerHTML = '';
    bar.appendChild(label);
    const tfOrder = ['30s','1m','5m','15m','1h','2h'];
    coins.forEach(([pid, info]) => {
      const chip = document.createElement('div');
      chip.className = 'price-chip';
      const p = info.price;
      let priceStr;
      if (p < 0.0001) priceStr = '$' + p.toFixed(8);
      else if (p < 0.01) priceStr = '$' + p.toFixed(6);
      else if (p < 1) priceStr = '$' + p.toFixed(4);
      else priceStr = '$' + p.toFixed(2);
      const changes = info.changes || {};
      const changesHtml = tfOrder.map(tf => {
        const val = changes[tf];
        if (val === undefined) return `<span class="price-change flat"><span class="tf-label">${tf}</span>—</span>`;
        const cls = val > 0.01 ? 'up' : val < -0.01 ? 'down' : 'flat';
        const sign = val > 0 ? '+' : '';
        return `<span class="price-change ${cls}"><span class="tf-label">${tf}</span>${sign}${val.toFixed(2)}%</span>`;
      }).join('');
      chip.innerHTML = `
        <div class="price-chip-header">
          <span class="coin-name">${pid.replace('-USD','')}</span>
          <span class="coin-price">${priceStr}</span>
        </div>
        <div class="price-changes">${changesHtml}</div>
      `;
      bar.appendChild(chip);
    });
  }

  // Sort agents by total_value desc
  const sorted = Object.entries(data.agents)
    .sort((a,b) => b[1].total_value - a[1].total_value);

  // Leaderboard
  const lb = document.getElementById('leaderboard-body');
  lb.innerHTML = sorted.map(([id, a], i) => {
    const roi = ((a.total_value - 200) / 200 * 100);
    const winTotal = a.wins + a.losses;
    const winPct = winTotal > 0 ? (a.wins / winTotal * 100).toFixed(0) : '—';
    const mem = a.memory_calls || 0;
    const memColor = mem > 0 ? '#a78bfa' : '#64748b';
    return `<tr>
      <td class="rank rank-${i+1}">${i+1}</td>
      <td class="agent-name">${id}</td>
      <td><span class="strategy-badge ${STRATEGY_CLASS[a.strategy] || ''}">${a.strategy}</span></td>
      <td style="color:#8892b0;font-size:12px">${a.model_id}</td>
      <td style="text-align:right;font-weight:700">${fmt(a.total_value)}</td>
      <td style="text-align:right" class="${pctClass(roi)}">${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%</td>
      <td style="text-align:right" class="${pctClass(a.realized_pnl)}">${fmt(a.realized_pnl)}</td>
      <td style="text-align:right">${a.trade_count}</td>
      <td style="text-align:right">${winPct}%</td>
      <td style="text-align:right;color:${memColor}">${mem}</td>
      <td style="text-align:right;color:#ff4757">${(a.max_drawdown*100).toFixed(1)}%</td>
    </tr>`;
  }).join('');

  // Agent cards
  const cards = document.getElementById('agent-cards');
  cards.innerHTML = sorted.map(([id, a]) => {
    const roi = ((a.total_value - 200) / 200 * 100);
    const posCount = Object.keys(a.positions).length;
    const borderColor = COLORS[id] || '#334155';
    return `<div class="agent-card" style="border-left: 3px solid ${borderColor}">
      <div class="agent-card-header">
        <span class="agent-card-name">${id}</span>
        <span class="strategy-badge ${STRATEGY_CLASS[a.strategy] || ''}">${a.strategy}</span>
      </div>
      <div class="stat-row"><span class="stat-label">Total Value</span><span class="stat-value">${fmt(a.total_value)}</span></div>
      <div class="stat-row"><span class="stat-label">Cash</span><span class="stat-value">${fmt(a.cash)}</span></div>
      <div class="stat-row"><span class="stat-label">ROI</span><span class="stat-value ${pctClass(roi)}">${roi >= 0?'+':''}${roi.toFixed(2)}%</span></div>
      <div class="stat-row"><span class="stat-label">Positions</span><span class="stat-value">${posCount}</span></div>
      <div class="stat-row"><span class="stat-label">Trades</span><span class="stat-value">${a.trade_count}</span></div>
      <div class="stat-row"><span class="stat-label">Realized P&L</span><span class="stat-value ${pctClass(a.realized_pnl)}">${fmt(a.realized_pnl)}</span></div>
      <div class="stat-row"><span class="stat-label">Memory Lookups</span><span class="stat-value" style="color:${(a.memory_calls||0) > 0 ? '#a78bfa' : '#64748b'}">${a.memory_calls||0}</span></div>
    </div>`;
  }).join('');

  // Positions
  const pos = document.getElementById('positions-body');
  let posRows = '';
  for (const [id, a] of sorted) {
    for (const [ticker, p] of Object.entries(a.positions)) {
      posRows += `<tr>
        <td style="color:${COLORS[id]||'#e0e6f0'}">${id}</td>
        <td>${ticker}</td>
        <td style="text-align:right">${p.qty}</td>
        <td style="text-align:right">${fmt(p.cost_basis)}</td>
      </tr>`;
    }
  }
  pos.innerHTML = posRows || '<tr><td colspan="4" style="color:#64748b;text-align:center">No open positions</td></tr>';

  // Trade log
  const tb = document.getElementById('trades-body');
  tb.innerHTML = data.trades.map(t => `<tr>
    <td style="color:#64748b">${t.ts}</td>
    <td style="color:${COLORS[t.agent]||'#e0e6f0'}">${t.agent}</td>
    <td><span class="${t.action === 'buy' ? 'buy-badge' : 'sell-badge'}">${t.action.toUpperCase()}</span></td>
    <td>${t.product}</td>
    <td style="text-align:right">${t.qty}</td>
    <td style="text-align:right">${fmt(t.price)}</td>
    <td style="text-align:right">${fmt(t.total)}</td>
    <td style="text-align:right">${fmt(t.fee)}</td>
  </tr>`).join('');

  // Mode badge
  const modeBadge = document.getElementById('mode-badge');
  if (data.mode === 'v2') {
    modeBadge.textContent = 'V2 A/B Test';
    modeBadge.style.background = '#1a0d2e';
    modeBadge.style.borderColor = '#c084fc';
    modeBadge.style.color = '#c084fc';
  } else if (data.mode === 'llm') {
    modeBadge.textContent = 'LLM Competition';
    modeBadge.style.background = '#1e1535';
    modeBadge.style.borderColor = '#a78bfa';
    modeBadge.style.color = '#a78bfa';
  } else if (data.mode === 'strategy') {
    modeBadge.textContent = 'Strategy Competition';
    modeBadge.style.background = '#0d2137';
    modeBadge.style.borderColor = '#60a5fa';
    modeBadge.style.color = '#60a5fa';
  } else {
    modeBadge.textContent = data.mode_label || 'Arena';
  }

  // Dynamic guide section
  const guide = document.getElementById('guide-section');
  if (data.mode === 'v2') {
    // V2 A/B mode: show comparison cards for each model
    const models = [...new Set(Object.values(data.agents).map(a => a.model_id))];
    guide.innerHTML = `
      <div class="strat-card" style="border-top-color:#c084fc;grid-column:1/-1">
        <h3>V2 A/B Test: Dumb vs Smart</h3>
        <div class="strat-subtitle" style="color:#c084fc">Each model runs in both V1 (raw candles) and V2 (signal briefs + plans + risk engine)</div>
        <ul>
          <li><strong style="color:#64748b">V1 Dumb:</strong> Raw candle data, basic tools (execute_trade, get_portfolio)</li>
          <li><strong style="color:#c084fc">V2 Smart:</strong> Pre-computed indicators (RSI, MACD, Bollinger, regime), trading plans with auto stop-loss/TP, persistent learnings</li>
          <li>Same model, same market data &mdash; only the intelligence layer differs</li>
          <li>Risk engine automatically exits at stop-loss/take-profit for V2 agents</li>
        </ul>
      </div>
    ` + models.map(mid => {
      const info = LLM_GUIDE[mid] || {color:'#8892b0', title:mid, provider:'Unknown', desc:''};
      // Find v1 and v2 agents for this model
      const v1 = Object.entries(data.agents).find(([id,a]) => a.model_id === mid && a.strategy === 'neutral');
      const v2 = Object.entries(data.agents).find(([id,a]) => a.model_id === mid && a.strategy === 'smart');
      const v1val = v1 ? v1[1].total_value : 200;
      const v2val = v2 ? v2[1].total_value : 200;
      const v1roi = ((v1val - 200) / 200 * 100).toFixed(2);
      const v2roi = ((v2val - 200) / 200 * 100).toFixed(2);
      const winner = v2val > v1val ? 'V2 Smart' : v1val > v2val ? 'V1 Dumb' : 'Tied';
      const winColor = v2val > v1val ? '#c084fc' : v1val > v2val ? '#64748b' : '#fbbf24';
      return `<div class="strat-card" style="border-top-color:${info.color}">
        <h3>${info.title}</h3>
        <div class="strat-subtitle" style="color:${info.color}">${info.provider}</div>
        <ul>
          <li>V1 Dumb: <strong class="${v1roi >= 0 ? 'positive' : 'negative'}">${v1roi >= 0 ? '+' : ''}${v1roi}%</strong> ROI</li>
          <li>V2 Smart: <strong class="${v2roi >= 0 ? 'positive' : 'negative'}">${v2roi >= 0 ? '+' : ''}${v2roi}%</strong> ROI</li>
        </ul>
        <div class="strat-note" style="color:${winColor}">Leading: ${winner}</div>
      </div>`;
    }).join('');
  } else if (data.mode === 'llm') {
    // LLM mode: show model cards
    const models = [...new Set(Object.values(data.agents).map(a => a.model_id))];
    guide.innerHTML = models.map(mid => {
      const info = LLM_GUIDE[mid] || {color:'#8892b0', title:mid, provider:'Unknown', desc:''};
      return `<div class="strat-card" style="border-top-color:${info.color}">
        <h3>${info.title}</h3>
        <div class="strat-subtitle" style="color:${info.color}">${info.provider}</div>
        <ul><li style="color:#a0aec0">${info.desc}</li>
        <li>Same neutral strategy as all competitors</li>
        <li>No personality bias &mdash; pure reasoning ability</li>
        <li>Free to trade however it sees fit</li></ul>
        <div class="strat-note">Model ID: ${mid}</div>
      </div>`;
    }).join('');
  } else {
    // Strategy mode: show strategy cards
    const strats = [...new Set(Object.values(data.agents).map(a => a.strategy))];
    guide.innerHTML = strats.map(s => {
      const info = STRATEGY_GUIDE[s];
      if (!info) return '';
      return `<div class="strat-card ${info.cls}">
        <h3>${info.title}</h3>
        <div class="strat-subtitle">${info.sub}</div>
        <ul>${info.points.map(p => `<li>${p}</li>`).join('')}</ul>
        <div class="strat-note">${info.note}</div>
      </div>`;
    }).join('');
  }

  // LLM Spend ticker (live from OpenRouter)
  if (data.llm_spend) {
    const s = data.llm_spend;
    document.getElementById('spend-total').textContent = '$' + s.total.toFixed(4);
    document.getElementById('spend-rate').textContent = '$' + s.per_hour.toFixed(2) + '/hr';
    document.getElementById('spend-remaining').textContent = s.credit_remaining.toFixed(2);
    document.getElementById('spend-model').textContent = s.model + ' via OpenRouter';
  }

  // Chart
  if (chart && data.balance_series) {
    const datasets = Object.entries(data.balance_series).map(([id, values]) => ({
      label: id,
      data: values,
      borderColor: COLORS[id] || '#8892b0',
      backgroundColor: (COLORS[id] || '#8892b0') + '20',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: true,
    }));
    const maxLen = Math.max(...datasets.map(d => d.data.length));
    chart.data.labels = Array.from({length: maxLen}, (_, i) => i);
    chart.data.datasets = datasets;
    chart.update('none');
  }
}

initChart();

// WebSocket connection with auto-reconnect
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 2000);
  ws.onerror = () => ws.close();
}
connect();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = query_db()
            await ws.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    _load_model_prices()
    uvicorn.run(app, host="0.0.0.0", port=8050)
