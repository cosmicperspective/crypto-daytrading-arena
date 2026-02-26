"""Comprehensive V2 integration validation script."""

import json
import sys
import time

errors = []
warnings = []

def check(label, condition, detail=""):
    if condition:
        print(f"  [PASS] {label}")
    else:
        msg = f"{label}: {detail}" if detail else label
        errors.append(msg)
        print(f"  [FAIL] {label} — {detail}")

def warn(label, detail=""):
    msg = f"{label}: {detail}" if detail else label
    warnings.append(msg)
    print(f"  [WARN] {label} — {detail}")


# ── 1. Module Imports ─────────────────────────────────────────────
print("\n=== 1. Module Imports ===")

try:
    from market_intel import (
        MarketIntelService, MarketRegime, Indicators,
        compute_rsi, compute_macd, compute_bollinger,
        compute_atr, compute_adx, detect_regime,
        _ema, _sma, _find_support_resistance, _generate_signal,
        INDICATOR_TIMEFRAMES,
    )
    check("market_intel imports", True)
except Exception as e:
    check("market_intel imports", False, str(e))

try:
    from risk_engine import RiskEngine, TradingPlan, PlanStatus, new_plan_id
    check("risk_engine imports", True)
except Exception as e:
    check("risk_engine imports", False, str(e))

try:
    from coinbase_consumer import CandleBook, PriceBook, Candle
    check("coinbase_consumer imports", True)
except Exception as e:
    check("coinbase_consumer imports", False, str(e))

try:
    from persistence import SQLiteStore
    check("persistence imports", True)
except Exception as e:
    check("persistence imports", False, str(e))

try:
    from trading_tools import (
        get_market_intel, create_trading_plan, modify_plan,
        get_agent_state, record_learning, init_v2_services,
    )
    check("trading_tools V2 imports", True)
except Exception as e:
    check("trading_tools V2 imports", False, str(e))

# ── 2. JSON Config Validation ─────────────────────────────────────
print("\n=== 2. JSON Config ===")

try:
    with open("arena_v2.json") as f:
        v2_config = json.load(f)
    agents = v2_config.get("agents", [])
    check("arena_v2.json loads", True)
    check("10 agents configured", len(agents) == 10, f"got {len(agents)}")

    v1_agents = [a for a in agents if a.get("agent_mode") == "v1"]
    v2_agents = [a for a in agents if a.get("agent_mode") == "v2"]
    check("5 V1 agents", len(v1_agents) == 5, f"got {len(v1_agents)}")
    check("5 V2 agents", len(v2_agents) == 5, f"got {len(v2_agents)}")

    # V1 should use "neutral" strategy, V2 should use "smart"
    v1_strategies = set(a["strategy"] for a in v1_agents)
    v2_strategies = set(a["strategy"] for a in v2_agents)
    check("V1 agents use neutral strategy", v1_strategies == {"neutral"}, str(v1_strategies))
    check("V2 agents use smart strategy", v2_strategies == {"smart"}, str(v2_strategies))

    # Each model should have exactly one V1 and one V2 agent
    v1_models = sorted(a["chat_node_name"] for a in v1_agents)
    v2_models = sorted(a["chat_node_name"] for a in v2_agents)
    check("V1 and V2 cover same models", v1_models == v2_models, f"v1={v1_models}, v2={v2_models}")

except Exception as e:
    check("arena_v2.json validation", False, str(e))

try:
    with open("models_llm.json") as f:
        models_config = json.load(f)
    models = models_config.get("models", [])
    check("models_llm.json loads", True)
    check("5 models configured", len(models) == 5, f"got {len(models)}")
except Exception as e:
    check("models_llm.json validation", False, str(e))


# ── 3. Indicator Accuracy ─────────────────────────────────────────
print("\n=== 3. Indicator Accuracy ===")

# RSI
rsi_gains = compute_rsi(list(range(1, 20)), 14)
check("RSI all gains = 100", rsi_gains == 100.0, f"got {rsi_gains}")

rsi_losses = compute_rsi(list(range(20, 1, -1)), 14)
check("RSI all losses ~0", rsi_losses is not None and rsi_losses < 1.0, f"got {rsi_losses}")

rsi_short = compute_rsi([1, 2, 3])
check("RSI insufficient data = None", rsi_short is None)

# MACD
m, s, h = compute_macd([10 + i * 0.5 for i in range(50)])
check("MACD uptrend positive", m is not None and m > 0, f"m={m}")

m2, s2, h2 = compute_macd(list(range(5)))
check("MACD insufficient data = None", m2 is None)

# Bollinger
u, mid, l = compute_bollinger([100.0] * 25)
check("Bollinger constant: upper=mid=lower", u == mid == l == 100.0, f"u={u}, m={mid}, l={l}")

u2, m2, l2 = compute_bollinger([10 + i * 0.1 + (i % 3) * 0.5 for i in range(25)])
check("Bollinger: upper > mid > lower", u2 > m2 > l2, f"u={u2}, m={m2}, l={l2}")

# ATR
highs = [11.0 + i * 0.1 for i in range(20)]
lows = [9.0 + i * 0.1 for i in range(20)]
closes = [10.0 + i * 0.1 for i in range(20)]
atr = compute_atr(highs, lows, closes)
check("ATR positive", atr is not None and atr > 0, f"atr={atr}")

# ADX (needs 29+ data points for period=14)
highs40 = [10 + i * 0.5 for i in range(40)]
lows40 = [9 + i * 0.5 for i in range(40)]
closes40 = [9.5 + i * 0.5 for i in range(40)]
adx, plus_di, minus_di = compute_adx(highs40, lows40, closes40)
check("ADX trending > 0", adx is not None and adx > 0, f"adx={adx}")


# ── 4. Regime Detection ──────────────────────────────────────────
print("\n=== 4. Regime Detection ===")

# Trending up
closes_up = [10 + i * 0.3 for i in range(50)]
ema21_up = sum(closes_up[-21:]) / 21
ema50_up = sum(closes_up[-50:]) / 50
regime, _ = detect_regime(closes_up, ema21_up, ema50_up, adx=35, atr=0.2,
                           bb_upper=closes_up[-1]+1, bb_lower=closes_up[-1]-1, volume_ratio=1.0)
check("Regime TRENDING_UP", regime == MarketRegime.TRENDING_UP, f"got {regime}")

# Trending down
closes_down = [20 - i * 0.3 for i in range(50)]
ema21_down = sum(closes_down[-21:]) / 21
ema50_down = sum(closes_down[-50:]) / 50
regime, _ = detect_regime(closes_down, ema21_down, ema50_down, adx=30, atr=0.2,
                           bb_upper=closes_down[-1]+1, bb_lower=closes_down[-1]-1, volume_ratio=1.0)
check("Regime TRENDING_DOWN", regime == MarketRegime.TRENDING_DOWN, f"got {regime}")

# Ranging
regime, _ = detect_regime([10.0]*20, 10.0, 10.0, adx=15, atr=0.1,
                           bb_upper=10.05, bb_lower=9.95, volume_ratio=1.0)
check("Regime RANGING", regime == MarketRegime.RANGING, f"got {regime}")

# Capitulation
regime, _ = detect_regime([10, 10, 10, 10, 6], 10.0, None, None, None, None, None, 3.0)
check("Regime CAPITULATION", regime == MarketRegime.CAPITULATION, f"got {regime}")


# ── 5. Signal Brief Generation ────────────────────────────────────
print("\n=== 5. Signal Brief ===")

from datetime import datetime, timezone

cb = CandleBook()
pb = PriceBook()
closes_sig = [0.10 + i * 0.001 for i in range(48)]
candles_raw = [
    [int(time.time()) - (48 - i) * 300, c * 0.995, c * 1.005, c, c, 1000.0]
    for i, c in enumerate(closes_sig)
]
candles_raw.reverse()
cb.update_from_api("DOGE-USD", 300, candles_raw)

pb.update({
    "product_id": "DOGE-USD", "price": "0.148",
    "best_bid": "0.1479", "best_bid_size": "1000",
    "best_ask": "0.1481", "best_ask_size": "1000",
    "side": "buy", "last_size": "100",
    "volume_24h": "5000000", "time": "",
})

svc = MarketIntelService(cb, pb)
svc.update("DOGE-USD")
ind = svc.get_indicators("DOGE-USD")
check("Indicators computed", ind is not None)
if ind:
    check("RSI computed", ind.rsi is not None, f"rsi={ind.rsi}")
    check("MACD computed", ind.macd_line is not None, f"macd={ind.macd_line}")
    check("Regime not UNKNOWN", ind.regime != MarketRegime.UNKNOWN, f"regime={ind.regime}")

brief = svc.format_brief("DOGE-USD")
check("Brief generated", brief is not None)
if brief:
    check("Brief has Signal Brief", "Signal Brief" in brief)
    check("Brief has Regime", "Regime" in brief)
    check("Brief has RSI", "RSI" in brief)
    check("Brief < 300 tokens (~1200 chars)", len(brief) < 1200, f"len={len(brief)}")


# ── 6. Risk Engine E2E ────────────────────────────────────────────
print("\n=== 6. Risk Engine E2E ===")

sell_log = []

def mock_sell(agent_id, product_id, quantity, side):
    sell_log.append((agent_id, product_id, quantity, side))
    return f"Sold {quantity} {product_id}"

# Take profit
pb_tp = PriceBook()
pb_tp.update({
    "product_id": "DOGE-USD", "price": "0.13",
    "best_bid": "0.13", "best_bid_size": "1000",
    "best_ask": "0.13", "best_ask_size": "1000",
    "side": "buy", "last_size": "100",
    "volume_24h": "5000000", "time": "",
})

engine = RiskEngine(pb_tp, mock_sell)
plan = TradingPlan(
    plan_id=new_plan_id(), agent_id="test", product_id="DOGE-USD",
    direction="long", quantity=1000.0, entry_price=0.10,
    stop_loss_price=0.09, take_profit_price=0.12,
    time_stop_minutes=30, thesis="test thesis", confidence=0.7,
)
engine.register_plan(plan)
triggered = engine.check_plans()
check("Take profit triggered", len(triggered) == 1 and triggered[0].status == PlanStatus.TAKE_PROFIT,
      f"triggered={len(triggered)}")
check("PnL positive on TP", triggered[0].pnl > 0, f"pnl={triggered[0].pnl}")
check("Sell executed on TP", len(sell_log) == 1, f"sell_log={sell_log}")

# Stop loss
sell_log.clear()
pb_sl = PriceBook()
pb_sl.update({
    "product_id": "DOGE-USD", "price": "0.085",
    "best_bid": "0.085", "best_bid_size": "1000",
    "best_ask": "0.085", "best_ask_size": "1000",
    "side": "buy", "last_size": "100",
    "volume_24h": "5000000", "time": "",
})

engine2 = RiskEngine(pb_sl, mock_sell)
plan2 = TradingPlan(
    plan_id=new_plan_id(), agent_id="test", product_id="DOGE-USD",
    direction="long", quantity=1000.0, entry_price=0.10,
    stop_loss_price=0.09, take_profit_price=0.12,
    time_stop_minutes=30, thesis="test stop loss", confidence=0.7,
)
engine2.register_plan(plan2)
triggered2 = engine2.check_plans()
check("Stop loss triggered", len(triggered2) == 1 and triggered2[0].status == PlanStatus.STOPPED_OUT,
      f"triggered={len(triggered2)}")
check("PnL negative on SL", triggered2[0].pnl < 0, f"pnl={triggered2[0].pnl}")

# Cooldown
sell_log.clear()
pb_cd = PriceBook()
pb_cd.update({
    "product_id": "DOGE-USD", "price": "0.08",
    "best_bid": "0.08", "best_bid_size": "1000",
    "best_ask": "0.08", "best_ask_size": "1000",
    "side": "buy", "last_size": "100",
    "volume_24h": "5000000", "time": "",
})

engine3 = RiskEngine(pb_cd, mock_sell, cooldown_after_losses=2, cooldown_minutes=5)
for _ in range(2):
    p = TradingPlan(
        plan_id=new_plan_id(), agent_id="test-cd", product_id="DOGE-USD",
        direction="long", quantity=1000.0, entry_price=0.10,
        stop_loss_price=0.09, take_profit_price=0.12,
        time_stop_minutes=30, thesis="cooldown test", confidence=0.7,
    )
    engine3.register_plan(p)
    engine3.check_plans()

on_cd, msg = engine3.is_agent_on_cooldown("test-cd")
check("Cooldown activated after 2 losses", on_cd, f"msg={msg}")


# ── 7. Persistence Schema ────────────────────────────────────────
print("\n=== 7. Persistence ===")

import os
test_db_path = "test_validate_v2.db"
if os.path.exists(test_db_path):
    os.remove(test_db_path)

db = SQLiteStore(test_db_path)

# Check all tables exist
import sqlite3
conn = sqlite3.connect(test_db_path)
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = sorted(row[0] for row in cursor.fetchall())
conn.close()

# V2-critical tables
for t in ["trading_plans", "learnings", "trades"]:
    check(f"Table '{t}' exists", t in tables, f"tables={tables}")

# Test learnings persistence
db.save_learning("test-agent", "Buy the dip on RSI < 30")
db.save_learning("test-agent", "Avoid trading during low volume")
learnings = db.load_learnings("test-agent")
check("Learnings saved and loaded", len(learnings) == 2, f"got {len(learnings)}")

# Cleanup (close connection first for Windows)
del db
try:
    os.remove(test_db_path)
except OSError:
    pass  # Windows file lock — harmless


# ── 8. Tool Registration Check ───────────────────────────────────
print("\n=== 8. Tool Registration ===")

try:
    from deploy_router_node import V1_TOOLS, V2_TOOLS
    check("V1_TOOLS has 5 tools", len(V1_TOOLS) == 5, f"got {len(V1_TOOLS)}")
    check("V2_TOOLS has 10 tools", len(V2_TOOLS) == 10, f"got {len(V2_TOOLS)}")

    # Tools are calfkit @agent_tool objects — use name attr or str repr
    def tool_name(t):
        return getattr(t, "__name__", None) or getattr(t, "name", None) or str(t)
    v2_exclusive = set(tool_name(t) for t in V2_TOOLS) - set(tool_name(t) for t in V1_TOOLS)
    check("5 V2-exclusive tools", len(v2_exclusive) == 5, f"got {len(v2_exclusive)}: {v2_exclusive}")
except Exception as e:
    check("Tool registration", False, str(e))


# ── 9. Run Arena Config ──────────────────────────────────────────
print("\n=== 9. Run Arena Config ===")

try:
    from run_arena import MODE_PRESETS, AgentSpec
    check("v2 mode preset exists", "v2" in MODE_PRESETS)
    preset = MODE_PRESETS["v2"]
    check("v2 uses models_llm.json", preset["models_file"] == "models_llm.json")
    check("v2 uses arena_v2.json", preset["tests_file"] == "arena_v2.json")

    # Check AgentSpec has agent_mode
    spec = AgentSpec(agent_name="test", chat_node_name="test", strategy="smart",
                     provider="openrouter", model_id="test", agent_mode="v2")
    check("AgentSpec supports agent_mode", spec.agent_mode == "v2")
except Exception as e:
    check("run_arena config", False, str(e))


# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
total = len(errors) + sum(1 for line in [] if "[PASS]" in line)
if errors:
    print(f"  FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED — V2 system validated!")
    if warnings:
        print(f"  ({len(warnings)} warning(s))")
        for w in warnings:
            print(f"    - {w}")
    print("  Ready to launch: uv run python run_arena.py --mode v2 --start all --use-uv --reset-db")
    sys.exit(0)
