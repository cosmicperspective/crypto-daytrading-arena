"""Comprehensive validation tests — logic, data, math, algorithms, strategy.

Tests every layer of the Darwin trading system:
1. Market data pipeline & symbol mapping
2. Technical indicator math correctness
3. Risk engine logic (long + short, stops, PnL)
4. Trading tools accounting (buy, sell, short, cover, portfolio)
5. Strategy prompt completeness
6. Cost tracking (OpenRouter live)
7. End-to-end trading plan lifecycle (long & short)
8. Sim config / fee math
9. Persistence round-trip
"""

import math
import os
import time

import pytest

from coinbase_consumer import PriceBook, CandleBook, Candle
from sim_config import SimConfig, load_sim_config
from trading_tools import AccountStore, AgentAccount, INITIAL_CASH
from risk_engine import RiskEngine, TradingPlan, PlanStatus, new_plan_id
from market_intel import (
    _ema, _sma, compute_rsi, compute_macd,
    compute_bollinger, compute_atr, compute_adx,
    detect_regime, _generate_signal, _find_support_resistance,
    MarketIntelService, MarketRegime, Indicators,
)


# ── Helpers ──────────────────────────────────────────────────────

def _pb(**prices: str) -> PriceBook:
    pb = PriceBook()
    for pid, price in prices.items():
        p = str(price)
        pb.update({
            "product_id": pid, "price": p,
            "best_bid": str(float(p) * 0.999),
            "best_bid_size": "1000",
            "best_ask": str(float(p) * 1.001),
            "best_ask_size": "1000",
            "side": "buy", "last_size": "100",
            "volume_24h": "5000000", "time": "",
        })
    return pb


def _paper_sim() -> SimConfig:
    return SimConfig(
        preset="paper",
        taker_fee_bps=0.0, maker_fee_bps=0.0,
        slippage_bps=0.0, impact_bps=0.0,
        simulate_latency_ms=0, partial_fill_prob=0.0,
        partial_fill_min_ratio=1.0, partial_fill_max_ratio=1.0,
        funding_bps_per_hour=0.0, borrow_bps_per_hour=0.0,
        min_trade_interval_s=0.0, max_order_usd=None,
        max_position_usd=None, max_leverage=1.0,
        allow_negative_cash=False,
    )


def _realistic_sim() -> SimConfig:
    return SimConfig(
        preset="realistic",
        taker_fee_bps=6.0, maker_fee_bps=4.0,
        slippage_bps=5.0, impact_bps=15.0,
        simulate_latency_ms=250, partial_fill_prob=0.0,
        partial_fill_min_ratio=1.0, partial_fill_max_ratio=1.0,
        funding_bps_per_hour=0.0, borrow_bps_per_hour=0.0,
        min_trade_interval_s=0.0, max_order_usd=None,
        max_position_usd=None, max_leverage=1.0,
        allow_negative_cash=False,
    )


def _make_plan(direction="long", entry=0.10, sl=0.09, tp=0.12, qty=1000.0):
    return TradingPlan(
        plan_id=new_plan_id(), agent_id="test",
        product_id="DOGE-USD", direction=direction,
        quantity=qty, entry_price=entry,
        stop_loss_price=sl, take_profit_price=tp,
        time_stop_minutes=30, thesis="test", confidence=0.7,
    )


# ══════════════════════════════════════════════════════════════════
# 1. MARKET DATA PIPELINE
# ══════════════════════════════════════════════════════════════════


class TestMarketDataPipeline:
    def test_price_book_update_and_read(self):
        pb = _pb(**{"DOGE-USD": "0.10", "SOL-USD": "150.00"})
        e = pb.get("DOGE-USD")
        assert e is not None
        assert float(e["price"]) == 0.10
        assert float(e["best_bid"]) < float(e["best_ask"])  # spread exists

    def test_candle_book_stores_ohlcv(self):
        cb = CandleBook()
        raw = [
            [1700000000, 95.0, 98.0, 96.0, 97.0, 50000.0],
            [1700000300, 96.0, 99.0, 97.0, 98.0, 60000.0],
        ]
        raw.sort(key=lambda x: x[0], reverse=True)
        cb.update_from_api("SOL-USD", 300, raw)
        candles = cb._candles.get(("SOL-USD", 300), [])
        assert len(candles) == 2
        assert candles[0].time.timestamp() < candles[1].time.timestamp()

    def test_kraken_symbol_mapping(self):
        from kraken_connector import KRAKEN_TO_INTERNAL, INTERNAL_TO_KRAKEN_REST, DEFAULT_PRODUCTS
        for p in DEFAULT_PRODUCTS:
            assert p in KRAKEN_TO_INTERNAL
            internal = KRAKEN_TO_INTERNAL[p]
            assert "-" in internal
            assert internal in INTERNAL_TO_KRAKEN_REST

    def test_coinbase_default_products(self):
        from coinbase_kafka_connector import DEFAULT_PRODUCTS
        for p in DEFAULT_PRODUCTS:
            assert "-USD" in p
        assert "DOGE-USD" in DEFAULT_PRODUCTS
        assert "SOL-USD" in DEFAULT_PRODUCTS


# ══════════════════════════════════════════════════════════════════
# 2. TECHNICAL INDICATOR MATH
# ══════════════════════════════════════════════════════════════════


class TestIndicatorMath:
    def test_ema_convergence(self):
        """EMA should converge toward constant input."""
        data = [100.0] * 50
        ema = _ema(data, 20)
        assert abs(ema[-1] - 100.0) < 0.001

    def test_sma_exact(self):
        """SMA of [1,2,3,4,5] period 5 = 3.0."""
        result = _sma([1, 2, 3, 4, 5], 5)
        assert result[-1] == pytest.approx(3.0)

    def test_rsi_bounds(self):
        """RSI must be between 0 and 100."""
        data = [100 + i * (-1)**i for i in range(30)]
        rsi = compute_rsi(data, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_pure_up(self):
        """All gains → RSI near 100."""
        data = list(range(1, 20))
        rsi = compute_rsi(data, 14)
        assert rsi is not None
        assert rsi > 95

    def test_rsi_pure_down(self):
        """All losses → RSI near 0."""
        data = list(range(20, 1, -1))
        rsi = compute_rsi(data, 14)
        assert rsi is not None
        assert rsi < 5

    def test_macd_crossover_detection(self):
        """Uptrending data should produce non-negative MACD."""
        data = [50 + i * 0.5 for i in range(40)]
        result = compute_macd(data)
        assert result is not None
        macd, signal, hist = result
        assert hist >= -0.001  # approximately bullish (floating point)

    def test_bollinger_band_width(self):
        """Constant price → zero bandwidth."""
        data = [100.0] * 25
        result = compute_bollinger(data, 20, 2)
        assert result is not None
        upper, mid, lower = result
        assert upper == pytest.approx(mid)
        assert lower == pytest.approx(mid)

    def test_bollinger_volatile_data(self):
        """Volatile data → upper > mid > lower."""
        data = [100 + 10 * ((-1)**i) for i in range(25)]
        result = compute_bollinger(data, 20, 2)
        assert result is not None
        upper, mid, lower = result
        assert upper > mid > lower

    def test_atr_positive(self):
        """ATR is always positive for any data with movement."""
        closes = [100 + i for i in range(20)]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        atr = compute_atr(highs, lows, closes, 14)
        assert atr is not None
        assert atr > 0

    def test_adx_trending_market(self):
        """Strong trend → ADX > 25."""
        closes = [100 + i * 2 for i in range(40)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        result = compute_adx(highs, lows, closes, 14)
        assert result is not None
        adx_val = result[0] if isinstance(result, tuple) else result
        assert adx_val > 25

    def test_support_resistance_levels(self):
        """Should find support below resistance."""
        closes = [100 + 5 * math.sin(i / 3) for i in range(50)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        support, resistance = _find_support_resistance(highs, lows, closes)
        # Returns single float values (strongest level)
        assert support is not None
        assert resistance is not None
        assert support < resistance


# ══════════════════════════════════════════════════════════════════
# 3. REGIME DETECTION & SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════


class TestRegimeAndSignal:
    def test_regime_trending_up(self):
        closes = [100 + i * 2 for i in range(40)]
        ema_21 = _ema(closes, 21)[-1]
        ema_50 = _ema(closes, 21)[-1] * 0.95  # simulate slower EMA below price
        adx = 35.0
        regime, _ = detect_regime(closes, ema_21, ema_50, adx, 2.0, 110.0, 90.0, 1.0)
        assert regime in (MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT)

    def test_regime_trending_down(self):
        closes = [200 - i * 2 for i in range(40)]
        ema_21 = _ema(closes, 21)[-1]
        ema_50 = _ema(closes, 21)[-1] * 1.05
        adx = 35.0
        regime, _ = detect_regime(closes, ema_21, ema_50, adx, 2.0, 200.0, 100.0, 1.0)
        assert regime in (MarketRegime.TRENDING_DOWN, MarketRegime.CAPITULATION)

    def test_regime_ranging(self):
        """Low ADX → RANGING."""
        closes = [100.0] * 40
        regime, _ = detect_regime(closes, 100.0, 100.0, 15.0, 1.0, 101.0, 99.0, 1.0)
        assert regime == MarketRegime.RANGING

    def test_signal_buy_conditions(self):
        """RSI < 40, MACD bullish → BUY."""
        ind = Indicators(
            product_id="TEST", timestamp=time.time(), price=95.0,
            rsi=35, macd_line=0.5, macd_signal=0.3, macd_histogram=0.2,
            bb_upper=110, bb_middle=100, bb_lower=90, bb_position="lower",
            ema_9=101, ema_21=100, ema_50=99,
            atr=2.0, adx=30, volume_ratio=1.5,
            support=95, resistance=110,
            regime=MarketRegime.TRENDING_UP, regime_strength="moderate",
        )
        sig = _generate_signal(ind)
        assert "BUY" in sig.upper()

    def test_signal_sell_conditions(self):
        """RSI > 70, MACD bearish → SELL."""
        ind = Indicators(
            product_id="TEST", timestamp=time.time(), price=108.0,
            rsi=75, macd_line=-0.5, macd_signal=-0.3, macd_histogram=-0.2,
            bb_upper=110, bb_middle=100, bb_lower=90, bb_position="upper",
            ema_9=99, ema_21=100, ema_50=101,
            atr=2.0, adx=30, volume_ratio=1.0,
            support=90, resistance=110,
            regime=MarketRegime.TRENDING_DOWN, regime_strength="moderate",
        )
        sig = _generate_signal(ind)
        assert "SELL" in sig.upper()

    def test_signal_hold_ranging(self):
        """Ranging market, neutral indicators → HOLD or weak signal."""
        ind = Indicators(
            product_id="TEST", timestamp=time.time(), price=100.0,
            rsi=50, macd_line=0.01, macd_signal=0.01, macd_histogram=0.0,
            bb_upper=101, bb_middle=100, bb_lower=99, bb_position="middle",
            ema_9=100, ema_21=100, ema_50=100,
            atr=1.0, adx=15, volume_ratio=1.0,
            support=98, resistance=102,
            regime=MarketRegime.RANGING, regime_strength="moderate",
        )
        sig = _generate_signal(ind)
        # Ranging + neutral indicators → HOLD or at most WEAK signal
        assert "HOLD" in sig.upper() or "WEAK" in sig.upper()


# ══════════════════════════════════════════════════════════════════
# 4. RISK ENGINE — LONG & SHORT MATH
# ══════════════════════════════════════════════════════════════════


class TestRiskEngineMath:
    def test_long_pnl_exact(self):
        """Long: buy at 0.10, price goes to 0.12 → PnL = 0.02 * 1000 = 20."""
        pb = _pb(**{"DOGE-USD": "0.12"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="long", entry=0.10, tp=0.11, qty=1000)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        # Exit at best_bid = 0.12 * 0.999 = 0.11988
        assert triggered[0].pnl == pytest.approx((0.12 * 0.999 - 0.10) * 1000, rel=0.01)

    def test_short_pnl_exact(self):
        """Short: sell at 0.10, price drops to 0.08 → PnL = (0.10 - 0.08) * 1000 = 20."""
        pb = _pb(**{"DOGE-USD": "0.08"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="short", entry=0.10, sl=0.11, tp=0.09, qty=1000)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        # Exit at best_ask = 0.08 * 1.001
        assert triggered[0].pnl == pytest.approx((0.10 - 0.08 * 1.001) * 1000, rel=0.01)

    def test_long_stop_triggers_at_exact_price(self):
        """Stop at 0.09, price exactly 0.09 → triggers."""
        pb = _pb(**{"DOGE-USD": "0.09"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="long", sl=0.09)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.STOPPED_OUT

    def test_short_stop_triggers_at_exact_price(self):
        """Short SL at 0.11, price exactly 0.11 → triggers."""
        pb = _pb(**{"DOGE-USD": "0.11"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="short", entry=0.10, sl=0.11, tp=0.08)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.STOPPED_OUT

    def test_time_stop_uses_minutes_correctly(self):
        """Time stop of 30 min, plan created 31 min ago → triggers."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="long")
        plan.created_at = time.time() - 31 * 60
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TIME_STOPPED

    def test_cooldown_math(self):
        """2 consecutive losses → cooldown_minutes * 60 seconds."""
        pb = _pb(**{"DOGE-USD": "0.05"})
        engine = RiskEngine(pb, lambda *a: "ok", cooldown_after_losses=2, cooldown_minutes=10)
        for _ in range(2):
            p = _make_plan(sl=0.09)
            engine.register_plan(p)
            engine.check_plans()
        on_cd, msg = engine.is_agent_on_cooldown("test")
        assert on_cd
        assert "10.0" in msg  # 10 min cooldown


# ══════════════════════════════════════════════════════════════════
# 5. TRADING TOOLS ACCOUNTING — PAPER & REALISTIC
# ══════════════════════════════════════════════════════════════════


class TestTradingAccounting:
    def test_initial_state(self):
        """Fresh account has correct starting cash and no positions."""
        s = AccountStore(_pb(**{"DOGE-USD": "0.10"}), _paper_sim())
        a = s.get_or_create("agent1")
        assert a.cash == INITIAL_CASH
        assert a.positions == {}
        assert a.realized_pnl == 0.0

    def test_buy_deducts_cash(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        r = s.execute_trade("a1", "DOGE-USD", 500, "buy")
        assert r.success
        a = s.get_or_create("a1")
        assert a.cash < INITIAL_CASH
        assert a.positions["DOGE-USD"] == 500

    def test_sell_adds_cash(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        s.execute_trade("a1", "DOGE-USD", 500, "buy")
        cash_after_buy = s.get_or_create("a1").cash
        s.execute_trade("a1", "DOGE-USD", 500, "sell")
        a = s.get_or_create("a1")
        assert a.cash > cash_after_buy
        assert "DOGE-USD" not in a.positions

    def test_short_sell_creates_negative_position(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        r = s.execute_trade("a1", "DOGE-USD", 500, "sell")
        assert r.success
        assert "Short sold" in r.message
        a = s.get_or_create("a1")
        assert a.positions["DOGE-USD"] == -500

    def test_cover_short_realizes_pnl(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        s.execute_trade("a1", "DOGE-USD", 500, "sell")  # short at ~0.10
        pb2 = _pb(**{"DOGE-USD": "0.08"})
        s._price_book = pb2
        r = s.execute_trade("a1", "DOGE-USD", 500, "buy")  # cover at ~0.08
        assert r.success
        assert "Covered" in r.message
        a = s.get_or_create("a1")
        assert a.realized_pnl > 0  # profit from price drop
        assert a.wins >= 1

    def test_portfolio_value_conservation(self):
        """Buy an asset → portfolio value ≈ initial (minus spread/fees)."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        a = s.get_or_create("a1")
        before = a.portfolio_value(pb)
        s.execute_trade("a1", "DOGE-USD", 500, "buy")
        after = a.portfolio_value(pb)
        # Small difference due to bid/ask spread
        assert abs(after - before) < 2.0

    def test_cannot_flip_long_to_short(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        s.execute_trade("a1", "DOGE-USD", 300, "buy")
        r = s.execute_trade("a1", "DOGE-USD", 500, "sell")
        assert not r.success
        assert "Close your long first" in r.message

    def test_cannot_flip_short_to_long(self):
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        s.execute_trade("a1", "DOGE-USD", 300, "sell")  # short 300
        r = s.execute_trade("a1", "DOGE-USD", 500, "buy")
        assert not r.success

    def test_realistic_fees_are_nonzero(self):
        """Realistic sim config should charge real fees."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _realistic_sim())
        s.execute_trade("a1", "DOGE-USD", 500, "buy")
        a = s.get_or_create("a1")
        assert a.fees_paid > 0

    def test_realistic_slippage(self):
        """Realistic config: total cost includes slippage + fees."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _realistic_sim())
        a = s.get_or_create("a1")
        cash_before = a.cash
        r_buy = s.execute_trade("a1", "DOGE-USD", 500, "buy")
        assert r_buy.success
        # Cost should be > 500 * 0.10 = $50 due to slippage + fees
        total_cost = cash_before - a.cash
        assert total_cost > 50.0
        assert a.fees_paid > 0

    def test_win_loss_counter(self):
        """Wins and losses are tracked correctly."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        s = AccountStore(pb, _paper_sim())
        s.execute_trade("a1", "DOGE-USD", 500, "buy")
        # Sell at profit
        pb2 = _pb(**{"DOGE-USD": "0.12"})
        s._price_book = pb2
        s.execute_trade("a1", "DOGE-USD", 500, "sell")
        a = s.get_or_create("a1")
        assert a.wins >= 1
        assert a.losses == 0


# ══════════════════════════════════════════════════════════════════
# 6. STRATEGY PROMPT COMPLETENESS
# ══════════════════════════════════════════════════════════════════


class TestStrategyPrompts:
    def test_all_strategies_exist(self):
        from deploy_router_node import STRATEGIES
        required = {"default", "momentum", "brainrot", "scalper", "neutral", "smart"}
        assert required.issubset(set(STRATEGIES.keys()))

    def test_smart_strategy_mentions_short(self):
        from deploy_router_node import STRATEGIES
        prompt = STRATEGIES["smart"]
        assert "short" in prompt.lower()
        assert "direction='short'" in prompt or "direction='long'" in prompt

    def test_smart_strategy_has_short_formulas(self):
        from deploy_router_node import STRATEGIES
        prompt = STRATEGIES["smart"]
        assert "entry + 1.5*ATR" in prompt  # short SL formula
        assert "entry - 3*ATR" in prompt    # short TP formula

    def test_neutral_mentions_short(self):
        from deploy_router_node import STRATEGIES
        assert "short" in STRATEGIES["neutral"].lower()

    def test_all_strategies_have_reasoning_addendum(self):
        from deploy_router_node import STRATEGIES
        for name, prompt in STRATEGIES.items():
            assert "Reasoning:" in prompt, f"{name} missing reasoning addendum"

    def test_v1_tools_differ_from_v2(self):
        from deploy_router_node import V1_TOOLS, V2_TOOLS
        assert len(V2_TOOLS) > len(V1_TOOLS)
        v2_names = {t.tool_schema.name for t in V2_TOOLS}
        assert "create_trading_plan" in v2_names
        assert "get_market_intel" in v2_names
        assert "modify_plan" in v2_names


# ══════════════════════════════════════════════════════════════════
# 7. COST TRACKING (OPENROUTER LIVE)
# ══════════════════════════════════════════════════════════════════


class TestCostTracking:
    def test_openrouter_key_set(self):
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("OPENROUTER_API_KEY", "")
        assert key.startswith("sk-or-v1-"), "OpenRouter key not set or invalid"

    def test_openrouter_spend_endpoint(self):
        from dotenv import load_dotenv
        import httpx
        load_dotenv()
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            pytest.skip("No OpenRouter key")
        r = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        data = r.json().get("data", {})
        assert "usage" in data, "No usage field in /auth/key response"
        assert "limit" in data
        assert "limit_remaining" in data
        assert data["limit_remaining"] > 0, "No credit remaining!"

    def test_openrouter_model_prices(self):
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/models", timeout=15)
        models = {m["id"]: m for m in r.json().get("data", [])}
        required = ["google/gemini-2.5-flash", "openai/gpt-5-nano", "anthropic/claude-haiku-4.5"]
        for mid in required:
            assert mid in models, f"Model {mid} not found on OpenRouter"
            p = models[mid].get("pricing", {})
            assert float(p.get("prompt", 0)) > 0, f"{mid} has no input price"
            assert float(p.get("completion", 0)) > 0, f"{mid} has no output price"

    def test_dashboard_uses_live_data(self):
        """Verify dashboard cost module returns real data, not estimates."""
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
        from web_dashboard import _poll_openrouter_spend, _load_model_prices, _or_model_prices
        spend = _poll_openrouter_spend()
        assert "total" in spend
        assert "remaining" in spend
        assert spend["remaining"] > 0
        _load_model_prices()
        assert len(_or_model_prices) > 10  # OpenRouter has many models


# ══════════════════════════════════════════════════════════════════
# 8. END-TO-END TRADING PLAN LIFECYCLE
# ══════════════════════════════════════════════════════════════════


class TestPlanLifecycleE2E:
    def test_long_plan_full_lifecycle(self):
        """Register → price rises → TP triggered → PnL positive → cooldown clear."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        sells = []
        engine = RiskEngine(pb, lambda *a: sells.append(a) or "ok")
        plan = _make_plan(direction="long", entry=0.10, sl=0.09, tp=0.11, qty=1000)
        engine.register_plan(plan)
        assert len(engine.get_active_plans()) == 1

        # Price rises past TP
        engine._price_book = _pb(**{"DOGE-USD": "0.12"})
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TAKE_PROFIT
        assert triggered[0].pnl > 0
        assert sells[-1][3] == "sell"  # close long = sell
        assert len(engine.get_active_plans()) == 0
        on_cd, _ = engine.is_agent_on_cooldown("test")
        assert not on_cd

    def test_short_plan_full_lifecycle(self):
        """Register → price drops → TP triggered → PnL positive → cooldown clear."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        sells = []
        engine = RiskEngine(pb, lambda *a: sells.append(a) or "ok")
        plan = _make_plan(direction="short", entry=0.10, sl=0.11, tp=0.09, qty=1000)
        engine.register_plan(plan)

        # Price drops past TP
        engine._price_book = _pb(**{"DOGE-USD": "0.08"})
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TAKE_PROFIT
        assert triggered[0].pnl > 0
        assert sells[-1][3] == "buy"  # close short = buy to cover

    def test_long_stop_loss_lifecycle(self):
        """Long plan → price drops → SL triggered → PnL negative → loss counted."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "ok", cooldown_after_losses=3)
        plan = _make_plan(direction="long", entry=0.10, sl=0.09, tp=0.12, qty=1000)
        engine.register_plan(plan)

        engine._price_book = _pb(**{"DOGE-USD": "0.085"})
        triggered = engine.check_plans()
        assert triggered[0].pnl < 0

    def test_short_stop_loss_lifecycle(self):
        """Short plan → price rises → SL triggered → PnL negative."""
        pb = _pb(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "ok")
        plan = _make_plan(direction="short", entry=0.10, sl=0.11, tp=0.08, qty=1000)
        engine.register_plan(plan)

        engine._price_book = _pb(**{"DOGE-USD": "0.115"})
        triggered = engine.check_plans()
        assert triggered[0].pnl < 0

    def test_multiple_plans_independent(self):
        """Two plans on different products trigger independently."""
        pb = _pb(**{"DOGE-USD": "0.10", "SOL-USD": "150.0"})
        engine = RiskEngine(pb, lambda *a: "ok")

        plan1 = TradingPlan(
            plan_id=new_plan_id(), agent_id="test", product_id="DOGE-USD",
            direction="long", quantity=1000, entry_price=0.10,
            stop_loss_price=0.09, take_profit_price=0.11,
            time_stop_minutes=30, thesis="doge long", confidence=0.7,
        )
        plan2 = TradingPlan(
            plan_id=new_plan_id(), agent_id="test", product_id="SOL-USD",
            direction="short", quantity=10, entry_price=150.0,
            stop_loss_price=155.0, take_profit_price=140.0,
            time_stop_minutes=30, thesis="sol short", confidence=0.6,
        )
        engine.register_plan(plan1)
        engine.register_plan(plan2)

        # DOGE hits TP, SOL doesn't move
        engine._price_book = _pb(**{"DOGE-USD": "0.12", "SOL-USD": "150.0"})
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].product_id == "DOGE-USD"
        assert len(engine.get_active_plans()) == 1  # SOL plan still active


# ══════════════════════════════════════════════════════════════════
# 9. PERSISTENCE ROUND-TRIP
# ══════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_save_and_load_account(self, tmp_path):
        from persistence import SQLiteStore
        db = SQLiteStore(str(tmp_path / "test.db"))
        acct = AgentAccount(cash=150.0, trade_count=5, realized_pnl=12.5, fees_paid=0.5, wins=3, losses=2)
        acct.positions["DOGE-USD"] = 500
        acct.cost_basis["DOGE-USD"] = 50.0
        db.save_account("agent1", acct)
        accounts = db.load_accounts(AgentAccount)
        assert "agent1" in accounts
        loaded = accounts["agent1"]
        assert loaded.cash == 150.0
        assert loaded.positions["DOGE-USD"] == 500
        assert loaded.cost_basis["DOGE-USD"] == 50.0
        assert loaded.realized_pnl == 12.5
        del db

    def test_save_and_load_short_position(self, tmp_path):
        """Persistence handles negative (short) positions."""
        from persistence import SQLiteStore
        db = SQLiteStore(str(tmp_path / "test.db"))
        acct = AgentAccount(cash=250.0, trade_count=1)
        acct.positions["DOGE-USD"] = -500
        acct.cost_basis["DOGE-USD"] = 50.0
        db.save_account("agent1", acct)
        accounts = db.load_accounts(AgentAccount)
        assert accounts["agent1"].positions["DOGE-USD"] == -500
        del db

    def test_save_plan(self, tmp_path):
        from persistence import SQLiteStore
        db = SQLiteStore(str(tmp_path / "test.db"))
        plan = _make_plan(direction="short", entry=0.10, sl=0.11, tp=0.08)
        db.save_plan(plan)
        # Verify it's in the DB
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        row = conn.execute("SELECT direction FROM trading_plans WHERE plan_id = ?", (plan.plan_id,)).fetchone()
        assert row is not None
        assert row[0] == "short"
        conn.close()
        del db


# ══════════════════════════════════════════════════════════════════
# 10. RUN ARENA CONFIGURATION
# ══════════════════════════════════════════════════════════════════


class TestArenaConfig:
    def test_mode_presets_exist(self):
        from run_arena import MODE_PRESETS
        assert "strategy" in MODE_PRESETS
        assert "llm" in MODE_PRESETS
        assert "v2" in MODE_PRESETS

    def test_arena_v2_config_loads(self):
        from run_arena import _load_agents, _load_models
        bootstrap, agents = _load_agents("arena_v2.json")
        models = _load_models("models_llm.json")
        assert len(agents) == 10  # 5 models × 2 modes
        assert len(models) == 5

    def test_exchange_flag_exists(self):
        """run_arena.py has --exchange flag with coinbase/kraken choices."""
        import argparse
        from run_arena import parse_args as _pa
        # parse_args() has no args param, so just verify the code structure
        import inspect
        src = inspect.getsource(_pa)
        assert "--exchange" in src
        assert "kraken" in src
        assert "coinbase" in src
