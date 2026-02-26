"""Tests for the V2 Market Intelligence Service."""

import math
import time

import pytest

from market_intel import (
    MarketIntelService,
    MarketRegime,
    _ema,
    _sma,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_adx,
    detect_regime,
    _find_support_resistance,
    _generate_signal,
    Indicators,
)
from coinbase_consumer import CandleBook, Candle, PriceBook
from datetime import datetime, timezone


# ── Helper ───────────────────────────────────────────────────────

def _make_candles(closes: list[float], base_ts: float | None = None) -> list[Candle]:
    """Build candle list from close prices (open=high=low=close, vol=1000)."""
    ts = base_ts or time.time() - len(closes) * 300
    return [
        Candle(
            time=datetime.fromtimestamp(ts + i * 300, tz=timezone.utc),
            open=c,
            high=c * 1.005,
            low=c * 0.995,
            close=c,
            volume=1000.0,
        )
        for i, c in enumerate(closes)
    ]


# ── EMA / SMA ───────────────────────────────────────────────────

class TestEMA:
    def test_ema_too_short(self):
        assert all(math.isnan(v) for v in _ema([1, 2], 5))

    def test_ema_seed_is_sma(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _ema(data, 3)
        # First valid value should be SMA of first 3
        assert result[2] == pytest.approx(2.0)

    def test_ema_length(self):
        data = list(range(1, 21))
        assert len(_ema(data, 5)) == 20

    def test_sma_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _sma(data, 3)
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)


# ── RSI ──────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_not_enough_data(self):
        assert compute_rsi([1, 2, 3]) is None

    def test_rsi_all_gains(self):
        # Monotonically increasing — RSI should be 100
        closes = list(range(1, 20))
        rsi = compute_rsi(closes, 14)
        assert rsi is not None
        assert rsi == pytest.approx(100.0)

    def test_rsi_all_losses(self):
        closes = list(range(20, 1, -1))
        rsi = compute_rsi(closes, 14)
        assert rsi is not None
        assert rsi == pytest.approx(0.0, abs=0.1)

    def test_rsi_range(self):
        # Mixed data — RSI should be between 0 and 100
        closes = [10, 11, 10.5, 11.5, 10.8, 11.2, 10.9, 11.1,
                  10.7, 11.3, 10.6, 11.4, 10.5, 11.0, 10.8, 11.2]
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert 0 <= rsi <= 100


# ── MACD ─────────────────────────────────────────────────────────

class TestMACD:
    def test_macd_not_enough_data(self):
        m, s, h = compute_macd(list(range(10)))
        assert m is None

    def test_macd_returns_values(self):
        closes = [float(x) for x in range(1, 50)]
        m, s, h = compute_macd(closes)
        assert m is not None
        assert s is not None
        assert h is not None

    def test_macd_trending_up(self):
        # Strong uptrend — MACD should be positive
        closes = [10 + i * 0.5 for i in range(50)]
        m, s, h = compute_macd(closes)
        assert m is not None and m > 0


# ── Bollinger Bands ──────────────────────────────────────────────

class TestBollinger:
    def test_bollinger_not_enough_data(self):
        u, m, l = compute_bollinger([1, 2, 3])
        assert u is None

    def test_bollinger_constant_price(self):
        closes = [100.0] * 25
        u, m, l = compute_bollinger(closes)
        assert m == pytest.approx(100.0)
        assert u == pytest.approx(100.0)  # zero std
        assert l == pytest.approx(100.0)

    def test_bollinger_upper_above_lower(self):
        closes = [10 + i * 0.1 + (i % 3) * 0.5 for i in range(25)]
        u, m, l = compute_bollinger(closes)
        assert u is not None and l is not None
        assert u > m > l


# ── ATR ──────────────────────────────────────────────────────────

class TestATR:
    def test_atr_not_enough_data(self):
        assert compute_atr([1], [1], [1]) is None

    def test_atr_positive(self):
        highs = [11.0 + i * 0.1 for i in range(20)]
        lows = [9.0 + i * 0.1 for i in range(20)]
        closes = [10.0 + i * 0.1 for i in range(20)]
        atr = compute_atr(highs, lows, closes)
        assert atr is not None and atr > 0


# ── ADX ──────────────────────────────────────────────────────────

class TestADX:
    def test_adx_not_enough_data(self):
        a, p, m = compute_adx([1, 2], [0, 1], [0.5, 1.5])
        assert a is None

    def test_adx_trending(self):
        # Strong trend: should produce ADX > 0
        highs = [10 + i * 0.5 for i in range(40)]
        lows = [9 + i * 0.5 for i in range(40)]
        closes = [9.5 + i * 0.5 for i in range(40)]
        a, p, m = compute_adx(highs, lows, closes)
        assert a is not None and a > 0


# ── Support/Resistance ───────────────────────────────────────────

class TestSupportResistance:
    def test_not_enough_data(self):
        s, r = _find_support_resistance([1], [1], [1])
        assert s is None and r is None

    def test_finds_levels(self):
        # Create swing highs and lows
        closes = [10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        s, r = _find_support_resistance(highs, lows, closes)
        # Should find some levels
        assert s is not None or r is not None


# ── Regime Detection ─────────────────────────────────────────────

class TestRegimeDetection:
    def test_unknown_with_no_data(self):
        regime, _ = detect_regime([], None, None, None, None, None, None, None)
        assert regime == MarketRegime.UNKNOWN

    def test_trending_up(self):
        closes = [10 + i * 0.3 for i in range(50)]
        ema21 = sum(closes[-21:]) / 21
        ema50 = sum(closes[-50:]) / 50
        regime, _ = detect_regime(closes, ema21, ema50, adx=35, atr=0.2,
                                   bb_upper=closes[-1]+1, bb_lower=closes[-1]-1,
                                   volume_ratio=1.0)
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down(self):
        closes = [20 - i * 0.3 for i in range(50)]
        ema21 = sum(closes[-21:]) / 21
        ema50 = sum(closes[-50:]) / 50
        regime, _ = detect_regime(closes, ema21, ema50, adx=30, atr=0.2,
                                   bb_upper=closes[-1]+1, bb_lower=closes[-1]-1,
                                   volume_ratio=1.0)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_ranging(self):
        # Narrow BB width (1%) so it doesn't trigger volatile (>5%)
        regime, _ = detect_regime([10.0]*20, 10.0, 10.0, adx=15, atr=0.1,
                                   bb_upper=10.05, bb_lower=9.95, volume_ratio=1.0)
        assert regime == MarketRegime.RANGING

    def test_capitulation(self):
        closes = [10, 10, 10, 10, 6]  # -40% drop
        regime, _ = detect_regime(closes, 10.0, None, None, None, None, None, 3.0)
        assert regime == MarketRegime.CAPITULATION


# ── Signal Generator ─────────────────────────────────────────────

class TestSignalGenerator:
    def test_strong_buy(self):
        ind = Indicators(product_id="TEST", timestamp=time.time())
        ind.rsi = 28  # oversold
        ind.macd_crossover = "bullish"
        ind.macd_histogram = 0.5
        ind.ema_cross_short = "bullish"
        ind.ema_cross_medium = "bullish"
        ind.volume_ratio = 2.0
        ind.regime = MarketRegime.TRENDING_UP
        ind.bb_upper = 100
        ind.price = 90
        sig = _generate_signal(ind)
        assert "BUY" in sig

    def test_strong_sell(self):
        ind = Indicators(product_id="TEST", timestamp=time.time())
        ind.rsi = 75
        ind.macd_crossover = "bearish"
        ind.macd_histogram = -0.5
        ind.ema_cross_short = "bearish"
        ind.ema_cross_medium = "bearish"
        ind.volume_ratio = 2.0
        ind.regime = MarketRegime.TRENDING_DOWN
        sig = _generate_signal(ind)
        assert "SELL" in sig

    def test_neutral(self):
        ind = Indicators(product_id="TEST", timestamp=time.time())
        sig = _generate_signal(ind)
        assert "NEUTRAL" in sig


# ── MarketIntelService integration ───────────────────────────────

class TestMarketIntelService:
    def test_update_with_data(self):
        cb = CandleBook()
        pb = PriceBook()

        # Insert enough 5-min candles
        closes = [0.10 + i * 0.001 for i in range(48)]
        candles_raw = [
            [int(time.time()) - (48 - i) * 300, c * 0.995, c * 1.005, c, c, 1000.0]
            for i, c in enumerate(closes)
        ]
        # Coinbase returns descending
        candles_raw.reverse()
        cb.update_from_api("DOGE-USD", 300, candles_raw)

        pb.update({
            "product_id": "DOGE-USD",
            "price": "0.148",
            "best_bid": "0.1479",
            "best_bid_size": "1000",
            "best_ask": "0.1481",
            "best_ask_size": "1000",
            "side": "buy",
            "last_size": "100",
            "volume_24h": "5000000",
            "time": "2025-01-01T00:00:00Z",
        })

        svc = MarketIntelService(cb, pb)
        svc.update("DOGE-USD")
        ind = svc.get_indicators("DOGE-USD")
        assert ind is not None
        assert ind.rsi is not None
        assert ind.regime != MarketRegime.UNKNOWN

    def test_format_brief(self):
        cb = CandleBook()
        pb = PriceBook()

        closes = [0.10 + i * 0.001 for i in range(48)]
        candles_raw = [
            [int(time.time()) - (48 - i) * 300, c * 0.995, c * 1.005, c, c, 1000.0]
            for i, c in enumerate(closes)
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
        brief = svc.format_brief("DOGE-USD")
        assert brief is not None
        assert "Signal Brief" in brief
        assert "Regime" in brief
        assert "RSI" in brief

    def test_format_brief_no_data(self):
        svc = MarketIntelService(CandleBook(), PriceBook())
        assert svc.format_brief("NOPE-USD") is None

    def test_format_all_briefs(self):
        svc = MarketIntelService(CandleBook(), PriceBook())
        result = svc.format_all_briefs(["DOGE-USD", "PEPE-USD"])
        assert "Insufficient data" in result
