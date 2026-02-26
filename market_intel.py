"""V2 Market Intelligence Service.

Pre-computes technical indicators, detects market regimes, and generates
narrative signal briefs for LLM agents.  All computation is pure Python
(no numpy/pandas dependency) since data sizes are small (~50 candles).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from coinbase_consumer import CandleBook, Candle, PriceBook, Timeframe

logger = logging.getLogger(__name__)

# Wider windows than the prompt candles — enough for MACD(26+9), RSI(14), BB(20), EMA(50)
INDICATOR_TIMEFRAMES = [
    Timeframe(300, 240, 0, "5-min candles (4h) for indicators"),  # ~48 candles
    Timeframe(60, 30, 0, "1-min candles (30m) for recent action"),  # ~30 candles
]


# ── Market Regime ────────────────────────────────────────────────


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    CAPITULATION = "CAPITULATION"
    UNKNOWN = "UNKNOWN"


# ── Indicator container ──────────────────────────────────────────


@dataclass
class Indicators:
    product_id: str
    timestamp: float
    price: float | None = None

    # RSI
    rsi: float | None = None
    rsi_signal: str = ""

    # MACD
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    macd_crossover: str = ""

    # Bollinger Bands
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_position: str = ""

    # EMAs
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_cross_short: str = ""   # 9 vs 21
    ema_cross_medium: str = ""  # 21 vs 50

    # ATR
    atr: float | None = None

    # ADX
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None

    # Volume
    volume_ratio: float | None = None
    volume_trend: str = ""

    # Rate of change (%)
    roc_5: float | None = None
    roc_15: float | None = None

    # Key levels
    support: float | None = None
    resistance: float | None = None

    # Regime
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_duration_minutes: int = 0
    regime_strength: str = ""


# ── Pure-Python indicator functions ──────────────────────────────


def _ema(data: list[float], period: int) -> list[float]:
    """EMA with SMA seed.  Returns list same length as *data*; leading
    entries (before enough data) are ``float('nan')``."""
    n = len(data)
    if n < period:
        return [float("nan")] * n
    k = 2.0 / (period + 1)
    out: list[float] = [float("nan")] * (period - 1)
    seed = sum(data[:period]) / period
    out.append(seed)
    for i in range(period, n):
        seed = data[i] * k + seed * (1 - k)
        out.append(seed)
    return out


def _sma(data: list[float], period: int) -> list[float]:
    n = len(data)
    if n < period:
        return [float("nan")] * n
    out: list[float] = [float("nan")] * (period - 1)
    s = sum(data[:period])
    out.append(s / period)
    for i in range(period, n):
        s += data[i] - data[i - period]
        out.append(s / period)
    return out


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[float | None, float | None, float | None]:
    if len(closes) < slow + signal_period:
        return None, None, None
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = [
        f - s if not (math.isnan(f) or math.isnan(s)) else float("nan")
        for f, s in zip(fast_ema, slow_ema)
    ]
    valid = [v for v in macd_line if not math.isnan(v)]
    if len(valid) < signal_period:
        return (valid[-1] if valid else None), None, None
    sig = _ema(valid, signal_period)
    m = valid[-1]
    s = sig[-1]
    h = m - s if not math.isnan(s) else None
    return m, s, h


def compute_bollinger(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float | None, float | None, float | None]:
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    var = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(var)
    return mid + num_std * std, mid, mid - num_std * std


def compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    if len(closes) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(closes)):
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def compute_adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> tuple[float | None, float | None, float | None]:
    n = len(closes)
    if n < period * 2 + 1:
        return None, None, None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr_list: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr_list.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    sp = sum(plus_dm[:period])
    sm = sum(minus_dm[:period])
    st = sum(tr_list[:period])
    dx_list: list[float] = []
    last_pdi = 0.0
    last_mdi = 0.0
    for i in range(period, len(tr_list)):
        sp = sp - sp / period + plus_dm[i]
        sm = sm - sm / period + minus_dm[i]
        st = st - st / period + tr_list[i]
        if st == 0:
            continue
        last_pdi = 100.0 * sp / st
        last_mdi = 100.0 * sm / st
        di_sum = last_pdi + last_mdi
        dx_list.append(100.0 * abs(last_pdi - last_mdi) / di_sum if di_sum else 0.0)
    if len(dx_list) < period:
        return None, None, None
    adx = sum(dx_list[:period]) / period
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i]) / period
    return adx, last_pdi, last_mdi


def _find_support_resistance(
    highs: list[float], lows: list[float], closes: list[float]
) -> tuple[float | None, float | None]:
    if len(closes) < 5:
        return None, None
    current = closes[-1]
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(2, len(closes) - 2):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
            swing_highs.append(highs[i])
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            swing_lows.append(lows[i])
    supports = [s for s in swing_lows if s < current]
    support = max(supports) if supports else (min(lows[-20:]) if len(lows) >= 20 else None)
    resistances = [r for r in swing_highs if r > current]
    resistance = min(resistances) if resistances else (max(highs[-20:]) if len(highs) >= 20 else None)
    return support, resistance


# ── Regime detector ──────────────────────────────────────────────


def detect_regime(
    closes: list[float],
    ema_21: float | None,
    ema_50: float | None,
    adx: float | None,
    atr: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    volume_ratio: float | None,
) -> tuple[MarketRegime, str]:
    if not closes or ema_21 is None:
        return MarketRegime.UNKNOWN, ""
    price = closes[-1]

    # Capitulation: sharp drop + volume spike
    if len(closes) >= 5:
        peak = max(closes[-5:])
        if peak > 0:
            drop_pct = (price - peak) / peak * 100
            if drop_pct < -3 and volume_ratio is not None and volume_ratio > 2.0:
                return MarketRegime.CAPITULATION, "strong"

    # Breakout: price crossing BB with volume
    if bb_upper is not None and bb_lower is not None:
        if price > bb_upper and volume_ratio is not None and volume_ratio > 1.5:
            return MarketRegime.BREAKOUT, "strong" if volume_ratio > 2.0 else "moderate"
        if price < bb_lower and volume_ratio is not None and volume_ratio > 1.5:
            return MarketRegime.BREAKOUT, "strong" if volume_ratio > 2.0 else "moderate"

    # Trending
    if adx is not None and adx > 25:
        if ema_50 is not None:
            if price > ema_21 and ema_21 > ema_50:
                return MarketRegime.TRENDING_UP, "strong" if adx > 40 else "moderate"
            if price < ema_21 and ema_21 < ema_50:
                return MarketRegime.TRENDING_DOWN, "strong" if adx > 40 else "moderate"
        if price > ema_21:
            return MarketRegime.TRENDING_UP, "moderate"
        return MarketRegime.TRENDING_DOWN, "moderate"

    # Volatile: wide Bollinger bands
    if bb_upper is not None and bb_lower is not None:
        mid = (bb_upper + bb_lower) / 2
        if mid > 0:
            bb_width_pct = (bb_upper - bb_lower) / mid * 100
            if bb_width_pct > 5:
                return MarketRegime.VOLATILE, "moderate"

    # Ranging
    if adx is not None and adx < 20:
        return MarketRegime.RANGING, "moderate"

    return MarketRegime.RANGING, "weak"


# ── Signal generator ─────────────────────────────────────────────


def _generate_signal(ind: Indicators) -> str:
    bull = 0
    bear = 0

    # RSI
    if ind.rsi is not None:
        if ind.rsi < 30:
            bull += 2
        elif ind.rsi < 40:
            bull += 1
        elif ind.rsi > 70:
            bear += 2
        elif ind.rsi > 60:
            bear += 1

    # MACD
    if ind.macd_crossover == "bullish":
        bull += 2
    elif ind.macd_crossover == "bearish":
        bear += 2
    if ind.macd_histogram is not None:
        (bull if ind.macd_histogram > 0 else bear).__class__  # noop; next line does it
        if ind.macd_histogram > 0:
            bull += 1
        else:
            bear += 1

    # EMA
    if ind.ema_cross_short == "bullish":
        bull += 1
    elif ind.ema_cross_short == "bearish":
        bear += 1
    if ind.ema_cross_medium == "bullish":
        bull += 1
    elif ind.ema_cross_medium == "bearish":
        bear += 1

    # Volume confirmation
    if ind.volume_ratio is not None and ind.volume_ratio > 1.5:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    # Regime
    regime_score = {
        MarketRegime.TRENDING_UP: (2, 0),
        MarketRegime.TRENDING_DOWN: (0, 2),
        MarketRegime.CAPITULATION: (0, 3),
        MarketRegime.BREAKOUT: (0, 0),  # handled below
    }.get(ind.regime, (0, 0))
    bull += regime_score[0]
    bear += regime_score[1]
    if ind.regime == MarketRegime.BREAKOUT:
        if ind.price and ind.bb_upper and ind.price > ind.bb_upper:
            bull += 2
        else:
            bear += 2

    net = bull - bear
    if net >= 5:
        return "STRONG BUY — multiple indicators aligned bullish"
    if net >= 3:
        return "MODERATE BUY — trend continuation likely"
    if net >= 1:
        return "WEAK BUY — slight bullish lean, consider small position"
    if net <= -5:
        return "STRONG SELL — multiple indicators aligned bearish"
    if net <= -3:
        return "MODERATE SELL — downtrend likely to continue"
    if net <= -1:
        return "WEAK SELL — slight bearish lean, consider reducing exposure"
    return "NEUTRAL — no clear edge, wait for setup"


# ── MarketIntelService ───────────────────────────────────────────


class MarketIntelService:
    """Pre-computes technical indicators and generates signal briefs.

    Operates on its own ``CandleBook`` with wider windows suitable for
    indicator computation (separate from the prompt candle data).
    """

    def __init__(self, candle_book: CandleBook, price_book: PriceBook) -> None:
        self._candle_book = candle_book
        self._price_book = price_book
        self._indicators: dict[str, Indicators] = {}
        self._regime_history: dict[str, list[tuple[float, MarketRegime]]] = {}

    @property
    def candle_book(self) -> CandleBook:
        return self._candle_book

    def update(self, product_id: str) -> None:
        """Recompute all indicators for *product_id*."""
        # Prefer 5-min candles (wider window), fall back to 1-min
        candles = self._candle_book._candles.get((product_id, 300), [])
        if len(candles) < 14:
            candles = self._candle_book._candles.get((product_id, 60), [])
        if len(candles) < 14:
            return

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        now = time.time()

        ind = Indicators(product_id=product_id, timestamp=now)

        # Current price
        pe = self._price_book.get(product_id)
        ind.price = float(pe["price"]) if pe else closes[-1]

        # RSI
        ind.rsi = compute_rsi(closes)
        if ind.rsi is not None:
            if ind.rsi > 70:
                ind.rsi_signal = "overbought"
            elif ind.rsi < 30:
                ind.rsi_signal = "oversold"
            else:
                ind.rsi_signal = "neutral"

        # MACD
        ind.macd_line, ind.macd_signal, ind.macd_histogram = compute_macd(closes)
        if ind.macd_line is not None and ind.macd_signal is not None:
            prev_m, prev_s, _ = compute_macd(closes[:-1])
            if prev_m is not None and prev_s is not None:
                if prev_m < prev_s and ind.macd_line > ind.macd_signal:
                    ind.macd_crossover = "bullish"
                elif prev_m > prev_s and ind.macd_line < ind.macd_signal:
                    ind.macd_crossover = "bearish"
                else:
                    ind.macd_crossover = "none"

        # Bollinger
        ind.bb_upper, ind.bb_middle, ind.bb_lower = compute_bollinger(closes)
        if ind.bb_upper is not None and ind.price is not None:
            if ind.price > ind.bb_upper:
                ind.bb_position = "above_upper"
            elif ind.bb_lower is not None and ind.price < ind.bb_lower:
                ind.bb_position = "below_lower"
            else:
                ind.bb_position = "middle"

        # EMAs
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ind.ema_9 = ema9[-1] if not math.isnan(ema9[-1]) else None
        ind.ema_21 = ema21[-1] if not math.isnan(ema21[-1]) else None
        if len(closes) >= 50:
            ema50 = _ema(closes, 50)
            ind.ema_50 = ema50[-1] if not math.isnan(ema50[-1]) else None
        if ind.ema_9 is not None and ind.ema_21 is not None:
            ind.ema_cross_short = "bullish" if ind.ema_9 > ind.ema_21 else "bearish"
        if ind.ema_21 is not None and ind.ema_50 is not None:
            ind.ema_cross_medium = "bullish" if ind.ema_21 > ind.ema_50 else "bearish"

        # ATR
        ind.atr = compute_atr(highs, lows, closes)

        # ADX
        ind.adx, ind.plus_di, ind.minus_di = compute_adx(highs, lows, closes)

        # Volume
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            cur_vol = volumes[-1]
            if avg_vol > 0:
                ind.volume_ratio = cur_vol / avg_vol
                if ind.volume_ratio > 1.5:
                    ind.volume_trend = "expanding"
                elif ind.volume_ratio < 0.5:
                    ind.volume_trend = "contracting"
                else:
                    ind.volume_trend = "normal"

        # Rate of change
        if len(closes) >= 6:
            ind.roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100
        if len(closes) >= 16:
            ind.roc_15 = (closes[-1] - closes[-16]) / closes[-16] * 100

        # Support / resistance
        ind.support, ind.resistance = _find_support_resistance(highs, lows, closes)

        # Regime
        regime, strength = detect_regime(
            closes, ind.ema_21, ind.ema_50, ind.adx, ind.atr,
            ind.bb_upper, ind.bb_lower, ind.volume_ratio,
        )
        ind.regime = regime
        ind.regime_strength = strength
        hist = self._regime_history.setdefault(product_id, [])
        if not hist or hist[-1][1] != regime:
            hist.append((now, regime))
            if len(hist) > 50:
                hist[:] = hist[-50:]
        ind.regime_duration_minutes = int((now - hist[-1][0]) / 60)

        self._indicators[product_id] = ind

    def update_all(self, product_ids: list[str]) -> None:
        for pid in product_ids:
            try:
                self.update(pid)
            except Exception:
                logger.exception("Failed to compute indicators for %s", pid)

    def get_indicators(self, product_id: str) -> Indicators | None:
        return self._indicators.get(product_id)

    # ── Narrative brief formatters ───────────────────────────────

    def format_brief(self, product_id: str) -> str | None:
        ind = self._indicators.get(product_id)
        if ind is None:
            return None
        lines = [f"{product_id} Signal Brief:"]

        # Regime
        regime_str = ind.regime.value.replace("_", " ").title()
        if ind.regime_strength:
            regime_str += f" ({ind.regime_strength}"
            if ind.regime_duration_minutes > 0:
                regime_str += f", {ind.regime_duration_minutes} min"
            regime_str += ")"
        lines.append(f"  Regime: {regime_str}")

        if ind.price is not None:
            lines.append(f"  Price: ${ind.price:.6f}")

        # MACD
        if ind.macd_line is not None:
            parts: list[str] = []
            if ind.macd_crossover in ("bullish", "bearish"):
                parts.append(f"{ind.macd_crossover} crossover")
            if ind.macd_histogram is not None:
                parts.append("histogram expanding" if ind.macd_histogram > 0 else "histogram contracting")
            lines.append(f"  MACD: {', '.join(parts) if parts else 'neutral'}")

        # RSI
        if ind.rsi is not None:
            tag = ind.rsi_signal or ("weak" if ind.rsi < 45 else "strong" if ind.rsi > 55 else "neutral")
            lines.append(f"  RSI: {ind.rsi:.0f} ({tag})")

        # Volume
        if ind.volume_ratio is not None:
            lines.append(f"  Volume: {ind.volume_ratio:.1f}x average ({ind.volume_trend})")

        # EMA structure
        if ind.ema_cross_short:
            ema_desc = f"9/21 {ind.ema_cross_short}"
            if ind.ema_cross_medium:
                ema_desc += f", 21/50 {ind.ema_cross_medium}"
            lines.append(f"  EMAs: {ema_desc}")

        # Bollinger
        if ind.bb_position:
            desc = {
                "above_upper": "Price above upper band (extended)",
                "below_lower": "Price below lower band (oversold)",
                "middle": "Price within bands",
            }.get(ind.bb_position, ind.bb_position)
            lines.append(f"  Bollinger: {desc}")

        # Key levels
        lvl: list[str] = []
        if ind.support is not None:
            lvl.append(f"Support ${ind.support:.6f}")
        if ind.resistance is not None:
            lvl.append(f"Resistance ${ind.resistance:.6f}")
        if lvl:
            lines.append(f"  Key Levels: {', '.join(lvl)}")

        # ATR
        if ind.atr is not None:
            lines.append(f"  ATR: ${ind.atr:.6f} (use for stop distance)")

        # Signal
        lines.append(f"  Signal: {_generate_signal(ind)}")
        return "\n".join(lines)

    def format_all_briefs(self, product_ids: list[str]) -> str:
        parts: list[str] = []
        for pid in product_ids:
            brief = self.format_brief(pid)
            parts.append(brief if brief else f"{pid}: Insufficient data for analysis")
        return "\n\n".join(parts)
