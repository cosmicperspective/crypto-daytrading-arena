from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimConfig:
    preset: str
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    impact_bps: float
    simulate_latency_ms: int
    partial_fill_prob: float
    partial_fill_min_ratio: float
    partial_fill_max_ratio: float
    funding_bps_per_hour: float
    borrow_bps_per_hour: float
    min_trade_interval_s: float
    max_order_usd: float | None
    max_position_usd: float | None
    max_leverage: float
    allow_negative_cash: bool


PRESETS: dict[str, SimConfig] = {
    "paper": SimConfig(
        preset="paper",
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        simulate_latency_ms=0,
        partial_fill_prob=0.0,
        partial_fill_min_ratio=1.0,
        partial_fill_max_ratio=1.0,
        funding_bps_per_hour=0.0,
        borrow_bps_per_hour=0.0,
        min_trade_interval_s=0.0,
        max_order_usd=None,
        max_position_usd=None,
        max_leverage=1.0,
        allow_negative_cash=False,
    ),
    "realistic": SimConfig(
        preset="realistic",
        taker_fee_bps=6.0,
        maker_fee_bps=4.0,
        slippage_bps=5.0,
        impact_bps=15.0,
        simulate_latency_ms=250,
        partial_fill_prob=0.2,
        partial_fill_min_ratio=0.4,
        partial_fill_max_ratio=0.9,
        funding_bps_per_hour=0.0,
        borrow_bps_per_hour=0.0,
        min_trade_interval_s=2.0,
        max_order_usd=25_000.0,
        max_position_usd=75_000.0,
        max_leverage=1.5,
        allow_negative_cash=True,
    ),
    "ultra": SimConfig(
        preset="ultra",
        taker_fee_bps=8.0,
        maker_fee_bps=6.0,
        slippage_bps=10.0,
        impact_bps=25.0,
        simulate_latency_ms=750,
        partial_fill_prob=0.5,
        partial_fill_min_ratio=0.2,
        partial_fill_max_ratio=0.8,
        funding_bps_per_hour=0.0,
        borrow_bps_per_hour=0.0,
        min_trade_interval_s=5.0,
        max_order_usd=10_000.0,
        max_position_usd=40_000.0,
        max_leverage=1.25,
        allow_negative_cash=True,
    ),
}


def _coerce_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _coerce_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_sim_config() -> SimConfig:
    preset_name = os.getenv("SIM_PRESET", "paper").strip().lower()
    base = PRESETS.get(preset_name, PRESETS["paper"])

    overrides = _load_overrides(os.getenv("SIM_CONFIG_FILE"))

    def ov(key: str, default: Any) -> Any:
        return overrides.get(key, default)

    return SimConfig(
        preset=preset_name,
        taker_fee_bps=_coerce_float(os.getenv("SIM_TAKER_FEE_BPS"), ov("taker_fee_bps", base.taker_fee_bps)),
        maker_fee_bps=_coerce_float(os.getenv("SIM_MAKER_FEE_BPS"), ov("maker_fee_bps", base.maker_fee_bps)),
        slippage_bps=_coerce_float(os.getenv("SIM_SLIPPAGE_BPS"), ov("slippage_bps", base.slippage_bps)),
        impact_bps=_coerce_float(os.getenv("SIM_IMPACT_BPS"), ov("impact_bps", base.impact_bps)),
        simulate_latency_ms=_coerce_int(
            os.getenv("SIM_LATENCY_MS"), ov("simulate_latency_ms", base.simulate_latency_ms)
        ),
        partial_fill_prob=_coerce_float(
            os.getenv("SIM_PARTIAL_FILL_PROB"), ov("partial_fill_prob", base.partial_fill_prob)
        ),
        partial_fill_min_ratio=_coerce_float(
            os.getenv("SIM_PARTIAL_MIN_RATIO"),
            ov("partial_fill_min_ratio", base.partial_fill_min_ratio),
        ),
        partial_fill_max_ratio=_coerce_float(
            os.getenv("SIM_PARTIAL_MAX_RATIO"),
            ov("partial_fill_max_ratio", base.partial_fill_max_ratio),
        ),
        funding_bps_per_hour=_coerce_float(
            os.getenv("SIM_FUNDING_BPS_PER_HOUR"),
            ov("funding_bps_per_hour", base.funding_bps_per_hour),
        ),
        borrow_bps_per_hour=_coerce_float(
            os.getenv("SIM_BORROW_BPS_PER_HOUR"),
            ov("borrow_bps_per_hour", base.borrow_bps_per_hour),
        ),
        min_trade_interval_s=_coerce_float(
            os.getenv("SIM_MIN_TRADE_INTERVAL_S"),
            ov("min_trade_interval_s", base.min_trade_interval_s),
        ),
        max_order_usd=(
            _coerce_float(os.getenv("SIM_MAX_ORDER_USD"), ov("max_order_usd", base.max_order_usd))
            if (os.getenv("SIM_MAX_ORDER_USD") or "max_order_usd" in overrides)
            else base.max_order_usd
        ),
        max_position_usd=(
            _coerce_float(os.getenv("SIM_MAX_POSITION_USD"), ov("max_position_usd", base.max_position_usd))
            if (os.getenv("SIM_MAX_POSITION_USD") or "max_position_usd" in overrides)
            else base.max_position_usd
        ),
        max_leverage=_coerce_float(os.getenv("SIM_MAX_LEVERAGE"), ov("max_leverage", base.max_leverage)),
        allow_negative_cash=_coerce_bool(
            os.getenv("SIM_ALLOW_NEGATIVE_CASH"), ov("allow_negative_cash", base.allow_negative_cash)
        ),
    )
