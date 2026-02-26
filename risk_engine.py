"""V2 Risk Engine.

Monitors active trading plans and automatically executes stop-losses,
take-profits, and time stops.  Also tracks cooldowns and daily loss limits.
The LLM is never in the loop for stop execution — this is deterministic Python.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from coinbase_consumer import PriceBook

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    ACTIVE = "active"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"
    TIME_STOPPED = "time_stopped"
    CLOSED = "closed"


@dataclass
class TradingPlan:
    plan_id: str
    agent_id: str
    product_id: str
    direction: str          # "long"  (short not yet supported)
    quantity: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    time_stop_minutes: int
    thesis: str
    confidence: float
    status: PlanStatus = PlanStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0


def new_plan_id() -> str:
    return uuid.uuid4().hex[:8]


class RiskEngine:
    """Monitors trading plans and auto-executes stop-loss / take-profit."""

    def __init__(
        self,
        price_book: PriceBook,
        execute_sell_fn: Callable[[str, str, float, str], str],
        cooldown_after_losses: int = 2,
        cooldown_minutes: float = 5.0,
    ) -> None:
        self._price_book = price_book
        self._execute_sell = execute_sell_fn  # (agent_id, product_id, quantity, action) -> msg
        self._plans: dict[str, TradingPlan] = {}
        self._completed: list[TradingPlan] = []

        self.cooldown_after_losses = cooldown_after_losses
        self.cooldown_minutes = cooldown_minutes

        self._consecutive_losses: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}

    # ── Plan management ──────────────────────────────────────────

    def register_plan(self, plan: TradingPlan) -> str:
        self._plans[plan.plan_id] = plan
        logger.info(
            "PLAN_REGISTERED | %s | %s %s qty=%.4f SL=%.6f TP=%.6f | %s",
            plan.agent_id, plan.direction, plan.product_id,
            plan.quantity, plan.stop_loss_price, plan.take_profit_price,
            plan.thesis[:80],
        )
        return plan.plan_id

    def get_active_plans(self, agent_id: str | None = None) -> list[TradingPlan]:
        plans = [p for p in self._plans.values() if p.status == PlanStatus.ACTIVE]
        if agent_id:
            plans = [p for p in plans if p.agent_id == agent_id]
        return plans

    def get_completed_plans(self, agent_id: str | None = None, limit: int = 20) -> list[TradingPlan]:
        plans = self._completed if agent_id is None else [p for p in self._completed if p.agent_id == agent_id]
        return plans[-limit:]

    def modify_plan(
        self,
        plan_id: str,
        new_stop_loss: float | None = None,
        new_take_profit: float | None = None,
    ) -> tuple[bool, str]:
        plan = self._plans.get(plan_id)
        if not plan or plan.status != PlanStatus.ACTIVE:
            return False, f"Plan {plan_id} not found or not active."
        msgs: list[str] = []
        if new_stop_loss is not None:
            if plan.direction == "long" and new_stop_loss > plan.stop_loss_price:
                plan.stop_loss_price = new_stop_loss
                msgs.append(f"Stop tightened to ${new_stop_loss:.6f}")
            elif plan.direction == "long":
                return False, "Can only tighten stops (move stop higher for longs)."
        if new_take_profit is not None:
            plan.take_profit_price = new_take_profit
            msgs.append(f"TP moved to ${new_take_profit:.6f}")
        return True, " | ".join(msgs) if msgs else "No changes."

    def close_plan(self, plan_id: str, reason: str = "manual") -> tuple[bool, str]:
        plan = self._plans.get(plan_id)
        if not plan or plan.status != PlanStatus.ACTIVE:
            return False, f"Plan {plan_id} not found or not active."
        entry = self._price_book.get(plan.product_id)
        exit_price = float(entry["best_bid"]) if entry else plan.entry_price
        plan.status = PlanStatus.CLOSED
        plan.closed_at = time.time()
        plan.exit_price = exit_price
        plan.exit_reason = reason
        plan.pnl = (exit_price - plan.entry_price) * plan.quantity if plan.direction == "long" else 0
        # Execute the sell
        msg = self._execute_sell(plan.agent_id, plan.product_id, plan.quantity, "sell")
        self._completed.append(plan)
        del self._plans[plan.plan_id]
        self._track_loss(plan)
        logger.info("PLAN_CLOSED | %s | %s pnl=%.4f reason=%s", plan.agent_id, plan.plan_id, plan.pnl, reason)
        return True, f"Plan closed. {msg}"

    # ── Background check loop ────────────────────────────────────

    def check_plans(self) -> list[TradingPlan]:
        """Check all active plans against live prices.  Auto-execute exits.
        Returns plans that were triggered this cycle."""
        triggered: list[TradingPlan] = []
        now = time.time()

        for plan_id in list(self._plans):
            plan = self._plans[plan_id]
            if plan.status != PlanStatus.ACTIVE:
                continue
            entry = self._price_book.get(plan.product_id)
            if entry is None:
                continue
            current = float(entry["price"])
            exit_reason: str | None = None

            # Time stop
            elapsed_min = (now - plan.created_at) / 60
            if plan.time_stop_minutes > 0 and elapsed_min >= plan.time_stop_minutes:
                exit_reason = "time_stop"
                plan.status = PlanStatus.TIME_STOPPED
            # Stop-loss
            elif plan.direction == "long" and current <= plan.stop_loss_price:
                exit_reason = "stop_loss"
                plan.status = PlanStatus.STOPPED_OUT
            # Take-profit
            elif plan.direction == "long" and current >= plan.take_profit_price:
                exit_reason = "take_profit"
                plan.status = PlanStatus.TAKE_PROFIT

            if exit_reason:
                exit_price = float(entry["best_bid"]) if plan.direction == "long" else float(entry["best_ask"])
                plan.closed_at = now
                plan.exit_price = exit_price
                plan.exit_reason = exit_reason
                plan.pnl = (exit_price - plan.entry_price) * plan.quantity if plan.direction == "long" else 0

                # Execute exit trade
                try:
                    self._execute_sell(plan.agent_id, plan.product_id, plan.quantity, "sell")
                except Exception:
                    logger.exception("Failed to execute exit for plan %s", plan_id)

                self._completed.append(plan)
                del self._plans[plan_id]
                self._track_loss(plan)
                triggered.append(plan)
                logger.info(
                    "PLAN_TRIGGERED | %s | %s %s exit=%.6f reason=%s pnl=%.4f",
                    plan.agent_id, plan.product_id, plan_id,
                    exit_price, exit_reason, plan.pnl,
                )

        return triggered

    def _track_loss(self, plan: TradingPlan) -> None:
        if plan.pnl < 0:
            self._consecutive_losses[plan.agent_id] = self._consecutive_losses.get(plan.agent_id, 0) + 1
            if self._consecutive_losses[plan.agent_id] >= self.cooldown_after_losses:
                self._cooldown_until[plan.agent_id] = time.time() + self.cooldown_minutes * 60
                logger.info(
                    "COOLDOWN | %s | %d losses → cooling down %.0f min",
                    plan.agent_id, self._consecutive_losses[plan.agent_id], self.cooldown_minutes,
                )
        else:
            self._consecutive_losses[plan.agent_id] = 0

    def is_agent_on_cooldown(self, agent_id: str) -> tuple[bool, str]:
        cd = self._cooldown_until.get(agent_id, 0)
        if time.time() < cd:
            remaining = (cd - time.time()) / 60
            return True, f"Cooldown: {remaining:.1f} min left after {self.cooldown_after_losses} consecutive losses"
        return False, ""

    # ── State for agent prompt ───────────────────────────────────

    def get_agent_state_text(self, agent_id: str) -> str:
        lines: list[str] = []
        active = self.get_active_plans(agent_id)
        if active:
            lines.append("Active Trading Plans:")
            for p in active:
                elapsed = (time.time() - p.created_at) / 60
                entry = self._price_book.get(p.product_id)
                cur = float(entry["price"]) if entry else p.entry_price
                unreal = (cur - p.entry_price) * p.quantity if p.direction == "long" else 0
                lines.append(
                    f"  [{p.plan_id}] {p.direction.upper()} {p.product_id} "
                    f"qty={p.quantity:.4f} entry=${p.entry_price:.6f} "
                    f"SL=${p.stop_loss_price:.6f} TP=${p.take_profit_price:.6f} "
                    f"P&L=${unreal:+.4f} ({elapsed:.0f}m/{p.time_stop_minutes}m) "
                    f"thesis: {p.thesis}"
                )
        else:
            lines.append("No active trading plans.")

        completed = self.get_completed_plans(agent_id, limit=5)
        if completed:
            lines.append("\nRecent Completed Plans:")
            for p in completed:
                lines.append(
                    f"  {p.direction.upper()} {p.product_id} "
                    f"entry=${p.entry_price:.6f} exit=${p.exit_price:.6f} "
                    f"P&L=${p.pnl:+.4f} reason={p.exit_reason}"
                )

        on_cd, cd_msg = self.is_agent_on_cooldown(agent_id)
        if on_cd:
            lines.append(f"\nWARNING: {cd_msg}")
        return "\n".join(lines)
