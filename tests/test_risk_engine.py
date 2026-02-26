"""Tests for the V2 Risk Engine."""

import time

import pytest

from coinbase_consumer import PriceBook
from risk_engine import RiskEngine, TradingPlan, PlanStatus, new_plan_id


# ── Helpers ──────────────────────────────────────────────────────

def _make_price_book(**prices: str) -> PriceBook:
    """Create a PriceBook with given product prices."""
    pb = PriceBook()
    for pid, price in prices.items():
        p = str(price)
        pb.update({
            "product_id": pid,
            "price": p,
            "best_bid": p,
            "best_bid_size": "1000",
            "best_ask": p,
            "best_ask_size": "1000",
            "side": "buy",
            "last_size": "100",
            "volume_24h": "5000000",
            "time": "",
        })
    return pb


def _make_plan(
    agent_id: str = "test-agent",
    product_id: str = "DOGE-USD",
    entry_price: float = 0.10,
    stop_loss: float = 0.09,
    take_profit: float = 0.12,
    time_stop: int = 30,
    quantity: float = 1000.0,
) -> TradingPlan:
    return TradingPlan(
        plan_id=new_plan_id(),
        agent_id=agent_id,
        product_id=product_id,
        direction="long",
        quantity=quantity,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        time_stop_minutes=time_stop,
        thesis="test thesis",
        confidence=0.7,
    )


class TestPlanManagement:
    def test_register_plan(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_plan()
        pid = engine.register_plan(plan)
        assert pid == plan.plan_id
        assert len(engine.get_active_plans()) == 1

    def test_get_active_plans_filtered(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "sold")
        engine.register_plan(_make_plan(agent_id="alice"))
        engine.register_plan(_make_plan(agent_id="bob"))
        assert len(engine.get_active_plans("alice")) == 1
        assert len(engine.get_active_plans()) == 2

    def test_modify_plan_tighten_stop(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)
        ok, msg = engine.modify_plan(plan.plan_id, new_stop_loss=0.095)
        assert ok
        assert plan.stop_loss_price == 0.095

    def test_modify_plan_cannot_widen_stop(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)
        ok, msg = engine.modify_plan(plan.plan_id, new_stop_loss=0.085)
        assert not ok

    def test_close_plan(self):
        pb = _make_price_book(**{"DOGE-USD": "0.105"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_plan()
        engine.register_plan(plan)
        ok, msg = engine.close_plan(plan.plan_id)
        assert ok
        assert len(engine.get_active_plans()) == 0
        assert len(engine.get_completed_plans()) == 1
        assert len(sell_calls) == 1


class TestStopLoss:
    def test_stop_loss_triggered(self):
        pb = _make_price_book(**{"DOGE-USD": "0.085"})  # below stop of 0.09
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)

        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.STOPPED_OUT
        assert triggered[0].pnl < 0  # loss
        assert len(sell_calls) == 1

    def test_stop_loss_not_triggered(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})  # above stop
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)
        assert len(engine.check_plans()) == 0
        assert len(engine.get_active_plans()) == 1


class TestTakeProfit:
    def test_take_profit_triggered(self):
        pb = _make_price_book(**{"DOGE-USD": "0.13"})  # above TP of 0.12
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_plan(take_profit=0.12)
        engine.register_plan(plan)

        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TAKE_PROFIT
        assert triggered[0].pnl > 0


class TestTimeStop:
    def test_time_stop_triggered(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_plan(time_stop=1)  # 1 minute
        plan.created_at = time.time() - 120  # created 2 min ago
        engine.register_plan(plan)

        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TIME_STOPPED


class TestCooldown:
    def test_cooldown_after_consecutive_losses(self):
        pb = _make_price_book(**{"DOGE-USD": "0.08"})
        engine = RiskEngine(pb, lambda *a: "sold", cooldown_after_losses=2, cooldown_minutes=5)

        # Create and trigger 2 losing plans
        for _ in range(2):
            plan = _make_plan(stop_loss=0.09)
            engine.register_plan(plan)
            engine.check_plans()

        on_cd, msg = engine.is_agent_on_cooldown("test-agent")
        assert on_cd
        assert "Cooldown" in msg

    def test_no_cooldown_after_wins(self):
        pb = _make_price_book(**{"DOGE-USD": "0.13"})
        engine = RiskEngine(pb, lambda *a: "sold", cooldown_after_losses=2)

        for _ in range(3):
            plan = _make_plan(take_profit=0.12)
            engine.register_plan(plan)
            engine.check_plans()

        on_cd, _ = engine.is_agent_on_cooldown("test-agent")
        assert not on_cd

    def test_win_resets_loss_counter(self):
        sell_calls = []
        engine = RiskEngine(
            _make_price_book(**{"DOGE-USD": "0.08"}),
            lambda *a: sell_calls.append(a) or "sold",
            cooldown_after_losses=2,
        )

        # One loss
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)
        engine.check_plans()

        # One win (change price)
        engine._price_book = _make_price_book(**{"DOGE-USD": "0.13"})
        plan = _make_plan(take_profit=0.12)
        engine.register_plan(plan)
        engine.check_plans()

        # One more loss
        engine._price_book = _make_price_book(**{"DOGE-USD": "0.08"})
        plan = _make_plan(stop_loss=0.09)
        engine.register_plan(plan)
        engine.check_plans()

        # Should NOT be on cooldown (win reset the counter)
        on_cd, _ = engine.is_agent_on_cooldown("test-agent")
        assert not on_cd


def _make_short_plan(
    agent_id: str = "test-agent",
    product_id: str = "DOGE-USD",
    entry_price: float = 0.10,
    stop_loss: float = 0.11,
    take_profit: float = 0.08,
    time_stop: int = 30,
    quantity: float = 1000.0,
) -> TradingPlan:
    return TradingPlan(
        plan_id=new_plan_id(),
        agent_id=agent_id,
        product_id=product_id,
        direction="short",
        quantity=quantity,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        time_stop_minutes=time_stop,
        thesis="short test thesis",
        confidence=0.7,
    )


class TestShortStopLoss:
    def test_short_stop_loss_triggered_when_price_rises(self):
        """Short SL at 0.11, price at 0.115 → should trigger."""
        pb = _make_price_book(**{"DOGE-USD": "0.115"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_short_plan(stop_loss=0.11)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.STOPPED_OUT
        assert triggered[0].pnl < 0  # loss on short when price rises
        # Exit action should be "buy" (cover short)
        assert sell_calls[0][3] == "buy"

    def test_short_stop_loss_not_triggered_below(self):
        """Short SL at 0.11, price at 0.095 → should NOT trigger."""
        pb = _make_price_book(**{"DOGE-USD": "0.095"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(stop_loss=0.11)
        engine.register_plan(plan)
        assert len(engine.check_plans()) == 0


class TestShortTakeProfit:
    def test_short_take_profit_triggered_when_price_drops(self):
        """Short TP at 0.08, price at 0.075 → should trigger."""
        pb = _make_price_book(**{"DOGE-USD": "0.075"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_short_plan(take_profit=0.08)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert len(triggered) == 1
        assert triggered[0].status == PlanStatus.TAKE_PROFIT
        assert triggered[0].pnl > 0  # profit on short when price drops
        assert sell_calls[0][3] == "buy"

    def test_short_take_profit_not_triggered_above(self):
        """Short TP at 0.08, price at 0.095 → should NOT trigger."""
        pb = _make_price_book(**{"DOGE-USD": "0.095"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(take_profit=0.08)
        engine.register_plan(plan)
        assert len(engine.check_plans()) == 0


class TestShortPnl:
    def test_short_pnl_calculation(self):
        """Short at 0.10, exit at 0.08 → pnl = (0.10 - 0.08) * 1000 = +20."""
        pb = _make_price_book(**{"DOGE-USD": "0.075"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(entry_price=0.10, take_profit=0.08, quantity=1000)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        # Exit price is best_ask (same as price in test helper)
        assert triggered[0].pnl == pytest.approx((0.10 - 0.075) * 1000)

    def test_short_loss_pnl(self):
        """Short at 0.10, exit at 0.115 → pnl = (0.10 - 0.115) * 1000 = -15."""
        pb = _make_price_book(**{"DOGE-USD": "0.115"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(entry_price=0.10, stop_loss=0.11, quantity=1000)
        engine.register_plan(plan)
        triggered = engine.check_plans()
        assert triggered[0].pnl == pytest.approx((0.10 - 0.115) * 1000)


class TestShortModifyPlan:
    def test_tighten_short_stop_lower(self):
        """Short stop at 0.11, tighten to 0.105 → should work."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(stop_loss=0.11)
        engine.register_plan(plan)
        ok, msg = engine.modify_plan(plan.plan_id, new_stop_loss=0.105)
        assert ok
        assert plan.stop_loss_price == 0.105

    def test_cannot_widen_short_stop(self):
        """Short stop at 0.11, try to widen to 0.12 → should fail."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan(stop_loss=0.11)
        engine.register_plan(plan)
        ok, msg = engine.modify_plan(plan.plan_id, new_stop_loss=0.12)
        assert not ok
        assert "lower" in msg.lower()


class TestShortClosePlan:
    def test_close_short_plan(self):
        pb = _make_price_book(**{"DOGE-USD": "0.095"})
        sell_calls = []
        engine = RiskEngine(pb, lambda *a: sell_calls.append(a) or "sold")
        plan = _make_short_plan()
        engine.register_plan(plan)
        ok, msg = engine.close_plan(plan.plan_id)
        assert ok
        assert len(engine.get_active_plans()) == 0
        # Close a short = buy to cover
        assert sell_calls[0][3] == "buy"
        completed = engine.get_completed_plans()
        assert completed[0].pnl == pytest.approx((0.10 - 0.095) * 1000)


class TestShortUnrealizedPnl:
    def test_short_unrealized_pnl_in_state_text(self):
        pb = _make_price_book(**{"DOGE-USD": "0.095"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_short_plan()
        engine.register_plan(plan)
        text = engine.get_agent_state_text("test-agent")
        assert "SHORT" in text
        assert "$+5.0000" in text  # (0.10 - 0.095) * 1000 = 5.0


class TestAgentStateText:
    def test_no_plans(self):
        engine = RiskEngine(_make_price_book(**{"DOGE-USD": "0.10"}), lambda *a: "sold")
        text = engine.get_agent_state_text("test-agent")
        assert "No active trading plans" in text

    def test_with_active_plan(self):
        engine = RiskEngine(_make_price_book(**{"DOGE-USD": "0.105"}), lambda *a: "sold")
        plan = _make_plan()
        engine.register_plan(plan)
        text = engine.get_agent_state_text("test-agent")
        assert "LONG" in text
        assert "DOGE-USD" in text
        assert "thesis" in text.lower()

    def test_with_completed_plans(self):
        pb = _make_price_book(**{"DOGE-USD": "0.13"})
        engine = RiskEngine(pb, lambda *a: "sold")
        plan = _make_plan(take_profit=0.12)
        engine.register_plan(plan)
        engine.check_plans()
        text = engine.get_agent_state_text("test-agent")
        assert "Completed" in text
