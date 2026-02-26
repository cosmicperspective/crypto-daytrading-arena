"""Tests for short selling in the trading tools / account store."""

import pytest

from coinbase_consumer import PriceBook
from sim_config import SimConfig
from trading_tools import AccountStore, AgentAccount, INITIAL_CASH


# ── Helpers ──────────────────────────────────────────────────────


def _make_price_book(**prices: str) -> PriceBook:
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


def _paper_sim() -> SimConfig:
    """Always use the paper preset for tests (zero fees/slippage, no rate limits)."""
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


def _make_store(pb: PriceBook) -> AccountStore:
    return AccountStore(pb, _paper_sim())


# ── Opening a short ─────────────────────────────────────────────


class TestOpenShort:
    def test_sell_without_position_opens_short(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        result = s.execute_trade("agent1", "DOGE-USD", 1000, "sell")
        assert result.success
        assert "Short sold" in result.message
        account = s.get_or_create("agent1")
        assert account.positions["DOGE-USD"] < 0
        assert account.positions["DOGE-USD"] == -1000

    def test_short_requires_cash_collateral(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        # Starting cash is $200. Shorting 2500 DOGE @ 0.10 = $250 collateral > $200 → fail
        result = s.execute_trade("agent1", "DOGE-USD", 2500, "sell")
        assert not result.success
        assert "Insufficient cash" in result.message

    def test_short_tracks_cost_basis(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        result = s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        assert result.success
        account = s.get_or_create("agent1")
        # Cost basis should be proceeds (entry value for the short)
        assert account.cost_basis["DOGE-USD"] > 0
        assert account.trade_count == 1

    def test_add_to_existing_short(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 200, "sell")
        s.execute_trade("agent1", "DOGE-USD", 300, "sell")
        account = s.get_or_create("agent1")
        assert account.positions["DOGE-USD"] == -500


# ── Covering a short ────────────────────────────────────────────


class TestCoverShort:
    def test_buy_covers_short(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        result = s.execute_trade("agent1", "DOGE-USD", 500, "buy")
        assert result.success
        assert "Covered" in result.message
        account = s.get_or_create("agent1")
        assert "DOGE-USD" not in account.positions

    def test_partial_cover(self):
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        result = s.execute_trade("agent1", "DOGE-USD", 200, "buy")
        assert result.success
        account = s.get_or_create("agent1")
        assert account.positions["DOGE-USD"] == pytest.approx(-300, abs=1)

    def test_cannot_over_cover(self):
        """Can't buy more than short size (no flip from short to long in one trade)."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        result = s.execute_trade("agent1", "DOGE-USD", 600, "buy")
        assert not result.success
        assert "only cover" in result.message.lower() or "short size" in result.message.lower()


# ── Short P&L ───────────────────────────────────────────────────


class TestShortPnl:
    def test_profitable_short(self):
        """Short at 0.10, cover at 0.08 → profit."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")

        # Price drops
        pb2 = _make_price_book(**{"DOGE-USD": "0.08"})
        s._price_book = pb2
        result = s.execute_trade("agent1", "DOGE-USD", 500, "buy")
        assert result.success
        account = s.get_or_create("agent1")
        # Should have made money (realized_pnl > 0 after fees)
        assert account.wins >= 1

    def test_losing_short(self):
        """Short at 0.10, cover at 0.12 → loss."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 200, "sell")

        # Price rises
        pb2 = _make_price_book(**{"DOGE-USD": "0.12"})
        s._price_book = pb2
        result = s.execute_trade("agent1", "DOGE-USD", 200, "buy")
        assert result.success
        account = s.get_or_create("agent1")
        assert account.losses >= 1


# ── Portfolio value ──────────────────────────────────────────────


class TestShortPortfolioValue:
    def test_portfolio_value_with_short(self):
        """Short position should reduce portfolio value when price rises."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        account = s.get_or_create("agent1")
        initial_value = account.portfolio_value(pb)

        s.execute_trade("agent1", "DOGE-USD", 500, "sell")

        # Price rises (bad for short)
        pb2 = _make_price_book(**{"DOGE-USD": "0.12"})
        value_up = account.portfolio_value(pb2)
        assert value_up < initial_value

        # Price drops (good for short)
        pb3 = _make_price_book(**{"DOGE-USD": "0.08"})
        value_down = account.portfolio_value(pb3)
        assert value_down > initial_value

    def test_flat_portfolio_value_unchanged(self):
        """Immediately after opening short at same price, portfolio value ≈ initial (minus fees)."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        account = s.get_or_create("agent1")
        initial = account.portfolio_value(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        # Value should be very close to initial (small difference is fees + slippage)
        after = account.portfolio_value(pb)
        assert abs(after - initial) < 2.0  # within $2 of initial (fees)


# ── Mixed positions ──────────────────────────────────────────────


class TestMixedPositions:
    def test_cannot_sell_long_and_open_short_in_one_order(self):
        """If holding 300 long, can't sell 500 (would flip to short)."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 300, "buy")
        result = s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        assert not result.success
        assert "Close your long first" in result.message

    def test_close_long_then_open_short(self):
        """Close long, then open short as separate trades."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 300, "buy")
        r1 = s.execute_trade("agent1", "DOGE-USD", 300, "sell")
        assert r1.success
        r2 = s.execute_trade("agent1", "DOGE-USD", 200, "sell")
        assert r2.success
        assert "Short sold" in r2.message
        account = s.get_or_create("agent1")
        assert account.positions["DOGE-USD"] == -200


# ── Portfolio display ────────────────────────────────────────────


class TestPortfolioDisplay:
    def test_short_shows_in_portfolio(self):
        """_get_portfolio should show SHORT side for negative positions."""
        pb = _make_price_book(**{"DOGE-USD": "0.10"})
        s = _make_store(pb)
        s.execute_trade("agent1", "DOGE-USD", 500, "sell")
        # Access via the internal function
        from trading_tools import _get_portfolio, store as _store
        old_store = _store
        import trading_tools
        trading_tools.store = s
        try:
            text = _get_portfolio("agent1")
            assert "SHORT" in text
            assert "DOGE-USD" in text
        finally:
            trading_tools.store = old_store
