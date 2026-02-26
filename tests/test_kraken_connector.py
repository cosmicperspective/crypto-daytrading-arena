"""Tests for the Kraken connector."""

import json
import time

import pytest

from kraken_connector import (
    KrakenKafkaConnector,
    TickerMessage,
    KRAKEN_TO_INTERNAL,
    INTERNAL_TO_KRAKEN_REST,
    DEFAULT_PRODUCTS,
    poll_kraken_candles,
)
from coinbase_consumer import CandleBook, PriceBook


# ── Symbol mapping ──────────────────────────────────────────────


class TestSymbolMapping:
    def test_all_default_products_have_internal_mapping(self):
        for p in DEFAULT_PRODUCTS:
            assert p in KRAKEN_TO_INTERNAL, f"{p} missing from KRAKEN_TO_INTERNAL"

    def test_all_internal_have_rest_mapping(self):
        for internal in KRAKEN_TO_INTERNAL.values():
            assert internal in INTERNAL_TO_KRAKEN_REST, f"{internal} missing from INTERNAL_TO_KRAKEN_REST"

    def test_internal_format_is_coinbase_style(self):
        for internal in KRAKEN_TO_INTERNAL.values():
            assert "-" in internal, f"{internal} should be COIN-USD format"
            assert "/" not in internal

    def test_kraken_rest_format(self):
        assert INTERNAL_TO_KRAKEN_REST["DOGE-USD"] == "XDGUSD"
        assert INTERNAL_TO_KRAKEN_REST["PEPE-USD"] == "PEPEUSD"
        assert INTERNAL_TO_KRAKEN_REST["SOL-USD"] == "SOLUSD"
        assert INTERNAL_TO_KRAKEN_REST["SUI-USD"] == "SUIUSD"
        assert INTERNAL_TO_KRAKEN_REST["FARTCOIN-USD"] == "FARTCOINUSD"


# ── Ticker handling ─────────────────────────────────────────────


class TestTickerHandling:
    def _make_connector(self):
        """Create a connector without broker/router for unit testing."""
        connector = KrakenKafkaConnector.__new__(KrakenKafkaConnector)
        connector._products = DEFAULT_PRODUCTS
        connector._internal_products = [
            KRAKEN_TO_INTERNAL.get(p, p.replace("/", "-")) for p in DEFAULT_PRODUCTS
        ]
        connector._latest = {}
        connector._last_published_prices = {}
        connector._last_publish_time = 0.0
        connector._sequence = 0
        connector._min_change_bps = 0.0
        connector._max_silent_seconds = 60.0
        return connector

    def test_handle_ticker_basic(self):
        c = self._make_connector()
        data = {
            "symbol": "DOGE/USD",
            "bid": 0.0961,
            "bid_qty": 10000.0,
            "ask": 0.0962,
            "ask_qty": 5000.0,
            "last": 0.09615,
            "volume": 100000000.0,
            "high": 0.1061,
            "low": 0.0955,
            "timestamp": "2026-02-26T12:00:00Z",
        }
        ticker = c._handle_ticker(data)
        assert ticker.product_id == "DOGE-USD"
        assert ticker.price == "0.09615"
        assert ticker.best_bid == "0.0961"
        assert ticker.best_ask == "0.0962"
        assert ticker.volume_24h == "100000000.0"
        assert ticker.time == "2026-02-26T12:00:00Z"

    def test_handle_ticker_updates_latest(self):
        c = self._make_connector()
        data = {
            "symbol": "PEPE/USD",
            "bid": 0.0000038,
            "bid_qty": 1000000.0,
            "ask": 0.0000039,
            "ask_qty": 1000000.0,
            "last": 0.00000385,
            "volume": 50000000000.0,
            "high": 0.000004,
            "low": 0.0000037,
            "timestamp": "",
        }
        c._handle_ticker(data)
        assert "PEPE-USD" in c._latest
        assert c._latest["PEPE-USD"].price == "3.85e-06"

    def test_handle_ticker_increments_sequence(self):
        c = self._make_connector()
        data = {"symbol": "DOGE/USD", "bid": 0.1, "bid_qty": 1, "ask": 0.1,
                "ask_qty": 1, "last": 0.1, "volume": 1, "high": 0.1, "low": 0.1, "timestamp": ""}
        c._handle_ticker(data)
        c._handle_ticker(data)
        assert c._sequence == 2

    def test_handle_ticker_unknown_symbol(self):
        c = self._make_connector()
        data = {"symbol": "XYZ/USD", "bid": 1, "bid_qty": 1, "ask": 1,
                "ask_qty": 1, "last": 1, "volume": 1, "high": 1, "low": 1, "timestamp": ""}
        ticker = c._handle_ticker(data)
        assert ticker.product_id == "XYZ-USD"  # Falls back to simple replacement


# ── Price change gate ───────────────────────────────────────────


class TestPriceChangeGate:
    def _make_connector(self, bps=15.0, max_silent=60.0):
        c = KrakenKafkaConnector.__new__(KrakenKafkaConnector)
        c._latest = {}
        c._last_published_prices = {}
        c._last_publish_time = 0.0
        c._min_change_bps = bps
        c._max_silent_seconds = max_silent
        return c

    def test_first_publish_always_passes(self):
        c = self._make_connector(bps=15.0)
        c._latest = {"DOGE-USD": TickerMessage(
            product_id="DOGE-USD", price="0.10", best_bid="0.10",
            best_bid_size="1000", best_ask="0.10", best_ask_size="1000",
            side="", last_size="0", open_24h="0", high_24h="0", low_24h="0",
            volume_24h="0", volume_30d="0", trade_id=0, sequence=0, time="",
        )}
        assert c._prices_changed_enough() is True

    def test_no_change_blocked(self):
        c = self._make_connector(bps=15.0)
        c._last_published_prices = {"DOGE-USD": 0.10}
        c._last_publish_time = time.time()
        c._latest = {"DOGE-USD": TickerMessage(
            product_id="DOGE-USD", price="0.10", best_bid="0.10",
            best_bid_size="1000", best_ask="0.10", best_ask_size="1000",
            side="", last_size="0", open_24h="0", high_24h="0", low_24h="0",
            volume_24h="0", volume_30d="0", trade_id=0, sequence=0, time="",
        )}
        assert c._prices_changed_enough() is False

    def test_large_change_passes(self):
        c = self._make_connector(bps=15.0)
        c._last_published_prices = {"DOGE-USD": 0.10}
        c._last_publish_time = time.time()
        c._latest = {"DOGE-USD": TickerMessage(
            product_id="DOGE-USD", price="0.1002", best_bid="0.10",
            best_bid_size="1000", best_ask="0.10", best_ask_size="1000",
            side="", last_size="0", open_24h="0", high_24h="0", low_24h="0",
            volume_24h="0", volume_30d="0", trade_id=0, sequence=0, time="",
        )}
        # 0.1002 vs 0.10 = 20 bps > 15 bps threshold
        assert c._prices_changed_enough() is True

    def test_max_silent_forces_publish(self):
        c = self._make_connector(bps=15.0, max_silent=5.0)
        c._last_published_prices = {"DOGE-USD": 0.10}
        c._last_publish_time = time.time() - 10  # 10s ago, threshold is 5s
        c._latest = {"DOGE-USD": TickerMessage(
            product_id="DOGE-USD", price="0.10", best_bid="0.10",
            best_bid_size="1000", best_ask="0.10", best_ask_size="1000",
            side="", last_size="0", open_24h="0", high_24h="0", low_24h="0",
            volume_24h="0", volume_30d="0", trade_id=0, sequence=0, time="",
        )}
        assert c._prices_changed_enough() is True

    def test_zero_bps_always_passes(self):
        c = self._make_connector(bps=0.0)
        c._last_published_prices = {"DOGE-USD": 0.10}
        c._last_publish_time = time.time()
        c._latest = {"DOGE-USD": TickerMessage(
            product_id="DOGE-USD", price="0.10", best_bid="0.10",
            best_bid_size="1000", best_ask="0.10", best_ask_size="1000",
            side="", last_size="0", open_24h="0", high_24h="0", low_24h="0",
            volume_24h="0", volume_30d="0", trade_id=0, sequence=0, time="",
        )}
        assert c._prices_changed_enough() is True


# ── Candle format conversion ────────────────────────────────────


class TestCandleConversion:
    def test_candle_book_update(self):
        """Simulate Kraken OHLC data converted to Coinbase format."""
        cb = CandleBook()
        # Kraken raw → Coinbase format: [ts, low, high, open, close, volume]
        raw = [
            [1700000000, 0.095, 0.098, 0.096, 0.097, 50000.0],
            [1700000300, 0.096, 0.099, 0.097, 0.098, 60000.0],
            [1700000600, 0.094, 0.097, 0.095, 0.096, 40000.0],
        ]
        # Descending order (Coinbase convention)
        raw.sort(key=lambda x: x[0], reverse=True)
        cb.update_from_api("DOGE-USD", 300, raw)
        candles = cb._candles.get(("DOGE-USD", 300), [])
        assert len(candles) == 3
        # Should be sorted ascending after update_from_api
        assert candles[0].time.timestamp() < candles[-1].time.timestamp()


# ── Live API tests (integration) ────────────────────────────────


class TestKrakenLiveAPI:
    """These tests hit the real Kraken API — skip if offline."""

    @pytest.fixture(autouse=True)
    def _check_connectivity(self):
        import httpx
        try:
            r = httpx.get("https://api.kraken.com/0/public/Time", timeout=5)
            r.raise_for_status()
        except Exception:
            pytest.skip("Kraken API not reachable")

    def test_ticker_endpoint(self):
        import httpx
        r = httpx.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XDGUSD"},
            timeout=10,
        )
        data = r.json()
        assert not data.get("error"), f"Kraken error: {data['error']}"
        assert "XDGUSD" in data["result"]
        td = data["result"]["XDGUSD"]
        assert float(td["c"][0]) > 0  # last trade price

    def test_ohlc_endpoint(self):
        import httpx
        r = httpx.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": "XDGUSD", "interval": 5, "since": int(time.time()) - 3600},
            timeout=10,
        )
        data = r.json()
        assert not data.get("error"), f"Kraken error: {data['error']}"
        for key, candles in data["result"].items():
            if key == "last":
                continue
            assert len(candles) > 0, "No candles returned"
            c = candles[0]
            assert len(c) == 8  # Kraken OHLC has 8 fields

    def test_all_products_exist(self):
        import httpx
        pairs = ",".join(INTERNAL_TO_KRAKEN_REST.values())
        r = httpx.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": pairs},
            timeout=10,
        )
        data = r.json()
        assert not data.get("error"), f"Kraken error: {data['error']}"
        for internal, kraken in INTERNAL_TO_KRAKEN_REST.items():
            assert kraken in data["result"], f"{kraken} ({internal}) not found on Kraken"

    def test_private_api_auth(self):
        """Verify API key auth works (read-only balance check)."""
        import hashlib, hmac, base64, os, urllib.request, urllib.parse
        from dotenv import load_dotenv
        load_dotenv()

        key = os.getenv("KRAKEN_API_KEY")
        secret = os.getenv("KRAKEN_API_SECRET")
        if not key or not secret:
            pytest.skip("KRAKEN_API_KEY/SECRET not set")

        urlpath = "/0/private/Balance"
        nonce = str(int(time.time() * 1000))
        data = {"nonce": nonce}
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest()).decode()

        req = urllib.request.Request("https://api.kraken.com" + urlpath, data=postdata.encode())
        req.add_header("API-Key", key)
        req.add_header("API-Sign", sigdigest)
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        assert not result.get("error"), f"Auth failed: {result['error']}"
        assert "result" in result
