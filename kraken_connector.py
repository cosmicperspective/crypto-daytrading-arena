"""
Kraken-to-Kafka connector that streams real-time market data from
the Kraken WebSocket v2 API and invokes an AgentRouterNode
for each price update (fire-and-forget).

Uses the ticker channel for real-time price updates.

Usage:
    uv run python kraken_connector.py
    uv run python kraken_connector.py --products DOGE/USD PEPE/USD SHIB/USD
    uv run python kraken_connector.py --interval 2 --min-change-bps 15

Prerequisites:
    - Kafka broker running (default: localhost:9092)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import httpx
import websockets
from pydantic import BaseModel

from calfkit.broker.broker import BrokerClient
from calfkit.nodes.agent_router_node import AgentRouterNode
from calfkit.runners.service_client import RouterServiceClient
from coinbase_consumer import CandleBook, Candle, PriceBook, Timeframe, TIMEFRAMES

logger = logging.getLogger(__name__)

KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
KRAKEN_REST_BASE = "https://api.kraken.com"

# Kraken symbol format: "DOGE/USD" for WS, "XDGUSD" for REST
# We normalise to Coinbase-style "DOGE-USD" for internal use.

DEFAULT_PRODUCTS = [
    "DOGE/USD",
    "PEPE/USD",
    "SOL/USD",
    "SUI/USD",
    "FARTCOIN/USD",
]

# Map from Kraken WS symbol → internal product ID (Coinbase-style)
# and from internal → Kraken REST pair name
KRAKEN_TO_INTERNAL = {
    "DOGE/USD": "DOGE-USD",
    "PEPE/USD": "PEPE-USD",
    "SOL/USD": "SOL-USD",
    "SUI/USD": "SUI-USD",
    "FARTCOIN/USD": "FARTCOIN-USD",
    # Legacy / alternate coins
    "SHIB/USD": "SHIB-USD",
    "BONK/USD": "BONK-USD",
    "WIF/USD": "WIF-USD",
    "FLOKI/USD": "FLOKI-USD",
}

INTERNAL_TO_KRAKEN_REST = {
    "DOGE-USD": "XDGUSD",
    "PEPE-USD": "PEPEUSD",
    "SOL-USD": "SOLUSD",
    "SUI-USD": "SUIUSD",
    "FARTCOIN-USD": "FARTCOINUSD",
    # Legacy / alternate coins
    "SHIB-USD": "SHIBUSD",
    "BONK-USD": "BONKUSD",
    "WIF-USD": "WIFUSD",
    "FLOKI-USD": "FLOKIUSD",
}

PRICE_TOPIC = "market_data.prices"

RECONNECT_DELAY_SECONDS = 3

PAUSE_FILE = Path(__file__).parent / "arena.pause"

# Kraken OHLC intervals (minutes) → Coinbase-style granularity (seconds)
KRAKEN_INTERVAL_MAP = {
    15: 900,   # 15-min candles
    5: 300,    # 5-min candles
    1: 60,     # 1-min candles
}


class TickerMessage(BaseModel):
    """Ticker message published to Kafka — same schema as Coinbase connector."""

    product_id: str
    price: str
    best_bid: str
    best_bid_size: str
    best_ask: str
    best_ask_size: str
    side: str
    last_size: str
    open_24h: str
    high_24h: str
    low_24h: str
    volume_24h: str
    volume_30d: str
    trade_id: int
    sequence: int
    time: str


class KrakenKafkaConnector:
    """Streams Kraken ticker data to an AgentRouterNode.

    Connects to the Kraken WebSocket v2 ticker channel and invokes the
    configured AgentRouterNode with each price update using fire-and-forget
    publishes via RouterServiceClient.
    """

    def __init__(
        self,
        broker: BrokerClient,
        router_node: AgentRouterNode,
        products: list[str],
        min_publish_interval: float = 0.0,
        candle_book: CandleBook | None = None,
        min_change_bps: float = 0.0,
        max_silent_seconds: float = 60.0,
    ) -> None:
        self._broker = broker
        self._client = RouterServiceClient(broker, router_node)
        self._products = products
        self._internal_products = [KRAKEN_TO_INTERNAL.get(p, p.replace("/", "-")) for p in products]
        self._min_interval = min_publish_interval
        self._running = True
        self._candle_book = candle_book
        self._min_change_bps = min_change_bps
        self._max_silent_seconds = max_silent_seconds

        self._latest: dict[str, TickerMessage] = {}
        self._last_published_prices: dict[str, float] = {}
        self._last_publish_time: float = 0.0
        self._sequence: int = 0

    async def start(self) -> None:
        await self._broker.start()
        logger.info("Kafka broker connected")

        try:
            while self._running:
                try:
                    await self._consume_and_publish()
                except websockets.ConnectionClosed:
                    if not self._running:
                        break
                    logger.warning(
                        "WebSocket connection lost. Reconnecting in %ds...",
                        RECONNECT_DELAY_SECONDS,
                    )
                    await asyncio.sleep(RECONNECT_DELAY_SECONDS)
                except Exception:
                    if not self._running:
                        break
                    logger.exception(
                        "Unexpected error. Reconnecting in %ds...",
                        RECONNECT_DELAY_SECONDS,
                    )
                    await asyncio.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            await self._broker.close()
            logger.info("Kafka broker closed")

    def stop(self) -> None:
        self._running = False

    def _prices_changed_enough(self) -> bool:
        if self._min_change_bps <= 0.0:
            return True
        if not self._last_published_prices:
            return True
        if self._last_publish_time > 0 and (
            time.time() - self._last_publish_time > self._max_silent_seconds
        ):
            return True
        for pid, ticker in self._latest.items():
            try:
                current = float(ticker.price)
            except (ValueError, TypeError):
                continue
            prev = self._last_published_prices.get(pid)
            if prev is None or prev == 0:
                return True
            change_bps = abs(current - prev) / prev * 10_000
            if change_bps >= self._min_change_bps:
                return True
        return False

    async def _publish_latest(self) -> None:
        if not self._latest:
            return

        if PAUSE_FILE.exists():
            logger.debug("Arena paused — skipping publish")
            return

        if not self._prices_changed_enough():
            return

        batch = list(self._latest.values())
        _exclude = {
            "best_bid_size",
            "best_ask_size",
            "last_size",
            "side",
            "trade_id",
            "sequence",
            "open_24h",
            "high_24h",
            "low_24h",
            "volume_24h",
            "volume_30d",
            "time",
        }
        batch_json = json.dumps([t.model_dump(exclude=_exclude) for t in batch])

        prompt_parts = [
            "Here is the latest ticker information. You should view your "
            "portfolio first before making any decisions to trade.\n"
            "price = last traded price, best_bid = price you sell at, "
            "best_ask = price you buy at.\n\n"
            f"{batch_json}",
        ]

        if self._candle_book is not None and self._candle_book.has_data():
            prompt_parts.append(
                "\n## Price History (OHLCV candlesticks)\n"
                "Below are candlesticks at three granularities — coarser for "
                "broader trend context, finer for recent price action.\n\n"
                f"{self._candle_book.format_prompt(self._internal_products)}"
            )

        await self._client.invoke(
            user_prompt="\n".join(prompt_parts),
            deps={"invoked_at": time.time()},
        )

        for t in batch:
            try:
                self._last_published_prices[t.product_id] = float(t.price)
            except (ValueError, TypeError):
                pass
        self._last_publish_time = time.time()

        summary = ", ".join(f"{t.product_id} @ ${t.price}" for t in batch)
        logger.info(
            "Published batch of %d ticker(s) to router: %s",
            len(batch),
            summary,
        )

    async def _periodic_publish(self) -> None:
        interval = max(self._min_interval, 1.0)
        while self._running:
            await asyncio.sleep(interval)
            await self._publish_latest()

    def _handle_ticker(self, data: dict) -> None:
        """Convert a Kraken WS v2 ticker message to our internal format."""
        symbol = data.get("symbol", "")
        product_id = KRAKEN_TO_INTERNAL.get(symbol, symbol.replace("/", "-"))
        self._sequence += 1

        ticker = TickerMessage(
            product_id=product_id,
            price=str(data.get("last", 0)),
            best_bid=str(data.get("bid", 0)),
            best_bid_size=str(data.get("bid_qty", 0)),
            best_ask=str(data.get("ask", 0)),
            best_ask_size=str(data.get("ask_qty", 0)),
            side="",
            last_size="0",
            open_24h="0",
            high_24h=str(data.get("high", 0)),
            low_24h=str(data.get("low", 0)),
            volume_24h=str(data.get("volume", 0)),
            volume_30d="0",
            trade_id=0,
            sequence=self._sequence,
            time=data.get("timestamp", ""),
        )
        self._latest[product_id] = ticker
        return ticker

    async def _consume_and_publish(self) -> None:
        self._latest.clear()

        async with websockets.connect(KRAKEN_WS_URL) as ws:
            # Subscribe to ticker channel
            await ws.send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "ticker",
                            "symbol": self._products,
                        },
                    }
                )
            )
            logger.info(
                "Subscribed to %d products on Kraken ticker: %s",
                len(self._products),
                ", ".join(self._products),
            )

            flush_task = asyncio.create_task(self._periodic_publish())

            candle_task: asyncio.Task | None = None
            if self._candle_book is not None:
                candle_task = asyncio.create_task(
                    poll_kraken_candles(
                        products=self._internal_products,
                        price_book=PriceBook(),
                        candle_book=self._candle_book,
                        interval=60.0,
                    )
                )

            try:
                async for raw in ws:
                    if not self._running:
                        break

                    msg = json.loads(raw)
                    channel = msg.get("channel")
                    if channel != "ticker":
                        continue

                    # Kraken sends data as a list of ticker objects
                    for item in msg.get("data", []):
                        ticker = self._handle_ticker(item)
                        await self._broker.publish(ticker, PRICE_TOPIC)
            finally:
                flush_task.cancel()
                try:
                    await flush_task
                except asyncio.CancelledError:
                    pass
                if candle_task is not None:
                    candle_task.cancel()
                    try:
                        await candle_task
                    except asyncio.CancelledError:
                        pass


async def poll_kraken_candles(
    products: list[str],
    price_book: PriceBook,
    candle_book: CandleBook,
    interval: float = 60.0,
    timeframes: list[Timeframe] | None = None,
) -> None:
    """Poll Kraken REST API for multi-timeframe candles.

    Kraken OHLC response format:
        [timestamp, open, high, low, close, vwap, volume, count]
    We convert to Coinbase format:
        [timestamp, low, high, open, close, volume]
    """
    tfs = timeframes if timeframes is not None else TIMEFRAMES
    async with httpx.AsyncClient(base_url=KRAKEN_REST_BASE, timeout=15.0) as client:
        while True:
            now = int(time.time())

            for product_id in products:
                kraken_pair = INTERNAL_TO_KRAKEN_REST.get(product_id)
                if not kraken_pair:
                    continue

                try:
                    for tf in tfs:
                        kraken_interval = tf.granularity // 60  # Kraken uses minutes
                        since = now - tf.start_minutes_ago * 60

                        resp = await client.get(
                            "/0/public/OHLC",
                            params={
                                "pair": kraken_pair,
                                "interval": kraken_interval,
                                "since": since,
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        if data.get("error"):
                            logger.warning("Kraken OHLC error for %s: %s", product_id, data["error"])
                            continue

                        # Extract candles (skip 'last' key)
                        raw_candles = []
                        for key, value in data.get("result", {}).items():
                            if key == "last":
                                continue
                            # Kraken format: [ts, open, high, low, close, vwap, volume, count]
                            # Convert to Coinbase format: [ts, low, high, open, close, volume]
                            for c in value:
                                ts = c[0]
                                # Filter by time window
                                end_ts = now - tf.end_minutes_ago * 60
                                if ts > end_ts:
                                    continue
                                raw_candles.append([
                                    ts,
                                    float(c[3]),   # low
                                    float(c[2]),   # high
                                    float(c[1]),   # open
                                    float(c[4]),   # close
                                    float(c[6]),   # volume
                                ])

                        # Sort descending (Coinbase convention) for update_from_api
                        raw_candles.sort(key=lambda x: x[0], reverse=True)
                        if raw_candles:
                            candle_book.update_from_api(product_id, tf.granularity, raw_candles)

                    # Fetch current ticker
                    resp = await client.get(
                        "/0/public/Ticker",
                        params={"pair": kraken_pair},
                    )
                    resp.raise_for_status()
                    ticker_data = resp.json()
                    for key, td in ticker_data.get("result", {}).items():
                        price_book.update({
                            "product_id": product_id,
                            "price": td["c"][0],       # last trade price
                            "best_bid": td["b"][0],     # bid
                            "best_bid_size": td["b"][2], # bid lot volume
                            "best_ask": td["a"][0],     # ask
                            "best_ask_size": td["a"][2], # ask lot volume
                            "side": "",
                            "last_size": td["c"][1],    # last trade lot size
                            "volume_24h": td["v"][1],   # 24h volume
                            "time": "",
                        })

                except Exception:
                    logger.exception("Kraken REST poll failed for %s", product_id)

            await asyncio.sleep(interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Kraken market data to a Kafka topic.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP_SERVERS or localhost:9092).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=DEFAULT_PRODUCTS,
        help="Kraken product symbols (default: %(default)s).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Minimum seconds between publishes (default: 0).",
    )
    parser.add_argument(
        "--min-change-bps",
        type=float,
        default=0.0,
        help="Only publish when price moves this many bps (default: 0).",
    )
    parser.add_argument(
        "--max-silent-seconds",
        type=float,
        default=60.0,
        help="Force publish after this many seconds of silence (default: 60).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace, router_node: AgentRouterNode) -> None:
    broker = BrokerClient(bootstrap_servers=args.bootstrap_servers)
    connector = KrakenKafkaConnector(
        broker=broker,
        router_node=router_node,
        products=args.products,
        min_publish_interval=args.interval,
        min_change_bps=args.min_change_bps,
        max_silent_seconds=args.max_silent_seconds,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, connector.stop)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    logger.info("Starting Kraken -> Kafka connector")
    logger.info("  Router topic:  %s", router_node.subscribed_topic)
    logger.info("  Broker:        %s", args.bootstrap_servers)
    logger.info("  Products:      %s", ", ".join(args.products))
    logger.info("  Min interval:  %ss", args.interval)
    logger.info("  Change gate:   %s bps", args.min_change_bps)
    logger.info("  Max silent:    %ss", args.max_silent_seconds)

    await connector.start()


def main() -> None:
    from calfkit.nodes.chat_node import ChatNode
    from calfkit.stores.in_memory import InMemoryMessageHistoryStore

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    router_node = AgentRouterNode(
        chat_node=ChatNode(),
        tool_nodes=[],
        message_history_store=InMemoryMessageHistoryStore(),
        system_prompt="You are a market data consumer.",
    )
    asyncio.run(run(args, router_node))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
