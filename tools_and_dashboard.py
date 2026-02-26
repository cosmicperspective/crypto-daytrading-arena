import argparse
import asyncio
import logging

from dotenv import load_dotenv
from rich.live import Live

from calfkit.broker.broker import BrokerClient
from calfkit.runners.service import NodesService
from agent_registry import AGENT_REGISTRY_TOPIC, AgentMeta
from coinbase_kafka_connector import (
    PRICE_TOPIC,
    TickerMessage,
)
from trading_tools import (
    _persistence,
    calculator,
    execute_trade,
    get_performance_stats,
    get_portfolio,
    get_trade_history,
    price_book,
    view,
)

# Tools & Price Feed — Deploys trading tool workers and subscribes
# to the Kafka price topic published by the connector.
#
# The price subscriber hydrates the shared price book that the trading
# tools read from when executing trades.
#
# Usage:
#     uv run python examples/daytrading_agents_arena/tools_and_dashboard.py
#
# Prerequisites:
#     - Kafka broker running at localhost:9092

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy trading tools, price feed, and dashboard.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        required=True,
        help="Kafka bootstrap servers address",
    )
    return parser.parse_args()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    print("=" * 50)
    print("Tools & Price Feed Deployment")
    print("=" * 50)

    print(f"\nConnecting to Kafka broker at {args.bootstrap_servers}...")
    broker = BrokerClient(bootstrap_servers=args.bootstrap_servers)
    service = NodesService(broker)

    # ── Tool nodes ───────────────────────────────────────────────
    print("\nRegistering trading tool nodes...")
    for tool in (execute_trade, get_portfolio, get_trade_history, get_performance_stats, calculator):
        service.register_node(tool)
        print(f"  - {tool.tool_schema.name} (topic: {tool.subscribed_topic})")

    # ── Price subscriber ─────────────────────────────────────────
    _price_tick_count = 0

    @broker.subscriber(PRICE_TOPIC, group_id="tools-dashboard")
    async def handle_price_update(ticker: TickerMessage) -> None:
        nonlocal _price_tick_count
        price_book.update(ticker.model_dump())
        # Persist price snapshot for dashboard history
        if _persistence is not None:
            import time as _time
            try:
                _persistence.insert_price(
                    ts=_time.time(),
                    product_id=ticker.product_id,
                    price=float(ticker.price),
                    best_bid=float(ticker.best_bid) if ticker.best_bid else None,
                    best_ask=float(ticker.best_ask) if ticker.best_ask else None,
                )
            except Exception:
                pass
            _price_tick_count += 1
            if _price_tick_count % 500 == 0:
                _persistence.cleanup_old_prices()
        view.rerender()

    @broker.subscriber(AGENT_REGISTRY_TOPIC, group_id="tools-dashboard")
    async def handle_agent_meta(meta: AgentMeta) -> None:
        view.update_agent_meta(meta)

    print("\nStarting portfolio dashboard (prices via Kafka)...")

    with Live(view._build_layout(), auto_refresh=False, screen=True) as live:
        view.attach_live(live)
        await service.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTools and price feed stopped.")
