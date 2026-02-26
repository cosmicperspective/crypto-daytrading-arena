import argparse
import asyncio
import logging

from dotenv import load_dotenv
from rich.live import Live

from calfkit.broker.broker import BrokerClient
from calfkit.runners.service import NodesService
from agent_registry import AGENT_REGISTRY_TOPIC, AgentMeta
from coinbase_kafka_connector import (
    DEFAULT_PRODUCTS,
    PRICE_TOPIC,
    TickerMessage,
)
from coinbase_consumer import CandleBook, PriceBook, poll_rest
from market_intel import MarketIntelService, INDICATOR_TIMEFRAMES
from risk_engine import RiskEngine
from trading_tools import (
    _execute_trade,
    _persistence,
    calculator,
    create_trading_plan,
    execute_trade,
    get_agent_state,
    get_market_intel,
    get_performance_stats,
    get_portfolio,
    get_trade_history,
    init_v2_services,
    modify_plan,
    price_book,
    record_learning,
    view,
)

# Tools & Price Feed — Deploys trading tool workers, V2 market intelligence,
# risk engine, and subscribes to the Kafka price topic.
#
# Usage:
#     uv run python tools_and_dashboard.py --bootstrap-servers localhost:9092

load_dotenv()

logger = logging.getLogger(__name__)


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


# ── V2 background tasks ─────────────────────────────────────────


async def _intel_poll_loop(
    intel_service: MarketIntelService,
    products: list[str],
    price_book_ref: PriceBook,
    interval: float = 60.0,
) -> None:
    """Periodically fetch candles and recompute indicators."""
    logger.info("V2 Intel polling started (interval=%ss, products=%s)", interval, products)
    while True:
        try:
            await poll_rest(
                products=products,
                price_book=price_book_ref,
                candle_book=intel_service.candle_book,
                interval=interval,
                timeframes=INDICATOR_TIMEFRAMES,
            )
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Intel poll error, retrying in %ss", interval)
            await asyncio.sleep(interval)


async def _intel_compute_loop(
    intel_service: MarketIntelService,
    products: list[str],
    interval: float = 15.0,
) -> None:
    """Recompute indicators periodically (separate from candle fetching)."""
    logger.info("V2 Intel compute loop started (interval=%ss)", interval)
    while True:
        try:
            intel_service.update_all(products)
        except Exception:
            logger.exception("Intel compute error")
        await asyncio.sleep(interval)


async def _risk_engine_loop(
    risk_engine: RiskEngine,
    interval: float = 5.0,
) -> None:
    """Check active plans against live prices every *interval* seconds."""
    logger.info("V2 Risk engine loop started (interval=%ss)", interval)
    while True:
        try:
            triggered = risk_engine.check_plans()
            if triggered:
                for p in triggered:
                    logger.info(
                        "RISK_EXIT | %s | %s %s reason=%s pnl=%.4f",
                        p.agent_id, p.product_id, p.plan_id, p.exit_reason, p.pnl,
                    )
                # Persist triggered plans
                if _persistence is not None:
                    for p in triggered:
                        _persistence.save_plan(p)
                view.rerender()
        except Exception:
            logger.exception("Risk engine check error")
        await asyncio.sleep(interval)


# ── Main ─────────────────────────────────────────────────────────


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    print("=" * 55)
    print("  Tools & Price Feed Deployment (V1 + V2)")
    print("=" * 55)

    print(f"\nConnecting to Kafka broker at {args.bootstrap_servers}...")
    broker = BrokerClient(bootstrap_servers=args.bootstrap_servers)
    service = NodesService(broker)

    # ── V1 tool nodes ────────────────────────────────────────────
    print("\nRegistering V1 tool nodes...")
    for tool in (execute_trade, get_portfolio, get_trade_history, get_performance_stats, calculator):
        service.register_node(tool)
        print(f"  - {tool.tool_schema.name}")

    # ── V2 services + tool nodes ─────────────────────────────────
    print("\nInitialising V2 services...")
    intel_candle_book = CandleBook()
    intel_service = MarketIntelService(intel_candle_book, price_book)
    risk_engine = RiskEngine(
        price_book=price_book,
        execute_sell_fn=_execute_trade,
    )
    init_v2_services(intel_service, risk_engine)

    print("Registering V2 tool nodes...")
    for tool in (get_market_intel, create_trading_plan, modify_plan, get_agent_state, record_learning):
        service.register_node(tool)
        print(f"  - {tool.tool_schema.name}")

    # ── Price subscriber ─────────────────────────────────────────
    _price_tick_count = 0

    @broker.subscriber(PRICE_TOPIC, group_id="tools-dashboard")
    async def handle_price_update(ticker: TickerMessage) -> None:
        nonlocal _price_tick_count
        price_book.update(ticker.model_dump())
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

    # ── Start V2 background tasks ────────────────────────────────
    print("\nStarting V2 background services...")
    intel_poll_task = asyncio.create_task(
        _intel_poll_loop(intel_service, DEFAULT_PRODUCTS, price_book, interval=60.0)
    )
    intel_compute_task = asyncio.create_task(
        _intel_compute_loop(intel_service, DEFAULT_PRODUCTS, interval=15.0)
    )
    risk_task = asyncio.create_task(
        _risk_engine_loop(risk_engine, interval=5.0)
    )
    print("  - Intel candle polling (60s)")
    print("  - Intel indicator compute (15s)")
    print("  - Risk engine plan monitor (5s)")

    print("\nStarting portfolio dashboard (prices via Kafka)...")

    with Live(view._build_layout(), auto_refresh=False, screen=True) as live:
        view.attach_live(live)
        try:
            await service.run()
        finally:
            intel_poll_task.cancel()
            intel_compute_task.cancel()
            risk_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTools and price feed stopped.")
