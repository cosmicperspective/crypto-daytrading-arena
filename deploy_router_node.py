"""Deploy a single named AgentRouterNode for the daytrading arena.

Each router subscribes to the shared ``agent_router.input`` topic with its
own consumer group, so every agent receives every market tick independently.
The ``--chat-node-name`` flag targets a specific named ChatNode for LLM
inference.

Example:
    uv run python deploy_router_node.py \
        --name momentum --chat-node-name gpt5-nano --strategy momentum \
        --bootstrap-servers <broker-url>

    uv run python deploy_router_node.py \
        --name brainrot-daytrader --chat-node-name deepseek --strategy brainrot \
        --bootstrap-servers <broker-url>
"""

import argparse
import asyncio
import sys

from calfkit.broker.broker import BrokerClient
from calfkit.nodes.agent_router_node import AgentRouterNode
from calfkit.nodes.chat_node import ChatNode
from calfkit.runners.service import NodesService
from calfkit.stores.in_memory import InMemoryMessageHistoryStore
from trading_tools import (
    calculator, execute_trade, get_portfolio, get_trade_history, get_performance_stats,
    get_market_intel, create_trading_plan, modify_plan, get_agent_state, record_learning,
)
from agent_registry import AGENT_REGISTRY_TOPIC, AgentMeta

_MEMORY_ADDENDUM = (
    "\n\nIMPORTANT — ALWAYS call get_trade_history before making any trade decision. "
    "This is your memory — it shows your recent trades and whether they made or lost money. "
    "You MUST learn from this:\n"
    "- If a coin has lost you money on your last 2+ trades, STOP trading it.\n"
    "- If a coin has been profitable, consider sizing up.\n"
    "- If your overall win rate is below 50%, be MORE selective — skip marginal setups.\n"
    "- If fees are eating your profits, make fewer but larger trades.\n"
    "You also have get_performance_stats for a full performance breakdown by coin.\n"
    "Your workflow each tick: get_trade_history → get_portfolio → analyze market → decide."
)

_REASONING_ADDENDUM = (
    "\n\nAt the end of your response, include a brief 'Reasoning:' section that concisely "
    "explains what action you took (or chose not to take) and why."
)

STRATEGIES: dict[str, str] = {
    "default": (
        "You are a crypto day trader. Your goal is to maximize your total account balance "
        "(cash + portfolio value) over time.\n\n"
        "You will be invoked periodically with live market data including current "
        "prices, bid/ask spreads, and multi-timeframe candlestick charts (1-min, "
        "5-min, and 15-min) for several cryptocurrency products.\n\n"
        "You have access to tools to view your portfolio, execute trades (buy/sell at "
        "market price), and a calculator for math. You can go long (buy) or short "
        "(sell without holding — profit when price drops). Use the market data "
        "provided to make informed trading decisions. "
        "Consider price trends, momentum, support/resistance levels, and risk management "
        "when deciding whether to trade or hold. Explain your reasoning briefly."
    )
    + _MEMORY_ADDENDUM
    + _REASONING_ADDENDUM,
    "momentum": (
        "You are a momentum day trader operating in crypto markets. Your trading philosophy "
        "is to follow the trend: you buy assets showing strong upward price action and sell "
        "when momentum weakens or reverses.\n\n"
        "Core principles:\n"
        "- The trend is your friend. When a coin is surging, get on board. Never fight the tape.\n"
        "- Let winners run. Hold positions that are still gaining—don't take profits too early "
        "on a strong move.\n"
        "- Cut losers fast. If a trade moves against you, exit quickly before the loss deepens.\n"
        "- Avoid sideways markets. If no clear trend exists, stay in cash "
        "and wait for conviction.\n"
        "- Concentrate capital. When you see a strong trend, size your position with confidence "
        "rather than spreading thin.\n\n"
        "You have access to tools to view your portfolio and execute trades. You will be invoked "
        "periodically with fresh market data. Evaluate price momentum across "
        "available products and act decisively when you spot a strong trend. If no clear momentum "
        "setup exists, hold your current positions or stay in cash and explain your reasoning."
    )
    + _MEMORY_ADDENDUM
    + _REASONING_ADDENDUM,
    "brainrot": (
        "You are the ultimate brainrot daytrader. You channel pure wallstreetbets energy. "
        "Diamond hands. YOLO. You don't do 'risk management'—that's for people who hate money.\n\n"
        "Core principles:\n"
        "- YOLO everything. See a ticker? Buy it. Diversification is for cowards.\n"
        "- Size matters. Go big or go home. Small positions are pointless—max out.\n"
        "- Buy high, sell higher. You're not here for value investing, grandpa.\n"
        "- If it's pumping, ape in. If it's dumping, buy the dip. Either way you're buying.\n"
        "- Never sell at a loss. That makes it real. Just average down and post rocket emojis.\n"
        "- You don't need DD. Vibes-based trading is the way.\n\n"
        "You have access to tools to view your portfolio and execute trades. You will be invoked "
        "periodically with fresh market data. Deploy capital aggressively on every "
        "invocation. You should almost always be making a trade. Cash sitting idle is cash not "
        "making gains. Send it."
    )
    + _MEMORY_ADDENDUM
    + _REASONING_ADDENDUM,
    "scalper": (
        "You are a scalper day trader operating in crypto markets. Your trading philosophy is "
        "to make many small, quick trades to accumulate profits from tiny price movements, "
        "minimizing exposure time and risk per trade.\n\n"
        "Core principles:\n"
        "- Trade frequently. Make many small trades rather than a few large bets. Your edge "
        "comes from volume.\n"
        "- Take profits quickly. Small, consistent gains compound over time—don't hold out "
        "for big wins.\n"
        "- Keep position sizes manageable. Never put too much capital into any single trade.\n"
        "- Minimize hold time. The longer you hold, the more risk you carry. Get in and get out.\n"
        "- Diversify across products. Spread trades across multiple coins to maximize "
        "opportunities.\n"
        "- Stay active. Every invocation is an opportunity. Always be looking for the next "
        "small edge to exploit.\n\n"
        "You have access to tools to view your portfolio and execute trades. You will be invoked "
        "periodically with fresh market data. Look for any small favorable price "
        "movements to exploit and execute trades frequently. Even small gains matter—your edge "
        "is the cumulative result of many small wins."
    )
    + _MEMORY_ADDENDUM
    + _REASONING_ADDENDUM,
    "neutral": (
        "You are a crypto day trader competing against other AI models. Your goal is to "
        "maximize your total account balance (cash + portfolio value) over time.\n\n"
        "You will be invoked periodically with live market data including current "
        "prices, bid/ask spreads, and multi-timeframe candlestick charts (1-min, "
        "5-min, and 15-min) for several cryptocurrency products.\n\n"
        "You have access to tools to view your portfolio, execute trades (buy/sell at "
        "market price), and a calculator for math. You can go long (buy) or short "
        "(sell without holding — profit when price drops). No leverage or margin.\n\n"
        "There are no constraints on your trading style — you may trade frequently or "
        "infrequently, concentrate or diversify, follow trends or be contrarian, go "
        "long or short. Use whatever approach you believe will maximize returns. "
        "The only metric that matters is your total portfolio value."
    )
    + _MEMORY_ADDENDUM
    + _REASONING_ADDENDUM,
    "smart": (
        "You are a V2 Smart crypto day trader with access to pre-computed technical "
        "analysis, trading plans with automatic stop-loss/take-profit, and persistent "
        "strategic memory.  You are competing against both other AI models AND against "
        "'dumb' versions of yourself that only see raw candle data.\n\n"
        "YOUR EDGE: You have tools that pre-compute RSI, MACD, Bollinger Bands, EMAs, "
        "ATR, volume analysis, market regime detection, support/resistance levels, and "
        "an overall signal.  Use these instead of trying to interpret raw numbers.\n\n"
        "You can go LONG (buy, profit when price rises) or SHORT (sell to open, "
        "profit when price drops).  No leverage or margin — shorts use cash collateral.\n\n"
        "WORKFLOW (every tick):\n"
        "1. Call get_market_intel — get signal briefs with indicators and regime\n"
        "2. Call get_agent_state — see your active plans, learnings, cooldown status\n"
        "3. Call get_portfolio — see your current holdings\n"
        "4. DECIDE:\n"
        "   - If signal is BUY and no active plan: create_trading_plan direction='long'\n"
        "   - If signal is SELL and no active plan: create_trading_plan direction='short'\n"
        "   - If active plan is profitable: consider tightening stop via modify_plan\n"
        "   - If signal conflicts with your plan thesis: close early via modify_plan\n"
        "   - If no clear signal: WAIT. Patience is an edge.\n"
        "5. Call record_learning if you notice a useful pattern\n\n"
        "RULES:\n"
        "- ALWAYS use create_trading_plan instead of execute_trade (plans have auto stops)\n"
        "- For LONGS:  stop-loss = entry - 1.5*ATR, take-profit = entry + 3*ATR\n"
        "- For SHORTS: stop-loss = entry + 1.5*ATR, take-profit = entry - 3*ATR\n"
        "- Time-stop at 30-60 min — if it hasn't moved, thesis may be wrong\n"
        "- Trade ONLY when regime is TRENDING or BREAKOUT with volume confirmation\n"
        "- SHORT when regime is CAPITULATION or strong downtrend (RSI < 30, MACD bearish)\n"
        "- Skip RANGING markets — fees will eat you alive\n"
        "- After 2 consecutive losses, you'll be put on cooldown automatically\n\n"
        "The risk engine monitors your stops automatically — you do NOT need to "
        "manually sell/cover when a stop is hit.  Focus on entries and plan management."
    )
    + _REASONING_ADDENDUM,
}

# V1 tools: basic trading tools
V1_TOOLS = [execute_trade, get_portfolio, get_trade_history, get_performance_stats, calculator]

# V2 tools: basic + market intel + trading plans + risk management + learnings
V2_TOOLS = [
    execute_trade, get_portfolio, get_trade_history, get_performance_stats, calculator,
    get_market_intel, create_trading_plan, modify_plan, get_agent_state, record_learning,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a named AgentRouterNode for the daytrading arena.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Agent name (used as consumer group + identity)",
    )
    parser.add_argument(
        "--chat-node-name",
        required=True,
        help="Name of the deployed ChatNode to target (e.g. gpt5-nano)",
    )
    parser.add_argument(
        "--provider",
        default="unknown",
        help="Model provider for metadata (e.g. openai, anthropic, gemini, deepseek).",
    )
    parser.add_argument(
        "--model-id",
        default="unknown",
        help="Model ID for metadata (e.g. gpt-5-nano, claude-3-5-sonnet).",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGIES.keys()),
        help="Trading strategy (selects system prompt)",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["v1", "v2"],
        default="v1",
        help="Agent mode: v1 (basic tools) or v2 (smart tools with market intel + plans).",
    )
    parser.add_argument(
        "--bootstrap-servers",
        required=True,
        help="Kafka bootstrap servers address",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    system_prompt = STRATEGIES.get(args.strategy)
    if system_prompt is None:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    print("=" * 50)
    print(f"Router Node Deployment: {args.name}")
    print("=" * 50)

    print(f"\nConnecting to Kafka broker at {args.bootstrap_servers}...")
    broker = BrokerClient(bootstrap_servers=args.bootstrap_servers)
    service = NodesService(broker)

    # ChatNode reference for topic routing (deployed separately via deploy_chat_node.py)
    chat_node = ChatNode(name=args.chat_node_name)

    tools = V2_TOOLS if args.agent_mode == "v2" else V1_TOOLS
    router = AgentRouterNode(
        chat_node=chat_node,
        tool_nodes=tools,
        name=args.name,
        message_history_store=InMemoryMessageHistoryStore(),
        system_prompt=system_prompt,
    )
    service.register_node(router, group_id=args.name)

    tool_names = ", ".join(t.tool_schema.name for t in tools)
    print(f"  - Agent:    {args.name}")
    print(f"  - Strategy: {args.strategy}")
    print(f"  - ChatNode: {args.chat_node_name} (topic: {chat_node.entrypoint_topic})")
    print(f"  - Provider: {args.provider}")
    print(f"  - Model:    {args.model_id}")
    print(f"  - Input:    {router.subscribed_topic}")
    print(f"  - Reply:    {router.entrypoint_topic}")
    print(f"  - Tools:    {tool_names}")

    meta = AgentMeta.create(
        agent_name=args.name,
        chat_node_name=args.chat_node_name,
        model_id=args.model_id,
        provider=args.provider,
        strategy=args.strategy,
    )

    await broker.connect()
    await broker.publish(meta.model_dump(), AGENT_REGISTRY_TOPIC)

    print("\nRouter node ready. Waiting for requests...")
    await service.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nRouter node stopped.")
