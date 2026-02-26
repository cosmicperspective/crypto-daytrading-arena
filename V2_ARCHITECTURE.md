# V2 Architecture: Crypto Daytrading Arena

## Core Insight

V1 asks LLMs to do what they're worst at — pattern recognition on raw numerical time series.
V2 flips this: move computation to Python, move judgment to LLMs.

**Stop asking LLMs to be calculators. Start asking them to be strategists.**

---

## Three-Layer Architecture

```
         ┌───────────────────────────────┐
         │   Layer 1: Market Intel       │  (Python, no LLM)
         │   Indicators, Regime, Signals │
         └──────────────┬────────────────┘
                        │ narrative signal briefs
         ┌──────────────▼────────────────┐
         │   Layer 2: Agent Brain        │  (LLM + persistent state)
         │   Plans, Learnings, Strategy  │
         └──────────────┬────────────────┘
                        │ trade intents
         ┌──────────────▼────────────────┐
         │   Layer 3: Risk Engine        │  (Python, no LLM)
         │   Stops, Limits, Execution    │
         └───────────────────────────────┘
```

---

## Layer 1: Market Intelligence Service

Pre-compute everything in Python and deliver narrative briefings instead of raw CSV candles.

### Technical Indicators (pandas-ta, ~10ms compute)
- RSI (14-period) with overbought/oversold flags
- MACD (12, 26, 9) with signal crossover detection
- Bollinger Bands (20, 2) with band touch/penetration
- EMA crossovers (9/21 short-term, 21/50 medium-term)
- Volume: current vs 20-period average, trend direction
- ATR (14-period) for position sizing and stop placement
- VWAP for current session
- Price velocity: rate of change over 1m, 5m, 15m, 1h

### Market Regime Detector (state machine, no ML)
- **TRENDING_UP**: Price > 21 EMA, 21 EMA > 50 EMA, ADX > 25
- **TRENDING_DOWN**: Price < 21 EMA, 21 EMA < 50 EMA, ADX > 25
- **RANGING**: ADX < 20, price within Bollinger Bands
- **VOLATILE**: ATR expanding, multiple band touches
- **BREAKOUT**: Price crossing Bollinger Band with volume expansion
- **CAPITULATION**: Sharp drop with volume spike

### Signal Brief (what agents receive instead of raw CSV)

**V1 (raw, ~800 tokens, LLM must parse):**
```
ts,open,high,low,close,volume
1772126100,0.0966,0.0968,0.0965,0.0967,234521
...
```

**V2 (narrative, ~200 tokens, immediately actionable):**
```
DOGE-USD Signal Brief:
- Regime: TRENDING_UP (strong, 45 min)
- Momentum: MACD bullish crossover 8 min ago, histogram expanding
- RSI: 62 (neutral, room to run)
- Volume: 1.8x average
- Key levels: Support $0.1823 (21 EMA), Resistance $0.1891 (swing high)
- ATR: $0.0034 (use for stop distance)
- Signal: MODERATE BUY — trend continuation likely
```

### Sentiment Layer
- **CoinGecko API**: Trending coins, community scores (every 5 min)
- **Fear & Greed Index**: Overall crypto sentiment (every hour)
- **Funding rates**: From Binance perpetual futures — overheated/capitulation signals (free)
- **Social mention velocity**: Reddit/X mention spikes precede price moves by 10-60 min

---

## Layer 2: Persistent Agent Brain

### The Problem
V1 agents buy DOGE at tick N. At tick N+1, they're brand new agents that must rediscover they hold DOGE and why. The trade *thesis* is lost.

### Trading Plans (new concept)

New tool — `create_trading_plan`:
```python
create_trading_plan(
    product_id="DOGE-USD",
    direction="long",
    quantity=500,
    stop_loss_price=0.0953,      # REQUIRED
    take_profit_price=0.0995,    # REQUIRED
    time_stop_minutes=45,        # Max hold time
    thesis="DOGE trending up with expanding volume",
    confidence=0.7
)
```

Executes entry AND registers a plan. Risk engine monitors stops automatically.

### Persistent State (loaded into prompt each tick)

```
agent_state:
  active_plan:
    product_id, direction, entry_price, stop_loss, take_profit
    thesis: "why I entered this trade"
    status: active | stopped_out | take_profit | closed

  learnings: [
    "My momentum trades win 65%. Mean-reversion only 30% — focus on momentum."
    "PEPE mean-reverts after RSI > 75 in ranging markets"
  ]

  product_bias:
    DOGE-USD: bullish (reason: ...)
    SHIB-USD: avoid (3 consecutive losses)

  completed_plans: [last 20 with P&L, regime at entry/exit]
```

### New Tools
- `create_trading_plan` — entry + plan registration
- `modify_plan` — tighten stop, take partial, close, extend time
- `record_learning` — persist strategic insights across ticks
- `get_market_intel` — pre-computed signal brief from Layer 1

### Prompt Shift

**V1**: "Here are numbers. Decide."
**V2**: "You're managing a long DOGE position with a clear thesis. Given this update, hold/tighten/partial/close?"

---

## Layer 3: Risk & Execution Engine

Deterministic Python service. LLMs cannot be trusted to enforce their own stop-losses.

### Automated Stop/TP Monitoring
- Subscribes to price ticks, checks all active plans
- Executes exits automatically when triggered
- No LLM in the loop

### Portfolio-Level Limits
- Max 30% of portfolio in one position
- Must keep 20% cash reserve
- Daily loss limit: halt for 30 min if down 5%
- Cooldown: 5 min after 2 consecutive losses
- Fee gate: reject trades where round-trip fees > 50% of ATR

---

## Backtesting via Replay

`replay_connector.py` reads from price_history table and publishes to Kafka at 10x/100x speed. Everything downstream is identical — same agents, same tools, same risk engine.

---

## Ensemble Approaches

- **Voting**: 3 agents, execute when 2/3 agree
- **Specialists**: Assign 1-2 coins per agent for deeper focus
- **Contrarian mirror**: One agent that challenges the consensus

---

## What NOT To Do

- Don't add more candle timeframes — pre-compute indicators instead
- Don't add order book depth — meme coin books are spoofed/noisy
- Don't go faster — 30s is too fast, move to 60-120s and spend budget on better reasoning
- Don't add leverage — amplifies noise
- Stronger models > more ticks

---

## Implementation Priority

| Phase | Change | Impact | Effort |
|-------|--------|--------|--------|
| **1** | Technical indicator engine + signal briefs | Highest | Medium |
| **1** | Mandatory stop-loss/take-profit plans | High | Medium |
| **1** | Persistent agent state | High | Medium |
| **2** | Sentiment data (CoinGecko + Fear/Greed) | Moderate | Low |
| **2** | Backtesting replay connector | High (long-term) | Medium |
| **2** | Portfolio risk limits | Moderate | Low |
| **3** | Adaptive learnings + record_learning | Potentially high | Medium |
| **3** | Ensemble voting coordinator | Potentially high | Medium |

---

## Token Economics

V2 signal briefs use ~60% fewer tokens than raw CSV candles.
State-aware prompts cut tool calls by ~30%.
Longer tick intervals (60-120s) cut total LLM calls by 50-75%.

**Net result: V2 can afford 3-4x more expensive models (Sonnet, Pro) and still cost less than V1.**

---

## Path to Profitability

Most likely winning strategy: **sentiment-driven momentum** — only trade when:
1. Regime is TRENDING
2. Social mentions are spiking
3. Historical win rate for this setup > 55%

Trade rarely, but with high conviction. That's what high transaction costs demand.
