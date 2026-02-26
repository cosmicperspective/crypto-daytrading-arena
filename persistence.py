from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterable

from typing import Any

from agent_registry import AgentMeta


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                agent_id TEXT PRIMARY KEY,
                cash REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                agent_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                qty REAL NOT NULL,
                cost_basis REAL NOT NULL,
                avg_entry_ts REAL,
                PRIMARY KEY(agent_id, product_id)
            );
            CREATE TABLE IF NOT EXISTS stats (
                agent_id TEXT PRIMARY KEY,
                trade_count INTEGER NOT NULL,
                realized_pnl REAL NOT NULL,
                fees_paid REAL NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                equity_peak REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                last_trade_ts REAL,
                memory_calls INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                product_id TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL,
                latency REAL
            );
            CREATE TABLE IF NOT EXISTS agent_meta (
                agent_id TEXT PRIMARY KEY,
                chat_node_name TEXT NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                strategy TEXT NOT NULL,
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                product_id TEXT NOT NULL,
                price REAL NOT NULL,
                best_bid REAL,
                best_ask REAL
            );
            CREATE INDEX IF NOT EXISTS idx_price_history_product_ts
                ON price_history(product_id, ts);
            """
        )
        self._conn.commit()

    def load_accounts(self, account_factory) -> dict[str, Any]:
        accounts: dict[str, Any] = {}
        cur = self._conn.cursor()
        for row in cur.execute("SELECT agent_id, cash FROM accounts"):
            accounts[row["agent_id"]] = account_factory(cash=row["cash"])

        for row in cur.execute(
            "SELECT agent_id, product_id, qty, cost_basis, avg_entry_ts FROM positions"
        ):
            agent_id = row["agent_id"]
            account = accounts.setdefault(agent_id, account_factory())
            account.positions[row["product_id"]] = row["qty"]
            account.cost_basis[row["product_id"]] = row["cost_basis"]
            if row["avg_entry_ts"] is not None:
                account.avg_entry_ts[row["product_id"]] = row["avg_entry_ts"]

        for row in cur.execute("SELECT * FROM stats"):
            agent_id = row["agent_id"]
            account = accounts.setdefault(agent_id, account_factory())
            account.trade_count = row["trade_count"]
            account.realized_pnl = row["realized_pnl"]
            account.fees_paid = row["fees_paid"]
            account.wins = row["wins"]
            account.losses = row["losses"]
            account.equity_peak = row["equity_peak"]
            account.max_drawdown = row["max_drawdown"]
            account.last_trade_ts = row["last_trade_ts"]
            try:
                account.memory_calls = row["memory_calls"]
            except (IndexError, KeyError):
                pass

        return accounts

    def save_account(self, agent_id: str, account: Any) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO accounts(agent_id, cash) VALUES(?, ?) "
            "ON CONFLICT(agent_id) DO UPDATE SET cash=excluded.cash",
            (agent_id, account.cash),
        )

        cur.execute("DELETE FROM positions WHERE agent_id = ?", (agent_id,))
        for product_id, qty in account.positions.items():
            cur.execute(
                "INSERT INTO positions(agent_id, product_id, qty, cost_basis, avg_entry_ts) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    agent_id,
                    product_id,
                    qty,
                    account.cost_basis.get(product_id, 0.0),
                    account.avg_entry_ts.get(product_id),
                ),
            )

        cur.execute(
            "INSERT INTO stats(agent_id, trade_count, realized_pnl, fees_paid, wins, losses, "
            "equity_peak, max_drawdown, last_trade_ts, memory_calls) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(agent_id) DO UPDATE SET "
            "trade_count=excluded.trade_count, realized_pnl=excluded.realized_pnl, "
            "fees_paid=excluded.fees_paid, wins=excluded.wins, losses=excluded.losses, "
            "equity_peak=excluded.equity_peak, max_drawdown=excluded.max_drawdown, "
            "last_trade_ts=excluded.last_trade_ts, memory_calls=excluded.memory_calls",
            (
                agent_id,
                account.trade_count,
                account.realized_pnl,
                account.fees_paid,
                account.wins,
                account.losses,
                account.equity_peak,
                account.max_drawdown,
                account.last_trade_ts,
                getattr(account, "memory_calls", 0),
            ),
        )

        self._conn.commit()

    def insert_trade(
        self,
        ts: str,
        agent_id: str,
        action: str,
        product_id: str,
        qty: float,
        price: float,
        fee: float,
        latency: float | None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO trades(ts, agent_id, action, product_id, qty, price, fee, latency) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, agent_id, action, product_id, qty, price, fee, latency),
        )
        self._conn.commit()

    def load_trades(self) -> list[tuple[str, str, str, str, float, float, float, float | None]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT ts, agent_id, action, product_id, qty, price, fee, latency "
            "FROM trades ORDER BY id ASC"
        ).fetchall()
        return [
            (
                row["ts"],
                row["agent_id"],
                row["action"],
                row["product_id"],
                row["qty"],
                row["price"],
                row["fee"],
                row["latency"],
            )
            for row in rows
        ]

    def save_agent_meta(self, meta: AgentMeta) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO agent_meta(agent_id, chat_node_name, model_id, provider, strategy, started_at) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(agent_id) DO UPDATE SET "
            "chat_node_name=excluded.chat_node_name, model_id=excluded.model_id, "
            "provider=excluded.provider, strategy=excluded.strategy, started_at=excluded.started_at",
            (
                meta.agent_name,
                meta.chat_node_name,
                meta.model_id,
                meta.provider,
                meta.strategy,
                meta.started_at,
            ),
        )
        self._conn.commit()

    def insert_price(
        self,
        ts: float,
        product_id: str,
        price: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO price_history(ts, product_id, price, best_bid, best_ask) "
            "VALUES(?, ?, ?, ?, ?)",
            (ts, product_id, price, best_bid, best_ask),
        )
        self._conn.commit()

    def cleanup_old_prices(self, max_age_seconds: float = 7200) -> None:
        """Remove price entries older than max_age_seconds (default 2h)."""
        import time as _time
        cutoff = _time.time() - max_age_seconds
        cur = self._conn.cursor()
        cur.execute("DELETE FROM price_history WHERE ts < ?", (cutoff,))
        self._conn.commit()

    def load_agent_meta(self) -> dict[str, AgentMeta]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM agent_meta").fetchall()
        meta: dict[str, AgentMeta] = {}
        for row in rows:
            meta[row["agent_id"]] = AgentMeta(
                agent_name=row["agent_id"],
                chat_node_name=row["chat_node_name"],
                model_id=row["model_id"],
                provider=row["provider"],
                strategy=row["strategy"],
                started_at=row["started_at"],
            )
        return meta
