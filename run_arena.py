from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable


# ── Mode presets ────────────────────────────────────────────────

MODE_PRESETS = {
    "strategy": {
        "models_file": "models_strategy.json",
        "tests_file": "arena_strategy.json",
        "description": "Strategy Competition — 4 strategies, 1 model (Gemini 2.5 Flash)",
    },
    "llm": {
        "models_file": "models_llm.json",
        "tests_file": "arena_llm.json",
        "description": "LLM Competition — multiple models, 1 neutral strategy",
    },
}


@dataclass
class ModelSpec:
    name: str
    provider: str
    model_id: str
    base_url: str | None = None


@dataclass
class AgentSpec:
    agent_name: str
    chat_node_name: str
    strategy: str
    provider: str
    model_id: str


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_models(path: str) -> list[ModelSpec]:
    data = _load_json(path)
    models = []
    for item in data.get("models", []):
        models.append(
            ModelSpec(
                name=item["name"],
                provider=item["provider"],
                model_id=item["model_id"],
                base_url=item.get("base_url"),
            )
        )
    return models


def _load_agents(path: str) -> tuple[str, list[AgentSpec]]:
    data = _load_json(path)
    bootstrap = data.get("bootstrap_servers", "localhost:9092")
    agents = []
    for item in data.get("agents", []):
        agents.append(
            AgentSpec(
                agent_name=item["agent_name"],
                chat_node_name=item["chat_node_name"],
                strategy=item["strategy"],
                provider=item.get("provider", "unknown"),
                model_id=item.get("model_id", "unknown"),
            )
        )
    return bootstrap, agents


def _cmd_python(use_uv: bool) -> list[str]:
    if use_uv:
        return ["uv", "run", "python"]
    return [sys.executable]


def _spawn(cmd: list[str], name: str, log_dir: str, dry_run: bool) -> subprocess.Popen | None:
    if dry_run:
        print(" ".join(cmd))
        return None

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file)


def _maybe_add_base_url(cmd: list[str], base_url: str | None) -> None:
    if base_url:
        cmd.extend(["--base-url", base_url])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the daytrading arena processes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  strategy  — 4 trading strategies compete using Gemini 2.5 Flash\n"
            "  llm       — 4 different LLMs compete using the same neutral strategy\n"
            "\nExamples:\n"
            "  python run_arena.py --mode strategy --start all\n"
            "  python run_arena.py --mode llm --start all\n"
            "  python run_arena.py --mode llm --start chatnodes routers --dry-run\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_PRESETS.keys()),
        default=None,
        help="Competition mode (auto-selects config files). See 'Modes' below.",
    )
    parser.add_argument("--models-file", default=None, help="Path to models JSON (overrides --mode)")
    parser.add_argument("--tests-file", default=None, help="Path to agents/tests JSON (overrides --mode)")
    parser.add_argument(
        "--start",
        nargs="+",
        choices=["all", "tools", "connector", "viewer", "chatnodes", "routers", "dashboard"],
        default=["chatnodes", "routers"],
        help="Which processes to start (default: chatnodes routers).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent inference workers per ChatNode (default: 2).",
    )
    parser.add_argument(
        "--connector-interval",
        type=float,
        default=2.0,
        help="Coinbase connector min publish interval in seconds (default: 2).",
    )
    parser.add_argument(
        "--min-change-bps",
        type=float,
        default=15.0,
        help="Price change gate in bps (default: 15).",
    )
    parser.add_argument(
        "--max-silent-seconds",
        type=float,
        default=30.0,
        help="Force publish after this many seconds of silence (default: 30).",
    )
    parser.add_argument("--use-uv", action="store_true", help="Use uv run python")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit")
    parser.add_argument("--log-dir", default="logs", help="Directory for process logs")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete arena.db before starting (fresh competition).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config files from --mode or explicit flags
    if args.mode:
        preset = MODE_PRESETS[args.mode]
        models_file = args.models_file or preset["models_file"]
        tests_file = args.tests_file or preset["tests_file"]
        print(f"\n  Mode: {preset['description']}")
    else:
        models_file = args.models_file or "models.json"
        tests_file = args.tests_file or "arena_tests.json"

    models = _load_models(models_file)
    bootstrap, agents = _load_agents(tests_file)

    # Expand "all" to all components
    start = set(args.start)
    if "all" in start:
        start = {"tools", "connector", "chatnodes", "routers", "dashboard"}

    print("=" * 55)
    print("  Crypto Daytrading Arena Launcher")
    print("=" * 55)
    if args.mode:
        print(f"  Mode:       {args.mode}")
    print(f"  Models:     {models_file} ({len(models)} model(s))")
    print(f"  Agents:     {tests_file} ({len(agents)} agent(s))")
    print(f"  Bootstrap:  {bootstrap}")
    print(f"  Starting:   {', '.join(sorted(start))}")
    print(f"  Workers:    {args.max_workers} per ChatNode")
    print()

    # Reset DB if requested
    if args.reset_db:
        db_path = os.path.join(os.path.dirname(__file__), "arena.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print("  Deleted arena.db for fresh start")
        else:
            print("  arena.db not found (already clean)")
        print()

    procs: list[subprocess.Popen] = []
    cmd_py = _cmd_python(args.use_uv)

    # 1. Tools & Price Feed
    if "tools" in start:
        cmd = cmd_py + ["tools_and_dashboard.py", "--bootstrap-servers", bootstrap]
        print(f"  [tools] tools_and_dashboard.py")
        p = _spawn(cmd, "tools_and_dashboard", args.log_dir, args.dry_run)
        if p:
            procs.append(p)

    # 2. Coinbase Connector
    if "connector" in start:
        cmd = cmd_py + [
            "coinbase_connector.py",
            "--bootstrap-servers", bootstrap,
            "--interval", str(args.connector_interval),
            "--min-change-bps", str(args.min_change_bps),
            "--max-silent-seconds", str(args.max_silent_seconds),
        ]
        print(f"  [connector] coinbase_connector.py (interval={args.connector_interval}s, gate={args.min_change_bps}bps)")
        p = _spawn(cmd, "coinbase_connector", args.log_dir, args.dry_run)
        if p:
            procs.append(p)

    # 3. ChatNodes
    if "chatnodes" in start:
        for model in models:
            cmd = cmd_py + [
                "deploy_chat_node.py",
                "--name", model.name,
                "--provider", model.provider,
                "--model-id", model.model_id,
                "--bootstrap-servers", bootstrap,
                "--max-workers", str(args.max_workers),
            ]
            _maybe_add_base_url(cmd, model.base_url)
            print(f"  [chatnode] {model.name} -> {model.model_id}")
            p = _spawn(cmd, f"chatnode_{model.name}", args.log_dir, args.dry_run)
            if p:
                procs.append(p)

    # 4. Router Nodes (Agents)
    if "routers" in start:
        for agent in agents:
            cmd = cmd_py + [
                "deploy_router_node.py",
                "--name", agent.agent_name,
                "--chat-node-name", agent.chat_node_name,
                "--strategy", agent.strategy,
                "--bootstrap-servers", bootstrap,
                "--provider", agent.provider,
                "--model-id", agent.model_id,
            ]
            print(f"  [router] {agent.agent_name} ({agent.strategy}) -> {agent.chat_node_name}")
            p = _spawn(cmd, f"router_{agent.agent_name}", args.log_dir, args.dry_run)
            if p:
                procs.append(p)

    # 5. Viewer
    if "viewer" in start:
        cmd = cmd_py + ["response_viewer.py", "--bootstrap-servers", bootstrap]
        print(f"  [viewer] response_viewer.py")
        p = _spawn(cmd, "response_viewer", args.log_dir, args.dry_run)
        if p:
            procs.append(p)

    # 6. Web Dashboard
    if "dashboard" in start:
        cmd = cmd_py + ["web_dashboard.py"]
        print(f"  [dashboard] web_dashboard.py (http://localhost:8050)")
        p = _spawn(cmd, "web_dashboard", args.log_dir, args.dry_run)
        if p:
            procs.append(p)

    if args.dry_run:
        print("\n  (dry run — no processes started)")
        return

    procs = [p for p in procs if p is not None]
    print(f"\n  {len(procs)} process(es) running. Press Ctrl+C to stop all.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        for p in procs:
            p.terminate()
        print("  All processes stopped.")


if __name__ == "__main__":
    main()
