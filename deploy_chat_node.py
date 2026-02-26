"""Deploy a single named ChatNode backed by a selected model provider.

Run one instance per model. The node listens on its private topic
``ai_prompted.<name>`` so that agent routers can target it by name.

Example:
    uv run python deploy_chat_node.py \
        --name gpt5-nano --provider openai --model-id gpt-5-nano --bootstrap-servers <broker-url> \
        --reasoning-effort low

    uv run python deploy_chat_node.py \
        --name deepseek --provider deepseek --model-id deepseek-chat --bootstrap-servers <broker-url> \
        --base-url <deepseek-base-url> --api-key $DEEPSEEK_API_KEY
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

from calfkit.broker.broker import BrokerClient
from calfkit.nodes.chat_node import ChatNode
from calfkit.providers.pydantic_ai.openai import OpenAIModelClient
from calfkit.runners.service import NodesService

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a named ChatNode for per-model inference.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="ChatNode name (becomes private topic ai_prompted.<name>)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "gemini", "deepseek", "openai-compatible", "openrouter"],
        help="Model provider (default: openai).",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID passed to OpenAIModelClient (e.g. gpt-5-nano, deepseek-chat)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible providers (default: OpenAI)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--bootstrap-servers",
        required=True,
        help="Kafka bootstrap servers address",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent inference workers (default: 1)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help='Reasoning effort for reasoning models (e.g. "low")',
    )
    return parser.parse_args()


def _resolve_api_key(provider: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_map = {
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.getenv(env_map.get(provider, "OPENAI_API_KEY"))


def _build_model_client(
    provider: str,
    model_id: str,
    base_url: str | None,
    api_key: str,
    reasoning_effort: str | None,
) -> Any:
    if provider in {"openai", "openai-compatible", "deepseek", "gemini", "openrouter"}:
        if provider == "openrouter" and not base_url:
            base_url = os.getenv("OPENROUTER_BASE_URL")
        return OpenAIModelClient(
            model_name=model_id,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )

    if provider == "anthropic":
        try:
            from calfkit.providers.pydantic_ai.anthropic import AnthropicModelClient  # type: ignore
        except Exception:
            print("ERROR: Anthropic provider not available in calfkit.")
            print("Install a calfkit version that includes AnthropicModelClient.")
            sys.exit(1)
        return AnthropicModelClient(
            model_name=model_id,
            api_key=api_key,
        )

    print(f"ERROR: Unsupported provider '{provider}'.")
    sys.exit(1)


async def main() -> None:
    args = parse_args()

    # Resolve API key: explicit flag > env var
    api_key = _resolve_api_key(args.provider, args.api_key)
    if not api_key:
        print("ERROR: No API key provided.")
        print("Pass --api-key or set a provider-specific key (e.g. OPENAI_API_KEY).")
        sys.exit(1)

    print("=" * 50)
    print(f"ChatNode Deployment: {args.name}")
    print("=" * 50)

    print(f"\nConnecting to Kafka broker at {args.bootstrap_servers}...")
    broker = BrokerClient(bootstrap_servers=args.bootstrap_servers)

    print(f"Configuring model client: {args.model_id}")
    model_client = _build_model_client(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
    )

    chat_node = ChatNode(model_client, name=args.name)
    service = NodesService(broker)
    service.register_node(chat_node, max_workers=args.max_workers)

    print(f"  - Name:  {args.name}")
    print(f"  - Provider: {args.provider}")
    print(f"  - Model: {args.model_id}")
    print(f"  - Topic: {chat_node.entrypoint_topic}")
    print(f"  - Workers: {args.max_workers}")
    if args.base_url:
        print(f"  - Base URL: {args.base_url}")
    if args.reasoning_effort:
        print(f"  - Reasoning effort: {args.reasoning_effort}")

    print("\nChat node ready. Waiting for requests...")
    await service.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat node stopped.")
