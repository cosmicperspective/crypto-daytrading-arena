from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

AGENT_REGISTRY_TOPIC = "agent_registry"


class AgentMeta(BaseModel):
    agent_name: str
    chat_node_name: str
    model_id: str
    provider: str
    strategy: str
    started_at: str

    @classmethod
    def create(
        cls,
        agent_name: str,
        chat_node_name: str,
        model_id: str,
        provider: str,
        strategy: str,
    ) -> "AgentMeta":
        return cls(
            agent_name=agent_name,
            chat_node_name=chat_node_name,
            model_id=model_id,
            provider=provider,
            strategy=strategy,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
