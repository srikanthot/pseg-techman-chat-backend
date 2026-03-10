"""
AgentSession — lightweight per-request state carrier.

Created by the route handler, passed into AgentRuntime.run() /
AgentRuntime.run_stream(), and populated as the pipeline executes.

This is our application-level session object — distinct from
agent_framework.AgentSession (AFAgentSession) which is the SDK's
internal session that carries InMemoryHistoryProvider state.

The two session objects work together:
  AgentSession       — holds the HTTP request context (question, session_id)
  AFAgentSession     — holds the Agent Framework SDK state (history, provider state)
"""

import uuid
from dataclasses import dataclass, field


@dataclass
class AgentSession:
    """Per-request state for a single agent pipeline execution."""

    question: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Populated by AgentRuntime after retrieval succeeds
    retrieved_results: list[dict] = field(default_factory=list)

    # Populated by AgentRuntime after generation completes (non-streaming path)
    answer_text: str = ""
