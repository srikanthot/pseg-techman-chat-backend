"""
RagContextProvider — Agent Framework BaseContextProvider for RAG injection.

This is the core Agent Framework SDK integration that makes the repo a true
AF-pattern implementation rather than a plain FastAPI + custom LLM loop.

How it works:
  1. AgentRuntime.run() / run_stream() calls retrieve() to fetch chunks from
     Azure AI Search, then calls rag_provider.store_results(af_session, results)
     to store them in the AF session's state dict under a known key.

  2. Before each LLM call, the Agent Framework SDK fires before_run() on every
     registered context_provider.  RagContextProvider.before_run():
       a. Pops the pre-retrieved results from session.state.
       b. Formats them into numbered evidence blocks via build_context_blocks().
       c. Calls context.extend_instructions(source_id, text) — the official SDK
          hook that appends grounded context to the agent's system prompt for
          this specific turn.

  3. after_run() is a no-op here.  Cosmos DB / audit logging can be wired in.

Why this pattern matters:
  - RAG context injection is a first-class SDK hook, not ad-hoc string
    concatenation inside the orchestrator.
  - Results are pre-fetched once and stored in session.state, so the provider
    never triggers a second Azure AI Search query.
  - The provider is reusable across any agent that needs RAG grounding.
"""

import logging
from typing import Any

from agent_framework import AgentSession, BaseContextProvider, SessionContext

from app.agent_runtime.context_providers import build_context_blocks
from app.config.settings import TRACE_MODE

logger = logging.getLogger(__name__)

# Key under which AgentRuntime stores pre-retrieved results in session.state.
# Using a private-convention name avoids collisions with other providers.
_PENDING_RESULTS_KEY = "_rag_pending_results"


class RagContextProvider(BaseContextProvider):
    """Injects pre-retrieved Azure AI Search chunks as grounded context.

    Registered in af_agent_factory.py via client.as_agent(context_providers=[...]).
    """

    def __init__(self) -> None:
        super().__init__("rag")

    # ── Called by AgentRuntime before agent.run() ─────────────────────────────

    def store_results(self, session: AgentSession, results: list[dict]) -> None:
        """Hand off pre-retrieved chunks to before_run() via session.state.

        Called by AgentRuntime after retrieval and gate pass, before agent.run().
        Storing in session.state (the shared dict visible to all providers and
        the before_run hook) avoids a second Azure AI Search call.
        """
        session.state[_PENDING_RESULTS_KEY] = results

    # ── Agent Framework SDK hook — fires before each LLM call ────────────────

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject retrieved chunks as additional system-level instructions.

        context.extend_instructions() is the official Agent Framework SDK method
        for appending text to the agent's system prompt on a per-provider basis.
        The framework concatenates all provider instructions before the model call.
        """
        results: list[dict] = session.state.pop(_PENDING_RESULTS_KEY, [])
        if not results:
            return

        context_blocks = build_context_blocks(results)
        context.extend_instructions(
            self.source_id,
            (
                "Context (retrieved from PSEG technical manuals):\n\n"
                f"{context_blocks}\n\n"
                "Answer the question using ONLY the context above. "
                "When the context covers the topic — even partially — provide a "
                "complete answer from the available information. "
                "Reference each source by its [N] label inline. "
                "Include a 'Sources:' section at the end."
            ),
        )

        if TRACE_MODE:
            for i, r in enumerate(results, start=1):
                section = " > ".join(
                    p for p in [
                        r.get("section1") or "",
                        r.get("section2") or "",
                        r.get("section3") or "",
                    ] if p
                )
                reranker_str = (
                    f"  reranker={r['reranker_score']:.4f}"
                    if r.get("reranker_score") is not None else ""
                )
                logger.info(
                    "TRACE | context_block[%d] src=%r section=%r score=%.4f%s\n%s",
                    i, r["source"], section or "(none)",
                    r["score"], reranker_str,
                    r["content"][:300],
                )

    # ── Agent Framework SDK hook — fires after each LLM call ─────────────────

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """No-op — Cosmos DB or audit logging can be wired here."""
