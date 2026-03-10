"""
AgentRuntime — Microsoft Agent Framework SDK orchestrator.

Architecture
------------
  POST /chat
       ↓
  routes.py              thin: validate → create AgentSession → call runtime.run()
       ↓                 returns ChatResponse (JSON)
  AgentRuntime.run()
    1. retrieve()         embed query → hybrid Azure AI Search (DefaultAzureCredential)
    2. GATE               score-first: top chunk score >= threshold → pass
    3. citations          built from retrieval — always, never from LLM answer text
    4. rag_provider       store pre-retrieved results in AFAgentSession.state
    5. af_agent.run()     Agent Framework ChatAgent → AzureOpenAIChatClient (managed id)
                            • InMemoryHistoryProvider  (optional, see ENABLE_IN_MEMORY_HISTORY)
                            • RagContextProvider.before_run()  injects [N] context blocks
                            • collects streamed tokens into a full answer string
    6. return             ChatResponse{answer, citations, session_id}

  POST /chat/stream
       ↓
  routes.py              thin: validate → create AgentSession → call runtime.run_stream()
       ↓                 returns StreamingResponse (SSE)
  AgentRuntime.run_stream()
    Same pipeline as run() except:
    • tokens are yielded as SSE data lines as they arrive
    • errors are emitted as named SSE error events (cannot raise HTTPException mid-stream)
    • citations are emitted as a named SSE citations event after the last token

Key differences vs agentv01
----------------------------
  - Managed identity: AzureOpenAIChatClient + DefaultAzureCredential (no API keys)
  - Citations: always built from retrieval, never gated on "Sources:" in answer text
  - Confidence gate: score-first (top chunk score); no hard count-only fail
  - /chat JSON non-streaming endpoint (run() method) alongside /chat/stream
  - ENABLE_IN_MEMORY_HISTORY toggle for InMemoryHistoryProvider (default: disabled)
  - 502 HTTPException for retrieval and generation failures on /chat
  - Named SSE error event + [DONE] for failures on /chat/stream
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from agent_framework import AgentSession as AFAgentSession
from fastapi import HTTPException

from app.agent_runtime.citation_provider import build_citations
from app.agent_runtime.prompts import CLARIFYING_RESPONSE
from app.agent_runtime.session import AgentSession
from app.api.schemas import ChatResponse, CitationsPayload
from app.config.settings import (
    ENABLE_IN_MEMORY_HISTORY,
    MIN_AVG_SCORE,
    MIN_RERANKER_SCORE,
    MIN_RESULTS,
    TOP_K,
    USE_SEMANTIC_RERANKER,
)
from app.llm.af_agent_factory import af_agent, rag_provider
from app.tools.retrieval_tool import retrieve

logger = logging.getLogger(__name__)

_PING_INTERVAL_SECONDS = 20  # SSE keepalive interval

# AF session cache — used only when ENABLE_IN_MEMORY_HISTORY=true.
# InMemoryHistoryProvider stores conversation history inside each AFAgentSession.state.
# Scoped to this process — resets on restart or scale-out event.
_af_sessions: dict[str, AFAgentSession] = {}


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_data(payload: str) -> bytes:
    """Encode a string as an SSE data line (newlines escaped)."""
    escaped = payload.replace("\n", "\\n")
    return f"data: {escaped}\n\n".encode("utf-8")


def _sse_event(event_name: str, payload: str) -> bytes:
    """Encode a named SSE event."""
    return f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")


# ── Confidence gate ───────────────────────────────────────────────────────────

def _check_gate(results: list[dict]) -> tuple[bool, str]:
    """Score-first confidence gate — return (passed, rejection_reason).

    A single highly relevant chunk passes the gate when its score meets the
    configured threshold.  MIN_RESULTS is NOT a hard prerequisite for passing —
    it is only consulted when the top score also fails, and then only to enrich
    the rejection message.

    Score ranges:
      Semantic reranker active : reranker_score  0.0 – 4.0  (threshold: MIN_RERANKER_SCORE)
      Hybrid / RRF only        : base RRF score  0.01–0.033 (threshold: MIN_AVG_SCORE)
    """
    if not results:
        return False, "no chunks retrieved"

    top = results[0]  # results are pre-sorted by effective score descending

    if USE_SEMANTIC_RERANKER and top.get("reranker_score") is not None:
        top_score = top["reranker_score"]
        if top_score >= MIN_RERANKER_SCORE:
            return True, ""
        reason = f"top reranker score {top_score:.3f} < threshold {MIN_RERANKER_SCORE}"
        if len(results) < MIN_RESULTS:
            reason += f"; retrieved {len(results)} chunk(s) — minimum is {MIN_RESULTS}"
        return False, reason

    top_score = top["score"]
    if top_score >= MIN_AVG_SCORE:
        return True, ""
    reason = f"top score {top_score:.4f} < threshold {MIN_AVG_SCORE}"
    if len(results) < MIN_RESULTS:
        reason += f"; retrieved {len(results)} chunk(s) — minimum is {MIN_RESULTS}"
    return False, reason


# ── AF session management ─────────────────────────────────────────────────────

def _get_or_create_af_session(session_id: str) -> AFAgentSession:
    """Return an Agent Framework session for this session_id.

    Stateless mode (ENABLE_IN_MEMORY_HISTORY=false, default):
        Always create a fresh AFAgentSession — nothing is stored between requests.
        Safe for App Service scale-out (no shared memory) and restarts.

    History mode (ENABLE_IN_MEMORY_HISTORY=true):
        Reuse the same AFAgentSession across requests with the same session_id
        so InMemoryHistoryProvider accumulates multi-turn conversation history.
        Only suitable for single-instance local dev or demos.
    """
    if not ENABLE_IN_MEMORY_HISTORY:
        return af_agent.create_session()
    if session_id not in _af_sessions:
        _af_sessions[session_id] = af_agent.create_session()
    return _af_sessions[session_id]


# ── AgentRuntime ──────────────────────────────────────────────────────────────

class AgentRuntime:
    """Orchestrates the full retrieve → gate → generate → cite pipeline.

    Uses Microsoft Agent Framework SDK primitives for LLM invocation, RAG
    context injection (RagContextProvider.before_run()), and optional
    multi-turn memory (InMemoryHistoryProvider).
    """

    # ── Non-streaming (/chat JSON) ────────────────────────────────────────────

    async def run(self, question: str, session: AgentSession) -> ChatResponse:
        """Execute the pipeline and return a complete ChatResponse.

        Raises
        ------
        HTTPException(502)
            If Azure AI Search retrieval or Azure OpenAI generation fails.
            The route handler propagates this directly to the HTTP client.
        """
        logger.info(
            "AgentRuntime.run | session=%s | question=%r",
            session.session_id, question[:80],
        )

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        try:
            results: list[dict] = await asyncio.to_thread(retrieve, question)
        except Exception as exc:
            logger.exception(
                "Retrieval failed for session=%s: %s", session.session_id, exc
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Failed to retrieve information from Azure AI Search. "
                    "Check service availability and managed identity permissions."
                ),
            )

        session.retrieved_results = results

        # ── 2. Gate ───────────────────────────────────────────────────────────
        passed, reason = _check_gate(results)
        if not passed:
            logger.info(
                "Gate rejected session=%s: %s", session.session_id, reason
            )
            return ChatResponse(
                answer=CLARIFYING_RESPONSE,
                citations=[],
                session_id=session.session_id,
            )

        # ── 3. Citations — always from retrieval, never from LLM answer text ─
        citations = build_citations(results)

        # ── 4. AF session + RAG context injection ─────────────────────────────
        af_session = _get_or_create_af_session(session.session_id)
        rag_provider.store_results(af_session, results)

        # ── 5. Generate via Agent Framework ChatAgent ─────────────────────────
        # Collect streamed tokens into a full answer string.
        # af_agent.run(stream=True) is an async generator; each update carries
        # the incremental token in update.text.
        try:
            answer_buf: list[str] = []
            async for update in af_agent.run(
                question, stream=True, session=af_session
            ):
                if update.text:
                    answer_buf.append(update.text)
            answer = "".join(answer_buf)
        except Exception as exc:
            logger.exception(
                "LLM generation failed for session=%s: %s", session.session_id, exc
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Failed to generate a response from Azure OpenAI. "
                    "Check service availability and managed identity permissions."
                ),
            )

        session.answer_text = answer
        return ChatResponse(
            answer=answer,
            citations=citations,
            session_id=session.session_id,
        )

    # ── Streaming (/chat/stream SSE) ──────────────────────────────────────────

    async def run_stream(
        self,
        question: str,
        session: AgentSession,
    ) -> AsyncGenerator[bytes, None]:
        """Execute the pipeline and yield SSE-formatted bytes.

        Pass this generator directly to FastAPI's StreamingResponse.

        SSE event contract
        ------------------
        ``data: <token>``               answer token (unnamed; accumulate client-side)
        ``event: citations  data: ...`` CitationsPayload JSON — always on success
        ``event: error      data: ...`` ``{"error": "..."}`` JSON — on failure, then [DONE]
        ``event: ping       data: ...`` keepalive heartbeat — ignore
        ``data: [DONE]``                stream end sentinel

        Cannot raise HTTPException once streaming has started — failures are emitted
        as named SSE error events so the client can distinguish them from tokens.
        """
        logger.info(
            "AgentRuntime.run_stream | session=%s | question=%r",
            session.session_id, question[:80],
        )

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        try:
            results: list[dict] = await asyncio.to_thread(retrieve, question)
        except Exception as exc:
            logger.exception(
                "Retrieval failed for session=%s: %s", session.session_id, exc
            )
            yield _sse_event(
                "error",
                json.dumps({
                    "error": (
                        "Failed to retrieve information from Azure AI Search. "
                        "Check service availability and managed identity permissions."
                    )
                }),
            )
            yield _sse_data("[DONE]")
            return

        session.retrieved_results = results

        # ── 2. Gate ───────────────────────────────────────────────────────────
        passed, reason = _check_gate(results)
        if not passed:
            logger.info(
                "Gate rejected session=%s: %s", session.session_id, reason
            )
            yield _sse_data(CLARIFYING_RESPONSE)
            yield _sse_event(
                "citations", CitationsPayload(citations=[]).model_dump_json()
            )
            yield _sse_data("[DONE]")
            return

        # ── 3. Citations — always from retrieval ──────────────────────────────
        citations = build_citations(results)
        citations_payload = CitationsPayload(citations=citations)

        # ── 4. AF session + RAG context injection ─────────────────────────────
        af_session = _get_or_create_af_session(session.session_id)
        rag_provider.store_results(af_session, results)

        # ── 5. Stream tokens via Agent Framework ChatAgent ────────────────────
        last_ping = time.monotonic()
        try:
            async for update in af_agent.run(
                question, stream=True, session=af_session
            ):
                now = time.monotonic()
                if now - last_ping >= _PING_INTERVAL_SECONDS:
                    yield _sse_event("ping", "keepalive")
                    last_ping = now

                if update.text:
                    yield _sse_data(update.text)

        except Exception as exc:
            logger.exception(
                "LLM streaming failed for session=%s: %s", session.session_id, exc
            )
            yield _sse_event(
                "error",
                json.dumps({
                    "error": (
                        "Failed to generate a response from Azure OpenAI. "
                        "Check service availability and managed identity permissions."
                    )
                }),
            )
            yield _sse_data("[DONE]")
            return

        # ── 6. Citations event — always emitted on success ────────────────────
        yield _sse_event("citations", citations_payload.model_dump_json())
        yield _sse_data("[DONE]")
