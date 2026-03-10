"""
API route handlers for the PSEG Tech Manual Chat Backend.

Endpoints:
  GET  /health          — liveness check (defined in main.py)
  POST /chat            — non-streaming JSON response (primary integration endpoint)
  POST /chat/stream     — Server-Sent Events streaming response
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, CitationsPayload
from app.config.settings import (
    MIN_RESULTS,
    MIN_AVG_SCORE,
    MIN_RERANKER_SCORE,
    USE_SEMANTIC_RERANKER,
)
from app.services import chat as chat_service
from app.services import search as search_service
from app.services.citations import build_citations
from app.services.prompts import CLARIFYING_RESPONSE

router = APIRouter()
logger = logging.getLogger(__name__)

_PING_INTERVAL_SECS = 20  # keepalive ping interval for SSE connections


# ── Confidence gate ───────────────────────────────────────────────────────────

def _check_gate(results: list[dict]) -> tuple[bool, str]:
    """Return (passed, rejection_reason).

    Gate logic — practical for technical manual Q&A:

    1. Require at least MIN_RESULTS chunks (default 1).
       A single highly relevant chunk is sufficient — we do not hard-fail
       just because fewer than 2 chunks were retrieved.

    2. Evaluate the TOP chunk's score (not the average).
       This avoids penalising responses when one excellent chunk is
       accompanied by weaker supporting chunks. Average-based gating
       would incorrectly reject such cases.

    Score ranges by mode:
      - Semantic reranker on:  reranker_score  0.0 – 4.0   (threshold: MIN_RERANKER_SCORE)
      - Semantic reranker off: base RRF score  0.01 – 0.033 (threshold: MIN_AVG_SCORE)
    """
    if len(results) < MIN_RESULTS:
        return False, f"retrieved {len(results)} chunk(s) — minimum is {MIN_RESULTS}"

    # results are pre-sorted by effective score descending; index 0 is the top chunk
    top = results[0]

    if USE_SEMANTIC_RERANKER and top.get("reranker_score") is not None:
        top_score = top["reranker_score"]
        if top_score < MIN_RERANKER_SCORE:
            return (
                False,
                f"top reranker score {top_score:.3f} < threshold {MIN_RERANKER_SCORE}",
            )
        return True, ""

    # Fall back to base RRF / hybrid score
    top_score = top["score"]
    if top_score < MIN_AVG_SCORE:
        return False, f"top score {top_score:.4f} < threshold {MIN_AVG_SCORE}"
    return True, ""


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_data(text: str) -> bytes:
    """Encode a token as an SSE data line, escaping embedded newlines."""
    escaped = text.replace("\n", "\\n")
    return f"data: {escaped}\n\n".encode("utf-8")


def _sse_event(event: str, data: str) -> bytes:
    """Encode a named SSE event."""
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


# ── POST /chat (non-streaming, JSON) ─────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Non-streaming chat",
    description=(
        "Submit a question and receive a complete JSON response with the answer "
        "and structured citations.\n\n"
        "This is the **recommended endpoint for Power Apps and PCF integrations** — "
        "no streaming required, simple request/response contract.\n\n"
        "**Citation behaviour:** citations are always derived from retrieval results. "
        "They are never dependent on whether the LLM includes `[1]` or `Sources:` "
        "in its answer text. If retrieval succeeds and the confidence gate passes, "
        "citations are always returned.\n\n"
        "**Error responses:** `502` if Azure AI Search or Azure OpenAI is unreachable. "
        "`200` with a clarifying question and empty citations if the confidence gate "
        "rejects the retrieval results."
    ),
    tags=["chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "POST /chat session=%s question=%r",
        session_id,
        request.question[:80],
    )

    # ── Retrieve ──────────────────────────────────────────────────────────────
    # retrieve() is synchronous (uses sync Azure SDK); run it in a thread pool
    # so it does not block the async event loop.
    try:
        results = await asyncio.to_thread(search_service.retrieve, request.question)
    except Exception as exc:
        logger.exception("Retrieval error for session=%s: %s", session_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to retrieve information from Azure AI Search. "
                "Check service availability and managed identity permissions."
            ),
        )

    # ── Confidence gate ───────────────────────────────────────────────────────
    passed, reason = _check_gate(results)
    if not passed:
        logger.info("Gate rejected session=%s reason=%s", session_id, reason)
        # Gate rejection is a valid application outcome (insufficient evidence),
        # not a server error — return 200 with a clarifying question.
        return ChatResponse(
            answer=CLARIFYING_RESPONSE,
            citations=[],
            session_id=session_id,
        )

    # ── Build citations from retrieval results ────────────────────────────────
    # Citations are always derived from what was retrieved, never from the LLM's
    # answer text. This makes citation output stable regardless of answer format.
    # Build them here, before generation, so they are available even if generation
    # raises an exception.
    citations = build_citations(results)

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        answer = await chat_service.complete(session_id, request.question, results)
    except Exception as exc:
        logger.exception("Chat completion error for session=%s: %s", session_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to generate a response from Azure OpenAI. "
                "Check service availability and managed identity permissions."
            ),
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        session_id=session_id,
    )


# ── POST /chat/stream (SSE streaming) ────────────────────────────────────────

async def _stream_generator(
    session_id: str,
    question: str,
) -> AsyncGenerator[bytes, None]:
    """Internal SSE byte generator: retrieve → gate → stream tokens → citations → DONE.

    Error handling differs from the JSON endpoint: once a StreamingResponse has
    started, we cannot raise HTTPException. Instead we emit named SSE error events
    so the client can distinguish them from answer tokens and handle them correctly.

    SSE event contract:
      data: <token>          — answer token (unnamed, accumulate to build full answer)
      event: citations       — CitationsPayload JSON (always emitted on success)
      event: error           — {"error": "..."} JSON (emitted on failure, then DONE)
      event: ping            — keepalive, ignore
      data: [DONE]           — stream end sentinel
    """

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        results = await asyncio.to_thread(search_service.retrieve, question)
    except Exception as exc:
        logger.exception("Retrieval error for session=%s: %s", session_id, exc)
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

    # ── Confidence gate ───────────────────────────────────────────────────────
    passed, reason = _check_gate(results)
    if not passed:
        logger.info("Gate rejected session=%s reason=%s", session_id, reason)
        yield _sse_data(CLARIFYING_RESPONSE)
        yield _sse_event("citations", json.dumps({"citations": []}))
        yield _sse_data("[DONE]")
        return

    # ── Build citations from retrieval results ────────────────────────────────
    # Computed from retrieval — not gated on LLM answer formatting.
    citations = build_citations(results)
    citations_payload = CitationsPayload(citations=citations)

    # ── Stream tokens ─────────────────────────────────────────────────────────
    last_ping = time.monotonic()

    try:
        async for token in chat_service.stream_complete(session_id, question, results):
            yield _sse_data(token)

            # Keepalive ping to prevent proxy / load-balancer timeouts
            now = time.monotonic()
            if now - last_ping >= _PING_INTERVAL_SECS:
                yield _sse_event("ping", "keepalive")
                last_ping = now

    except Exception as exc:
        logger.exception("Streaming error for session=%s: %s", session_id, exc)
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

    # ── Citations (always emitted on success) ─────────────────────────────────
    yield _sse_event("citations", citations_payload.model_dump_json())
    yield _sse_data("[DONE]")


@router.post(
    "/chat/stream",
    summary="Streaming chat (SSE)",
    description=(
        "Submit a question and receive a streaming Server-Sent Events response.\n\n"
        "Tokens are streamed as they are generated. Named SSE events:\n\n"
        "- `event: citations` — `CitationsPayload` JSON, always emitted after the last token\n"
        "- `event: error` — `{\"error\": \"...\"}` JSON, emitted if retrieval or generation fails\n"
        "- `event: ping` — keepalive, ignore\n"
        "- `data: [DONE]` — stream end sentinel\n\n"
        "Citations are always derived from retrieval results — not from answer formatting."
    ),
    tags=["chat"],
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "POST /chat/stream session=%s question=%r",
        session_id,
        request.question[:80],
    )

    return StreamingResponse(
        _stream_generator(session_id, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx output buffering
            "Connection": "keep-alive",
        },
    )
