"""
API route handlers for the PSEG Tech Manual Chat Backend.

Endpoints:
  GET  /health          — liveness check
  POST /chat            — non-streaming JSON response (primary integration endpoint)
  POST /chat/stream     — Server-Sent Events streaming response
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
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

    Gate rejects if:
    - fewer chunks than MIN_RESULTS were retrieved, OR
    - average effective score falls below the configured threshold.
    """
    if len(results) < MIN_RESULTS:
        return False, f"only {len(results)} chunks retrieved (minimum {MIN_RESULTS})"

    if USE_SEMANTIC_RERANKER:
        reranker_scores = [
            r["reranker_score"]
            for r in results
            if r.get("reranker_score") is not None
        ]
        if reranker_scores:
            avg = sum(reranker_scores) / len(reranker_scores)
            if avg < MIN_RERANKER_SCORE:
                return False, f"avg reranker score {avg:.3f} < threshold {MIN_RERANKER_SCORE}"
            return True, ""

    # Fall back to base score gate
    avg = sum(r["score"] for r in results) / len(results)
    if avg < MIN_AVG_SCORE:
        return False, f"avg score {avg:.4f} < threshold {MIN_AVG_SCORE}"
    return True, ""


def _has_inline_citations(answer: str) -> bool:
    """Return True if the LLM used inline citation markers."""
    return "Sources:" in answer or "[1]" in answer


# ── POST /chat (non-streaming, JSON) ─────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Non-streaming chat",
    description=(
        "Submit a question and receive a complete JSON response with the answer "
        "and structured citations. This is the recommended endpoint for Power Apps "
        "and PCF integrations."
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
    try:
        results = await asyncio.to_thread(search_service.retrieve, request.question)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        return ChatResponse(
            answer="I encountered an error retrieving information from the manuals. Please try again.",
            citations=[],
            session_id=session_id,
        )

    # ── Confidence gate ───────────────────────────────────────────────────────
    passed, reason = _check_gate(results)
    if not passed:
        logger.info("Gate rejected: %s", reason)
        return ChatResponse(
            answer=CLARIFYING_RESPONSE,
            citations=[],
            session_id=session_id,
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        answer = await chat_service.complete(session_id, request.question, results)
    except Exception as exc:
        logger.exception("Chat completion error: %s", exc)
        return ChatResponse(
            answer="I encountered an error generating a response. Please try again.",
            citations=[],
            session_id=session_id,
        )

    # ── Build citations ───────────────────────────────────────────────────────
    citations = build_citations(results) if _has_inline_citations(answer) else []

    return ChatResponse(
        answer=answer,
        citations=citations,
        session_id=session_id,
    )


# ── POST /chat/stream (SSE streaming) ────────────────────────────────────────

def _sse_data(text: str) -> bytes:
    """Encode a token as an SSE data line, escaping embedded newlines."""
    escaped = text.replace("\n", "\\n")
    return f"data: {escaped}\n\n".encode("utf-8")


def _sse_event(event: str, data: str) -> bytes:
    """Encode a named SSE event."""
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


async def _stream_generator(
    session_id: str,
    question: str,
) -> AsyncGenerator[bytes, None]:
    """Internal SSE generator: retrieve → gate → stream tokens → citations → DONE."""

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        results = await asyncio.to_thread(search_service.retrieve, question)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        yield _sse_data(
            "I encountered an error retrieving information from the manuals. Please try again."
        )
        yield _sse_event("citations", json.dumps({"citations": []}))
        yield _sse_data("[DONE]")
        return

    # ── Confidence gate ───────────────────────────────────────────────────────
    passed, reason = _check_gate(results)
    if not passed:
        logger.info("Gate rejected: %s", reason)
        yield _sse_data(CLARIFYING_RESPONSE)
        yield _sse_event("citations", json.dumps({"citations": []}))
        yield _sse_data("[DONE]")
        return

    # ── Stream tokens ─────────────────────────────────────────────────────────
    answer_buf: list[str] = []
    last_ping = time.monotonic()

    try:
        async for token in chat_service.stream_complete(session_id, question, results):
            answer_buf.append(token)
            yield _sse_data(token)

            # Keepalive ping to prevent proxy/load-balancer timeouts
            now = time.monotonic()
            if now - last_ping >= _PING_INTERVAL_SECS:
                yield _sse_event("ping", "keepalive")
                last_ping = now

    except Exception as exc:
        logger.exception("Streaming error: %s", exc)
        yield _sse_data("\n\nI encountered an error while generating the response.")

    # ── Citations ─────────────────────────────────────────────────────────────
    answer = "".join(answer_buf)
    citations = build_citations(results) if _has_inline_citations(answer) else []
    payload = CitationsPayload(citations=citations)
    yield _sse_event("citations", payload.model_dump_json())
    yield _sse_data("[DONE]")


@router.post(
    "/chat/stream",
    summary="Streaming chat (SSE)",
    description=(
        "Submit a question and receive a streaming Server-Sent Events response. "
        "Tokens are streamed as they are generated. A 'citations' named event is "
        "emitted at the end, followed by a [DONE] sentinel."
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
