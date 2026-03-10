"""
FastAPI route handlers — intentionally thin per the Agent Framework pattern.

Each route does exactly three things:
  1. Validate the incoming request (Pydantic handles this automatically).
  2. Create an AgentSession carrying the question and session_id.
  3. Delegate to AgentRuntime — all business logic lives there.

No retrieval, gate, generation, or citation logic belongs here.

Endpoints
---------
POST /chat
    Non-streaming JSON response.  Primary integration endpoint for Power Apps
    and PCF components.  Returns ChatResponse{answer, citations, session_id}.
    Raises 502 if Azure AI Search or Azure OpenAI is unreachable.
    Returns 200 with a clarifying question if the confidence gate rejects results.

POST /chat/stream
    Server-Sent Events streaming response.  Tokens arrive as unnamed data lines;
    structured citations arrive as a named "citations" event after the last token.
    Failures are emitted as named "error" events so clients can distinguish them
    from answer tokens.
"""

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent_runtime.agent import AgentRuntime
from app.agent_runtime.session import AgentSession
from app.api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Single shared AgentRuntime — stateless, safe for concurrent requests
_runtime = AgentRuntime()


# ── POST /chat (non-streaming JSON) ──────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Non-streaming chat",
    description=(
        "Submit a question and receive a complete JSON response.\n\n"
        "**Recommended for Power Apps / PCF** — simple request/response, no streaming.\n\n"
        "**Citations** are always built from retrieval results — never from LLM answer "
        "text.  Stable and predictable regardless of answer formatting.\n\n"
        "**HTTP errors:** `502` if Azure AI Search or Azure OpenAI is unreachable; "
        "`200` with a clarifying question if the confidence gate rejects the results."
    ),
    tags=["chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "POST /chat  session=%s  question=%r", session_id, request.question[:80]
    )
    session = AgentSession(question=request.question, session_id=session_id)
    # AgentRuntime.run() raises HTTPException(502) on backend failures;
    # FastAPI propagates it directly — no try/except needed here.
    return await _runtime.run(request.question, session)


# ── POST /chat/stream (SSE streaming) ────────────────────────────────────────

@router.post(
    "/chat/stream",
    summary="Streaming chat (SSE)",
    description=(
        "Submit a question and receive a streaming Server-Sent Events response.\n\n"
        "**SSE event contract:**\n\n"
        "- `data: <token>` (unnamed) — answer token; accumulate client-side\n"
        "- `event: citations` — `CitationsPayload` JSON; always emitted after last token\n"
        "- `event: error` — `{\"error\": \"...\"}` JSON; emitted on failure, then `[DONE]`\n"
        "- `event: ping` — keepalive heartbeat; ignore\n"
        "- `data: [DONE]` — stream end sentinel\n\n"
        "**Citations** are always from retrieval — not from answer formatting."
    ),
    tags=["chat"],
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "POST /chat/stream  session=%s  question=%r",
        session_id, request.question[:80],
    )
    session = AgentSession(question=request.question, session_id=session_id)
    return StreamingResponse(
        _runtime.run_stream(request.question, session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx output buffering
            "Connection": "keep-alive",
        },
    )
