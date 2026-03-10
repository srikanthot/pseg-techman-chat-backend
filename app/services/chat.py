"""
Azure OpenAI chat completion using Managed Identity.

Uses AsyncAzureOpenAI so both the streaming and non-streaming paths are
fully async — no thread-pool blocking in the event loop.

Conversation history:
  Disabled by default (ENABLE_IN_MEMORY_HISTORY=false).
  The backend is stateless by default, which is safe for App Service
  scale-out and restarts. Enable history only for single-instance
  development or when loss of history on restart is acceptable.
  When disabled, every request is independent (stateless mode).
"""

import logging
from typing import AsyncGenerator

from openai import AsyncAzureOpenAI

from app.config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    ENABLE_IN_MEMORY_HISTORY,
)
from app.services.credentials import get_openai_token_provider
from app.services.prompts import SYSTEM_PROMPT, build_context_blocks

logger = logging.getLogger(__name__)

# Lazy-initialised singleton async client
_async_client: AsyncAzureOpenAI | None = None

# In-memory conversation history: session_id → list[{role, content}]
# Only populated when ENABLE_IN_MEMORY_HISTORY=true.
# Scoped to a single process instance — resets on restart or scale-out.
_histories: dict[str, list[dict]] = {}

_MAX_HISTORY_MESSAGES = 20  # 10 turns (user + assistant per turn)


def _get_client() -> AsyncAzureOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=get_openai_token_provider(),
            api_version=AZURE_OPENAI_API_VERSION,
        )
        logger.info(
            "AsyncAzureOpenAI (chat) initialised: endpoint=%s deployment=%s "
            "history=%s",
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_CHAT_DEPLOYMENT,
            "enabled" if ENABLE_IN_MEMORY_HISTORY else "disabled (stateless)",
        )
    return _async_client


def _get_history(session_id: str) -> list[dict]:
    """Return the conversation history for this session, or [] if disabled."""
    if not ENABLE_IN_MEMORY_HISTORY:
        return []
    return _histories.get(session_id, [])


def _save_turn(session_id: str, question: str, answer: str) -> None:
    """Persist the user/assistant turn. No-op when history is disabled."""
    if not ENABLE_IN_MEMORY_HISTORY:
        return
    if session_id not in _histories:
        _histories[session_id] = []
    _histories[session_id].append({"role": "user", "content": question})
    _histories[session_id].append({"role": "assistant", "content": answer})
    # Trim to keep memory bounded
    _histories[session_id] = _histories[session_id][-_MAX_HISTORY_MESSAGES:]


def _build_messages(session_id: str, question: str, context_blocks: str) -> list[dict]:
    """Assemble the full messages list for this turn.

    History is only included when ENABLE_IN_MEMORY_HISTORY=true.
    In stateless mode the list is simply [system, user].
    """
    history = _get_history(session_id)
    user_content = (
        f"Question:\n{question}\n\n"
        f"Context (retrieved from technical manuals):\n{context_blocks}\n\n"
        "Answer the question using ONLY the context above. "
        "Reference each source by its [N] label inline. "
        'Include a "Sources:" section at the end.'
    )
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )


async def complete(session_id: str, question: str, results: list[dict]) -> str:
    """Non-streaming chat completion. Returns the full answer string.

    Raises on Azure OpenAI failure — the route handler converts this to
    an appropriate HTTPException.
    """
    context_blocks = build_context_blocks(results)
    messages = _build_messages(session_id, question, context_blocks)

    response = await _get_client().chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content or ""
    _save_turn(session_id, question, answer)
    logger.info("complete() session=%s answer_len=%d", session_id, len(answer))
    return answer


async def stream_complete(
    session_id: str,
    question: str,
    results: list[dict],
) -> AsyncGenerator[str, None]:
    """Streaming chat completion. Yields token strings one at a time.

    Raises on Azure OpenAI failure — the route handler catches this and
    emits a named SSE error event.
    Conversation history is saved after the full answer is assembled.
    """
    context_blocks = build_context_blocks(results)
    messages = _build_messages(session_id, question, context_blocks)

    stream = await _get_client().chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
        stream=True,
    )

    answer_buf: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            token = delta.content
            answer_buf.append(token)
            yield token

    answer = "".join(answer_buf)
    _save_turn(session_id, question, answer)
    logger.info(
        "stream_complete() session=%s answer_len=%d", session_id, len(answer)
    )
