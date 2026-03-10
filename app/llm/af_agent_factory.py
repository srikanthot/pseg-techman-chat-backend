"""
Microsoft Agent Framework SDK — AzureOpenAIChatClient with Managed Identity.

This module is the primary Agent Framework SDK integration point.
It creates two module-level singletons consumed by AgentRuntime:

  af_agent     — the fully configured Agent Framework ChatAgent instance
  rag_provider — the shared RagContextProvider instance; AgentRuntime calls
                 rag_provider.store_results(af_session, results) before
                 each af_agent.run() call so the provider can inject
                 pre-fetched chunks without a second Azure AI Search query.

Key Agent Framework SDK primitives used here:
  AzureOpenAIChatClient   — wraps Azure OpenAI; accepts credential=DefaultAzureCredential()
                            so no API key is ever needed or stored.
                            Reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
                            AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, and
                            AZURE_OPENAI_TOKEN_ENDPOINT from environment variables;
                            handles token acquisition internally.
  client.as_agent()       — the official SDK factory that returns a ChatAgent
                            configured with a system prompt and context_providers.
  InMemoryHistoryProvider — Agent Framework built-in provider that stores
                            multi-turn conversation history in session.state.
                            Included only when ENABLE_IN_MEMORY_HISTORY=true.
  RagContextProvider      — our custom BaseContextProvider subclass (see
                            agent_runtime/af_rag_context_provider.py) that
                            injects retrieved chunks as grounded context via
                            context.extend_instructions() before each LLM call.

Managed Identity vs API key:
  API key approach : AzureOpenAIChatClient(api_key="...")
  this repo        : AzureOpenAIChatClient(credential=DefaultAzureCredential())
                     — no key, token auto-refreshes via managed identity
"""

import logging

from agent_framework import InMemoryHistoryProvider
from agent_framework.azure import AzureOpenAIChatClient

from app.agent_runtime.af_rag_context_provider import RagContextProvider
from app.agent_runtime.prompts import SYSTEM_PROMPT
from app.config.settings import ENABLE_IN_MEMORY_HISTORY
from app.llm.credentials import get_credential

logger = logging.getLogger(__name__)

# ── Shared RagContextProvider ─────────────────────────────────────────────────
# A single instance is shared between this factory and AgentRuntime so that
# AgentRuntime can call rag_provider.store_results(af_session, results) to hand
# off pre-retrieved chunks before each agent.run() invocation.
rag_provider = RagContextProvider()

# ── AzureOpenAIChatClient (Managed Identity) ──────────────────────────────────
# Pass credential=DefaultAzureCredential() — the Agent Framework SDK internally
# calls get_bearer_token_provider(credential, AZURE_OPENAI_TOKEN_ENDPOINT) and
# passes the result to AsyncAzureOpenAI as azure_ad_token_provider.
# The client reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME directly from environment variables.
_client = AzureOpenAIChatClient(
    credential=get_credential(),
)

# ── Context providers list ────────────────────────────────────────────────────
# Build the list at import time so it is fixed for the lifetime of the process.
# InMemoryHistoryProvider is optional — disabled by default (stateless mode).
_context_providers = [rag_provider]

if ENABLE_IN_MEMORY_HISTORY:
    # Prepend so history is resolved before RAG context is injected.
    _context_providers.insert(0, InMemoryHistoryProvider())
    logger.info(
        "Agent Framework: InMemoryHistoryProvider enabled — "
        "multi-turn history stored in session.state (single-process only)"
    )
else:
    logger.info(
        "Agent Framework: stateless mode — "
        "ENABLE_IN_MEMORY_HISTORY=false, each request is independent"
    )

# ── ChatAgent singleton ───────────────────────────────────────────────────────
# client.as_agent() is the Agent Framework SDK factory method that:
#   1. Binds the system prompt (instructions) to every conversation.
#   2. Registers context_providers whose before_run() hooks fire before each
#      LLM call to inject history and/or RAG context.
#   3. Returns an agent object whose .run(question, stream=True, session=...)
#      drives the full prompt-assembly → LLM → streaming-response cycle.
af_agent = _client.as_agent(
    name="PSEGTechManualAgent",
    instructions=SYSTEM_PROMPT,
    context_providers=_context_providers,
)

logger.info(
    "Agent Framework ChatAgent ready — name=%s history=%s",
    "PSEGTechManualAgent",
    "enabled" if ENABLE_IN_MEMORY_HISTORY else "disabled (stateless)",
)
