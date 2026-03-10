"""
Azure OpenAI query-embedding generation via Managed Identity.

No API key — authentication is via azure_ad_token_provider backed by
DefaultAzureCredential (az login locally, managed identity on App Service).

The Azure AI Search index has no built-in vectorizer, so query embeddings
must be generated here before issuing a hybrid search.  This module exposes
a single synchronous embed() function designed to be called via
asyncio.to_thread() from the async retrieval path so it never blocks the
event loop.
"""

import logging

from openai import AzureOpenAI

from app.config.settings import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)
from app.llm.credentials import get_openai_token_provider

logger = logging.getLogger(__name__)

# Lazy-init singleton — one client shared across all requests
_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=get_openai_token_provider(),
            api_version=AZURE_OPENAI_API_VERSION,
        )
        logger.info(
            "AzureOpenAI (embeddings) initialised — endpoint=%s deployment=%s",
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        )
    return _client


def embed(text: str) -> list[float]:
    """Return an embedding vector for *text*.

    Called synchronously from retrieval_tool.retrieve() which is itself
    run via asyncio.to_thread().

    Raises
    ------
    openai.OpenAIError
        Propagated to the caller — RetrievalTool logs and falls back to
        keyword-only search on failure.
    """
    response = _get_client().embeddings.create(
        model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding
