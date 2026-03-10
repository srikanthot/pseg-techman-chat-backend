"""
Azure OpenAI embeddings using Managed Identity.

Generates query vectors at search time (the index has no built-in vectorizer).
Uses a lazy-initialised sync AzureOpenAI client — safe to call from a
thread pool (retrieve() is always invoked via asyncio.to_thread).
"""

import logging
from openai import AzureOpenAI
from app.config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
)
from app.services.credentials import get_openai_token_provider

logger = logging.getLogger(__name__)

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
            "AzureOpenAI (embeddings) initialised: endpoint=%s deployment=%s",
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        )
    return _client


def embed(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Raises on failure — the caller (retrieve) handles the exception and
    falls back to keyword-only search.
    """
    response = _get_client().embeddings.create(
        model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding
