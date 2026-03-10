"""
Shared Azure credential and token-provider singletons.

Authentication strategy — no API keys anywhere:
  Local development : az login / VS Code Azure sign-in (DefaultAzureCredential chain)
  Azure App Service : system-assigned managed identity (DefaultAzureCredential chain)

Two singletons are exposed:
  get_credential()             → DefaultAzureCredential (used by SearchClient)
  get_openai_token_provider()  → callable that returns a fresh Bearer token for
                                 Azure OpenAI — passed to AzureOpenAIChatClient
                                 and AzureOpenAI (embeddings) as
                                 azure_ad_token_provider=...
"""

import logging

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from app.config.settings import AZURE_OPENAI_TOKEN_SCOPE

logger = logging.getLogger(__name__)

_credential: DefaultAzureCredential | None = None
_openai_token_provider = None


def get_credential() -> DefaultAzureCredential:
    """Return the shared DefaultAzureCredential instance (lazy-init)."""
    global _credential
    if _credential is None:
        logger.info("Initialising DefaultAzureCredential")
        _credential = DefaultAzureCredential()
    return _credential


def get_openai_token_provider():
    """Return a callable that fetches / refreshes an Azure AD token for OpenAI.

    The returned callable is compatible with the ``azure_ad_token_provider``
    parameter of both ``openai.AzureOpenAI`` and
    ``agent_framework.azure.AzureOpenAIChatClient``.  The SDK calls it
    automatically whenever a token is needed or about to expire.
    """
    global _openai_token_provider
    if _openai_token_provider is None:
        _openai_token_provider = get_bearer_token_provider(
            get_credential(),
            AZURE_OPENAI_TOKEN_SCOPE,
        )
        logger.info(
            "OpenAI token provider initialised (scope=%s)", AZURE_OPENAI_TOKEN_SCOPE
        )
    return _openai_token_provider
