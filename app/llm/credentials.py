"""
Shared Azure credential and token-provider singletons.

Authentication strategy — no API keys anywhere:
  Local development : az login / VS Code Azure sign-in (DefaultAzureCredential chain)
  Azure App Service : system-assigned managed identity (DefaultAzureCredential chain)

Two singletons are exposed:

  get_credential()             → DefaultAzureCredential
                                 Used by:
                                   • SearchClient (retrieval_tool.py)
                                   • AzureOpenAIChatClient (af_agent_factory.py)
                                     — passed as credential=; the AF SDK internally
                                       calls get_bearer_token_provider(credential,
                                       AZURE_OPENAI_TOKEN_ENDPOINT) before each call.

  get_openai_token_provider()  → callable that returns a fresh Bearer token for
                                 Azure OpenAI — passed as azure_ad_token_provider=
                                 to the plain openai.AzureOpenAI embeddings client
                                 in aoai_embeddings.py.  Not used by
                                 AzureOpenAIChatClient (which takes credential= instead).
"""

import logging

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from app.config.settings import AZURE_OPENAI_TOKEN_ENDPOINT

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
    parameter of ``openai.AzureOpenAI`` (used for the embeddings client in
    aoai_embeddings.py).  The plain openai SDK requires a token-provider callable;
    ``AzureOpenAIChatClient`` (Agent Framework SDK) takes ``credential=`` directly
    and handles token acquisition internally.
    """
    global _openai_token_provider
    if _openai_token_provider is None:
        _openai_token_provider = get_bearer_token_provider(
            get_credential(),
            AZURE_OPENAI_TOKEN_ENDPOINT,
        )
        logger.info(
            "OpenAI token provider initialised (endpoint=%s)", AZURE_OPENAI_TOKEN_ENDPOINT
        )
    return _openai_token_provider
