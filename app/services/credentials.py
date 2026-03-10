"""
Shared Azure credential instances using Managed Identity (DefaultAzureCredential).

Authentication strategy:
- Locally: uses your Azure CLI / VS Code / Azure PowerShell sign-in (no keys needed)
- Azure App Service: uses the system-assigned managed identity automatically

No API keys are stored or read anywhere in this module.
"""

import logging
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from app.config.settings import AZURE_OPENAI_TOKEN_SCOPE

logger = logging.getLogger(__name__)

# Singleton — one credential object reused across all Azure SDK clients
_credential: DefaultAzureCredential | None = None
_openai_token_provider = None


def get_credential() -> DefaultAzureCredential:
    """Return the shared DefaultAzureCredential instance.

    On first call, this initialises the credential chain:
    1. Environment variables (CI/CD use-case)
    2. Workload Identity (AKS)
    3. Managed Identity (Azure App Service, Azure Functions, etc.)
    4. Azure CLI (local development)
    5. Azure Developer CLI
    6. VS Code / Visual Studio
    7. Azure PowerShell
    """
    global _credential
    if _credential is None:
        logger.info("Initialising DefaultAzureCredential")
        _credential = DefaultAzureCredential()
    return _credential


def get_openai_token_provider():
    """Return a callable that fetches a fresh Azure AD token for Azure OpenAI.

    The returned callable is passed to AzureOpenAI / AsyncAzureOpenAI as the
    `azure_ad_token_provider` parameter. The SDK calls it automatically when
    a token is needed or about to expire.
    """
    global _openai_token_provider
    if _openai_token_provider is None:
        _openai_token_provider = get_bearer_token_provider(
            get_credential(),
            AZURE_OPENAI_TOKEN_SCOPE,
        )
    return _openai_token_provider
