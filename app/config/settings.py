"""
Application configuration loaded from environment variables.

No secrets are stored here. All authentication uses Managed Identity
(DefaultAzureCredential) — works locally via Azure CLI login and on
Azure App Service via system-assigned managed identity.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT: str = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: str = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

# Azure AD token scope for OpenAI.
# For Azure Government (GCC High): https://cognitiveservices.azure.us/.default
AZURE_OPENAI_TOKEN_SCOPE: str = os.getenv(
    "AZURE_OPENAI_TOKEN_SCOPE",
    "https://cognitiveservices.azure.com/.default",
)

# ── Azure AI Search ───────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT: str = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "rag-psegtechm-index-finalv2")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins.
# Leave blank for wildcard mode (allow_origins=["*"], allow_credentials=False).
# Set to specific origins to enable credential-bearing requests:
#   ALLOWED_ORIGINS=https://apps.powerapps.com,https://your-org.dynamics.com
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else []
)
# Wildcard mode is active when ALLOWED_ORIGINS is empty.
# Per the CORS spec, allow_origins=["*"] MUST NOT be combined with
# allow_credentials=True — browsers will reject such responses.
CORS_ALLOW_CREDENTIALS: bool = bool(ALLOWED_ORIGINS)

# ── Index field mappings ──────────────────────────────────────────────────────
# These must match the actual field names in your Azure AI Search index schema.
SEARCH_CONTENT_FIELD: str = os.getenv("SEARCH_CONTENT_FIELD", "chunk")
SEARCH_SEMANTIC_CONTENT_FIELD: str = os.getenv("SEARCH_SEMANTIC_CONTENT_FIELD", "chunk_for_semantic")
SEARCH_VECTOR_FIELD: str = os.getenv("SEARCH_VECTOR_FIELD", "text_vector")
SEARCH_FILENAME_FIELD: str = os.getenv("SEARCH_FILENAME_FIELD", "source_file")
SEARCH_URL_FIELD: str = os.getenv("SEARCH_URL_FIELD", "source_url")
SEARCH_CHUNK_ID_FIELD: str = os.getenv("SEARCH_CHUNK_ID_FIELD", "chunk_id")
SEARCH_TITLE_FIELD: str = os.getenv("SEARCH_TITLE_FIELD", "title")
SEARCH_SECTION1_FIELD: str = os.getenv("SEARCH_SECTION1_FIELD", "header_1")
SEARCH_SECTION2_FIELD: str = os.getenv("SEARCH_SECTION2_FIELD", "header_2")
SEARCH_SECTION3_FIELD: str = os.getenv("SEARCH_SECTION3_FIELD", "header_3")
# Leave blank if your index has no page number field
SEARCH_PAGE_FIELD: str = os.getenv("SEARCH_PAGE_FIELD", "")

# ── Retrieval tuning ──────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))                          # max chunks returned after all filtering
RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "15"))  # raw candidates from Azure Search
VECTOR_K: int = int(os.getenv("VECTOR_K", "50"))                   # nearest-neighbor count for vector query

# ── Quality / confidence gate ─────────────────────────────────────────────────
USE_SEMANTIC_RERANKER: bool = os.getenv("USE_SEMANTIC_RERANKER", "true").lower() == "true"
SEMANTIC_CONFIG_NAME: str = os.getenv("SEMANTIC_CONFIG_NAME", "manual-semantic-config")
QUERY_LANGUAGE: str = os.getenv("QUERY_LANGUAGE", "en-us")

# Minimum number of retrieved chunks before attempting generation.
# Default is 1 — a single highly relevant chunk is sufficient.
# Raise to 2+ for stricter environments.
MIN_RESULTS: int = int(os.getenv("MIN_RESULTS", "1"))

# Gate threshold applied to the TOP chunk's score (not the average).
# Scoring depends on reranker mode:
#   - With semantic reranker: reranker_score range 0.0 – 4.0
#   - Without reranker:       RRF / hybrid score range 0.01 – 0.033
MIN_AVG_SCORE: float = float(os.getenv("MIN_AVG_SCORE", "0.01"))
MIN_RERANKER_SCORE: float = float(os.getenv("MIN_RERANKER_SCORE", "0.2"))

# ── Diversity filtering ───────────────────────────────────────────────────────
DIVERSITY_BY_SOURCE: bool = os.getenv("DIVERSITY_BY_SOURCE", "true").lower() == "true"
MAX_CHUNKS_PER_SOURCE: int = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2"))
DOMINANT_SOURCE_SCORE_RATIO: float = float(os.getenv("DOMINANT_SOURCE_SCORE_RATIO", "1.5"))
MAX_CHUNKS_DOMINANT_SOURCE: int = int(os.getenv("MAX_CHUNKS_DOMINANT_SOURCE", "4"))
SCORE_GAP_MIN_RATIO: float = float(os.getenv("SCORE_GAP_MIN_RATIO", "0.55"))

# ── Session / conversation history ────────────────────────────────────────────
# Disabled by default — the backend is stateless, which is safe for App Service
# scale-out and restarts. Enable only for single-instance local development or
# when you accept that history will reset on restart / scale events.
ENABLE_IN_MEMORY_HISTORY: bool = (
    os.getenv("ENABLE_IN_MEMORY_HISTORY", "false").lower() == "true"
)

# ── Debug ─────────────────────────────────────────────────────────────────────
TRACE_MODE: bool = os.getenv("TRACE_MODE", "false").lower() == "true"
