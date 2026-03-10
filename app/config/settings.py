"""
Application configuration — loaded from environment variables via python-dotenv.

Authentication model: Managed Identity (DefaultAzureCredential).
No API keys. No secrets. Works locally via `az login` and on Azure App Service
via the system-assigned managed identity without any code changes.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT: str = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: str = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

# Managed Identity token scope for Azure OpenAI.
# Commercial Azure (default): https://cognitiveservices.azure.com/.default
# Azure Government / GCC High:  https://cognitiveservices.azure.us/.default
AZURE_OPENAI_TOKEN_SCOPE: str = os.getenv(
    "AZURE_OPENAI_TOKEN_SCOPE",
    "https://cognitiveservices.azure.com/.default",
)

# ── Azure AI Search ───────────────────────────────────────────────────────────
# Authentication: DefaultAzureCredential passed directly to SearchClient.
# No AZURE_SEARCH_API_KEY is read or required.
AZURE_SEARCH_ENDPOINT: str = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "rag-psegtechm-index-finalv2")

# ── CORS ──────────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS is a comma-separated list of allowed origins.
#
# Default (when env var is not set):
#   http://localhost:3000,http://localhost:8000
#   Covers local Swagger UI, curl, and local frontends during development.
#
# For production (Power Apps / PCF), override with your actual domain(s):
#   ALLOWED_ORIGINS=https://apps.powerapps.com,https://your-org.crm.dynamics.com
#
# Why we never use allow_origins=["*"]:
#   The CORS spec forbids combining allow_origins=["*"] with allow_credentials=True.
#   All modern browsers enforce this — they will block such responses.
#   By always using a specific origin list we can safely set allow_credentials=True.
_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:8000"
_raw_origins: str = os.getenv("ALLOWED_ORIGINS", _DEFAULT_CORS_ORIGINS).strip()
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── Index field mappings ──────────────────────────────────────────────────────
# Must match the actual field names in your Azure AI Search index schema exactly.
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
SEARCH_PAGE_FIELD: str = os.getenv("SEARCH_PAGE_FIELD", "")  # blank = index has no page field

# ── Retrieval tuning ──────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))                    # max chunks after all filtering
RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "15"))  # raw Azure Search pool
VECTOR_K: int = int(os.getenv("VECTOR_K", "50"))             # kNN neighbours for vector query

# ── Confidence gate ───────────────────────────────────────────────────────────
USE_SEMANTIC_RERANKER: bool = os.getenv("USE_SEMANTIC_RERANKER", "true").lower() == "true"
SEMANTIC_CONFIG_NAME: str = os.getenv("SEMANTIC_CONFIG_NAME", "manual-semantic-config")
QUERY_LANGUAGE: str = os.getenv("QUERY_LANGUAGE", "en-us")

# Minimum number of chunks required before the gate allows generation.
# Default is 1 — a single highly relevant chunk is sufficient.
# The gate does NOT hard-fail just because fewer than 2 chunks were retrieved.
# Raise to 2+ only for stricter multi-citation requirements.
MIN_RESULTS: int = int(os.getenv("MIN_RESULTS", "1"))

# Thresholds applied to the TOP chunk's score — not the average score.
# Using the top score avoids rejecting responses where one strong chunk exists
# alongside lower-scoring supporting chunks.
#   Semantic reranker mode: reranker_score range 0.0 – 4.0
#   Hybrid / RRF mode:      base score      range 0.01 – 0.033
MIN_AVG_SCORE: float = float(os.getenv("MIN_AVG_SCORE", "0.01"))
MIN_RERANKER_SCORE: float = float(os.getenv("MIN_RERANKER_SCORE", "0.2"))

# ── Diversity filtering ───────────────────────────────────────────────────────
DIVERSITY_BY_SOURCE: bool = os.getenv("DIVERSITY_BY_SOURCE", "true").lower() == "true"
MAX_CHUNKS_PER_SOURCE: int = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2"))
DOMINANT_SOURCE_SCORE_RATIO: float = float(os.getenv("DOMINANT_SOURCE_SCORE_RATIO", "1.5"))
MAX_CHUNKS_DOMINANT_SOURCE: int = int(os.getenv("MAX_CHUNKS_DOMINANT_SOURCE", "4"))
SCORE_GAP_MIN_RATIO: float = float(os.getenv("SCORE_GAP_MIN_RATIO", "0.55"))

# ── In-memory conversation history ────────────────────────────────────────────
# DISABLED by default — the backend is stateless.
#
# Stateless (false, default):
#   Every request is independent. Safe for App Service scale-out (instances
#   do not share in-process memory) and restarts (no silent data loss).
#
# Enabled (true):
#   Lightweight in-process history per session_id (last 10 turns retained).
#   Only suitable for single-instance local dev or demos.
#   History is lost on restart or scale-out — not suitable for production.
ENABLE_IN_MEMORY_HISTORY: bool = (
    os.getenv("ENABLE_IN_MEMORY_HISTORY", "false").lower() == "true"
)

# ── Debug ─────────────────────────────────────────────────────────────────────
TRACE_MODE: bool = os.getenv("TRACE_MODE", "false").lower() == "true"
