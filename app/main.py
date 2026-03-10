"""
FastAPI application entry point for the PSEG Tech Manual Agent Backend.

Production startup command (Azure App Service):
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app

Local development:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.settings import ALLOWED_ORIGINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs startup logging before yielding to requests."""
    logger.info("PSEG Tech Manual Agent Backend starting up")
    logger.info(
        "Authentication: DefaultAzureCredential (Managed Identity / Azure CLI)"
    )
    logger.info(
        "Orchestration: Microsoft Agent Framework SDK "
        "(AzureOpenAIChatClient + RagContextProvider)"
    )
    logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)
    yield
    logger.info("PSEG Tech Manual Agent Backend shutting down")


app = FastAPI(
    title="PSEG Tech Manual Agent Backend",
    description=(
        "Agent Framework SDK-based RAG API for PSEG field technician manuals.\n\n"
        "**Authentication:** Managed Identity (DefaultAzureCredential) — no API keys.\n\n"
        "**Orchestration:** Microsoft Agent Framework SDK — AzureOpenAIChatClient, "
        "RagContextProvider (BaseContextProvider), InMemoryHistoryProvider.\n\n"
        "**Primary endpoint:** `POST /chat` — complete JSON response for "
        "Power Apps / PCF integration.\n\n"
        "**Streaming endpoint:** `POST /chat/stream` — Server-Sent Events for "
        "live token streaming."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS is always an explicit list of origins — never ["*"].
# Parsed from the ALLOWED_ORIGINS environment variable (comma-separated).
# Default for local dev: http://localhost:3000,http://localhost:8000
# Production: set ALLOWED_ORIGINS to your Power Apps / PCF domains.
#
# Using an explicit origin list means allow_credentials=True is valid:
# the CORS spec only forbids combining allow_origins=["*"] with credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

app.include_router(router)


@app.get("/health", tags=["health"], summary="Health check")
async def health() -> dict:
    """Returns 200 OK if the service is running.

    Used by Azure App Service health probes and uptime monitors.
    """
    return {"status": "ok"}
