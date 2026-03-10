"""
FastAPI application entry point for the PSEG Tech Manual Chat Backend.

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
from app.config.settings import ALLOWED_ORIGINS, CORS_ALLOW_CREDENTIALS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs startup logic before yielding to requests."""
    logger.info("PSEG Tech Manual Chat Backend starting up")
    logger.info(
        "Authentication: DefaultAzureCredential (Managed Identity / Azure CLI)"
    )
    if ALLOWED_ORIGINS:
        logger.info("CORS: specific origins=%s credentials=True", ALLOWED_ORIGINS)
    else:
        logger.info("CORS: wildcard (*) mode — credentials=False")
    yield
    logger.info("PSEG Tech Manual Chat Backend shutting down")


app = FastAPI(
    title="PSEG Tech Manual Chat Backend",
    description=(
        "RAG-based chat API for PSEG field technician manuals.\n\n"
        "**Authentication:** Managed Identity (DefaultAzureCredential) — no API keys.\n\n"
        "**Primary endpoint:** `POST /chat` — returns a complete JSON response "
        "suitable for Power Apps / PCF integration.\n\n"
        "**Streaming endpoint:** `POST /chat/stream` — Server-Sent Events for "
        "live token streaming."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Rules:
#   - allow_origins=["*"] + allow_credentials=True is INVALID per CORS spec.
#     Browsers reject such responses with a CORS error.
#   - When ALLOWED_ORIGINS is set:  use specific origins + allow_credentials=True
#   - When ALLOWED_ORIGINS is empty: wildcard mode + allow_credentials=False
#     (safe for most Power Apps / PCF call patterns that use bearer tokens)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=CORS_ALLOW_CREDENTIALS,
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
