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
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for broad integration compatibility.
# Restrict allow_origins to specific domains in a locked-down environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["health"], summary="Health check")
async def health() -> dict:
    """Returns 200 OK if the service is running.

    Used by Azure App Service health probes and uptime monitors.
    """
    return {"status": "ok"}
