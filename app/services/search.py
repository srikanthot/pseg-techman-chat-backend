"""
Azure AI Search retrieval using Managed Identity.

Implements hybrid search: BM25 keyword + Ada-002 vector, with optional
semantic reranking, confidence gate, adaptive diversity filter, and
score-gap filter — preserving all behaviour from the original project.

The SearchClient uses DefaultAzureCredential (no API key).
retrieve() is a synchronous function designed to be called via
asyncio.to_thread() from async route handlers.
"""

import logging
import re
from typing import Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType

from app.config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    SEARCH_CONTENT_FIELD,
    SEARCH_SEMANTIC_CONTENT_FIELD,
    SEARCH_VECTOR_FIELD,
    SEARCH_FILENAME_FIELD,
    SEARCH_URL_FIELD,
    SEARCH_CHUNK_ID_FIELD,
    SEARCH_TITLE_FIELD,
    SEARCH_SECTION1_FIELD,
    SEARCH_SECTION2_FIELD,
    SEARCH_SECTION3_FIELD,
    SEARCH_PAGE_FIELD,
    TOP_K,
    RETRIEVAL_CANDIDATES,
    VECTOR_K,
    USE_SEMANTIC_RERANKER,
    SEMANTIC_CONFIG_NAME,
    QUERY_LANGUAGE,
    DIVERSITY_BY_SOURCE,
    MAX_CHUNKS_PER_SOURCE,
    DOMINANT_SOURCE_SCORE_RATIO,
    MAX_CHUNKS_DOMINANT_SOURCE,
    SCORE_GAP_MIN_RATIO,
    TRACE_MODE,
)
from app.services.credentials import get_credential
from app.services.embeddings import embed

logger = logging.getLogger(__name__)

# Singleton SearchClient — thread-safe, reuse across requests
_search_client: SearchClient | None = None

# Conversational filler to strip from keyword queries (improves BM25 precision)
_FILLER_RE = re.compile(
    r"\b(right now|currently|can you|what should|tell me|please|how do i|how to|"
    r"what is|what are|i need to|i want to|give me|explain|describe)\b",
    re.IGNORECASE,
)

# Patterns that identify Table of Contents / index pages
_TOC_RE = re.compile(
    r"(table of contents|\.{4,}|\. \. \.|contents.*\d{1,3}$|\bindex\b)",
    re.IGNORECASE | re.MULTILINE,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=get_credential(),
        )
        logger.info(
            "SearchClient initialised: endpoint=%s index=%s",
            AZURE_SEARCH_ENDPOINT,
            AZURE_SEARCH_INDEX,
        )
    return _search_client


def _select_fields() -> list[str]:
    """Build the list of index fields to retrieve (exclude vector fields)."""
    fields = [
        SEARCH_CONTENT_FIELD,
        SEARCH_FILENAME_FIELD,
        SEARCH_URL_FIELD,
        SEARCH_CHUNK_ID_FIELD,
        SEARCH_TITLE_FIELD,
        SEARCH_SECTION1_FIELD,
        SEARCH_SECTION2_FIELD,
        SEARCH_SECTION3_FIELD,
    ]
    if SEARCH_SEMANTIC_CONTENT_FIELD:
        fields.append(SEARCH_SEMANTIC_CONTENT_FIELD)
    if SEARCH_PAGE_FIELD:
        fields.append(SEARCH_PAGE_FIELD)
    # Remove empty strings (optional fields not configured)
    return [f for f in fields if f]


def _distill_keyword_query(question: str) -> str:
    """Strip conversational filler to improve BM25 recall on technical terms."""
    distilled = _FILLER_RE.sub("", question).strip()
    return distilled if len(distilled) >= 10 else question


def _normalize(doc) -> dict:
    """Map a raw Azure Search document to the canonical result schema."""

    def get(field: str) -> str:
        if not field:
            return ""
        return str(doc.get(field) or "")

    reranker_raw = doc.get("@search.reranker_score")
    return {
        "content": get(SEARCH_CONTENT_FIELD),
        "semantic_content": get(SEARCH_SEMANTIC_CONTENT_FIELD),
        "title": get(SEARCH_TITLE_FIELD),
        "source": get(SEARCH_FILENAME_FIELD),
        "url": get(SEARCH_URL_FIELD),
        "chunk_id": get(SEARCH_CHUNK_ID_FIELD),
        "section1": get(SEARCH_SECTION1_FIELD),
        "section2": get(SEARCH_SECTION2_FIELD),
        "section3": get(SEARCH_SECTION3_FIELD),
        "page": get(SEARCH_PAGE_FIELD) if SEARCH_PAGE_FIELD else "",
        "score": float(doc.get("@search.score") or 0.0),
        "reranker_score": float(reranker_raw) if reranker_raw is not None else None,
    }


def _effective_score(r: dict) -> float:
    """Return the reranker score if available, otherwise the base score."""
    if r.get("reranker_score") is not None:
        return r["reranker_score"]
    return r["score"]


def _is_toc_chunk(content: str) -> bool:
    """Return True if the chunk appears to be a Table of Contents page."""
    return bool(_TOC_RE.search(content[:500]))


def _adaptive_diversity(results: list[dict]) -> list[dict]:
    """Cap chunks per source, allowing a higher cap for a dominant source.

    A source is dominant when its top score is >= DOMINANT_SOURCE_SCORE_RATIO
    times the next-best source's top score. This rewards coherence when one
    document clearly owns the topic.
    """
    if not results:
        return results

    # Find the top score per source
    source_top: dict[str, float] = {}
    for r in results:
        src = r["source"]
        eff = _effective_score(r)
        if src not in source_top or eff > source_top[src]:
            source_top[src] = eff

    dominant_source: Optional[str] = None
    if len(source_top) >= 2:
        sorted_sources = sorted(source_top.items(), key=lambda x: x[1], reverse=True)
        top_src, top_score = sorted_sources[0]
        _, second_score = sorted_sources[1]
        if second_score > 0 and top_score / second_score >= DOMINANT_SOURCE_SCORE_RATIO:
            dominant_source = top_src

    filtered: list[dict] = []
    counts: dict[str, int] = {}
    for r in results:
        src = r["source"]
        cap = MAX_CHUNKS_DOMINANT_SOURCE if src == dominant_source else MAX_CHUNKS_PER_SOURCE
        if counts.get(src, 0) < cap:
            filtered.append(r)
            counts[src] = counts.get(src, 0) + 1
    return filtered


def _filter_score_gap(results: list[dict]) -> list[dict]:
    """Discard chunks whose score is too far below the top chunk's score."""
    if not results:
        return results
    top = _effective_score(results[0])
    threshold = top * SCORE_GAP_MIN_RATIO
    return [r for r in results if _effective_score(r) >= threshold]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Run hybrid search (BM25 + vector) against Azure AI Search.

    Pipeline:
    1.  Distil keyword query (strip conversational filler)
    2.  Generate query embedding (falls back to keyword-only on failure)
    3.  Execute hybrid search with optional semantic reranking
    4.  Normalise raw documents → canonical result schema
    5.  Sort by effective score (reranker_score if present, else base score)
    6.  Filter Table-of-Contents / index pages
    7.  Apply adaptive per-source diversity cap
    8.  Apply score-gap filter (drop chunks too far below top)
    9.  Trim to top_k
    10. Trace log if TRACE_MODE=true

    Returns a list of result dicts ready for context injection into the LLM.
    """
    client = _get_search_client()

    # Step 1 — Distil keyword query
    keyword_query = _distill_keyword_query(question)
    logger.debug("Keyword query: %r", keyword_query)

    # Step 2 — Generate embedding (keyword-only fallback if this fails)
    query_vector: Optional[list[float]] = None
    try:
        query_vector = embed(question)
    except Exception as exc:
        logger.warning("Embedding failed — using keyword-only search: %s", exc)

    # Step 3 — Build vector query
    vector_queries = []
    if query_vector:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vector,
                fields=SEARCH_VECTOR_FIELD,
                k_nearest_neighbors=VECTOR_K,
            )
        )

    search_kwargs: dict = {
        "search_text": keyword_query,
        "select": _select_fields(),
        "top": RETRIEVAL_CANDIDATES,
        "vector_queries": vector_queries if vector_queries else None,
    }

    # Step 4 — Execute search (semantic reranker or plain hybrid)
    try:
        if USE_SEMANTIC_RERANKER:
            search_kwargs["query_type"] = QueryType.SEMANTIC
            search_kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG_NAME
            search_kwargs["query_language"] = QUERY_LANGUAGE
            logger.debug("Using semantic reranker: config=%s", SEMANTIC_CONFIG_NAME)
        raw_results = list(client.search(**search_kwargs))
    except Exception as exc:
        if USE_SEMANTIC_RERANKER:
            # Semantic reranker may not be available — fall back to hybrid
            logger.warning(
                "Semantic search failed (%s) — falling back to hybrid search", exc
            )
            search_kwargs.pop("query_type", None)
            search_kwargs.pop("semantic_configuration_name", None)
            search_kwargs.pop("query_language", None)
            raw_results = list(client.search(**search_kwargs))
        else:
            raise

    # Step 5 — Normalise and sort by effective score
    results = [_normalize(doc) for doc in raw_results]
    results.sort(key=_effective_score, reverse=True)

    # Step 5b — Filter TOC / index pages
    results = [r for r in results if not _is_toc_chunk(r["content"])]

    # Step 6 — Adaptive per-source diversity cap
    if DIVERSITY_BY_SOURCE:
        results = _adaptive_diversity(results)

    # Step 7 — Score-gap filter
    results = _filter_score_gap(results)

    # Step 8 — Trim to top_k
    results = results[:top_k]

    # Step 9 — Trace log
    if TRACE_MODE:
        for i, r in enumerate(results, 1):
            logger.info(
                "TRACE [%d] src=%r score=%.4f reranker=%.3f "
                "section=%r preview=%r",
                i,
                r["source"],
                r["score"],
                r["reranker_score"] or 0.0,
                f"{r['section1']} > {r['section2']}".strip(" >"),
                r["content"][:120],
            )

    logger.info(
        "retrieve() → %d chunks for question: %r",
        len(results),
        question[:80],
    )
    return results
