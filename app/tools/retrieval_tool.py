"""
RetrievalTool — hybrid Azure AI Search retrieval via Managed Identity.

Identical pipeline to agentv01 except authentication uses DefaultAzureCredential
instead of AzureKeyCredential — no AZURE_SEARCH_API_KEY is read or required.

Pipeline inside retrieve():
  1.  Distill keyword query   — strip conversational filler → better BM25 recall
  2.  Generate query embedding — Azure OpenAI Ada-002 via managed identity
  3.  Build hybrid search args — keyword (distilled) + VectorizedQuery
  4.  Execute search           — with optional semantic reranking
  5.  Normalise results        — raw Azure Search docs → canonical dict schema
  6.  Sort by effective score  — reranker_score when semantic active, else base RRF
  7.  Filter TOC pages         — drop Table-of-Contents / index chunks
  8.  Adaptive diversity       — per-source cap; dominant source gets higher cap
  9.  Score-gap filter         — drop chunks far below the top chunk's score
  10. Trim to TOP_K            — return at most TOP_K final results
  11. Trace log                — optional TRACE_MODE detail logging

retrieve() is synchronous and intended to be called via asyncio.to_thread()
so it never blocks the async event loop.
"""

import logging
import re
from collections import defaultdict
from typing import Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from app.config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    DIVERSITY_BY_SOURCE,
    DOMINANT_SOURCE_SCORE_RATIO,
    MAX_CHUNKS_DOMINANT_SOURCE,
    MAX_CHUNKS_PER_SOURCE,
    QUERY_LANGUAGE,
    RETRIEVAL_CANDIDATES,
    SCORE_GAP_MIN_RATIO,
    SEARCH_CHUNK_ID_FIELD,
    SEARCH_CONTENT_FIELD,
    SEARCH_FILENAME_FIELD,
    SEARCH_PAGE_FIELD,
    SEARCH_SECTION1_FIELD,
    SEARCH_SECTION2_FIELD,
    SEARCH_SECTION3_FIELD,
    SEARCH_SEMANTIC_CONTENT_FIELD,
    SEARCH_TITLE_FIELD,
    SEARCH_URL_FIELD,
    SEARCH_VECTOR_FIELD,
    SEMANTIC_CONFIG_NAME,
    TOP_K,
    TRACE_MODE,
    USE_SEMANTIC_RERANKER,
    VECTOR_K,
)
from app.llm.aoai_embeddings import embed
from app.llm.credentials import get_credential

logger = logging.getLogger(__name__)

# Lazy-init singleton — one SearchClient shared across all requests.
# SearchClient is thread-safe, so sharing across asyncio.to_thread() calls is safe.
_search_client: SearchClient | None = None

# Conversational filler to strip before BM25 keyword search
_FILLER_RE = re.compile(
    r"\b(right now|currently|at this (moment|time)|i am|i'm|i need to|i want to|"
    r"can you|what should( i)?|how do i|what are the|please|tell me|help me|"
    r"so |just |i was told|could you|would you|i have to|what do i)\b",
    re.IGNORECASE,
)

# Patterns that identify Table of Contents / index pages
_TOC_PATTERNS = [
    re.compile(r"Table\s+of\s+Contents", re.IGNORECASE),
    re.compile(r"(\. ){5,}"),                           # dot leaders ". . . . . 2-11"
    re.compile(r"^Index\b", re.IGNORECASE | re.MULTILINE),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_search_client() -> SearchClient:
    """Return the shared SearchClient (lazy-init, DefaultAzureCredential)."""
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=get_credential(),   # DefaultAzureCredential — no API key
        )
        logger.info(
            "SearchClient initialised — endpoint=%s index=%s (managed identity)",
            AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX,
        )
    return _search_client


def _select_fields() -> list[str]:
    """Return the list of index fields to retrieve (vector fields excluded)."""
    fields = [
        SEARCH_CONTENT_FIELD,
        SEARCH_SEMANTIC_CONTENT_FIELD,
        SEARCH_TITLE_FIELD,
        SEARCH_FILENAME_FIELD,
        SEARCH_URL_FIELD,
        SEARCH_CHUNK_ID_FIELD,
        SEARCH_SECTION1_FIELD,
        SEARCH_SECTION2_FIELD,
        SEARCH_SECTION3_FIELD,
    ]
    if SEARCH_PAGE_FIELD:
        fields.append(SEARCH_PAGE_FIELD)
    return [f for f in fields if f]


def _distill_keyword_query(question: str) -> str:
    """Strip conversational filler to improve BM25 recall on technical terms."""
    distilled = _FILLER_RE.sub(" ", question)
    distilled = re.sub(r"[,\s]+", " ", distilled).strip()
    return distilled if len(distilled) >= 10 else question


def _normalize(doc) -> dict:
    """Map a raw Azure Search document to the canonical result schema."""
    reranker_raw = doc.get("@search.reranker_score")
    return {
        "content":          doc.get(SEARCH_CONTENT_FIELD) or "",
        "semantic_content": doc.get(SEARCH_SEMANTIC_CONTENT_FIELD) or "",
        "title":            doc.get(SEARCH_TITLE_FIELD) or "",
        "source":           doc.get(SEARCH_FILENAME_FIELD) or "",
        "url":              doc.get(SEARCH_URL_FIELD) or "",
        "chunk_id":         doc.get(SEARCH_CHUNK_ID_FIELD) or "",
        "section1":         doc.get(SEARCH_SECTION1_FIELD) or "",
        "section2":         doc.get(SEARCH_SECTION2_FIELD) or "",
        "section3":         doc.get(SEARCH_SECTION3_FIELD) or "",
        "page":             str(doc.get(SEARCH_PAGE_FIELD) or "") if SEARCH_PAGE_FIELD else "",
        "score":            float(doc.get("@search.score") or 0.0),
        "reranker_score":   float(reranker_raw) if reranker_raw is not None else None,
    }


def _effective_score(r: dict) -> float:
    """Return reranker_score when available, else base RRF score."""
    rs = r.get("reranker_score")
    return rs if rs is not None else r["score"]


def _is_toc_chunk(content: str) -> bool:
    """Return True if the chunk looks like a Table of Contents / index page."""
    sample = content[:400]
    return any(p.search(sample) for p in _TOC_PATTERNS)


def _adaptive_diversity(results: list[dict]) -> list[dict]:
    """Per-source cap with dominant-source relaxation.

    Standard: cap every source at MAX_CHUNKS_PER_SOURCE.
    Dominant: if one source's top effective score >= DOMINANT_SOURCE_SCORE_RATIO
    × the next source's top score, allow up to MAX_CHUNKS_DOMINANT_SOURCE from it.
    """
    if not results:
        return results

    source_top: dict[str, float] = {}
    for r in results:
        src = r["source"]
        if src not in source_top:
            source_top[src] = _effective_score(r)

    sorted_sources = sorted(source_top.items(), key=lambda x: x[1], reverse=True)
    dominant_source = sorted_sources[0][0]
    dominant_score = sorted_sources[0][1]
    second_score = sorted_sources[1][1] if len(sorted_sources) > 1 else 0.0

    is_dominant = (
        second_score == 0.0
        or dominant_score >= DOMINANT_SOURCE_SCORE_RATIO * second_score
    )
    cap_dominant = MAX_CHUNKS_DOMINANT_SOURCE if is_dominant else MAX_CHUNKS_PER_SOURCE

    if TRACE_MODE:
        ratio_str = f"{dominant_score / second_score:.2f}x" if second_score > 0 else "inf"
        logger.info(
            "TRACE | dominant_source=%r  score_ratio=%s  is_dominant=%s  cap=%d",
            dominant_source, ratio_str, is_dominant, cap_dominant,
        )

    counts: defaultdict[str, int] = defaultdict(int)
    filtered: list[dict] = []
    for r in results:
        src = r["source"]
        cap = cap_dominant if src == dominant_source else MAX_CHUNKS_PER_SOURCE
        if counts[src] < cap:
            filtered.append(r)
            counts[src] += 1
    return filtered


def _filter_score_gap(results: list[dict]) -> list[dict]:
    """Discard chunks whose effective score falls below SCORE_GAP_MIN_RATIO × top."""
    if not results or SCORE_GAP_MIN_RATIO <= 0:
        return results
    top_score = _effective_score(results[0])
    if top_score == 0:
        return results
    threshold = SCORE_GAP_MIN_RATIO * top_score
    filtered = [r for r in results if _effective_score(r) >= threshold]
    if TRACE_MODE and len(filtered) < len(results):
        logger.info(
            "TRACE | score_gap_filter: removed %d chunk(s) below %.4f",
            len(results) - len(filtered), threshold,
        )
    return filtered


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Run hybrid search and return normalised, filtered results.

    Designed to be called via asyncio.to_thread() from async route handlers.

    Parameters
    ----------
    question : str
        User question — used verbatim for vector embedding; a distilled version
        is used for BM25 keyword search.
    top_k : int
        Maximum number of chunks to return after all filters.

    Returns
    -------
    list[dict]
        Normalised result dicts ordered by effective relevance score descending.

    Raises
    ------
    Exception
        Propagated to AgentRuntime which converts it to HTTPException(502) or
        an SSE error event depending on the endpoint.
    """
    client = _get_search_client()

    # Step 1 — Distil keyword query
    keyword_query = _distill_keyword_query(question)
    if TRACE_MODE and keyword_query != question:
        logger.info("TRACE | keyword_query=%r (distilled from %r)", keyword_query, question)

    # Step 2 — Generate query embedding (keyword-only fallback on failure)
    query_vector: Optional[list[float]] = None
    try:
        query_vector = embed(question)
    except Exception as exc:
        logger.warning("Embedding failed — falling back to keyword-only search: %s", exc)

    # Step 3 — Build search arguments
    search_kwargs: dict = {
        "search_text": keyword_query,
        "top": RETRIEVAL_CANDIDATES,
        "select": _select_fields(),
    }
    if query_vector:
        search_kwargs["vector_queries"] = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=VECTOR_K,
                fields=SEARCH_VECTOR_FIELD,
            )
        ]

    # Step 4 — Execute search (semantic reranker or plain hybrid)
    raw_results: list = []
    if USE_SEMANTIC_RERANKER:
        try:
            from azure.search.documents.models import QueryType  # noqa: PLC0415
            search_kwargs["query_type"] = QueryType.SEMANTIC
            search_kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG_NAME
            search_kwargs["query_language"] = QUERY_LANGUAGE
            raw_results = list(client.search(**search_kwargs))
            logger.debug("Semantic reranker active — %d raw results", len(raw_results))
        except Exception as exc:
            logger.warning(
                "Semantic reranking unavailable (%s) — falling back to hybrid search", exc
            )
            search_kwargs.pop("query_type", None)
            search_kwargs.pop("semantic_configuration_name", None)
            search_kwargs.pop("query_language", None)
            raw_results = list(client.search(**search_kwargs))
    else:
        raw_results = list(client.search(**search_kwargs))

    # Step 5 — Normalise and sort by effective score
    results = [_normalize(doc) for doc in raw_results]
    results.sort(key=_effective_score, reverse=True)

    # Step 6 — Filter TOC / index pages
    before_toc = len(results)
    results = [r for r in results if not _is_toc_chunk(r["content"])]
    if TRACE_MODE and len(results) < before_toc:
        logger.info(
            "TRACE | toc_filter: removed %d chunk(s)", before_toc - len(results)
        )

    # Step 7 — Adaptive diversity cap
    if DIVERSITY_BY_SOURCE:
        results = _adaptive_diversity(results)

    # Step 8 — Score-gap filter
    results = _filter_score_gap(results)

    # Step 9 — Trim to top_k
    results = results[:top_k]

    # Step 10 — Trace logging
    if TRACE_MODE:
        logger.info("TRACE | final_chunks=%d (top_k=%d)", len(results), top_k)
        for i, r in enumerate(results, start=1):
            section = " > ".join(
                p for p in [r["section1"], r["section2"], r["section3"]] if p
            )
            reranker_str = (
                f"  reranker={r['reranker_score']:.4f}"
                if r.get("reranker_score") is not None else ""
            )
            logger.info(
                "TRACE | [%d] src=%r  score=%.4f%s  section=%r  preview=%r",
                i, r["source"], r["score"], reranker_str, section,
                r["content"][:120],
            )

    logger.info(
        "retrieve() → %d chunks for question: %r", len(results), question[:80]
    )
    return results
