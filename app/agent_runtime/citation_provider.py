"""
CitationProvider — deduplicates and structures retrieved results into Citations.

Citations are ALWAYS derived from retrieval results.  They are never gated on
whether the LLM includes "[1]" or "Sources:" in its answer text — that approach
produces unstable citation output depending on answer formatting.

build_citations() is called by AgentRuntime immediately after the confidence
gate passes, before generation, so citations are available even if generation
fails.  The same citation list is returned for both /chat (JSON) and
/chat/stream (SSE citations event).

Deduplication key: chunk_id (globally unique per indexed chunk).
Falls back to source+url if chunk_id is missing.
Results arrive ordered by effective relevance score descending, so the first
occurrence of each key is the most relevant chunk for that source.
"""

from app.api.schemas import Citation


def _section_path(r: dict) -> str:
    """Build a section breadcrumb from header_1/2/3 fields."""
    parts = [r.get("section1") or "", r.get("section2") or "", r.get("section3") or ""]
    return " > ".join(p for p in parts if p)


def build_citations(results: list[dict]) -> list[Citation]:
    """Build a deduplicated, ordered Citation list from retrieval results.

    Parameters
    ----------
    results:
        Normalised result dicts from retrieval_tool.retrieve().
        Ordered highest relevance first.

    Returns
    -------
    list[Citation]
        One Citation per unique chunk (keyed by chunk_id or source+url),
        in order of first appearance (highest relevance first).
    """
    seen: set[str] = set()
    citations: list[Citation] = []

    for r in results:
        key = r.get("chunk_id") or f"{r.get('source', '')}|{r.get('url', '')}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=r.get("source", ""),
                title=r.get("title", ""),
                section=_section_path(r),
                page=r.get("page", ""),
                url=r.get("url", ""),
                chunk_id=r.get("chunk_id", ""),
            )
        )

    return citations
