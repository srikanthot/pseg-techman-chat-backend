"""
Citation deduplication and formatting.

Converts raw retrieval result dicts into structured Citation objects,
deduplicating by chunk_id (or source+url as fallback), preserving
order so the highest-relevance source appears first.
"""

from app.api.schemas import Citation
from app.services.prompts import _section_path


def build_citations(results: list[dict]) -> list[Citation]:
    """Build a deduplicated list of Citation objects from retrieval results."""
    seen: set[str] = set()
    citations: list[Citation] = []

    for r in results:
        chunk_id = r.get("chunk_id", "")
        source = r.get("source", "")
        url = r.get("url", "")

        # Prefer chunk_id for deduplication; fall back to source+url
        dedup_key = chunk_id if chunk_id else f"{source}|{url}"

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        citations.append(
            Citation(
                source=source,
                title=r.get("title", ""),
                section=_section_path(r),
                page=r.get("page", ""),
                url=url,
                chunk_id=chunk_id,
            )
        )

    return citations
