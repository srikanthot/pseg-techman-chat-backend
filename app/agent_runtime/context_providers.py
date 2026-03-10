"""
Context formatting — converts retrieved result dicts into numbered evidence blocks.

build_context_blocks() is called by RagContextProvider.before_run() to format
the pre-retrieved Azure AI Search chunks into a single string that is injected
into the agent's system prompt via context.extend_instructions().

Each block carries a [N] label, source metadata, and the raw chunk content.
The LLM is instructed to answer ONLY from these blocks and to cite them by [N].
"""


def _section_path(r: dict) -> str:
    """Build a readable section breadcrumb from header_1/2/3 fields."""
    parts = [r.get("section1") or "", r.get("section2") or "", r.get("section3") or ""]
    return " > ".join(p for p in parts if p)


def build_context_blocks(results: list[dict]) -> str:
    """Format retrieved chunks into numbered, labeled evidence blocks for the LLM.

    Parameters
    ----------
    results:
        Normalised result dicts from retrieval_tool.retrieve() — keys: content,
        title, source, url, chunk_id, section1, section2, section3, score.
        Ordered by effective relevance score descending.

    Returns
    -------
    str
        A single string containing one evidence block per chunk, separated by
        horizontal rules (``---``).  Block format::

            [N]
            Title: ...
            Source: ...
            Section: ...
            URL: ...
            Chunk ID: ...
            Content:
            <raw chunk text>
    """
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        lines = [f"[{i}]"]
        if r.get("title"):
            lines.append(f"Title: {r['title']}")
        lines.append(f"Source: {r['source']}")
        section = _section_path(r)
        if section:
            lines.append(f"Section: {section}")
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        if r.get("chunk_id"):
            lines.append(f"Chunk ID: {r['chunk_id']}")
        lines.append("Content:")
        lines.append(r["content"])
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)
