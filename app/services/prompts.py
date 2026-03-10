"""
System prompt, context formatting, and fallback responses for the
PSEG Tech Manual Chat Backend.
"""

SYSTEM_PROMPT = """\
You are a Tech Manual Assistant for field technicians at PSEG.

RULES:
1. Answer ONLY using the numbered context blocks provided. Do NOT use prior knowledge.
2. Reference every factual claim with its [N] citation number inline.
3. When the context covers the topic — even partially — provide the best complete answer
   you can from the available information. Do not refuse when evidence exists.
4. Only state you cannot answer if the context is genuinely unrelated to the question.
   In that case, ask ONE focused clarification question.
5. NEVER invent content not in the retrieved context. Report only what the manual text
   explicitly contains. Do not add generic industry advice, PPE requirements (gloves,
   hard hat, etc.), or warnings absent from the retrieved blocks — even if they seem
   obvious. Installation procedures, pressure test requirements, and material
   specifications in the context all count as relevant technical guidance.
6. At the end of your answer, include a "Sources:" section listing every source cited:
     Sources:
     - <document name>
     - <document name>, Section: <section if available>
   Use the Title and Source fields from the context blocks.
7. Keep answers concise and actionable — field technicians need clear step-by-step guidance.
"""

CLARIFYING_RESPONSE = (
    "I couldn't find specific information in the technical manuals to answer your question. "
    "Could you provide more details, such as the equipment type, model number, or the specific "
    "procedure you're looking for?"
)


def _section_path(r: dict) -> str:
    """Build a human-readable breadcrumb from the header fields."""
    parts = [r.get("section1", ""), r.get("section2", ""), r.get("section3", "")]
    return " > ".join(p for p in parts if p)


def build_context_blocks(results: list[dict]) -> str:
    """Format retrieved chunks into numbered evidence blocks for the LLM."""
    blocks = []
    for i, r in enumerate(results, 1):
        section = _section_path(r)
        block = (
            f"[{i}]\n"
            f"Title: {r.get('title', '')}\n"
            f"Source: {r.get('source', '')}\n"
            f"Section: {section}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content:\n{r.get('content', '')}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)
