"""MANGOS Guardrails — input/output safety middleware for RAG/LLM services.

    from guardrails import guard_input, guard_output

Wrap any RAG backend: run guard_input() on the user message, guard_output()
on the model answer against retrieved contexts.
"""
from .pii import detect as detect_pii, redact as redact_pii
from .injection import scan as scan_injection
from .grounding import check as check_grounding


def guard_input(user_text: str) -> dict:
    """Screen an incoming user message: block injection, redact PII."""
    inj = scan_injection(user_text)
    redacted, pii = redact_pii(user_text)
    return {
        "allow": not inj["is_injection"],
        "injection": inj,
        "pii_found": pii,
        "sanitized_text": redacted,
    }


def guard_output(answer: str, contexts: list[str], threshold: float = 0.5) -> dict:
    """Screen a model answer: enforce grounding, redact leaked PII."""
    g = check_grounding(answer, contexts, threshold)
    redacted, pii = redact_pii(g["safe_answer"])
    return {"grounded": g["grounded"], "grounding_score": g["score"],
            "pii_redacted": bool(pii), "final_answer": redacted}


__all__ = ["guard_input", "guard_output", "detect_pii", "redact_pii",
           "scan_injection", "check_grounding"]
