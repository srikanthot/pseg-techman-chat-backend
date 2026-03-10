"""Regex-based PII detection and redaction (no external services)."""
from __future__ import annotations
import re

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def detect(text: str) -> list[dict]:
    hits = []
    for kind, rx in PATTERNS.items():
        for m in rx.finditer(text or ""):
            hits.append({"type": kind, "value": m.group(), "span": [m.start(), m.end()]})
    return hits


def redact(text: str) -> tuple[str, list[dict]]:
    """Return (redacted_text, detections)."""
    hits = detect(text)
    out = text or ""
    for kind, rx in PATTERNS.items():
        out = rx.sub(f"[REDACTED_{kind.upper()}]", out)
    return out, hits
