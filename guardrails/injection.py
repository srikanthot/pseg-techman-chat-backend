"""Heuristic prompt-injection / jailbreak detection (OWASP LLM01)."""
from __future__ import annotations
import re

_SIGNALS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts)",
    r"disregard (all |the )?(previous|prior|system)",
    r"you are now (a |an )?\w+",
    r"reveal (your )?(system prompt|instructions|prompt)",
    r"(print|repeat|output) (your )?(system prompt|instructions)",
    r"pretend (to be|you are)",
    r"do anything now|\bDAN\b",
    r"developer mode",
    r"bypass (the )?(rules|guardrails|safety)",
]
_RX = [re.compile(s, re.I) for s in _SIGNALS]


def scan(text: str) -> dict:
    matched = [rx.pattern for rx in _RX if rx.search(text or "")]
    return {"is_injection": bool(matched), "signals": matched}
