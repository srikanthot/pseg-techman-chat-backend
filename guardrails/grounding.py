"""Grounding / faithfulness check — refuse ungrounded answers."""
from __future__ import annotations
import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to and or is are was were be in on at for with that this it as by from".split())


def _content(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def grounding_score(answer: str, contexts: list[str]) -> float:
    a = _content(answer)
    if not a:
        return 1.0
    ctx = _content(" ".join(contexts or []))
    return len(a & ctx) / len(a)


def check(answer: str, contexts: list[str], threshold: float = 0.5) -> dict:
    score = round(grounding_score(answer, contexts), 3)
    grounded = score >= threshold
    return {
        "grounded": grounded,
        "score": score,
        "safe_answer": answer if grounded else
        "I don't have enough grounded information in the sources to answer that confidently.",
    }
