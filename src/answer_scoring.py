"""Deterministic answer-quality scoring shared by Parts A, B, and C.

A single scorer is used for the Part A strategy comparison, Tree-of-Thought
path selection, APE prompt selection, and the chosen/rejected decision in the
Part C preference dataset. Keeping one definition means the preference labels
are consistent with the comparisons reported earlier in the assignment.
"""

from __future__ import annotations

import re
from typing import Iterable

STRUCTURE_PATTERNS = (
    re.compile(r"^\s*(?:step\s*\d+|\d+[.)]|[-*\u2022])", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(?:answer|conclusion|reasoning|observation|thought)\s*:", re.IGNORECASE | re.MULTILINE),
)

SPECIFICITY_PATTERN = re.compile(
    r"\d|%|mg|mL|kg|ratio|probability|prevalence|sensitivity|specificity|threshold|interval", re.IGNORECASE
)

HEDGE_PATTERN = re.compile(
    r"depends on|may |can |should be|not (?:clinical )?advice|uncertain|varies|consider", re.IGNORECASE
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def grounding_terms(reference_texts: Iterable[str], top_k: int = 25) -> list[str]:
    """Pick the most frequent content terms from reference corpus passages."""
    stop = {
        "the", "and", "of", "to", "in", "a", "is", "for", "that", "with", "on", "as", "by", "this", "an",
        "or", "from", "are", "be", "can", "which", "when", "at", "it", "may", "not", "but", "has", "have",
        "than", "their", "they", "its", "into", "also", "more", "most", "other", "should", "been", "was",
    }
    counts: dict[str, int] = {}
    for text in reference_texts:
        for token in set(_tokens(text)):
            if len(token) > 3 and token not in stop:
                counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return ranked[:top_k]


def score_answer(answer: str, query: str = "", keywords: Iterable[str] | None = None) -> dict:
    """Score an answer from 0-100 on five transparent components."""
    answer = (answer or "").strip()
    if not answer:
        return {"grounding": 0.0, "structure": 0.0, "completeness": 0.0, "specificity": 0.0,
                "non_repetition": 0.0, "total": 0.0, "words": 0}

    answer_tokens = set(_tokens(answer))
    words = len(_tokens(answer))

    keywords = list(keywords or [])
    if keywords:
        hit = sum(1 for k in keywords if k.lower() in answer_tokens)
        grounding = 35.0 * hit / len(keywords)
    else:
        query_terms = [t for t in set(_tokens(query)) if len(t) > 3]
        hit = sum(1 for t in query_terms if t in answer_tokens)
        grounding = 35.0 * hit / max(len(query_terms), 1)

    structure_hits = sum(1 for pattern in STRUCTURE_PATTERNS if pattern.search(answer))
    structure = 20.0 * min(structure_hits, 2) / 2

    # Reward a 45-220 word band; penalise both one-liners and padding.
    if words < 45:
        completeness = 20.0 * words / 45
    elif words <= 220:
        completeness = 20.0
    else:
        completeness = max(0.0, 20.0 - (words - 220) / 20.0)

    specificity_hits = len(SPECIFICITY_PATTERN.findall(answer))
    hedge_hits = len(HEDGE_PATTERN.findall(answer))
    specificity = min(15.0, 2.0 * specificity_hits + 1.5 * hedge_hits)

    sentences = _sentences(answer)
    normalised = [re.sub(r"\W+", " ", s.lower()).strip() for s in sentences]
    duplicate_ratio = 0.0 if not normalised else (len(normalised) - len(set(normalised))) / len(normalised)
    non_repetition = 10.0 * (1.0 - duplicate_ratio)

    total = grounding + structure + completeness + specificity + non_repetition
    return {
        "grounding": round(grounding, 2),
        "structure": round(structure, 2),
        "completeness": round(completeness, 2),
        "specificity": round(specificity, 2),
        "non_repetition": round(non_repetition, 2),
        "total": round(total, 2),
        "words": words,
    }


def best_of(candidates: list[str], query: str = "", keywords: Iterable[str] | None = None) -> tuple[int, dict]:
    """Return (index, score) of the highest-scoring candidate answer."""
    if not candidates:
        raise ValueError("best_of requires at least one candidate")
    scored = [score_answer(c, query, keywords) for c in candidates]
    best_index = max(range(len(scored)), key=lambda i: scored[i]["total"])
    return best_index, scored[best_index]


def answer_signature(answer: str, top_k: int = 8) -> str:
    """Stable content signature used for self-consistency majority voting."""
    stop = {"the", "and", "of", "to", "in", "a", "is", "for", "that", "with", "on", "as", "by", "this",
            "an", "or", "from", "are", "be", "can", "which", "when", "at", "it", "may"}
    tokens = [t for t in _tokens(answer) if len(t) > 3 and t not in stop]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))[:top_k]
    return "|".join(sorted(ranked))
