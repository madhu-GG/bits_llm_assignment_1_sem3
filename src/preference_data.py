"""Part C1 — build the DPO preference dataset from the Part A and Part B outputs.

`chosen` is always a reasoning-augmented or vote-selected answer; `rejected` is
always the zero-shot single-pass answer for the same prompt. A pair is kept only
when the shared scorer in `answer_scoring` prefers the chosen response, so the
labels stay consistent with the comparison tables in Parts A and B.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import prompt_strategies as ps
import reasoning_strategies as rs
from answer_scoring import score_answer
from assignment2_queries import DOMAIN_QUERIES, HELD_OUT_QUERIES, REASONING_PROBLEMS
from domain_lm import retrieve_topics

try:
    from medical_data import TOPICS
except ImportError:  # pragma: no cover
    TOPICS = []

MIN_SCORE_MARGIN = 3.0


def topic_queries() -> list[str]:
    """Extra prompts derived from the Assignment 1 corpus topics."""
    return [f"Explain {topic['title'].lower()} and what it means for clinical practice."
            for topic in TOPICS]


def prompt_pool() -> list[str]:
    """Every training prompt. Held-out Part C3 queries are deliberately excluded."""
    pool = list(DOMAIN_QUERIES)
    pool += [problem["question"] for problem in REASONING_PROBLEMS]
    pool += topic_queries()
    held_out = {q.strip().lower() for q in HELD_OUT_QUERIES}
    return [p for p in pool if p.strip().lower() not in held_out]


# Each generator returns (source_label, answer_text) for one prompt.
def _chosen_generators(lm, query):
    yield "cot", rs.run_cot(lm, query)["answer"]
    yield "self_consistency_5", ps.run_self_consistency(lm, query)["answer"]
    yield "instruction", ps.run_instruction(lm, query)["answer"]
    yield "self_reflection", rs.run_self_reflection(lm, query)["answer"]
    yield "ape", ps.run_ape(lm, query)["answer"]


def build_preference_pairs(lm, queries=None, min_pairs: int = 60,
                           margin: float = MIN_SCORE_MARGIN) -> tuple[list[dict], dict]:
    queries = list(queries or prompt_pool())
    pairs, skipped = [], 0
    for query in queries:
        keywords = rs.keywords_for(query)
        rejected = lm.one(ps.zero_shot_prompt(query), max_new_tokens=ps.MAX_NEW_TOKENS, temperature=0.0)
        rejected_score = score_answer(rejected, query, keywords)
        for source, chosen in _chosen_generators(lm, query):
            chosen_score = score_answer(chosen, query, keywords)
            if chosen_score["total"] - rejected_score["total"] < margin:
                skipped += 1
                continue
            pairs.append({
                "prompt": query,
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
                "chosen_source": source,
                "rejected_source": "zero_shot",
                "chosen_score": chosen_score["total"],
                "rejected_score": rejected_score["total"],
                "score_margin": round(chosen_score["total"] - rejected_score["total"], 2),
                "chosen_words": chosen_score["words"],
                "rejected_words": rejected_score["words"],
                "topic": retrieve_topics(query, k=1)[0]["title"] if retrieve_topics(query, k=1) else "",
                "backend": lm.backend,
            })
    stats = summarise(pairs)
    stats["skipped_low_margin"] = skipped
    stats["prompts_used"] = len(queries)
    stats["meets_minimum"] = len(pairs) >= min_pairs
    return pairs, stats


def summarise(pairs) -> dict:
    if not pairs:
        return {"pairs": 0}
    chosen_words = [p["chosen_words"] for p in pairs]
    rejected_words = [p["rejected_words"] for p in pairs]
    by_source: dict[str, int] = {}
    for pair in pairs:
        by_source[pair["chosen_source"]] = by_source.get(pair["chosen_source"], 0) + 1
    return {
        "pairs": len(pairs),
        "unique_prompts": len({p["prompt"] for p in pairs}),
        "pairs_by_chosen_source": dict(sorted(by_source.items())),
        "avg_chosen_words": round(statistics.mean(chosen_words), 1),
        "avg_rejected_words": round(statistics.mean(rejected_words), 1),
        "avg_length_delta_words": round(statistics.mean(chosen_words) - statistics.mean(rejected_words), 1),
        "avg_score_margin": round(statistics.mean(p["score_margin"] for p in pairs), 2),
        "min_score_margin": round(min(p["score_margin"] for p in pairs), 2),
    }


def save_jsonl(pairs, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate(pairs, min_pairs: int = 60) -> list[str]:
    problems = []
    if len(pairs) < min_pairs:
        problems.append(f"only {len(pairs)} pairs; the assignment requires at least {min_pairs}")
    held_out = {q.strip().lower() for q in HELD_OUT_QUERIES}
    for index, pair in enumerate(pairs):
        for key in ("prompt", "chosen", "rejected"):
            if not pair.get(key):
                problems.append(f"row {index}: missing or empty '{key}'")
        if pair.get("chosen", "").strip() == pair.get("rejected", "").strip():
            problems.append(f"row {index}: chosen and rejected are identical")
        if pair.get("prompt", "").strip().lower() in held_out:
            problems.append(f"row {index}: prompt leaks a held-out Part C3 evaluation query")
    return problems
