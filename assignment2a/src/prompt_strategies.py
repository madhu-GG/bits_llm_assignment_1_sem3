"""Part A — prompting paradigms, prompt chaining, and function-calling output."""

from __future__ import annotations

import json
import re
from collections import Counter

import a2_paths  # noqa: F401
from answer_scoring import answer_signature, best_of, score_answer
from assignment2_queries import (
    CLINICAL_FACT_SCHEMA,
    FEW_SHOT_EXAMPLES,
    SYSTEM_ROLE,
)
from domain_lm import retrieve_topics

MAX_NEW_TOKENS = 200


# --------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------

def zero_shot_prompt(query: str) -> str:
    return f"Answer the following medical-literature question.\n\nQuestion: {query}\nAnswer:"


def few_shot_prompt(query: str, examples=None) -> str:
    examples = examples or FEW_SHOT_EXAMPLES
    shots = "\n\n".join(f"Q: {ex['question']}\nA: {ex['answer']}" for ex in examples[:3])
    return (
        "Answer the final question in the same style as the examples.\n\n"
        f"{shots}\n\nQ: {query}\nA:"
    )


def instruction_prompt(query: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Follow this step-by-step instruction format:\n"
        "1. State the definition from the corpus.\n"
        "2. List the governing considerations as a numbered list.\n"
        "3. Name the related corpus topic.\n"
        "4. End with the educational-use caveat.\n\n"
        f"Question: {query}\nAnswer:"
    )


def ape_meta_prompt(query: str) -> str:
    return (
        "You are a prompt engineer. Write five different instruction wordings that would make a "
        "medical-literature model answer the question below accurately and in a well-structured way. "
        "Return one instruction per line.\n\n"
        f"Question: {query}"
    )


# Fallback instruction templates used when the model's APE proposals are unusable.
APE_TEMPLATES = [
    "{role}\n\nAnswer the question using only corpus-supported statements.\n\nQuestion: {q}\nAnswer:",
    "{role}\n\nGive a numbered answer: definition, two governing considerations, one caution.\n\n"
    "Question: {q}\nAnswer:",
    "{role}\n\nExplain the mechanism first, then the practical implication, then the limitation.\n\n"
    "Question: {q}\nAnswer:",
    "{role}\n\nYou are preparing a study note. 1. Define. 2. Explain why it matters. "
    "3. State what must be monitored. 4. Add the educational caveat.\n\nQuestion: {q}\nAnswer:",
    "{role}\n\nAnswer concisely but completely, naming the related corpus topic and the uncertainty.\n\n"
    "Question: {q}\nAnswer:",
]


# --------------------------------------------------------------------------
# Strategy runners
# --------------------------------------------------------------------------

def _keywords_for(query: str) -> list[str]:
    topics = retrieve_topics(query, k=1)
    if not topics:
        return []
    from answer_scoring import grounding_terms
    topic = topics[0]
    return grounding_terms([topic["abstract"] + " " + " ".join(topic["points"])], top_k=20)


def run_zero_shot(lm, query: str) -> dict:
    answer = lm.one(zero_shot_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {"strategy": "zero_shot", "query": query, "answer": answer,
            "score": score_answer(answer, query, _keywords_for(query))}


def run_few_shot(lm, query: str) -> dict:
    answer = lm.one(few_shot_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {"strategy": "few_shot_3", "query": query, "answer": answer,
            "score": score_answer(answer, query, _keywords_for(query))}


def run_instruction(lm, query: str) -> dict:
    answer = lm.one(instruction_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {"strategy": "instruction", "query": query, "answer": answer,
            "score": score_answer(answer, query, _keywords_for(query))}


def run_self_consistency(lm, query: str, samples: int = 5, temperature: float = 0.7) -> dict:
    """Sample the same prompt 5x and keep the majority-vote answer."""
    prompt = instruction_prompt(query)
    candidates = lm.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature, n=samples)
    signatures = [answer_signature(c) for c in candidates]
    votes = Counter(signatures)
    winning_signature, vote_count = votes.most_common(1)[0]
    members = [c for c, s in zip(candidates, signatures) if s == winning_signature]
    keywords = _keywords_for(query)
    # Within the winning cluster, keep the representative with the best score.
    index, score = best_of(members, query, keywords)
    return {
        "strategy": "self_consistency_5",
        "query": query,
        "answer": members[index],
        "score": score,
        "samples": candidates,
        "vote_counts": dict(votes),
        "majority_size": vote_count,
        "agreement": round(vote_count / samples, 2),
    }


def _parse_variants(text: str, limit: int = 5) -> list[str]:
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", line).strip()
            for line in text.splitlines()]
    return [line for line in lines if len(line.split()) >= 5][:limit]


def run_ape(lm, query: str, num_variants: int = 5) -> dict:
    """Generate candidate instructions with the LLM, then keep the best-scoring output."""
    proposal_text = lm.one(ape_meta_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.7)
    proposals = _parse_variants(proposal_text, num_variants)
    prompts, sources = [], []
    for proposal in proposals:
        prompts.append(f"{SYSTEM_ROLE}\n\n{proposal}\n\nQuestion: {query}\nAnswer:")
        sources.append("llm_proposed")
    for template in APE_TEMPLATES[: num_variants - len(prompts)]:
        prompts.append(template.format(role=SYSTEM_ROLE, q=query))
        sources.append("template_fallback")

    keywords = _keywords_for(query)
    trials = []
    for prompt, source in zip(prompts, sources):
        answer = lm.one(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
        trials.append({"prompt": prompt, "source": source, "answer": answer,
                       "score": score_answer(answer, query, keywords)})
    best = max(trials, key=lambda t: t["score"]["total"])
    return {
        "strategy": "ape",
        "query": query,
        "answer": best["answer"],
        "score": best["score"],
        "selected_prompt": best["prompt"],
        "variants_tried": len(trials),
        "variant_scores": [t["score"]["total"] for t in trials],
        "raw_proposals": proposal_text,
    }


STRATEGIES = {
    "zero_shot": run_zero_shot,
    "few_shot_3": run_few_shot,
    "instruction": run_instruction,
    "self_consistency_5": run_self_consistency,
    "ape": run_ape,
}


def run_all_strategies(lm, queries) -> list[dict]:
    rows = []
    for query in queries:
        for name, runner in STRATEGIES.items():
            result = runner(lm, query)
            result["backend"] = lm.backend
            rows.append(result)
    return rows


def strategy_summary(rows) -> list[dict]:
    """Mean score per strategy across all queries."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row["score"])
    summary = []
    for name, scores in grouped.items():
        n = len(scores)
        summary.append({
            "strategy": name,
            "queries": n,
            "mean_total": round(sum(s["total"] for s in scores) / n, 2),
            "mean_grounding": round(sum(s["grounding"] for s in scores) / n, 2),
            "mean_structure": round(sum(s["structure"] for s in scores) / n, 2),
            "mean_words": round(sum(s["words"] for s in scores) / n, 1),
        })
    return sorted(summary, key=lambda r: -r["mean_total"])


# --------------------------------------------------------------------------
# A2 — prompt chaining
# --------------------------------------------------------------------------

def chain_step1_prompt(query: str, context: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Step 1 of 2 — fact extraction. Extract the relevant facts from the context that are needed to "
        "answer the question. Return a bulleted list of facts only, no conclusion.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nFacts:"
    )


def chain_step2_prompt(query: str, facts: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Step 2 of 2 — synthesise the answer using only the extracted facts below. "
        "Produce a numbered answer and end with the educational caveat.\n\n"
        f"Extracted facts:\n{facts}\n\nQuestion: {query}\nSynthesised answer:"
    )


def build_context(query: str, k: int = 3) -> str:
    blocks = []
    for topic in retrieve_topics(query, k=k):
        points = "\n".join(f"- {p}" for p in topic["points"])
        blocks.append(f"## {topic['title']}\n{topic['abstract']}\n{points}")
    return "\n\n".join(blocks)


def run_prompt_chain(lm, query: str) -> dict:
    context = build_context(query)
    facts = lm.one(chain_step1_prompt(query, context), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    answer = lm.one(chain_step2_prompt(query, facts), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    keywords = _keywords_for(query)
    single = lm.one(zero_shot_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {
        "query": query,
        "context_chars": len(context),
        "step1_facts": facts,
        "chained_answer": answer,
        "chained_score": score_answer(answer, query, keywords),
        "single_step_answer": single,
        "single_step_score": score_answer(single, query, keywords),
    }


# --------------------------------------------------------------------------
# A2 — function calling / structured JSON output
# --------------------------------------------------------------------------

def function_call_prompt(query: str, context: str, schema=CLINICAL_FACT_SCHEMA) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        f"You may call the function `{schema['name']}`. {schema['description']}\n"
        f"JSON schema:\n{json.dumps(schema['parameters'], indent=2)}\n\n"
        "Return only JSON matching the schema. Do not add prose before or after the JSON.\n\n"
        f"Context:\n{context}\n\nRequest: {query}\nJSON:"
    )


class SchemaValidationError(ValueError):
    pass


def parse_structured_output(raw: str, schema=CLINICAL_FACT_SCHEMA) -> dict:
    """Extract and validate the JSON object produced by the model."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise SchemaValidationError("No JSON object found in the model output.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Output is not valid JSON: {exc}") from exc

    properties = schema["parameters"]["properties"]
    missing = [key for key in schema["parameters"]["required"] if key not in payload]
    if missing:
        raise SchemaValidationError(f"Missing required keys: {missing}")
    unexpected = [key for key in payload if key not in properties]
    if unexpected:
        raise SchemaValidationError(f"Unexpected keys not in schema: {unexpected}")
    for key, value in payload.items():
        expected = properties[key]["type"]
        if expected == "string" and not isinstance(value, str):
            raise SchemaValidationError(f"Key '{key}' must be a string.")
        if expected == "number" and not isinstance(value, (int, float)):
            raise SchemaValidationError(f"Key '{key}' must be a number.")
    return payload


def run_function_call(lm, query: str, schema=CLINICAL_FACT_SCHEMA, retries: int = 2) -> dict:
    context = build_context(query, k=1)
    prompt = function_call_prompt(query, context, schema)
    attempts = []
    for attempt in range(1, retries + 2):
        raw = lm.one(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0 if attempt == 1 else 0.5)
        try:
            parsed = parse_structured_output(raw, schema)
            attempts.append({"attempt": attempt, "raw": raw, "valid": True})
            return {"query": query, "schema": schema["name"], "parsed": parsed,
                    "attempts": attempts, "valid": True}
        except SchemaValidationError as exc:
            attempts.append({"attempt": attempt, "raw": raw, "valid": False, "error": str(exc)})
            prompt = (f"{prompt}\n\nYour previous output was rejected: {exc}\n"
                      "Return only a valid JSON object this time.\nJSON:")
    return {"query": query, "schema": schema["name"], "parsed": None, "attempts": attempts, "valid": False}
