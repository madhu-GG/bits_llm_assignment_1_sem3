"""Part B — chain-of-thought, tree-of-thought, step-back, PAL, ReACT, self-reflection."""

from __future__ import annotations

import ast
import math
import re

from answer_scoring import grounding_terms, score_answer
from assignment2_queries import SYSTEM_ROLE
from domain_lm import retrieve_topics

MAX_NEW_TOKENS = 200


def keywords_for(query: str, k: int = 1) -> list[str]:
    topics = retrieve_topics(query, k=k)
    if not topics:
        return []
    blob = " ".join(t["abstract"] + " " + " ".join(t["points"]) for t in topics)
    return grounding_terms([blob], top_k=20)


# --------------------------------------------------------------------------
# B1 — Chain-of-Thought
# --------------------------------------------------------------------------

def standard_prompt(query: str) -> str:
    return f"Answer the following medical-literature question.\n\nQuestion: {query}\nAnswer:"


def cot_prompt(query: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        f"Question: {query}\n"
        "Let's think step by step. Number each reasoning step, then give a line beginning with 'Answer:'."
    )


def run_cot(lm, query: str) -> dict:
    keywords = keywords_for(query)
    standard = lm.one(standard_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    cot = lm.one(cot_prompt(query), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {
        "technique": "cot",
        "query": query,
        "standard_answer": standard,
        "standard_score": score_answer(standard, query, keywords),
        "answer": cot,
        "score": score_answer(cot, query, keywords),
        "final_answer": extract_final_answer(cot),
        "reasoning_steps": len(re.findall(r"(?mi)^\s*(?:step\s*\d+|\d+[.)])", cot)),
    }


def extract_final_answer(text: str) -> str:
    match = re.search(r"(?mi)^\s*answer\s*:\s*(.+)$", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# --------------------------------------------------------------------------
# B1 — Tree-of-Thought
# --------------------------------------------------------------------------

TOT_LENSES = ["mechanism", "measurement", "risk-and-monitoring"]


def tot_prompt(query: str, lens: str, index: int) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        f"Candidate reasoning path {index}. Approach: reason primarily through the {lens} of the problem.\n"
        f"Question: {query}\n"
        "Give the reasoning for this path, then a line beginning with 'Answer:'."
    )


def run_tot(lm, query: str, num_paths: int = 3) -> dict:
    keywords = keywords_for(query)
    paths = []
    for index in range(num_paths):
        lens = TOT_LENSES[index % len(TOT_LENSES)]
        text = lm.one(tot_prompt(query, lens, index + 1), max_new_tokens=MAX_NEW_TOKENS, temperature=0.7)
        paths.append({"path": index + 1, "lens": lens, "text": text,
                      "score": score_answer(text, query, keywords)})
    best = max(paths, key=lambda p: p["score"]["total"])
    return {
        "technique": "tot",
        "query": query,
        "paths": paths,
        "selected_path": best["path"],
        "selected_lens": best["lens"],
        "answer": best["text"],
        "score": best["score"],
        "path_scores": [p["score"]["total"] for p in paths],
    }


# --------------------------------------------------------------------------
# B1 — Step-Back prompting
# --------------------------------------------------------------------------

def stepback_question_prompt(query: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Write one step-back question that asks for the general principle behind the specific question below. "
        "Return the question only.\n\n"
        f"Specific question: {query}\nStep-back question:"
    )


def stepback_answer_prompt(query: str, principle_question: str, principle: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        f"General principle question: {principle_question}\n"
        f"General principle: {principle}\n\n"
        "Now use the general principle to answer the specific question. End with a line beginning 'Answer:'.\n\n"
        f"Specific question: {query}"
    )


def run_stepback(lm, query: str) -> dict:
    keywords = keywords_for(query)
    principle_question = lm.one(stepback_question_prompt(query), max_new_tokens=80, temperature=0.0)
    principle = lm.one(
        f"{SYSTEM_ROLE}\n\nState the general principle.\n\nQuestion: {principle_question}\nAnswer:",
        max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    answer = lm.one(stepback_answer_prompt(query, principle_question, principle),
                    max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    return {
        "technique": "step_back",
        "query": query,
        "stepback_question": principle_question,
        "principle": principle,
        "answer": answer,
        "score": score_answer(answer, query, keywords),
    }


# --------------------------------------------------------------------------
# B2 — Self-Reflection
# --------------------------------------------------------------------------

def reflection_prompt(query: str, draft: str) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        f"Question: {query}\n"
        f"Your previous answer:\n{draft}\n\n"
        "Critique the previous answer for missing corpus facts, missing limiting conditions, and missing "
        "monitoring implications. Then revise your previous answer. Use the headings 'Critique:' and "
        "'Revised answer:'."
    )


def run_self_reflection(lm, query: str, draft: str | None = None) -> dict:
    keywords = keywords_for(query)
    draft = draft if draft is not None else lm.one(standard_prompt(query),
                                                   max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    revision = lm.one(reflection_prompt(query, draft), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    revised = revision.split("Revised answer:", 1)[-1].strip() if "Revised answer:" in revision else revision
    return {
        "technique": "self_reflection",
        "query": query,
        "draft": draft,
        "draft_score": score_answer(draft, query, keywords),
        "critique_and_revision": revision,
        "answer": revised,
        "score": score_answer(revised, query, keywords),
    }


# --------------------------------------------------------------------------
# B2 — PAL (Program-Aided Language model)
# --------------------------------------------------------------------------

_ALLOWED_MODULES = {"math"}
_FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview",
    "__import__", "__builtins__",
}
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum, "len": len,
    "float": float, "int": int, "str": str, "bool": bool, "range": range,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "print": print,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "pow": pow, "divmod": divmod,
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise UnsafeCodeError(f"Import of '{name}' is not allowed inside the PAL sandbox.")
    return __import__(name, globals, locals, fromlist, level)


class UnsafeCodeError(ValueError):
    pass


def _validate_ast(tree: ast.AST) -> None:
    """Reject generated code that could reach the host environment."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in getattr(node, "names", [])]
            module = (getattr(node, "module", "") or "").split(".")[0]
            for name in names + ([module] if module else []):
                if name not in _ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Import of '{name}' is not allowed; only {_ALLOWED_MODULES}.")
        elif isinstance(node, (ast.While, ast.AsyncFor, ast.AsyncWith, ast.AsyncFunctionDef,
                               ast.Await, ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal,
                               ast.Lambda, ast.ClassDef, ast.With, ast.Try, ast.Raise)):
            raise UnsafeCodeError(f"Construct '{type(node).__name__}' is not allowed in PAL code.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCodeError("Dunder attribute access is not allowed.")
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in _ALLOWED_MODULES):
                raise UnsafeCodeError("Attribute access is only allowed on the math module.")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")


def safe_execute(code: str, result_var: str = "result", max_chars: int = 4000) -> dict:
    """Execute model-generated arithmetic code inside a restricted sandbox.

    Model output is untrusted input, so the code is AST-validated before it runs:
    only `math` may be imported, loops/attribute escapes/dunders are rejected,
    and builtins are reduced to a numeric allow-list.
    """
    if len(code) > max_chars:
        raise UnsafeCodeError("Generated program is too long to execute.")
    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)
    builtins_map = dict(_SAFE_BUILTINS)
    builtins_map["__import__"] = _restricted_import
    sandbox_globals = {"__builtins__": builtins_map, "math": math}
    sandbox_locals: dict = {}
    exec(compile(tree, filename="<pal>", mode="exec"), sandbox_globals, sandbox_locals)  # noqa: S102
    if result_var not in sandbox_locals:
        raise UnsafeCodeError(f"The program did not assign '{result_var}'.")
    return {"result": sandbox_locals[result_var], "variables": {
        k: v for k, v in sandbox_locals.items() if not k.startswith("_")}}


def pal_prompt(problem: dict) -> str:
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Solve the problem by writing Python code instead of reasoning in prose. "
        "Use only arithmetic and the math module. Assign the final numeric answer to a variable named "
        "`result`. Return the code inside a ```python fenced block.\n"
        f"Formula hint: {problem['hint']}\n\n"
        f"Question: {problem['question']}\nPython:"
    )


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    return (fenced.group(1) if fenced else text).strip()


def run_pal(lm, problem: dict, retries: int = 2) -> dict:
    prompt = pal_prompt(problem)
    attempts = []
    for attempt in range(1, retries + 2):
        raw = lm.one(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0 if attempt == 1 else 0.5)
        code = extract_code(raw)
        try:
            execution = safe_execute(code)
        except (UnsafeCodeError, SyntaxError, Exception) as exc:  # noqa: BLE001 - reported, not raised
            attempts.append({"attempt": attempt, "code": code, "error": f"{type(exc).__name__}: {exc}"})
            prompt = (f"{prompt}\n\nYour previous program failed with: {exc}\n"
                      "Return corrected code only.\nPython:")
            continue
        value = execution["result"]
        expected = problem.get("expected")
        correct = None
        if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
            correct = abs(float(value) - float(expected)) <= max(0.05, abs(expected) * 0.01)
        attempts.append({"attempt": attempt, "code": code, "result": value})
        return {
            "technique": "pal",
            "problem_id": problem["id"],
            "question": problem["question"],
            "code": code,
            "result": value,
            "unit": problem.get("unit"),
            "expected": expected,
            "matches_expected": correct,
            "attempts": attempts,
            "answer": f"{value} {problem.get('unit', '')}".strip(),
        }
    return {"technique": "pal", "problem_id": problem["id"], "question": problem["question"],
            "code": None, "result": None, "attempts": attempts, "answer": "PAL execution failed"}


# --------------------------------------------------------------------------
# B2 — ReACT
# --------------------------------------------------------------------------

def _corpus_blocks():
    from medical_data import TOPICS
    return TOPICS


def tool_search_corpus(query: str, k: int = 2) -> str:
    hits = retrieve_topics(query, k=k)
    if not hits:
        return "No matching corpus topic."
    return " | ".join(f"{t['title']}: {t['abstract'].split('. ')[0]}." for t in hits)


def tool_lookup_topic(title: str) -> str:
    for topic in _corpus_blocks():
        if title.strip().lower() in topic["title"].lower():
            points = " ".join(f"({i}) {p}" for i, p in enumerate(topic["points"], 1))
            return f"{topic['title']} — {topic['abstract']} Points: {points}"
    return f"No topic titled '{title}' in the corpus."


def tool_calculate(expression: str) -> str:
    try:
        return str(safe_execute(f"result = {expression}")["result"])
    except Exception as exc:  # noqa: BLE001 - surfaced as an observation
        return f"Calculation error: {exc}"


TOOLS = {
    "search_corpus": tool_search_corpus,
    "lookup_topic": tool_lookup_topic,
    "calculate": tool_calculate,
}

ACTION_PATTERN = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def react_prompt(question: str, scratchpad: str) -> str:
    tool_doc = "\n".join(f"- {name}[argument]" for name in TOOLS)
    return (
        f"{SYSTEM_ROLE}\n\n"
        "Answer the question using a Thought / Action / Observation loop. Available actions:\n"
        f"{tool_doc}\n- finish[final answer]\n\n"
        "Write exactly one Thought line and one Action line per turn, then stop and wait for the "
        "Observation.\n\n"
        f"Question: {question}\n{scratchpad}"
    )


def run_react(lm, question: str, max_cycles: int = 5) -> dict:
    scratchpad = ""
    trace = []
    final = None
    for cycle in range(1, max_cycles + 1):
        turn = lm.one(react_prompt(question, scratchpad), max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
        turn = turn.split("Observation:")[0].strip()
        thought = ""
        thought_match = re.search(r"Thought:\s*(.+)", turn)
        if thought_match:
            thought = thought_match.group(1).splitlines()[0].strip()
        action_match = ACTION_PATTERN.search(turn)
        if not action_match:
            trace.append({"cycle": cycle, "thought": thought, "action": None,
                          "observation": "No parsable action; stopping."})
            break
        tool, argument = action_match.group(1), action_match.group(2).strip()
        if tool == "finish":
            final = argument
            trace.append({"cycle": cycle, "thought": thought, "action": "finish",
                          "argument": argument, "observation": "Loop complete."})
            break
        observation = TOOLS[tool](argument) if tool in TOOLS else f"Unknown action '{tool}'."
        trace.append({"cycle": cycle, "thought": thought, "action": tool,
                      "argument": argument, "observation": observation})
        scratchpad += f"Thought: {thought}\nAction: {tool}[{argument}]\nObservation: {observation}\n"

    if final is None:
        final = lm.one(
            f"{SYSTEM_ROLE}\n\nUse the trace below to write the final answer.\n\n"
            f"Question: {question}\n{scratchpad}\nAnswer:",
            max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    keywords = keywords_for(question)
    return {
        "technique": "react",
        "query": question,
        "cycles": len(trace),
        "trace": trace,
        "answer": final,
        "score": score_answer(final, question, keywords),
    }
