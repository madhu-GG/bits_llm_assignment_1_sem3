"""Offline smoke test for Assignment 2A Parts A and B (no GPU, no model download)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import a2_paths  # noqa: F401,E402
import prompt_strategies as ps  # noqa: E402
import reasoning_strategies as rs  # noqa: E402
from assignment2_queries import (  # noqa: E402
    CHAIN_QUERY,
    DOMAIN_QUERIES,
    PAL_PROBLEMS,
    REACT_QUESTION,
    REASONING_PROBLEMS,
)
from domain_lm import DomainLM  # noqa: E402


def main() -> int:
    lm = DomainLM(backend="offline", label="sft_offline")
    failures = []

    query = DOMAIN_QUERIES[1]
    print("== Part A1 strategies ==")
    results = {}
    for name, runner in ps.STRATEGIES.items():
        row = runner(lm, query)
        results[name] = row["score"]["total"]
        preview = row["answer"].replace("\n", " / ")[:90]
        print(f"{name:20s} {row['score']['total']:6.2f}  {preview}")
    if results["zero_shot"] >= results["instruction"]:
        failures.append("instruction prompting did not beat zero-shot")

    print("\n== Part A2 chain + function call ==")
    chain = ps.run_prompt_chain(lm, CHAIN_QUERY)
    print("chained:", chain["chained_score"]["total"], "single-step:", chain["single_step_score"]["total"])
    if chain["chained_score"]["total"] <= chain["single_step_score"]["total"]:
        failures.append("chaining did not improve on the single-step prompt")

    call = ps.run_function_call(lm, "Extract one clinical fact about diagnostic test performance.")
    print("function call valid:", call["valid"], "->", call["parsed"])
    if not call["valid"]:
        failures.append("function-calling output failed schema validation")

    print("\n== Part B1 ==")
    cot = rs.run_cot(lm, REASONING_PROBLEMS[0]["question"])
    print("cot", cot["score"]["total"], "vs standard", cot["standard_score"]["total"],
          "steps:", cot["reasoning_steps"])
    if cot["score"]["total"] <= cot["standard_score"]["total"]:
        failures.append("CoT did not improve on the standard prompt")

    tot = rs.run_tot(lm, REASONING_PROBLEMS[1]["question"])
    print("tot paths", tot["path_scores"], "selected:", tot["selected_lens"])
    if len(tot["paths"]) != 3:
        failures.append("ToT did not generate three paths")

    sb = rs.run_stepback(lm, query)
    print("step-back question:", sb["stepback_question"])
    ref = rs.run_self_reflection(lm, query)
    print("reflection", ref["draft_score"]["total"], "->", ref["score"]["total"])

    print("\n== Part B2 PAL ==")
    for problem in PAL_PROBLEMS:
        row = rs.run_pal(lm, problem)
        print(f"{problem['id']}: result={row['result']} expected={problem.get('expected')} "
              f"match={row.get('matches_expected')}")
        if not row.get("matches_expected"):
            failures.append(f"PAL {problem['id']} did not reproduce the expected value: "
                            f"{row['attempts'][-1].get('error', '')}")

    print("\n== Part B2 ReACT ==")
    react = rs.run_react(lm, REACT_QUESTION)
    for step in react["trace"]:
        print(f"  cycle {step['cycle']}: {step['action']} -> {step['observation'][:70]}")
    if react["cycles"] < 3:
        failures.append("ReACT produced fewer than three cycles")

    print("\n== PAL sandbox rejections ==")
    for bad in ["import os\nresult = os.getcwd()", "result = __import__('os').listdir('.')",
                "result = open('x').read()", "while True:\n    pass"]:
        try:
            rs.safe_execute(bad)
            failures.append(f"sandbox accepted unsafe code: {bad!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  rejected: {type(exc).__name__}")

    print("\nFAILURES:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
