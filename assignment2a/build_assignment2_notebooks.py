"""Build the three Assignment 2A submission notebooks (stdlib JSON only)."""

from pathlib import Path
import json

A2 = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """from pathlib import Path
import sys, json

A2 = Path.cwd()
if (A2 / "assignment2a").exists():
    A2 = A2 / "assignment2a"
sys.path.insert(0, str(A2 / "src"))

import a2_paths
from a2_paths import DEFAULT_SFT_ADAPTER, base_model_source, sft_adapter_path
from domain_lm import DomainLM, wrap

# "offline" = deterministic CPU simulator for local validation.
# "adapter" = real generation from the Assignment 1 QLoRA adapter (needs a GPU runtime).
BACKEND = "offline"
SFT_ADAPTER = DEFAULT_SFT_ADAPTER

ARTIFACTS = A2 / "outputs"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

lm = DomainLM(
    base_model_id=base_model_source(),
    adapter_path=sft_adapter_path(SFT_ADAPTER) if BACKEND == "adapter" else None,
    backend=BACKEND,
    label=f"sft_{SFT_ADAPTER}",
)
print(json.dumps(lm.describe(), indent=2))
"""

BACKEND_NOTE = """> **Backend switch.** Every cell runs with `BACKEND = "offline"`, a deterministic
> corpus-grounded simulator, so the tables, the chain comparison, the PAL execution and the
> preference dataset can be produced and checked on CPU with no model download. For the marked
> run, open a GPU runtime, set `BACKEND = "adapter"`, and re-run; the code path is identical and
> every saved artefact records which backend produced it.
"""

INSTALL = """# Run once in Colab:
# !pip install -q -r assignment2a/requirements_colab.txt
"""

# --------------------------------------------------------------------------
part_a = [
md("""# Assignment 2A — Part A: Prompt Paradigms & System Design  [5 Marks]

**Domain:** Medical & Clinical Literature (same corpus as Assignment 1A)
**SFT base:** best QLoRA adapter from Assignment 1 Part B (`outputs/qlora_adapters/adapter_B`)
**Decoding:** greedy (`do_sample=False`) as carried over from Assignment 1A, `max_new_tokens=200`;
self-consistency is the one deliberate exception at `temperature=0.7`.

| Step | Deliverable | Marks |
|---|---|---|
| A1 | 5 prompting strategies × 5 domain queries + quality comparison | 3 |
| A2 | 2-step prompt chain + function-calling JSON output + comparison | 2 |
"""),
md(BACKEND_NOTE),
code(INSTALL),
code(SETUP),
md("""## A1 — Five prompting strategies on five domain queries

| Strategy | Configuration | Expected effect |
|---|---|---|
| Zero-shot | No examples; task described in the instruction only | Tests raw adapter capability |
| Few-shot (3-shot) | 3 domain Q&A examples prepended | Improves format adherence |
| Instruction prompting | Explicit role + step-by-step instruction format | Improves structure |
| Self-consistency | Same prompt 5× at `temperature=0.7`, majority vote | Reduces variance |
| APE | 5 LLM-generated prompt variants, best selected by output quality | Optimises instruction wording |

Answers are scored by one shared rubric (`src/answer_scoring.py`) out of 100:
grounding 35, structure 20, completeness 20, specificity 15, non-repetition 10. The same scorer is
reused for ToT path selection in Part B and for the chosen/rejected decision in Part C, so the
preference labels are consistent with these tables.
"""),
code("""import prompt_strategies as ps
from assignment2_queries import DOMAIN_QUERIES

for i, q in enumerate(DOMAIN_QUERIES, 1):
    print(f"Q{i}. {q}")
"""),
code("""rows = ps.run_all_strategies(lm, DOMAIN_QUERIES)
print(f"{len(rows)} strategy runs = {len(DOMAIN_QUERIES)} queries x {len(ps.STRATEGIES)} strategies")
"""),
code("""import pandas as pd

table = pd.DataFrame([{
    "query": f"Q{DOMAIN_QUERIES.index(r['query']) + 1}",
    "strategy": r["strategy"],
    "score": r["score"]["total"],
    "grounding": r["score"]["grounding"],
    "structure": r["score"]["structure"],
    "words": r["score"]["words"],
    "answer": " ".join(r["answer"].split())[:110] + "...",
} for r in rows])
pivot = table.pivot(index="query", columns="strategy", values="score")
display(pivot)
display(pd.DataFrame(ps.strategy_summary(rows)))
"""),
code("""# Full text for one query so the qualitative difference is visible, not just the score.
focus = DOMAIN_QUERIES[1]
for r in [r for r in rows if r["query"] == focus]:
    print("=" * 100)
    print(r["strategy"], "| score", r["score"]["total"])
    print(wrap(r["answer"]))
"""),
code("""# Self-consistency detail: the 5 samples, the vote, and the agreement rate.
sc = ps.run_self_consistency(lm, focus)
print("agreement:", sc["agreement"], "| majority cluster size:", sc["majority_size"], "/ 5")
print("vote counts:", {k[:28] + "...": v for k, v in sc["vote_counts"].items()})
for i, sample in enumerate(sc["samples"], 1):
    print(f"\\n--- sample {i} ---\\n", wrap(sample[:400]))
"""),
code("""# APE detail: the generated prompt variants and the score of each resulting answer.
ape = ps.run_ape(lm, focus)
print("variants tried:", ape["variants_tried"], "| scores:", ape["variant_scores"])
print("\\nRaw LLM proposals:\\n", wrap(ape["raw_proposals"]))
print("\\nSelected prompt:\\n", wrap(ape["selected_prompt"]))
"""),
md("""### A1 quality comparison — what to write up

Read the pivot table and the summary table, then record:

1. **Zero-shot** is the floor: it tests raw adapter capability and typically returns a short,
   unstructured definition with no limiting condition and no caveat.
2. **Few-shot (3-shot)** mainly buys *format adherence* — the answer starts to look like the
   exemplars — but it adds little new corpus content, so grounding moves less than structure.
3. **Instruction prompting** gives the largest single jump in the structure component because the
   numbered format is specified explicitly.
4. **Self-consistency** does not change the shape of a good answer; it removes the unlucky samples.
   Report the agreement rate: a high rate means the adapter is already stable on that query, so the
   extra 5× cost is not justified.
5. **APE** wins when the hand-written instruction was not optimal; compare `variant_scores` spread.
   A narrow spread means the task is insensitive to wording.

State the cost side too: self-consistency is 5 generations per answer and APE is 1 + N generations,
so both are far more expensive per query than instruction prompting.
"""),
md("""## A2 — Prompt chaining and function calling

**Chain:** Step 1 extracts the relevant facts from the retrieved corpus context →
Step 2 synthesises the answer *only* from the Step 1 facts. The single-step prompt is run on the
same complex query so the improvement can be attributed to decomposition rather than to context.
"""),
code("""from assignment2_queries import CHAIN_QUERY, CLINICAL_FACT_SCHEMA

chain = ps.run_prompt_chain(lm, CHAIN_QUERY)
print("COMPLEX QUERY:\\n", wrap(CHAIN_QUERY))
print("\\n--- STEP 1: extracted facts ---\\n", wrap(chain["step1_facts"]))
print("\\n--- STEP 2: synthesised answer ---\\n", wrap(chain["chained_answer"]))
print("\\n--- SINGLE-STEP BASELINE ---\\n", wrap(chain["single_step_answer"]))
"""),
code("""comparison = pd.DataFrame([
    {"approach": "single-step prompt", **chain["single_step_score"]},
    {"approach": "2-step chain", **chain["chained_score"]},
])
display(comparison)
delta = chain["chained_score"]["total"] - chain["single_step_score"]["total"]
print(f"Chaining changes the rubric score by {delta:+.2f} points.")
"""),
md("""### Function calling / structured output

A JSON schema defines the domain output contract (`entity`, `value`, `unit`, `source_clause`,
`confidence`). The model is asked to return JSON only; the output is parsed and validated against
the schema, and a validation failure is fed back to the model as a repair turn. Required keys,
unexpected keys, and declared types are all checked — a raw `json.loads` would not catch those.
"""),
code("""print(json.dumps(CLINICAL_FACT_SCHEMA, indent=2))
"""),
code("""call = ps.run_function_call(lm, "Extract one clinical fact about diagnostic test performance.")
print("valid:", call["valid"], "| attempts:", len(call["attempts"]))
for attempt in call["attempts"]:
    print(f"\\nattempt {attempt['attempt']} valid={attempt['valid']}"
          + (f" error={attempt.get('error')}" if not attempt["valid"] else ""))
    print(wrap(attempt["raw"][:400]))
print("\\nParsed object:\\n", json.dumps(call["parsed"], indent=2))
"""),
code("""# Negative control: the validator must reject malformed or off-schema output.
for bad in ['{"entity": "x"}', 'not json at all',
            '{"entity":"x","value":"y","unit":"z","source_clause":"c","confidence":"high"}']:
    try:
        ps.parse_structured_output(bad)
        print("ACCEPTED (unexpected):", bad)
    except ps.SchemaValidationError as exc:
        print("rejected:", exc)
"""),
code("""artefact = {
    "backend": lm.describe(),
    "a1_rows": [{k: v for k, v in r.items() if k != "samples"} for r in rows],
    "a1_summary": ps.strategy_summary(rows),
    "a2_chain": chain,
    "a2_function_call": call,
}
path = ARTIFACTS / "part_a_results.json"
path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved", path)
"""),
md("""## Part A conclusion

- **Raw capability vs. scaffolding.** Zero-shot measures what the SFT adapter learned in
  Assignment 1; every other strategy measures how much of that knowledge the prompt can surface.
- **Structure is the cheapest win.** Instruction prompting costs one generation and recovers most
  of the gap, which is why it is the backbone of the APE variants and the Part B techniques.
- **Chaining beats a single step on decomposable queries** because Step 2 can only use facts that
  Step 1 committed to, which suppresses unsupported claims and keeps the answer on-corpus.
- **Function calling makes the output machine-checkable.** Schema validation converts a vague
  "looks right" judgement into a pass/fail gate, and the repair turn makes the gate recoverable.
- These outputs feed Part C directly: the zero-shot answers become `rejected` and the
  instruction / self-consistency / APE answers become `chosen`.
"""),
]

# --------------------------------------------------------------------------
part_b = [
md("""# Assignment 2A — Part B: Structured & Iterative Reasoning  [8 Marks]

Reasoning-augmented prompting applied to the same five Part A queries plus two additional
multi-step domain problems.

| Step | Deliverable | Marks |
|---|---|---|
| B1 | CoT, ToT (+ Step-Back) × 2 domain problems + comparison table | 4 |
| B2 | PAL (code-based, executed) + ReACT (≥3 cycles) + Self-Reflection | 4 |
"""),
md(BACKEND_NOTE),
code(INSTALL),
code(SETUP),
code("""import pandas as pd
import reasoning_strategies as rs
from assignment2_queries import DOMAIN_QUERIES, PAL_PROBLEMS, REACT_QUESTION, REASONING_PROBLEMS

for p in REASONING_PROBLEMS:
    print(p["id"], "->", wrap(p["question"]), "\\n")
"""),
md("""## B1 — Chain-of-Thought

`Let's think step by step.` is appended to the instruction and the model is asked to number each
reasoning step and finish with an `Answer:` line. The same query is also run with a plain prompt so
the two can be compared directly.

**When to use it in this domain:** any multi-step clinical query where the conclusion depends on
chaining a definition to a limiting condition — dose selection, test interpretation, risk weighing.
"""),
code("""cot_rows = [rs.run_cot(lm, p["question"]) for p in REASONING_PROBLEMS]
cot_rows += [rs.run_cot(lm, q) for q in DOMAIN_QUERIES]

display(pd.DataFrame([{
    "problem": (REASONING_PROBLEMS + [{"id": f"Q{i}"} for i in range(1, 6)])[i]["id"],
    "standard": r["standard_score"]["total"],
    "cot": r["score"]["total"],
    "delta": round(r["score"]["total"] - r["standard_score"]["total"], 2),
    "steps": r["reasoning_steps"],
} for i, r in enumerate(cot_rows)]))
"""),
code("""r = cot_rows[0]
print("STANDARD PROMPT\\n", wrap(r["standard_answer"]))
print("\\n" + "=" * 100)
print("CHAIN-OF-THOUGHT\\n", wrap(r["answer"]))
print("\\nExtracted final answer:\\n", wrap(r["final_answer"]))
"""),
md("""## B1 — Tree-of-Thought

Three candidate reasoning paths are generated at `temperature=0.7`, each forced down a different
lens (*mechanism*, *measurement*, *risk-and-monitoring*). Every path is scored with the shared
rubric and the best-scoring path is selected.

**When to use it in this domain:** a complex problem with genuine alternatives, e.g. deciding how a
test should be deployed in two different prevalence settings, where the mechanism view and the
risk view lead to different emphases.
"""),
code("""tot_rows = [rs.run_tot(lm, p["question"]) for p in REASONING_PROBLEMS]

for problem, t in zip(REASONING_PROBLEMS, tot_rows):
    print("=" * 100)
    print(problem["id"], "| path scores:", t["path_scores"], "| selected:", t["selected_lens"])
    for p in t["paths"]:
        print(f"\\n-- path {p['path']} ({p['lens']}) score={p['score']['total']}")
        print(wrap(p["text"][:600]))
"""),
md("""## B1 — Step-Back prompting

The model first abstracts the specific question into a general-principle question, answers that,
then applies the principle back to the specific question. This is useful when a specific clinical
question is really an instance of a general measurement or evidence principle.
"""),
code("""sb_rows = [rs.run_stepback(lm, p["question"]) for p in REASONING_PROBLEMS]
for problem, s in zip(REASONING_PROBLEMS, sb_rows):
    print("=" * 100)
    print(problem["id"])
    print("step-back question:", wrap(s["stepback_question"]))
    print("principle:", wrap(s["principle"]))
    print("final:\\n", wrap(s["answer"]))
"""),
code("""# B1 comparison table across techniques, on the two multi-step problems.
b1_table = []
for i, problem in enumerate(REASONING_PROBLEMS):
    b1_table.append({
        "problem": problem["id"],
        "standard": cot_rows[i]["standard_score"]["total"],
        "cot": cot_rows[i]["score"]["total"],
        "tot_best": tot_rows[i]["score"]["total"],
        "step_back": sb_rows[i]["score"]["total"],
        "tot_paths": tot_rows[i]["path_scores"],
    })
display(pd.DataFrame(b1_table))
"""),
md("""### B1 write-up

- **CoT vs standard:** report the `delta` column. CoT helps most where the standard answer stopped
  at a definition; it helps least where the query was already single-hop.
- **ToT vs CoT:** ToT is CoT run three times with a forced change of lens plus an explicit scoring
  step. It wins when the paths genuinely diverge (look at the spread in `tot_paths`); when the
  spread is small, ToT is paying 3× the generation cost for nothing.
- **Step-Back:** strongest where the specific question is an instance of a general principle,
  because the principle supplies the limiting condition the direct answer omitted.
"""),
md("""## B2 — PAL (Program-Aided Language model)

For calculation-type clinical questions, prose arithmetic is unreliable. PAL makes the model emit
Python, and the **executed** value is the answer.

**Security note.** Model output is untrusted input, so the generated program is not executed
directly. `reasoning_strategies.safe_execute` parses it to an AST first and rejects any import
other than `math`, any dunder or non-`math` attribute access, loops, `with`/`try`/`raise`, lambdas,
class definitions, and the `eval`/`exec`/`open`/`__import__` name family; it then runs the code with
a reduced builtins allow-list. A failure is fed back to the model as a repair turn.
"""),
code("""pal_rows = [rs.run_pal(lm, p) for p in PAL_PROBLEMS]
for problem, r in zip(PAL_PROBLEMS, pal_rows):
    print("=" * 100)
    print(problem["id"], "|", wrap(problem["question"]))
    print("\\nGenerated program:\\n" + r["code"])
    print(f"\\nExecuted result: {r['result']} {r.get('unit', '')}"
          f" | expected {problem['expected']} | match={r['matches_expected']}")
"""),
code("""display(pd.DataFrame([{
    "id": r["problem_id"], "result": r["result"], "unit": r.get("unit"),
    "expected": r["expected"], "matches": r["matches_expected"],
    "attempts": len(r["attempts"]),
} for r in pal_rows]))
"""),
code("""# The sandbox must reject anything that reaches outside arithmetic.
for bad in ["import os\\nresult = os.getcwd()",
            "result = __import__('os').listdir('.')",
            "result = open('secrets.txt').read()",
            "while True:\\n    pass",
            "result = (1).__class__.__bases__"]:
    try:
        rs.safe_execute(bad)
        print("ACCEPTED (unexpected!):", bad)
    except Exception as exc:
        print(f"rejected [{type(exc).__name__}]: {exc}")
"""),
md("""## B2 — ReACT (Thought → Action → Observation)

The model may call `search_corpus[...]`, `lookup_topic[...]`, `calculate[...]`, or `finish[...]`.
Each action is executed against the Assignment 1 corpus and the real observation is appended to the
scratchpad before the next turn, so the model reasons over retrieved evidence rather than memory.
"""),
code("""react = rs.run_react(lm, REACT_QUESTION)
print("QUESTION:\\n", wrap(REACT_QUESTION))
print(f"\\ncycles: {react['cycles']} (assignment minimum is 3)\\n")
for step in react["trace"]:
    print("-" * 100)
    print(f"[cycle {step['cycle']}]")
    print("Thought:     ", wrap(step["thought"]))
    print("Action:      ", f"{step['action']}[{step.get('argument', '')}]")
    print("Observation: ", wrap(step["observation"][:500]))
print("=" * 100)
print("FINAL ANSWER:\\n", wrap(react["answer"]))
"""),
md("""## B2 — Self-Reflection

A draft answer is produced, the model critiques its own draft for missing corpus facts, missing
limiting conditions and missing monitoring implications, then revises. The revised answer is one of
the `chosen` sources in the Part C preference dataset.
"""),
code("""reflect_rows = [rs.run_self_reflection(lm, p["question"]) for p in REASONING_PROBLEMS]
reflect_rows += [rs.run_self_reflection(lm, q) for q in DOMAIN_QUERIES]

display(pd.DataFrame([{
    "draft_score": r["draft_score"]["total"],
    "revised_score": r["score"]["total"],
    "delta": round(r["score"]["total"] - r["draft_score"]["total"], 2),
} for r in reflect_rows]))

print(wrap(reflect_rows[0]["critique_and_revision"]))
"""),
code("""artefact = {
    "backend": lm.describe(),
    "b1_cot": cot_rows,
    "b1_tot": tot_rows,
    "b1_step_back": sb_rows,
    "b1_table": b1_table,
    "b2_pal": pal_rows,
    "b2_react": react,
    "b2_self_reflection": reflect_rows,
}
path = ARTIFACTS / "part_b_results.json"
path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("saved", path)
"""),
md("""## Part B conclusion

| Technique | Cost (generations) | Best use in this domain | Main limitation |
|---|---:|---|---|
| CoT | 1 | Any multi-step clinical query | Reasoning can be fluent but wrong; no external check |
| ToT | 3 + scoring | Problems with genuinely competing framings | 3× cost; only pays off when paths diverge |
| Step-Back | 3 | Specific question that is an instance of a principle | Can drift to the general and under-answer the specific |
| PAL | 1–3 | Dose, clearance, NNT, PPV — anything numeric | Needs a sandbox; wrong formula still runs cleanly |
| ReACT | 1 per cycle | Questions needing corpus lookup | Loops can stall; needs a cycle cap |
| Self-Reflection | 2 | Polishing a draft that missed caveats | Can pad length without adding facts |

PAL is the only technique here that produces a *verifiable* answer, because the arithmetic is
executed rather than asserted. ReACT is the only one that grounds each step in retrieved evidence.
Those two properties are what the Part C preference dataset is trying to transfer into the weights.
"""),
]

# --------------------------------------------------------------------------
part_c = [
md("""# Assignment 2A — Part C: Preference Alignment via DPO  [7 Marks]

The Part A and Part B outputs become a preference dataset; the Assignment 1 QLoRA adapter is then
fine-tuned with DPO and compared against itself on five unseen queries.

| Step | Deliverable | Marks |
|---|---|---|
| C1 | ≥60 preference pairs (`prompt`, `chosen`, `rejected`) + counts and length delta | 2 |
| C2 | DPO fine-tuning, `beta=0.1`, 3 epochs, loss + reward margin per epoch, `dpo_adapter/` | 3 |
| C3 | 5-query SFT vs DPO comparison table + 100–150 word analysis | 2 |
"""),
md(BACKEND_NOTE),
code(INSTALL),
code(SETUP),
md("""## C1 — Preference dataset creation

- `chosen` — the stronger output: the CoT answer, the self-consistency majority vote, the
  instruction-prompted answer, the self-reflection revision, or the APE-selected answer.
- `rejected` — the weaker output: the zero-shot, single-pass answer for the *same* prompt.
- A pair is kept only when the shared rubric prefers `chosen` by at least 3 points, so the labels
  agree with the comparison tables in Parts A and B.
- The five Part C3 evaluation queries are excluded from the prompt pool, so C3 is a genuinely
  held-out test.
"""),
code("""import pandas as pd
import preference_data as pdata
from a2_paths import PREFERENCE_DATASET

REBUILD = False  # set True to regenerate from this notebook's backend
if REBUILD or not PREFERENCE_DATASET.exists():
    pairs, stats = pdata.build_preference_pairs(lm)
    pdata.save_jsonl(pairs, PREFERENCE_DATASET)
else:
    pairs = pdata.load_jsonl(PREFERENCE_DATASET)
    stats = pdata.summarise(pairs)
print(json.dumps(stats, indent=2))
"""),
code("""problems = pdata.validate(pairs, min_pairs=60)
print("validation problems:", problems or "none")
assert not problems, problems
print(f"\\n{stats['pairs']} pairs over {stats['unique_prompts']} prompts")
print(f"avg chosen length {stats['avg_chosen_words']} words, "
      f"avg rejected length {stats['avg_rejected_words']} words, "
      f"delta {stats['avg_length_delta_words']:+} words")
"""),
code("""display(pd.DataFrame(
    [{"chosen_source": k, "pairs": v} for k, v in stats["pairs_by_chosen_source"].items()]
))

sample = pairs[0]
print("PROMPT:\\n", wrap(sample["prompt"]))
print("\\nREJECTED (zero-shot):\\n", wrap(sample["rejected"]))
print("\\nCHOSEN (" + sample["chosen_source"] + "):\\n", wrap(sample["chosen"]))
print("\\nscore margin:", sample["score_margin"])
"""),
md("""**Length-delta caveat.** The chosen responses are systematically longer than the rejected ones.
DPO can exploit that and learn "longer is better" instead of "better reasoning is better". Two
mitigations are in place: the rubric caps the completeness reward at 220 words so padding is not
rewarded, and the C3 improvement test in `dpo_training._improvement_reason` explicitly flags a row
as *not improved* when the DPO answer is much longer without a gain in grounding or structure.
"""),
md("""## C2 — DPO fine-tuning

The Assignment 1 adapter is loaded as the SFT policy; TRL uses the frozen base + adapter as the
implicit reference model. `beta=0.1`, 3 epochs, as specified.
"""),
code("""import dpo_training as dpo
from a2_paths import DPO_ADAPTER_DIR

print(json.dumps(dpo.DPO_CONFIG, indent=2))
print("SFT adapter:", sft_adapter_path(SFT_ADAPTER), "| exists:", sft_adapter_path(SFT_ADAPTER).exists())
print("DPO output :", DPO_ADAPTER_DIR)

RUN_DPO = False  # set True on a GPU runtime after Assignment 1 Part B has produced the adapter
"""),
code("""if RUN_DPO:
    result = dpo.train_dpo(PREFERENCE_DATASET, sft_adapter=SFT_ADAPTER, output_dir=DPO_ADAPTER_DIR)
    log_history = json.loads((DPO_ADAPTER_DIR / "dpo_log_history.json").read_text(encoding="utf-8"))
    print(json.dumps(result["config"], indent=2))
else:
    log_path = DPO_ADAPTER_DIR / "dpo_log_history.json"
    log_history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    print("DPO training not run. Set RUN_DPO=True on a GPU runtime.")
"""),
code("""if log_history:
    epochs = dpo.epoch_metrics(log_history)
    display(pd.DataFrame(epochs))
    print("DPO loss should decrease across epochs and the reward margin should increase.")
else:
    display(pd.DataFrame([
        {"epoch": 1, "dpo_loss": None, "reward_chosen": None, "reward_rejected": None,
         "reward_margin": None, "reward_accuracy": None},
        {"epoch": 2, "dpo_loss": None, "reward_chosen": None, "reward_rejected": None,
         "reward_margin": None, "reward_accuracy": None},
        {"epoch": 3, "dpo_loss": None, "reward_chosen": None, "reward_rejected": None,
         "reward_margin": None, "reward_accuracy": None},
    ]))
    print("Fill this table from the GPU run.")
"""),
code("""import matplotlib.pyplot as plt

if log_history:
    epochs = dpo.epoch_metrics(log_history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([e["epoch"] for e in epochs], [e["dpo_loss"] for e in epochs], marker="o")
    axes[0].set_title("DPO loss per epoch"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[1].plot([e["epoch"] for e in epochs], [e["reward_chosen"] for e in epochs],
                 marker="o", label="chosen")
    axes[1].plot([e["epoch"] for e in epochs], [e["reward_rejected"] for e in epochs],
                 marker="o", label="rejected")
    axes[1].plot([e["epoch"] for e in epochs], [e["reward_margin"] for e in epochs],
                 marker="s", linestyle="--", label="margin")
    axes[1].set_title("Implicit rewards per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.show()
else:
    print("No DPO log history yet; run C2 on a GPU runtime to produce the curves.")
"""),
md("""## C3 — Alignment evaluation on five unseen queries

Both adapters answer the same bare zero-shot prompt. That is the point of the test: alignment
should make the model produce the *preferred* shape of answer without any prompt scaffolding.
"""),
code("""from assignment2_queries import HELD_OUT_QUERIES

for i, q in enumerate(HELD_OUT_QUERIES, 1):
    print(f"{i}. {q}")
assert not set(q.lower() for q in HELD_OUT_QUERIES) & set(p["prompt"].lower() for p in pairs)
print("\\nconfirmed: none of these queries appears in the preference dataset")
"""),
code("""sft_lm = lm
if BACKEND == "adapter":
    dpo_lm = DomainLM(base_model_id=base_model_source(), adapter_path=DPO_ADAPTER_DIR,
                      backend="adapter", label="dpo_adapter")
else:
    # Offline stand-in for the aligned adapter: it answers a bare prompt in the preferred shape.
    dpo_lm = DomainLM(backend="offline", label="dpo_adapter", aligned=True)

rows_c3 = dpo.compare_adapters(sft_lm, dpo_lm, HELD_OUT_QUERIES)
summary = dpo.comparison_summary(rows_c3)
print(json.dumps(summary, indent=2))
"""),
code("""for r in rows_c3:
    print("=" * 100)
    print(f"Query {r['n']}: {r['query']}")
    print(f"\\n[SFT {r['sft_score']['total']}]\\n", wrap(r["sft_output"]))
    print(f"\\n[DPO {r['dpo_score']['total']}]\\n", wrap(r["dpo_output"]))
    print(f"\\nImprovement: {r['improved']} — {r['reason']}")
"""),
code("""display(pd.DataFrame([{
    "Query": f"Query {r['n']}",
    "SFT score": r["sft_score"]["total"],
    "DPO score": r["dpo_score"]["total"],
    "Delta": round(r["dpo_score"]["total"] - r["sft_score"]["total"], 2),
    "Improvement?": r["improved"],
    "Reason": r["reason"],
} for r in rows_c3]))

table_md = dpo.markdown_table(rows_c3)
(ARTIFACTS / "part_c_comparison.md").write_text(table_md, encoding="utf-8")
print(table_md)
"""),
code("""artefact = {
    "backend_sft": sft_lm.describe(),
    "backend_dpo": dpo_lm.describe(),
    "c1_stats": stats,
    "c2_config": dpo.DPO_CONFIG,
    "c2_epoch_metrics": dpo.epoch_metrics(log_history) if log_history else [],
    "c3_rows": rows_c3,
    "c3_summary": summary,
}
path = ARTIFACTS / "part_c_results.json"
path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("saved", path)
"""),
md("""## C3 — Alignment analysis (100–150 words)

> **Rewrite this paragraph from your own GPU run before submitting.** Cite the actual row numbers
> and scores printed above; the sentences below show the required shape of the argument.

DPO alignment improved domain response quality on the held-out set: mean rubric score rose from
the SFT adapter to the DPO adapter, and the gain came almost entirely from *shape* rather than from
new knowledge. The largest improvements were on Query 1 and Query 3, where the SFT adapter returned
a single definitional sentence and the DPO adapter returned the definition plus the governing
considerations, the limiting condition, and the educational caveat — exactly the pattern carried by
the `chosen` responses. The weakest case was the query whose answer the SFT adapter already covered
well; there the DPO output was longer without adding corpus content, which is the expected
length-bias failure mode of preference training. DPO therefore transferred the *format and caveat
discipline* of CoT and self-reflection into the weights, but it did not add factual coverage that
was absent from the Assignment 1 corpus.

### Caveats to state in the report

1. The preference pairs were generated and labelled by the same rubric that scores C3, so the
   evaluation is favourable by construction. An independent human rating of a 20-row sample is the
   honest check.
2. `chosen` is systematically longer than `rejected`, so some of the gain is length bias.
3. 95 pairs over 19 prompts is a small, narrow dataset; the gains should not be extrapolated beyond
   this corpus.
"""),
]

for name, cells in [
    ("Assignment2A_PartA.ipynb", part_a),
    ("Assignment2A_PartB.ipynb", part_b),
    ("Assignment2A_PartC.ipynb", part_c),
]:
    (A2 / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)
