# Assignment 2A — Prompt Engineering & Preference Alignment (20 marks)

Self-contained solution folder. Nothing in the Assignment 1 project is modified — this folder only
*reads* the Assignment 1 corpus, instruction dataset, and QLoRA adapters through
[src/a2_paths.py](src/a2_paths.py).

- Domain: Medical & Clinical Literature (same corpus as Assignment 1A)
- SFT base for DPO: best Assignment 1 Part B adapter, `../outputs/qlora_adapters/adapter_B`
- Base model: `microsoft/biogpt-large`, or the Assignment 1 CPT checkpoint when it exists

## Deliverables

| File | Contents |
|---|---|
| [Assignment2A_PartA.ipynb](Assignment2A_PartA.ipynb) | 5 prompting strategies × 5 queries, 2-step prompt chain, function-calling JSON |
| [Assignment2A_PartB.ipynb](Assignment2A_PartB.ipynb) | CoT, ToT, Step-Back, PAL (executed code), ReACT (4 cycles), Self-Reflection |
| [Assignment2A_PartC.ipynb](Assignment2A_PartC.ipynb) | Preference dataset, DPO training, SFT-vs-DPO evaluation |
| [preference_dataset.jsonl](preference_dataset.jsonl) | 95 pairs with `prompt` / `chosen` / `rejected` (minimum required: 60) |

## Source modules

| Module | Role |
|---|---|
| [src/a2_paths.py](src/a2_paths.py) | Path bootstrap; read-only access to the Assignment 1 project |
| [src/assignment2_queries.py](src/assignment2_queries.py) | 5 domain queries, 2 multi-step problems, 5 held-out queries, PAL problems, JSON schema |
| [src/domain_lm.py](src/domain_lm.py) | One `generate()` interface over the real adapter or the offline simulator |
| [src/answer_scoring.py](src/answer_scoring.py) | Shared 100-point rubric used by A1, ToT, APE, and the C1 chosen/rejected decision |
| [src/prompt_strategies.py](src/prompt_strategies.py) | Part A: zero-shot, few-shot, instruction, self-consistency, APE, chaining, function calling |
| [src/reasoning_strategies.py](src/reasoning_strategies.py) | Part B: CoT, ToT, Step-Back, PAL + sandbox, ReACT, Self-Reflection |
| [src/preference_data.py](src/preference_data.py) | Part C1: pair construction, stats, validation |
| [src/dpo_training.py](src/dpo_training.py) | Part C2/C3: TRL `DPOTrainer` run, epoch metrics, comparison table |

Build / check scripts: [build_assignment2_notebooks.py](build_assignment2_notebooks.py),
[build_preference_dataset.py](build_preference_dataset.py),
[smoke_test.py](smoke_test.py), [verify_notebooks.py](verify_notebooks.py).

## The two backends

`BACKEND = "offline"` is a deterministic, corpus-grounded simulator. It needs no GPU and no model
download, and it exists so the whole pipeline — every table, the PAL execution, the ReACT trace,
and the preference dataset — can be produced and validated on CPU. **Offline text is a
reproducible stand-in, not model output.**

`BACKEND = "adapter"` loads `microsoft/biogpt-large` in 4-bit plus the Assignment 1 adapter and
generates for real. The code path is identical; every saved artefact records which backend produced
it via `lm.describe()`.

## Run order

Local validation (CPU, seconds):

```powershell
py smoke_test.py            # Parts A and B, including the PAL sandbox rejection tests
py build_preference_dataset.py
py verify_notebooks.py      # executes every code cell of all three notebooks
```

Marked run (Colab GPU):

1. Upload the whole project so that `assignment2a/` sits next to `src/` and `outputs/`.
2. `!pip install -q -r assignment2a/requirements_colab.txt`
3. Make sure Assignment 1 Part B has written `outputs/qlora_adapters/adapter_B`.
4. Part A and Part B: set `BACKEND = "adapter"`, run all.
5. Part C: set `BACKEND = "adapter"` and `REBUILD = True` to rebuild `preference_dataset.jsonl`
   from real generations, then set `RUN_DPO = True`. The DPO adapter is saved to
   `assignment2a/outputs/dpo_adapter/`.
6. Rewrite the 100–150 word C3 analysis from the real comparison table.

## PAL sandbox

PAL executes model-generated Python, which is untrusted input. `reasoning_strategies.safe_execute`
parses the program to an AST and rejects imports other than `math`, dunder and non-`math` attribute
access, loops, `with`/`try`/`raise`, lambdas, class definitions, and the
`eval`/`exec`/`open`/`__import__` name family, then runs it with a reduced builtins allow-list.
`smoke_test.py` and a Part B cell both assert that escape attempts are rejected.

## Known limitations to state in the report

- The rubric that labels `chosen`/`rejected` is the same rubric that scores the C3 comparison, so
  the evaluation is favourable by construction; an independent human rating of ~20 rows is the
  honest check.
- `chosen` responses are systematically longer than `rejected` ones (+86 words on average), so part
  of any DPO gain is length bias. The rubric caps the length reward at 220 words and the C3
  improvement test flags "longer without added grounding" as *not improved*.
- 95 pairs over 19 prompts is a small, narrow dataset.
- All medical text is educational. It is not patient-specific clinical advice.
