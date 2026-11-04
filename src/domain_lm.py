"""Generation wrapper used by every Assignment 2A notebook.

`DomainLM` exposes one `generate()` call that the prompting, reasoning, and
alignment code shares. It has two backends:

* ``backend="adapter"`` loads the base model plus the best QLoRA adapter from
  Assignment 1 Part B (or the DPO adapter from Part C) and generates for real.
  This is the backend used for the marked GPU run.
* ``backend="offline"`` is a deterministic, corpus-grounded simulator that runs
  on CPU with no model download. It exists so the full Part A/B/C pipeline,
  the comparison tables, and the preference dataset can be produced and
  validated locally. Offline text is a reproducible stand-in, not model output,
  and every artefact records which backend produced it.
"""

from __future__ import annotations

import random
import re
import textwrap
from pathlib import Path

from assignment2_queries import SAFETY_NOTE

try:  # the topic knowledge base from Assignment 1
    from medical_data import TOPICS
except ImportError:  # pragma: no cover - only when src/ is not on sys.path
    TOPICS = []


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "of", "to", "in", "a", "is", "for", "that", "with", "on", "as", "by", "this",
         "an", "or", "from", "are", "be", "can", "which", "when", "at", "it", "may", "what", "why",
         "how", "does", "do", "explain", "should", "between", "their"}


def _keywords(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 3 and t not in _STOP}


def retrieve_topics(query: str, k: int = 2) -> list[dict]:
    """Rank corpus topics by keyword overlap with the query."""
    query_terms = _keywords(query)
    scored = []
    for topic in TOPICS:
        blob = topic["title"] + " " + topic["abstract"] + " " + " ".join(topic["points"])
        overlap = len(query_terms & _keywords(blob))
        scored.append((overlap, topic["title"], topic))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [topic for score, _, topic in scored[:k] if score > 0] or [t[2] for t in scored[:k]]


def detect_style(prompt: str) -> str:
    """Classify the prompt so the offline backend can imitate strategy effects."""
    lowered = prompt.lower()
    if "you are a prompt engineer" in lowered:
        return "ape_meta"
    if "assign the final numeric answer" in lowered:
        return "pal"
    if "use the trace below" in lowered:
        return "react_final"
    if "thought / action / observation loop" in lowered:
        return "react"
    if "return only json" in lowered or "json schema" in lowered:
        return "json"
    if "candidate reasoning path" in lowered:
        return "tot"
    if "critique the previous answer" in lowered:
        return "reflect"
    if "write one step-back question" in lowered:
        return "stepback_question"
    if "state the general principle" in lowered:
        return "principle"
    if "use the general principle to answer" in lowered:
        return "stepback"
    if "think step by step" in lowered:
        return "cot"
    if "fact extraction" in lowered or "extract the relevant facts" in lowered:
        return "extract"
    if "synthesise the answer" in lowered or "synthesize the answer" in lowered:
        return "synthesis"
    if lowered.count("q:") >= 2:
        return "fewshot"
    if "follow this step-by-step instruction format" in lowered:
        return "instruction"
    if lowered.startswith("you are a careful medical-literature assistant"):
        return "ape_variant"
    return "bare"


_PROMPT_AWARE_STYLES = {"ape_meta", "pal", "react", "ape_variant"}

APE_PROPOSALS = [
    "Answer using only corpus-supported statements and name the source topic.",
    "Give a numbered answer: definition, two governing considerations, one caution.",
    "Explain the mechanism first, then the practical implication, then the limitation.",
    "Write a study note: 1. Define. 2. Why it matters. 3. What to monitor. 4. Educational caveat.",
    "Answer concisely but completely, stating the uncertainty that remains.",
]

# Offline PAL programs for the three shipped calculation problems.
_PAL_PROGRAMS = {
    "cockcroft": """```python
age = 68
weight_kg = 62.0
serum_creatinine = 1.4
female_factor = 0.85
crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine) * female_factor
result = round(crcl, 1)
print("creatinine clearance mL/min:", result)
```""",
    "absolute risk reduction": """```python
import math
control = 0.180
treatment = 0.125
arr = control - treatment
rrr = arr / control
nnt = math.ceil(1 / arr)
print("ARR:", round(arr, 4), "RRR:", round(rrr, 4), "NNT:", nnt)
result = nnt
```""",
    "positive predictive value": """```python
sensitivity = 0.95
specificity = 0.80
prevalence = 0.02
true_positive = sensitivity * prevalence
false_positive = (1 - specificity) * (1 - prevalence)
ppv = true_positive / (true_positive + false_positive)
result = round(ppv * 100, 1)
print("PPV %:", result)
```""",
}


def _pal_program(prompt: str) -> str:
    lowered = prompt.lower()
    for key, program in _PAL_PROGRAMS.items():
        if key in lowered:
            return program
    return "```python\nresult = None\n```"


_REACT_SCRIPT = [
    ("I must find which corpus topic defines likelihood ratios.",
     "search_corpus", "likelihood ratio sensitivity specificity"),
    ("I should read that topic in full to get the negative likelihood ratio statement.",
     "lookup_topic", "Diagnostic test performance"),
    ("I still need the corpus statement linking predictive values to prevalence.",
     "search_corpus", "predictive values prevalence"),
    ("I now have the topic, the negative likelihood ratio meaning, and the prevalence link.",
     "finish", "The corpus topic is 'Diagnostic test performance'. A negative likelihood ratio tells the "
               "clinician how much a negative result lowers the odds of the target condition; the smaller "
               "the value, the stronger the rule-out. Likelihood ratios are properties of the test, so they "
               "stay usable across settings, whereas predictive values shift with prevalence - which is why "
               "the same negative result rules out disease more convincingly in a low-prevalence clinic. "
               "Educational summary of published medical literature. Not patient-specific clinical advice."),
]


def _react_turn(prompt: str, query: str) -> str:
    cycle = min(prompt.count("Observation:"), len(_REACT_SCRIPT) - 1)
    thought, action, argument = _REACT_SCRIPT[cycle]
    return f"Thought: {thought}\nAction: {action}[{argument}]"


class _OfflineBackend:
    """Deterministic corpus-grounded text generator for CPU validation."""

    def __init__(self, seed: int = 7, aligned: bool = False):
        self.seed = seed
        # `aligned` simulates the behaviour DPO teaches: the model stops emitting a
        # bare one-liner for an undecorated prompt and answers in the preferred shape.
        self.aligned = aligned

    def _rng(self, prompt: str, index: int, temperature: float) -> random.Random:
        if temperature <= 0:
            return random.Random(hash((prompt, self.seed)) & 0xFFFFFFFF)
        return random.Random(hash((prompt, self.seed, index)) & 0xFFFFFFFF)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, index: int) -> str:
        style = detect_style(prompt)
        if self.aligned and style in ("bare", "fewshot"):
            style = "aligned"
        query = _extract_query(prompt)
        topics = retrieve_topics(query, k=2)
        rng = self._rng(prompt, index, temperature)
        if style in _PROMPT_AWARE_STYLES:
            text = self._compose_prompt_aware(style, prompt, query, topics, rng)
        else:
            text = self._compose(style, query, topics, rng, temperature)
        return _truncate_words(text, max_new_tokens)

    def _compose_prompt_aware(self, style, prompt, query, topics, rng) -> str:
        """Styles whose output depends on the surrounding prompt, not only the query."""
        if style == "ape_meta":
            return "\n".join(f"{i}. {line}" for i, line in enumerate(APE_PROPOSALS, 1))
        if style == "pal":
            return _pal_program(prompt)
        if style == "react":
            return _react_turn(prompt, query)
        if style == "ape_variant":
            return self._ape_variant(prompt, query, topics, rng)
        raise ValueError(f"Unhandled prompt-aware style: {style}")

    def _ape_variant(self, prompt, query, topics, rng) -> str:
        """Vary formatting by instruction wording so APE selection is meaningful."""
        instruction = prompt.split("\n\n")[1] if "\n\n" in prompt else prompt
        primary = topics[0]
        points = primary["points"]
        shape = hash(instruction) % 4
        abstract = primary["abstract"]
        if shape == 0:
            return f"{abstract} {points[0]}"
        if shape == 1:
            body = "\n".join(f"{i}. {p}" for i, p in enumerate(points[:3], 1))
            return f"Definition: {abstract}\n{body}\nCaution: findings depend on the measurement context."
        if shape == 2:
            return (f"Mechanism: {abstract}\nPractical implication: {points[0]}\n"
                    f"Limitation: {points[-1]}")
        body = "\n".join(f"{i}. {p}" for i, p in enumerate(points[:4], 1))
        return (f"1. Definition: {abstract}\n2. Why it matters: {points[0]}\n"
                f"3. What to monitor: {points[1]}\n4. {SAFETY_NOTE}\n{body}")

    def _compose(self, style, query, topics, rng, temperature) -> str:
        primary = topics[0] if topics else {"title": "the domain corpus", "abstract": "", "points": []}
        secondary = topics[1] if len(topics) > 1 else primary
        points = list(primary["points"])
        if temperature > 0:
            rng.shuffle(points)
        abstract = primary["abstract"]
        first_sentence = abstract.split(". ")[0].strip(". ") + "."

        if style == "bare":
            # Zero-shot: short, unstructured, no caveat. Deliberately the weak baseline.
            return f"{first_sentence} {points[0] if points else ''}".strip()

        if style == "aligned":
            body = "\n".join(f"{i}. {p}" for i, p in enumerate(points[:3], 1))
            return (f"{abstract}\n{body}\n"
                    f"Limitation: the conclusion depends on the measured values and the patient context "
                    f"(see also {secondary['title'].lower()}).\n{SAFETY_NOTE}")

        if style == "fewshot":
            return (f"{abstract} {points[0] if points else ''} "
                    f"{points[1] if len(points) > 1 else ''}").strip()

        if style == "instruction":
            body = "\n".join(f"{i}. {p}" for i, p in enumerate(points[:4], 1))
            return (f"Overview: {abstract}\n{body}\n"
                    f"Related corpus topic: {secondary['title']}.\n{SAFETY_NOTE}")

        if style == "cot":
            steps = [
                f"Step 1 - Identify the concept: the query concerns {primary['title'].lower()}.",
                f"Step 2 - Recall the corpus definition: {abstract}",
                f"Step 3 - Apply the governing considerations: {points[0] if points else ''}",
                f"Step 4 - Check the limiting condition: {points[1] if len(points) > 1 else ''}",
                f"Step 5 - Cross-check against {secondary['title'].lower()}: "
                f"{secondary['points'][0] if secondary['points'] else ''}",
            ]
            answer = (f"Answer: {first_sentence} In practice this means "
                      f"{points[0][0].lower() + points[0][1:] if points else 'the corpus guidance applies'} "
                      f"and the conclusion depends on the measurement context. {SAFETY_NOTE}")
            return "\n".join(steps) + "\n" + answer

        if style == "stepback":
            return (f"Step-back principle: {abstract}\n"
                    f"Applying the principle to the question: {points[0] if points else ''}\n"
                    f"Answer: {first_sentence} {points[1] if len(points) > 1 else ''} {SAFETY_NOTE}")

        if style == "stepback_question":
            return f"What general principle governs {primary['title'].lower()}?"

        if style == "principle":
            return (f"General principle: {abstract} "
                    f"{points[0] if points else ''}")

        if style == "react_final":
            return (f"Answer: {first_sentence} {points[0] if points else ''} "
                    f"The corpus links this to {secondary['title'].lower()}. {SAFETY_NOTE}")

        if style == "tot":
            lens = rng.choice(["mechanism", "measurement", "risk-and-monitoring"])
            chosen = points[:3] if lens == "mechanism" else points[1:4]
            body = "\n".join(f"- {p}" for p in chosen)
            return (f"Approach: {lens}.\nReasoning: {abstract}\n{body}\n"
                    f"Answer: {first_sentence} The {lens} view explains why the corpus treats this as "
                    f"context-dependent rather than a fixed rule. {SAFETY_NOTE}")

        if style == "reflect":
            return (f"Critique: the earlier answer stated the definition but omitted the limiting conditions "
                    f"and the monitoring implication.\n"
                    f"Revised answer: {abstract}\n"
                    f"1. {points[0] if points else ''}\n"
                    f"2. {points[1] if len(points) > 1 else ''}\n"
                    f"3. {points[2] if len(points) > 2 else ''}\n{SAFETY_NOTE}")

        if style == "extract":
            facts = "\n".join(f"- {p}" for p in points[:3])
            return (f"Relevant facts from {primary['title']}:\n{facts}\n"
                    f"- Supporting definition: {abstract}")

        if style == "synthesis":
            return (f"Synthesis: {abstract}\n"
                    f"1. {points[0] if points else ''}\n"
                    f"2. {points[1] if len(points) > 1 else ''}\n"
                    f"3. Cross-topic link to {secondary['title'].lower()}: "
                    f"{secondary['points'][0] if secondary['points'] else ''}\n"
                    f"Conclusion: the decision depends on the measured values and the patient context. "
                    f"{SAFETY_NOTE}")

        if style == "json":
            clause = points[0] if points else abstract
            return (
                '{"entity": "%s", "value": "%s", "unit": "none", "source_clause": "%s", "confidence": 0.72}'
                % (primary["title"], first_sentence.rstrip("."), clause.replace('"', "'"))
            )

        return abstract


def _extract_query(prompt: str) -> str:
    """Recover the user question from a decorated prompt."""
    for marker in ("Specific question:", "Question:", "Request:", "Query:", "Q:"):
        if marker in prompt:
            tail = prompt.rsplit(marker, 1)[1]
            for line in tail.splitlines():
                if line.strip():
                    return line.strip()
    return prompt.strip()


def _truncate_words(text: str, max_new_tokens: int) -> str:
    """Apply an approximate token budget while keeping the line structure intact."""
    limit = max(20, int(max_new_tokens * 0.75))  # ~0.75 words per token in English
    kept, used = [], 0
    for line in text.splitlines():
        words = line.split()
        if used + len(words) <= limit:
            kept.append(line)
            used += len(words)
            continue
        remaining = limit - used
        if remaining > 0:
            kept.append(" ".join(words[:remaining]).rstrip(",;:") + " ...")
        break
    return "\n".join(kept) if kept else text


def choose_generation_backend(adapter_path=None, runtime=None) -> str:
    """Use the Assignment 1 adapter on CUDA or Apple MPS; otherwise offline.

    Loading 1.5B for hundreds of generations on CPU would make Run all last
    many hours. CUDA (Colab T4) and MPS (local M-series GPU) both run the
    real adapter when Assignment 1 has written it.
    """
    try:
        from runtime import detect_runtime
    except ImportError:  # notebook may have only src/ on the path
        from src.runtime import detect_runtime
    runtime = runtime or detect_runtime()
    path = Path(adapter_path) if adapter_path else None
    accelerator = bool(runtime.get("cuda") or runtime.get("mps"))
    if accelerator and path is not None and path.exists():
        return "adapter"
    return "offline"


class DomainLM:
    """Uniform generation interface over the Assignment 1 adapter or the offline backend."""

    def __init__(self, base_model_id="microsoft/biogpt-large", adapter_path=None,
                 backend="offline", seed=7, max_seq_length=None, label=None, aligned=False):
        self.base_model_id = base_model_id
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.backend = backend
        self.seed = seed
        self.aligned = aligned
        self.max_seq_length = max_seq_length
        self.label = label or (Path(self.adapter_path).name if self.adapter_path else backend)
        self._offline = _OfflineBackend(seed=seed, aligned=aligned)
        self._model = None
        self._tokenizer = None
        if backend == "adapter":
            self._load_adapter()

    def _load_adapter(self):
        import torch
        from transformers import AutoTokenizer
        try:
            from runtime import detect_runtime, load_causal_lm, training_budget
        except ImportError:
            from src.runtime import detect_runtime, load_causal_lm, training_budget

        runtime = detect_runtime()
        budget = training_budget(runtime)
        if self.max_seq_length is None:
            self.max_seq_length = int(budget.get("sequence_length") or 1024)
        four_bit = bool(runtime.get("cuda") and budget.get("four_bit"))
        model = load_causal_lm(
            self.base_model_id,
            four_bit=four_bit,
            gradient_checkpointing=False,
            runtime=runtime,
        )
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        tok_src = self.adapter_path if self.adapter_path and Path(self.adapter_path, "tokenizer_config.json").exists() else self.base_model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model, self._tokenizer = model, tokenizer
        torch.manual_seed(self.seed)

    def close(self) -> None:
        """Drop the loaded adapter. Do not hold SFT and DPO in memory together on a T4."""
        try:
            from runtime import release_model_memory
        except ImportError:
            from src.runtime import release_model_memory
        release_model_memory(self._model, self._tokenizer)
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.0,
                 n: int = 1, top_p: float = 0.9) -> list[str]:
        """Return ``n`` completions. temperature=0 means greedy/deterministic."""
        if self.backend == "offline":
            return [self._offline.generate(prompt, max_new_tokens, temperature, i) for i in range(n)]

        import torch
        tokenizer, model = self._tokenizer, self._model
        seq = self.max_seq_length or 1024
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = temperature > 0
        outputs = []
        for _ in range(n):
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(completion.strip())
        return outputs

    def one(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, n=1, **kwargs)[0]

    def describe(self) -> dict:
        return {"label": self.label, "backend": self.backend, "base_model_id": self.base_model_id,
                "adapter_path": self.adapter_path, "seed": self.seed, "aligned": self.aligned}


def wrap(text: str, width: int = 100) -> str:
    """Readable console printing for notebook cells."""
    return "\n".join(textwrap.fill(line, width) if line.strip() else line for line in text.splitlines())
