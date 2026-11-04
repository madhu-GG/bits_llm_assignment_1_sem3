"""Part C2/C3 — DPO fine-tuning on top of the Assignment 1 QLoRA adapter, and
the SFT-vs-DPO alignment evaluation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from a2_paths import DPO_ADAPTER_DIR, base_model_source, sft_adapter_path
from answer_scoring import score_answer
from assignment2_queries import HELD_OUT_QUERIES
from preference_data import load_jsonl
from prompt_strategies import zero_shot_prompt
from reasoning_strategies import keywords_for

try:
    from runtime import detect_runtime, load_causal_lm, training_budget
except ImportError:
    from src.runtime import detect_runtime, load_causal_lm, training_budget

DPO_CONFIG = {
    "beta": 0.1,
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 768,
    "max_prompt_length": 256,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "seed": 7,
}


def to_trl_dataset(pairs):
    """TRL's DPOTrainer only consumes prompt/chosen/rejected; metadata is dropped."""
    from datasets import Dataset
    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    return Dataset.from_list(rows)


def _resolve_targets(model, requested):
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    resolved = []
    for target in requested:
        if target not in available and target == "o_proj" and "out_proj" in available:
            target = "out_proj"
        if target in available:
            resolved.append(target)
    if not resolved:
        raise ValueError(f"None of {requested} exist in the model; inspect model.named_modules().")
    return resolved


def train_dpo(preference_path, sft_adapter="adapter_B", output_dir=DPO_ADAPTER_DIR, config=None):
    """Fine-tune the Assignment 1 adapter with DPO.

    CUDA uses 4-bit QLoRA and 3 epochs. CPU/MPS cannot hold a second full copy
    of 1.5B weights, so DPO trains the existing LoRA adapter for a handful of
    steps at a short sequence length. ``ref_model=None`` keeps the frozen
    backbone as the implicit reference (no second 6 GB copy).
    """
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    runtime = detect_runtime()
    budget = training_budget(runtime)
    if not budget.get("can_load_model", True):
        raise RuntimeError(budget.get("reason") or "Not enough RAM to load BioGPT-Large for DPO.")

    cfg = {**DPO_CONFIG, **(config or {})}
    cfg["num_train_epochs"] = int(budget.get("dpo_epochs") or cfg["num_train_epochs"])
    cfg["per_device_train_batch_size"] = int(budget.get("per_device_train_batch_size") or 1)
    cfg["gradient_accumulation_steps"] = int(budget.get("gradient_accumulation_steps") or 1)
    if budget.get("dpo_max_length"):
        cfg["max_length"] = int(budget["dpo_max_length"])
    if budget.get("dpo_max_prompt_length"):
        cfg["max_prompt_length"] = int(budget["dpo_max_prompt_length"])
    if budget.get("gpu_tier") == "t4":
        cfg["max_length"] = min(int(cfg["max_length"]), 512)
        cfg["max_prompt_length"] = min(int(cfg["max_prompt_length"]), 192)
    dpo_steps = int(budget.get("dpo_max_steps") or 0)
    pairs = load_jsonl(preference_path)
    dataset = to_trl_dataset(pairs)

    source = base_model_source()
    adapter_dir = sft_adapter_path(sft_adapter)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Assignment 1 adapter not found at {adapter_dir}. Run Assignment 1 Part B first.")

    four_bit = bool(runtime.get("cuda") and budget.get("four_bit"))
    base = load_causal_lm(source, four_bit=four_bit, gradient_checkpointing=True, runtime=runtime)
    if four_bit:
        base = prepare_model_for_kbit_training(base)
    # The SFT adapter is the policy starting point; TRL uses the frozen base+adapter
    # as the implicit reference model when ref_model is None.
    policy = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # On CUDA, TRL can wrap a fresh LoRA. On CPU the existing SFT LoRA is
    # already the only set of trainable params that will fit.
    peft_config = None
    if runtime.get("cuda"):
        peft_config = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            bias="none", task_type="CAUSAL_LM",
            target_modules=_resolve_targets(base, ["q_proj", "v_proj"]),
        )

    dtype = runtime.get("dtype")
    args_kwargs = dict(
        output_dir=str(output_dir),
        beta=cfg["beta"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        logging_steps=1,
        save_strategy="no" if dpo_steps else "epoch",
        fp16=bool(runtime.get("cuda") and dtype == "float16"),
        bf16=bool(runtime.get("cuda") and dtype == "bfloat16"),
        report_to="none",
        seed=cfg["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    if dpo_steps > 0:
        args_kwargs["max_steps"] = dpo_steps
        cfg["max_steps"] = dpo_steps
    if not runtime.get("cuda") and not runtime.get("mps"):
        args_kwargs["use_cpu"] = True
    try:
        args = DPOConfig(**args_kwargs)
    except TypeError:
        args_kwargs.pop("use_cpu", None)
        args = DPOConfig(**args_kwargs)

    trainer_kwargs = dict(model=policy, ref_model=None, args=args, train_dataset=dataset)
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:  # older TRL releases
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    history = [row for row in trainer.state.log_history if "loss" in row or "train_loss" in row]
    (Path(output_dir) / "dpo_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return {"config": cfg, "pairs": len(pairs), "output_dir": str(output_dir), "history": history}


def epoch_metrics(log_history) -> list[dict]:
    """Collapse TRL's step logs into a per-epoch DPO loss and reward-margin table."""
    buckets: dict[int, dict[str, list[float]]] = {}
    for row in log_history:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        key = int(epoch) if float(epoch).is_integer() else int(float(epoch)) + 1
        bucket = buckets.setdefault(key, {})
        for metric in ("loss", "rewards/chosen", "rewards/rejected", "rewards/margins",
                       "rewards/accuracies"):
            if metric in row:
                bucket.setdefault(metric, []).append(float(row[metric]))
    table = []
    for epoch in sorted(buckets):
        values = buckets[epoch]
        table.append({
            "epoch": epoch,
            "dpo_loss": round(statistics.mean(values["loss"]), 4) if "loss" in values else None,
            "reward_chosen": round(statistics.mean(values["rewards/chosen"]), 4) if "rewards/chosen" in values else None,
            "reward_rejected": round(statistics.mean(values["rewards/rejected"]), 4) if "rewards/rejected" in values else None,
            "reward_margin": round(statistics.mean(values["rewards/margins"]), 4) if "rewards/margins" in values else None,
            "reward_accuracy": round(statistics.mean(values["rewards/accuracies"]), 4) if "rewards/accuracies" in values else None,
        })
    return table


# --------------------------------------------------------------------------
# C3 — alignment evaluation
# --------------------------------------------------------------------------

def _improvement_reason(sft_score, dpo_score) -> tuple[str, str]:
    delta = dpo_score["total"] - sft_score["total"]
    if delta <= 1.0:
        gaps = []
        if dpo_score["words"] > sft_score["words"] * 1.8:
            gaps.append("longer without adding corpus content")
        if dpo_score["grounding"] <= sft_score["grounding"]:
            gaps.append("no gain in corpus grounding")
        if dpo_score["structure"] <= sft_score["structure"]:
            gaps.append("no gain in structure")
        return "N", "; ".join(gaps) or f"total score moved only {delta:+.1f}"
    reasons = []
    if dpo_score["structure"] > sft_score["structure"]:
        reasons.append(f"structure {sft_score['structure']:.0f}->{dpo_score['structure']:.0f}")
    if dpo_score["grounding"] > sft_score["grounding"]:
        reasons.append(f"grounding {sft_score['grounding']:.1f}->{dpo_score['grounding']:.1f}")
    if dpo_score["completeness"] > sft_score["completeness"]:
        reasons.append(f"completeness {sft_score['completeness']:.1f}->{dpo_score['completeness']:.1f}")
    if dpo_score["specificity"] > sft_score["specificity"]:
        reasons.append(f"specificity {sft_score['specificity']:.1f}->{dpo_score['specificity']:.1f}")
    return "Y", f"total {delta:+.1f} ({', '.join(reasons) or 'broad improvement'})"


def generate_zero_shot(lm, queries=None, max_new_tokens=200) -> list[str]:
    queries = list(queries or HELD_OUT_QUERIES)
    return [lm.one(zero_shot_prompt(query), max_new_tokens=max_new_tokens, temperature=0.0)
            for query in queries]


def compare_outputs(sft_outputs, dpo_outputs, queries=None) -> list[dict]:
    """Score two already-generated output lists. Does not load a second model."""
    queries = list(queries or HELD_OUT_QUERIES)
    if len(sft_outputs) != len(dpo_outputs) or len(sft_outputs) != len(queries):
        raise ValueError("SFT outputs, DPO outputs, and queries must be the same length.")
    rows = []
    for index, (query, sft_out, dpo_out) in enumerate(zip(queries, sft_outputs, dpo_outputs), 1):
        keywords = keywords_for(query)
        sft_score = score_answer(sft_out, query, keywords)
        dpo_score = score_answer(dpo_out, query, keywords)
        improved, reason = _improvement_reason(sft_score, dpo_score)
        rows.append({
            "n": index,
            "query": query,
            "sft_output": sft_out,
            "dpo_output": dpo_out,
            "sft_score": sft_score,
            "dpo_score": dpo_score,
            "improved": improved,
            "reason": reason,
        })
    return rows


def compare_adapters(sft_lm, dpo_lm, queries=None, max_new_tokens=200) -> list[dict]:
    """Convenience wrapper. On a T4, generate SFT, close that LM, then generate DPO."""
    queries = list(queries or HELD_OUT_QUERIES)
    sft_outputs = generate_zero_shot(sft_lm, queries, max_new_tokens)
    dpo_outputs = generate_zero_shot(dpo_lm, queries, max_new_tokens)
    return compare_outputs(sft_outputs, dpo_outputs, queries)


def comparison_summary(rows) -> dict:
    wins = sum(1 for r in rows if r["improved"] == "Y")
    return {
        "queries": len(rows),
        "improved": wins,
        "not_improved": len(rows) - wins,
        "mean_sft_score": round(statistics.mean(r["sft_score"]["total"] for r in rows), 2),
        "mean_dpo_score": round(statistics.mean(r["dpo_score"]["total"] for r in rows), 2),
        "mean_delta": round(statistics.mean(r["dpo_score"]["total"] - r["sft_score"]["total"] for r in rows), 2),
    }


def markdown_table(rows, truncate: int = 220) -> str:
    def cell(text):
        text = " ".join(str(text).split())
        return (text[:truncate] + "…") if len(text) > truncate else text

    header = "| Domain Query | SFT Adapter Output | DPO Adapter Output | Improvement? (Y/N + reason) |\n"
    header += "|---|---|---|---|\n"
    body = "".join(
        f"| {cell(r['query'])} | {cell(r['sft_output'])} | {cell(r['dpo_output'])} | "
        f"{r['improved']} — {cell(r['reason'])} |\n"
        for r in rows
    )
    return header + body
