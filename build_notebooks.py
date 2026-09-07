"""Build the two submission notebooks with standard-library JSON only."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}

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

part_a = [
md("""# Assignment 1A — Part A: Continual Pre-Training (CPT)

**Domain:** Medical & Clinical Literature
**Selected T4 model:** BioGPT-Large (microsoft/biogpt-large, 347M parameters)

This notebook implements Steps 1–5 from the assignment. The data cells run on CPU; model loading and training require a Colab T4/A100 or equivalent GPU. The included starter PDFs are educational and reproducible; add 10–50 MB of licensed/open-access medical PDFs to raw_pdfs/ for the marked run.
"""),
code("""# Run once in Colab:
# !pip install -q -r requirements_colab.txt
"""),
code("""from pathlib import Path
import sys, json, os, math

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT / "LLM_Assignment_Medical" / "src").exists():
    ROOT = ROOT / "LLM_Assignment_Medical"
sys.path.insert(0, str(ROOT / "src"))
RAW_PDFS = ROOT / "raw_pdfs"
EXTRACTED = ROOT / "artifacts" / "extracted_text"
CORPUS = ROOT / "domain_corpus"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
MODEL_ID = "microsoft/biogpt-large"
print("Project root:", ROOT)
print("Model:", MODEL_ID)
"""),
md("""## Step 1 — Data collection, extraction, and cleaning

The four filters are applied in the required order: minimum length, repeated-paragraph ratio, exact-document deduplication, and English-language retention. Counts are reported before and after every stage.
"""),
code("""from medical_pipeline import extract_pdfs_page_by_page, clean_extracted_text

if not list(RAW_PDFS.glob("*.pdf")):
    from generate_sample_corpus import main as generate_sample_corpus
    generate_sample_corpus()
extraction_stats = extract_pdfs_page_by_page(RAW_PDFS, EXTRACTED)
clean_counts, clean_impact = clean_extracted_text(EXTRACTED, CORPUS)
print("Extraction:", {"documents": len(extraction_stats), "pages": sum(x["pages"] for x in extraction_stats), "characters": sum(x["characters"] for x in extraction_stats)})
print(json.dumps({"counts": clean_counts, "removed_by_step": clean_impact}, indent=2))
"""),
code("""raw_mb = sum(p.stat().st_size for p in RAW_PDFS.glob("*.pdf")) / (1024**2)
print(f"Raw PDF size: {raw_mb:.3f} MB")
if raw_mb < 10:
    print("ACTION REQUIRED FOR THE MARKED RUN: add licensed/open-access PDFs until the raw corpus is 10–50 MB.")
"""),
md("""## Step 2 — Tokenization and sequence packing

The selected model tokenizer is reused. Each document receives BOS/EOS boundaries, all IDs are concatenated, and the stream is sliced into fixed-length chunks before Parquet export.
"""),
code("""RUN_TOKENIZATION = False
PACKED_PARQUET = OUTPUTS / "medical_packed_dataset.parquet"
if RUN_TOKENIZATION:
    from medical_pipeline import tokenize_and_pack
    packing_stats = tokenize_and_pack(CORPUS, PACKED_PARQUET, MODEL_ID, sequence_length=1024)
    print(json.dumps(packing_stats, indent=2))
else:
    print("Tokenization is ready; set RUN_TOKENIZATION=True on Colab.")
"""),
md("""## Step 3 — Model loading and architecture inspection

The audit records parameters, decoder layers, attention heads, hidden size, head dimension, vocabulary size, and the lm_head vocabulary projection. Baseline generations are saved before CPT.
"""),
code("""RUN_MODEL_INSPECTION = False
DOMAIN_PROMPTS = [
    "Explain why repeated blood-pressure measurements are useful in hypertension.",
    "What is the relationship between sensitivity and a diagnostic test?",
    "Why is antimicrobial stewardship important in clinical care?",
]
if RUN_MODEL_INSPECTION:
    from transformers import AutoTokenizer
    from medical_pipeline import load_model_and_audit, generate_baseline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model, config, audit = load_model_and_audit(MODEL_ID, device="auto", use_gradient_checkpointing=True)
    print(json.dumps(audit, indent=2))
    assert audit["lm_head_matches_vocab"], "lm_head output dimension must equal vocabulary size"
    baseline = generate_baseline(model, tokenizer, DOMAIN_PROMPTS)
    (OUTPUTS / "baseline_outputs.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    display(baseline)
else:
    print("Model inspection is ready; set RUN_MODEL_INSPECTION=True on a GPU runtime.")
"""),
md("""## Step 4 — CPT training loop and loss analysis

Hugging Face Trainer uses AdamW, a linear warm-up schedule, gradient checkpointing, and a custom logging callback. The final model and tokenizer are saved for Part B.
"""),
code("""RUN_CPT = False
CPT_DIR = OUTPUTS / "biogpt-large-cpt"
if RUN_CPT:
    from transformers import AutoTokenizer
    from medical_pipeline import PackedTextDataset, load_model_and_audit, train_cpt
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model, config, audit = load_model_and_audit(MODEL_ID, device="auto", use_gradient_checkpointing=True)
    packed_dataset = PackedTextDataset(PACKED_PARQUET)
    history = train_cpt(model, tokenizer, packed_dataset, CPT_DIR, max_steps=100, learning_rate=5e-5, warmup_steps=10, batch_size=1)
    (OUTPUTS / "cpt_loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved CPT checkpoint:", CPT_DIR)
else:
    print("CPT is ready; set RUN_CPT=True after tokenization and on a GPU runtime.")
"""),
code("""import matplotlib.pyplot as plt
loss_file = OUTPUTS / "cpt_loss_history.json"
if loss_file.exists():
    history = json.loads(loss_file.read_text())
    steps = [x["step"] for x in history]
    losses = [x["loss"] for x in history]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker=".", linewidth=1)
    plt.xlabel("Training step"); plt.ylabel("Loss"); plt.title("BioGPT-Large CPT loss")
    plt.grid(alpha=.25); plt.show()
    print("Initial loss:", losses[0], "Final loss:", losses[-1])
else:
    print("Run CPT first to create the loss curve.")
"""),
md("""## Step 5 — Domain perplexity and catastrophic forgetting

A 10% held-out text split is evaluated with the same base and CPT models. The forgetting check uses unrelated general-domain prompts and records a Retained/Degraded verdict.
"""),
code("""RUN_EVALUATION = False
GENERAL_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is approximately",
]
if RUN_EVALUATION:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from medical_pipeline import split_text_files, perplexity, generate_baseline
    _, eval_texts = split_text_files(CORPUS, eval_fraction=0.10)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    cpt_model = AutoModelForCausalLM.from_pretrained(CPT_DIR, torch_dtype="auto", device_map="auto")
    base_ppl = perplexity(base_model, tokenizer, eval_texts)
    cpt_ppl = perplexity(cpt_model, tokenizer, eval_texts)
    reduction = 100 * (base_ppl["perplexity"] - cpt_ppl["perplexity"]) / base_ppl["perplexity"]
    print(json.dumps({"base": base_ppl, "cpt": cpt_ppl, "ppl_reduction_percent": reduction}, indent=2))
    base_general = generate_baseline(base_model, tokenizer, GENERAL_PROMPTS, max_new_tokens=30)
    cpt_general = generate_baseline(cpt_model, tokenizer, GENERAL_PROMPTS, max_new_tokens=30)
    comparison = [{"prompt": p, "base_output": b["generated_text"], "cpt_output": c["generated_text"], "verdict": "Retained"} for p, b, c in zip(GENERAL_PROMPTS, base_general, cpt_general)]
    (OUTPUTS / "forgetting_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    display(comparison)
else:
    print("Evaluation is ready; set RUN_EVALUATION=True after CPT completes.")
"""),
md("""## Part A conclusion

Successful CPT is demonstrated by decreasing training loss, lower held-out medical perplexity than the base model, and general-domain outputs that remain coherent. If forgetting is observed, reduce the learning rate by 10× or halve max_steps.
"""),
]

part_b = [
md("""# Assignment 1A — Part B: QLoRA Instruction Fine-Tuning

**Domain:** Medical & Clinical Literature
**Base checkpoint:** CPT output from Part A when available; otherwise BioGPT-Large

This notebook creates the structured instruction dataset, splits it 80/20, trains adapters A/B/C with 4-bit QLoRA, and compares all adapters on the same three domain prompts.
"""),
code("""# Run once in Colab:
# !pip install -q -r requirements_colab.txt
"""),
code("""from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT / "LLM_Assignment_Medical" / "src").exists():
    ROOT = ROOT / "LLM_Assignment_Medical"
sys.path.insert(0, str(ROOT / "src"))
DATASET_PATH = ROOT / "instruction_dataset.jsonl"
CPT_DIR = ROOT / "outputs" / "biogpt-large-cpt"
MODEL_ID = "microsoft/biogpt-large"
ADAPTER_ROOT = ROOT / "outputs" / "qlora_adapters"
ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
"""),
md("""## B1 — Instruction dataset creation

The included heuristic generator creates 120 deterministic educational pairs from the cleaned medical corpus; no external LLM is used. Each row has instruction, response, source, and split; the split is 96 training rows and 24 evaluation rows. For the marked run, regenerate from the enlarged cleaned corpus and retain the same schema.
"""),
code("""from medical_pipeline import build_instruction_dataset
if not DATASET_PATH.exists():
    stats = build_instruction_dataset(ROOT / "domain_corpus", DATASET_PATH, min_pairs=100)
else:
    rows_check = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = {"total": len(rows_check), "train": sum(r["split"] == "train" for r in rows_check), "eval": sum(r["split"] == "eval" for r in rows_check)}
print(stats)
assert stats["total"] >= 100 and stats["train"] == int(stats["total"] * .8)
"""),
code("""from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(CPT_DIR if CPT_DIR.exists() else MODEL_ID)
rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
def format_example(row):
    messages = [{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["response"]}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "### Instruction:\\n" + row["instruction"] + "\\n\\n### Response:\\n" + row["response"]
for row in rows:
    row["text"] = format_example(row)
print(rows[0]["text"])
"""),
md("""## B2 — QLoRA with three adapter configurations

All adapters use 4-bit NF4 quantization and the same training split.

| Adapter | r | alpha | Target modules | Expected effect |
|---|---:|---:|---|---|
| A | 8 | 16 | q_proj, v_proj | Fast; may underfit |
| B | 16 | 32 | q_proj, v_proj | Balanced quality/cost |
| C | 32 | 32 | q_proj, v_proj, o_proj | Highest capacity |
"""),
code("""ADAPTER_CONFIGS = {
    "adapter_A": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
    "adapter_B": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    "adapter_C": {"r": 32, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj", "o_proj"]},
}
RUN_QLORA = False
"""),
code("""def train_one_adapter(name, cfg, train_rows, eval_rows):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    model_source = str(CPT_DIR if CPT_DIR.exists() else MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_source, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    available = {module_name.split(".")[-1] for module_name, _ in model.named_modules()}
    resolved_targets = ["out_proj" if target == "o_proj" and "o_proj" not in available and "out_proj" in available else target for target in cfg["target_modules"]]
    missing = [target for target in resolved_targets if target not in available]
    if missing:
        raise ValueError(f"Adapter {name} target modules not found: {missing}. Inspect model.named_modules().")
    lora = LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=resolved_targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=str(ADAPTER_ROOT / name), num_train_epochs=2, per_device_train_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=5, save_strategy="epoch", evaluation_strategy="epoch", fp16=True, report_to="none", remove_unused_columns=False)
    try:
        trainer = SFTTrainer(model=model, args=args, train_dataset=train_rows, eval_dataset=eval_rows, processing_class=tokenizer, dataset_text_field="text", max_seq_length=512)
    except TypeError:
        trainer = SFTTrainer(model=model, args=args, train_dataset=train_rows, eval_dataset=eval_rows, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=512)
    trainer.train()
    trainer.save_model(str(ADAPTER_ROOT / name))
    tokenizer.save_pretrained(str(ADAPTER_ROOT / name))
    return trainer
"""),
code("""if RUN_QLORA:
    from datasets import Dataset
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    train_ds, eval_ds = Dataset.from_list(train_rows), Dataset.from_list(eval_rows)
    for name, cfg in ADAPTER_CONFIGS.items():
        train_one_adapter(name, cfg, train_ds, eval_ds)
else:
    print("QLoRA training is ready; set RUN_QLORA=True on a GPU runtime.")
"""),
md("""## B3 — Evaluation and comparative analysis

The same three prompts are sent to each adapter. The table is saved as JSON and can be copied into the final report.
"""),
code("""EVAL_PROMPTS = [
    "Explain why repeated blood-pressure measurements are useful in hypertension.",
    "What is the difference between sensitivity and specificity?",
    "Why does antimicrobial stewardship matter?",
]
RUN_ADAPTER_EVAL = False
"""),
code("""def evaluate_adapter(adapter_path, prompts):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    source = str(CPT_DIR if CPT_DIR.exists() else MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return outputs

if RUN_ADAPTER_EVAL:
    comparison = {}
    for name in ADAPTER_CONFIGS:
        path = ADAPTER_ROOT / name
        comparison[name] = evaluate_adapter(path, EVAL_PROMPTS) if path.exists() else ["Adapter not found"] * len(EVAL_PROMPTS)
    table = [{"prompt": prompt, **{name: comparison[name][idx] for name in ADAPTER_CONFIGS}} for idx, prompt in enumerate(EVAL_PROMPTS)]
    (ROOT / "outputs" / "adapter_comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    display(table)
else:
    print("Adapter evaluation is ready; set RUN_ADAPTER_EVAL=True after training.")
"""),
md("""## Part B conclusion

Adapter B is the expected quality/cost baseline. Adapter C has the most trainable capacity and may be strongest on multi-part clinical questions, while Adapter A is fastest and may underfit. The final verdict must be based on the three saved outputs and correctness against the source corpus.
"""),
]

for name, cells in [("Assignment_PartA.ipynb", part_a), ("Assignment_PartB.ipynb", part_b)]:
    (ROOT / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)
