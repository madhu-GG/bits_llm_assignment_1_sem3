"""Reusable solution utilities for Assignment 1A (Medical & Clinical Literature).

The module keeps preprocessing dependency-light and loads ML dependencies only
inside the functions that need them. This makes the data pipeline testable on a
CPU-only machine and the training stages runnable on Colab/A100 after installing
requirements_colab.txt.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable


ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "for", "that", "with", "on", "as", "by",
    "this", "an", "or", "from", "are", "be", "can", "which", "when", "at", "it", "may",
}


def _pdf_reader(path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf before extracting PDFs: pip install pypdf") from exc
    return PdfReader(str(path))


def extract_pdfs_page_by_page(pdf_dir, extracted_dir):
    pdf_dir, extracted_dir = Path(pdf_dir), Path(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        reader = _pdf_reader(pdf_path)
        pages = []
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            pages.append(f"[PAGE {page_number}]\n{text}")
        text_path = extracted_dir / f"{pdf_path.stem}.txt"
        text_path.write_text("\n\n".join(pages), encoding="utf-8")
        results.append({"file": pdf_path.name, "pages": len(pages), "characters": sum(map(len, pages))})
    return results


def _paragraphs(text: str):
    chunks = re.split(r"\n\s*\n+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in chunks if p.strip()]


def _looks_english(text: str):
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return False
    stopword_hits = sum(word in ENGLISH_STOPWORDS for word in words)
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(len(text), 1)
    return ascii_ratio >= 0.85 and stopword_hits >= max(2, min(10, len(words) // 30))


def clean_extracted_text(extracted_dir, cleaned_dir, min_chars=50, duplicate_fraction=0.30):
    """Apply length, repetition, exact-document, and English filters in order."""
    extracted_dir, cleaned_dir = Path(extracted_dir), Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(extracted_dir.glob("*.txt"))
    counts = {"before": len(files), "after_length": 0, "after_repetition": 0, "after_deduplication": 0, "after_language": 0}
    candidates = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) >= min_chars:
            candidates.append((path, text))
    counts["after_length"] = len(candidates)

    candidates2 = []
    for path, text in candidates:
        paras = _paragraphs(text)
        duplicate_ratio = (len(paras) - len(set(paras))) / max(len(paras), 1)
        if duplicate_ratio <= duplicate_fraction:
            candidates2.append((path, text))
    counts["after_repetition"] = len(candidates2)

    seen = set()
    candidates3 = []
    for path, text in candidates2:
        key = re.sub(r"\s+", " ", text).strip()
        if key not in seen:
            seen.add(key)
            candidates3.append((path, text))
    counts["after_deduplication"] = len(candidates3)

    final = []
    for path, text in candidates3:
        if _looks_english(text):
            final.append((path, text))
    counts["after_language"] = len(final)

    for old in cleaned_dir.glob("*.txt"):
        old.unlink()
    for path, text in final:
        (cleaned_dir / path.name).write_text(text, encoding="utf-8")
    impact = {
        "length_filter": counts["before"] - counts["after_length"],
        "repetition_filter": counts["after_length"] - counts["after_repetition"],
        "deduplication": counts["after_repetition"] - counts["after_deduplication"],
        "language_filter": counts["after_deduplication"] - counts["after_language"],
    }
    if max(impact.values()) == 0:
        counts["greatest_impact_step"] = "none (no documents removed by the sample filters)"
    else:
        counts["greatest_impact_step"] = max(impact, key=impact.get)
    return counts, impact


def build_instruction_dataset(cleaned_dir, output_jsonl, min_pairs=100, train_ratio=0.8):
    from medical_data import instruction_pairs
    cleaned_dir = Path(cleaned_dir)
    source_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(cleaned_dir.glob("*.txt")))
    if not source_text.strip():
        raise ValueError("The cleaned corpus is empty; run PDF extraction and cleaning first.")
    # Keep only pairs whose topic is present in the cleaned source corpus. The
    # starter generator is deterministic, but this gate makes the JSONL source-
    # grounded and prevents unrelated topics from entering a new corpus run.
    pairs = [row for row in instruction_pairs() if row["source"] in source_text]
    if len(pairs) < min_pairs:
        raise ValueError(f"Expected at least {min_pairs} pairs, got {len(pairs)}")
    split_index = int(len(pairs) * train_ratio)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(pairs):
            row = dict(item)
            row["split"] = "train" if idx < split_index else "eval"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"total": len(pairs), "train": split_index, "eval": len(pairs) - split_index, "path": str(output_jsonl)}


def tokenize_and_pack(cleaned_dir, parquet_path, model_id, sequence_length=1024):
    """Tokenize with the selected model tokenizer and save packed rows as Parquet."""
    try:
        import pandas as pd
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install pandas, pyarrow, and transformers before tokenization.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = eos_id if eos_id is not None else tokenizer.unk_token_id
        warnings.warn("Tokenizer has no BOS token; using EOS/UNK as a boundary fallback.")
    if eos_id is None:
        eos_id = bos_id
        warnings.warn("Tokenizer has no EOS token; using BOS as a boundary fallback.")
    all_ids, lengths = [], []
    for path in sorted(Path(cleaned_dir).glob("*.txt")):
        ids = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
        ids = [bos_id] + ids + [eos_id]
        all_ids.extend(ids)
        lengths.append(len(ids))
    count = len(all_ids) // sequence_length
    rows = [{"input_ids": all_ids[i*sequence_length:(i+1)*sequence_length], "attention_mask": [1]*sequence_length} for i in range(count)]
    if not rows:
        raise ValueError("No complete packed sequence was produced; add more corpus text or reduce sequence_length.")
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return {"tokenizer": model_id, "total_tokens": len(all_ids), "average_document_tokens": sum(lengths)/max(len(lengths), 1), "packed_sequences": count, "sequence_length": sequence_length, "path": str(parquet_path)}


def load_model_and_audit(model_id, device="auto", use_gradient_checkpointing=True):
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install torch and transformers in Colab before model loading.") from exc
    config = AutoConfig.from_pretrained(model_id)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    else:
        dtype = torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    hidden = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if hidden and heads else None)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lm_head = getattr(model, "lm_head", None)
    audit = {"model_id": model_id, "device": device, "dtype": str(dtype), "total_parameters": total, "trainable_parameters": trainable, "decoder_layers": layers, "attention_heads": heads, "hidden_size": hidden, "head_dimension": head_dim, "vocab_size": config.vocab_size, "lm_head_output_dimension": getattr(lm_head, "out_features", None), "lm_head_matches_vocab": getattr(lm_head, "out_features", None) == config.vocab_size}
    return model, config, audit


def generate_baseline(model, tokenizer, prompts, max_new_tokens=80):
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        results.append({"prompt": prompt, "generated_text": tokenizer.decode(output[0], skip_special_tokens=True)})
    return results


class PackedTextDataset:
    """PyTorch Dataset wrapper for packed Parquet rows."""
    def __new__(cls, parquet_path):
        try:
            import pandas as pd
            import torch
        except ImportError as exc:
            raise ImportError("Install torch and pandas before creating PackedTextDataset.") from exc
        frame = pd.read_parquet(parquet_path)
        class _Dataset(torch.utils.data.Dataset):
            def __len__(self): return len(frame)
            def __getitem__(self, idx):
                return {"input_ids": torch.tensor(frame.iloc[idx]["input_ids"], dtype=torch.long), "attention_mask": torch.tensor(frame.iloc[idx]["attention_mask"], dtype=torch.long)}
        return _Dataset()


def train_cpt(model, tokenizer, dataset, output_dir, max_steps=100, learning_rate=5e-5, warmup_steps=10, batch_size=1):
    try:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments
    except ImportError as exc:
        raise ImportError("Install transformers before CPT training.") from exc
    class LossHistoryCallback(TrainerCallback):
        def __init__(self): self.history = []
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.history.append({"step": state.global_step, "loss": float(logs["loss"])})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    callback = LossHistoryCallback()
    args = TrainingArguments(output_dir=str(output_dir), max_steps=max_steps, learning_rate=learning_rate, warmup_steps=warmup_steps, lr_scheduler_type="linear", optim="adamw_torch", per_device_train_batch_size=batch_size, gradient_accumulation_steps=8, logging_steps=1, save_steps=max_steps, save_total_limit=1, fp16=True, bf16=False, report_to="none", remove_unused_columns=False)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator, callbacks=[callback])
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return callback.history


def perplexity(model, tokenizer, texts: Iterable[str], max_length=1024):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch before perplexity evaluation.") from exc
    model.eval()
    losses, tokens = [], 0
    for text in texts:
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, labels=batch["input_ids"])
        n = int(batch["attention_mask"].sum().item())
        losses.append(float(out.loss.item()) * max(n - 1, 1))
        tokens += max(n - 1, 1)
    mean_nll = sum(losses) / max(tokens, 1)
    return {"mean_nll": mean_nll, "perplexity": math.exp(min(mean_nll, 20)), "tokens": tokens}


def split_text_files(cleaned_dir, eval_fraction=0.10):
    files = sorted(Path(cleaned_dir).glob("*.txt"))
    split_at = max(1, int(len(files) * (1 - eval_fraction)))
    return [p.read_text(encoding="utf-8") for p in files[:split_at]], [p.read_text(encoding="utf-8") for p in files[split_at:]]
