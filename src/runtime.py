"""Colab T4 memory toolkit (about 15 GB GPU and 15 GB CPU RAM).

The Hugging Face ``microsoft/biogpt-large`` checkpoint is ~1.5B weights.
On this machine that means:

* fp32 weights on CPU ≈ 6 GB, bf16 weights ≈ 3 GB
* full AdamW (two 32-bit states) ≈ 12 GB — does not fit next to activations
* 8-bit Adam ≈ 1.5 GB optimiser; 4-bit NF4 weights ≈ 0.8 GB

Rules used by both notebooks:

1. Set env vars *before* importing torch.
2. Load the model straight onto ``cuda:0``. Never materialise it on CPU first.
3. Keep one model in memory. Delete it before the next stage.
4. Do **not** move a trained model back to CPU on teardown — that duplicates
   ~3 GB into the 15 GB system RAM and can OOM the process.
5. Cap Hugging Face offload so the CPU has room for Python, pypdf, and Parquet.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Leave headroom on the student Colab box (T4 ~15 GB, host RAM ~15 GB).
GPU_BUDGET_GIB = "12GiB"
CPU_OFFLOAD_GIB = "4GiB"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def project_root() -> Path:
    """Find the assignment folder even if Colab unpacked it one level down."""
    here = Path.cwd()
    markers = ("Assignment1A.ipynb", "Assignment2A.ipynb", "dataset_sources.txt", "raw_pdfs.tgz")
    if any((here / name).exists() for name in markers):
        return here
    nested = here / "LLM_Assignment_Medical_Solution"
    if nested.exists():
        return nested
    return here


def apply_memory_env() -> None:
    """Must run before importing torch. Cuts CUDA fragmentation and extra threads."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _cpu_ram() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / 1024**3, 2), round(vm.available / 1024**3, 2)
    except ImportError:
        return 0.0, 0.0


def detect_runtime() -> dict:
    """Pick the best device. A Colab T4 should report cuda + bfloat16 or float16."""
    cpu_total, cpu_avail = _cpu_ram()
    base = {
        "device": "cpu",
        "dtype": "float32",
        "cuda": False,
        "mps": False,
        "gpu_name": None,
        "gpu_total_gb": 0.0,
        "gpu_free_gb": 0.0,
        "cpu_total_gb": cpu_total,
        "cpu_available_gb": cpu_avail,
        "colab": in_colab(),
    }
    try:
        import torch
    except ImportError:
        return base

    if torch.cuda.is_available():
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info()
        base.update({
            "device": "cuda",
            "dtype": "bfloat16" if bf16 else "float16",
            "cuda": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(total_b / 1024**3, 2),
            "gpu_free_gb": round(free_b / 1024**3, 2),
            "gpu_cc": f"{props.major}.{props.minor}",
        })
        return base
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        base.update({"device": "mps", "dtype": "float32", "mps": True, "gpu_name": "mps"})
        return base
    return base


def has_gpu() -> bool:
    return detect_runtime()["cuda"]


def memory_snapshot(tag: str = "") -> dict:
    """Small live report so leaks between notebook cells are visible."""
    cpu_total, cpu_avail = _cpu_ram()
    snap = {
        "tag": tag,
        "cpu_total_gb": cpu_total,
        "cpu_available_gb": cpu_avail,
    }
    try:
        import psutil
        snap["cpu_process_gb"] = round(psutil.Process().memory_info().rss / 1024**3, 2)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            snap.update({
                "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "gpu_free_gb": round(free_b / 1024**3, 2),
                "gpu_total_gb": round(total_b / 1024**3, 2),
            })
    except ImportError:
        pass
    print(snap)
    return snap


def release_model_memory(*objects) -> None:
    """Drop model/trainer refs and return GPU blocks to CUDA.

    Do not ``.to("cpu")`` first. On a 15 GB host that copies the whole
    checkpoint into system RAM and is a common Colab OOM.
    """
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def disable_optional_scipy_backend() -> None:
    """Stop Transformers from importing SciPy for unused loss functions."""
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return
    import_utils._scipy_available = False
    import_utils.is_scipy_available = lambda: False
    transformers_utils.is_scipy_available = lambda: False


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_tier(runtime: dict | None = None) -> str:
    """Classify the accelerator. Do not use free-VRAM at detect time — that
    shrinks after the baseline load and would mis-label an A100 as a T4.
    """
    runtime = runtime or detect_runtime()
    if not runtime.get("cuda"):
        return "cpu"
    name = (runtime.get("gpu_name") or "").lower()
    total = float(runtime.get("gpu_total_gb") or 0)
    if any(tag in name for tag in ("a100", "h100", "h200", "l40", "b200")) or total >= 32:
        return "large"
    if "t4" in name or total <= 16.5:
        return "t4"
    return "mid"


def training_budget(runtime: dict | None = None) -> dict:
    """Batch size, accumulation, and CPT length for the detected machine.

    T4 (~15 GB) measured ~57 s/optimiser step on this 1.5B checkpoint with
    gradient checkpointing and 8-bit Adam. 300 steps would be ~4.7 hours.
    Faster lab GPUs ran 100 steps in ~2 minutes, so they can do a packed epoch.
    """
    runtime = runtime or detect_runtime()
    if not runtime.get("cuda"):
        return {
            "gpu_tier": "cpu",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "cpt_max_steps": 0,
            "sft_max_steps": 0,
            "dpo_max_steps": 0,
            "four_bit": False,
            "eight_bit_adam": False,
            "sequence_length": 1024,
            "max_new_tokens": 40,
            "warmup_ratio": 0.1,
            "max_memory": None,
            "reason": "no GPU — data prep still runs; training is skipped",
        }

    tier = gpu_tier(runtime)
    gpu_total = float(runtime.get("gpu_total_gb") or 0)
    if tier == "t4":
        # ~60 steps ≈ 1 hour on the measured Colab T4. 8-bit Adam is required
        # so optimiser states fit next to bf16 weights.
        gpu_cap, cpu_cap = GPU_BUDGET_GIB, CPU_OFFLOAD_GIB
        cpt_cap, sft_steps = 60, 40
        eight_bit = True
        reason = (
            "T4 (~15 GB): 60 CPT steps (~1 h at ~57 s/step), 8-bit Adam, "
            "4-bit QLoRA, load straight to GPU"
        )
    elif tier == "mid":
        gpu_cap = f"{max(12, int(gpu_total) - 4)}GiB"
        cpu_cap = "8GiB"
        cpt_cap, sft_steps = 150, 60
        eight_bit = True
        reason = (
            "Mid-size GPU: 150 CPT steps, 8-bit Adam, 4-bit QLoRA, "
            "load straight to GPU"
        )
    else:
        gpu_cap = f"{max(20, int(gpu_total) - 5)}GiB"
        cpu_cap = "16GiB"
        cpt_cap, sft_steps = 400, 80
        eight_bit = False
        reason = (
            "Large GPU (A100/L40S-class): one packed epoch of CPT, "
            "full AdamW, 4-bit QLoRA for SFT"
        )

    return {
        "gpu_tier": tier,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "cpt_max_steps": cpt_cap,
        "sft_max_steps": sft_steps,
        "dpo_epochs": 3,
        "four_bit": True,
        "eight_bit_adam": eight_bit,
        "sequence_length": 1024,
        "max_new_tokens": 40,
        "warmup_ratio": 0.1,
        "max_memory": {0: gpu_cap, "cpu": cpu_cap},
        "reason": reason,
    }


def cpt_steps_for_dataset(n_sequences: int, budget: dict | None = None) -> int:
    """Packed-epoch length, then cap it for the GPU tier.

    T4 is capped in ``training_budget`` (~60 steps). Large GPUs keep up to
    one packed epoch. 5 steps cannot move domain perplexity.
    """
    budget = budget or training_budget()
    cap = int(budget.get("cpt_max_steps") or 0)
    if cap <= 0:
        return 0
    batch = int(budget["per_device_train_batch_size"]) * int(budget["gradient_accumulation_steps"])
    one_epoch = max(1, (int(n_sequences) + batch - 1) // max(batch, 1))
    return min(cap, one_epoch)


def resolve_lora_targets(model, requested: list[str]) -> list[str]:
    """Map assignment names (o_proj) onto BioGPT names (out_proj)."""
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    resolved = []
    for target in requested:
        name = target
        if name not in available and name == "o_proj" and "out_proj" in available:
            name = "out_proj"
        if name in available and name not in resolved:
            resolved.append(name)
    if not resolved:
        raise ValueError(f"None of {requested} exist on this model. Available: {sorted(available)}")
    return resolved


def load_causal_lm(
    model_id: str,
    revision: str | None = None,
    *,
    four_bit: bool = False,
    gradient_checkpointing: bool = True,
    runtime: dict | None = None,
):
    """Load one causal LM onto the GPU without parking it in CPU RAM first."""
    import torch
    from transformers import AutoModelForCausalLM

    disable_optional_scipy_backend()
    runtime = runtime or detect_runtime()
    budget = training_budget(runtime)
    dtype = torch.bfloat16 if runtime.get("dtype") == "bfloat16" else (
        torch.float16 if runtime.get("cuda") else torch.float32
    )
    kwargs = {
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    if revision:
        kwargs["revision"] = revision
    if runtime.get("cuda"):
        kwargs["device_map"] = {"": 0}
        kwargs["max_memory"] = budget["max_memory"]
        if four_bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs.pop("dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if runtime.get("device") in {"cpu", "mps"} and not four_bit:
        model = model.to(runtime["device"])
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    elif hasattr(model, "config"):
        model.config.use_cache = True
    return model


def model_audit(model) -> dict:
    """Decoder depth, attention geometry, parameter counts, vocab vs lm_head."""
    config = model.config
    layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    hidden = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if hidden and heads else None)
    out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    lm_out = None
    if out_emb is not None:
        lm_out = getattr(out_emb, "out_features", None)
        if lm_out is None and getattr(out_emb, "weight", None) is not None:
            lm_out = int(out_emb.weight.shape[0])
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "decoder_layers": layers,
        "attention_heads": heads,
        "hidden_size": hidden,
        "head_dimension": head_dim,
        "vocab_size": config.vocab_size,
        "lm_head_output_dimension": lm_out,
        "lm_head_matches_vocab": lm_out == config.vocab_size,
    }


def trainer_memory_kwargs(runtime: dict | None = None, budget: dict | None = None) -> dict:
    """Hugging Face Trainer flags that stay inside a 15 GB T4."""
    runtime = runtime or detect_runtime()
    budget = budget or training_budget(runtime)
    dtype = runtime.get("dtype")
    return {
        "per_device_train_batch_size": budget["per_device_train_batch_size"],
        "gradient_accumulation_steps": budget["gradient_accumulation_steps"],
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "save_strategy": "no",
        "logging_steps": 10,
        "report_to": "none",
        "remove_unused_columns": False,
        "fp16": bool(runtime.get("cuda") and dtype == "float16"),
        "bf16": bool(runtime.get("cuda") and dtype == "bfloat16"),
        "optim": "adamw_bnb_8bit" if budget.get("eight_bit_adam") else "adamw_torch",
        "warmup_ratio": budget.get("warmup_ratio", 0.1),
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
    }
