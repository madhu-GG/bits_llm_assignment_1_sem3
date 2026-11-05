"""Colab T4 memory toolkit (about 15 GB GPU and 15 GB CPU RAM).

The Hugging Face ``microsoft/biogpt-large`` checkpoint is ~1.5B weights.
On this machine that means:

* fp32 weights on CPU ≈ 6 GB, bf16 weights ≈ 3 GB
* full AdamW (two 32-bit states) ≈ 12 GB — does not fit next to activations
* 8-bit Adam ≈ 1.5 GB optimiser; 4-bit NF4 weights ≈ 0.8 GB

Rules used by both notebooks:

1. Set env vars *before* importing torch.
2. On CUDA, load the model straight onto ``cuda:0``. Never materialise it on
   CPU first. On a CPU/MPS runtime the weights live in host RAM (~6 GB fp32).
3. Keep one model in memory. Delete it before the next stage.
4. Do **not** move a trained model back to CPU on teardown — that duplicates
   ~3 GB into the 15 GB system RAM and can OOM the process.
5. Cap Hugging Face offload so the CPU has room for Python, pypdf, and Parquet.
6. CPU cannot hold full AdamW on 1.5B weights. CPT then uses LoRA (same idea
   as SFT) plus a short sequence so Run all still finishes without a GPU.
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
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info()
        # Ampere (8.0+) has bf16 tensor cores. T4 is 7.5, so QLoRA compute
        # and AMP stay on float16 even when is_bf16_supported() is True.
        use_bf16 = props.major >= 8 and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        base.update({
            "device": "cuda",
            "dtype": "bfloat16" if use_bf16 else "float16",
            "cuda": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(total_b / 1024**3, 2),
            "gpu_free_gb": round(free_b / 1024**3, 2),
            "gpu_cc": f"{props.major}.{props.minor}",
        })
        return base
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Unified memory: report host RAM as the GPU budget. fp16 on M-series
        # is typically stable and halves the 1.5B footprint vs fp32.
        base.update({
            "device": "mps",
            "dtype": "float16",
            "mps": True,
            "gpu_name": "Apple MPS",
            "gpu_total_gb": cpu_total,
            "gpu_free_gb": cpu_avail,
        })
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
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory()
            snap.update({
                "gpu_allocated_gb": round(allocated / 1024**3, 2),
                "gpu_name": "Apple MPS",
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
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
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


def autocast_dtype(runtime: dict | None = None) -> str:
    """AMP / bitsandbytes compute dtype for this machine.

    Pre-Ampere CUDA (T4 is CC 7.5) has no bf16 tensor cores. Mixing bf16
    compute with float LoRA weights raises:
    ``gemm input type BFloat16 and output type float is only supported for
    CUDA devices with compute capability 8.0 or higher``.
    """
    runtime = runtime or detect_runtime()
    if runtime.get("cuda"):
        cc = str(runtime.get("gpu_cc") or "0.0")
        try:
            major = int(cc.split(".")[0])
        except ValueError:
            major = 0
        if major < 8:
            return "float16"
        return runtime.get("dtype") or "bfloat16"
    return runtime.get("dtype") or "float32"


def gpu_tier(runtime: dict | None = None) -> str:
    """Classify the accelerator. Do not use free-VRAM at detect time — that
    shrinks after the baseline load and would mis-label an A100 as a T4.
    """
    runtime = runtime or detect_runtime()
    if runtime.get("cuda"):
        name = (runtime.get("gpu_name") or "").lower()
        total = float(runtime.get("gpu_total_gb") or 0)
        if any(tag in name for tag in ("a100", "h100", "h200", "l40", "b200")) or total >= 32:
            return "large"
        if "t4" in name or total <= 16.5:
            return "t4"
        return "mid"
    if runtime.get("mps"):
        return "mps"
    return "cpu"


def _host_ram_gb(runtime: dict) -> float:
    return float(runtime.get("cpu_total_gb") or 0)


def _cpu_like_budget(runtime: dict, *, tier: str) -> dict:
    """Params that let BioGPT-Large train without CUDA.

    Full AdamW on 1.5B fp32 weights needs ~6 GB model + ~6 GB grads + ~12 GB
    optimiser ≈ 24 GB. A Colab CPU box is ~13 GB, so that path OOMs.

    What does fit: freeze the backbone, train LoRA (a few million params),
    keep sequences short so activations stay small, and take a handful of
    steps so Run all finishes. bitsandbytes 4-bit/8-bit Adam need CUDA, so
    both stay off. The LoRA is merged back into the base weights before
    save, so later cells still load a normal ``biogpt-large-cpt`` folder.
    """
    ram = _host_ram_gb(runtime)
    # psutil missing → ram=0; still try the 13 GB Colab-CPU recipe.
    can_load = ram >= 10.0 or ram == 0.0
    if tier == "mps":
        seq = 512 if ram >= 16.0 else 256
        cpt_steps, sft_steps = (16, 8) if can_load else (0, 0)
        dpo_steps = 20 if can_load else 0
        a2_sc, a2_ape, a2_react, a2_tokens = 3, 3, 3, 128
        reason = (
            f"Apple MPS ({ram:.1f} GB unified): LoRA CPT seq {seq}, "
            f"{cpt_steps} CPT / {sft_steps} SFT / {dpo_steps} DPO steps, "
            "fp16, no 4-bit. Assignment 2 uses 3-sample self-consistency "
            "and 3 APE variants so a local run stays near an hour."
        )
    else:
        seq = 128 if 0.0 < ram < 12.0 else 256
        cpt_steps, sft_steps = (8, 6) if can_load else (0, 0)
        dpo_steps = 8 if can_load else 0
        a2_sc, a2_ape, a2_react, a2_tokens = 3, 3, 3, 80
        reason = (
            f"CPU ({ram:.1f} GB RAM): LoRA CPT so AdamW fits, seq {seq}, "
            f"{cpt_steps} CPT steps / {sft_steps} SFT steps, fp32, no 4-bit "
            "(bitsandbytes needs CUDA). Slow; this is a complete run, not a skip."
        )
    if not can_load:
        reason = (
            f"Host RAM {ram:.1f} GB is too small to load BioGPT-Large "
            "(~6 GB fp32). Need about 10 GB. Data prep still runs."
        )
    return {
        "gpu_tier": tier,
        "can_load_model": can_load,
        "per_device_train_batch_size": 1,
        # Extra accumulation is extra full forwards. On CPU that only costs time.
        "gradient_accumulation_steps": 1,
        "cpt_max_steps": cpt_steps,
        "sft_max_steps": sft_steps,
        "dpo_epochs": 1,
        "dpo_max_steps": dpo_steps,
        "a2_self_consistency_samples": a2_sc,
        "a2_ape_variants": a2_ape,
        "a2_react_cycles": a2_react,
        "a2_max_new_tokens": a2_tokens,
        "four_bit": False,
        "eight_bit_adam": False,
        "lora_cpt": bool(cpt_steps),
        "lora_cpt_r": 8,
        "lora_cpt_alpha": 16,
        "lora_cpt_targets": ("q_proj", "v_proj"),
        "sequence_length": seq,
        "sft_max_length": min(256, seq),
        "dpo_max_length": min(256, seq),
        "dpo_max_prompt_length": min(128, seq // 2 or 64),
        "max_new_tokens": 24,
        "warmup_ratio": 0.1,
        "logging_steps": 1,
        "optim": "adamw_torch",
        "max_memory": None,
        "reason": reason,
    }


def training_budget(runtime: dict | None = None) -> dict:
    """Batch size, accumulation, and CPT length for the detected machine.

    T4 (~15 GB) measured ~57 s/optimiser step on this 1.5B checkpoint with
    gradient checkpointing and 8-bit Adam. 300 steps would be ~4.7 hours.
    Faster lab GPUs ran 100 steps in ~2 minutes, so they can do a packed epoch.
    CPU/MPS cannot hold full AdamW; they use LoRA CPT and a short sequence.
    """
    runtime = runtime or detect_runtime()
    tier = gpu_tier(runtime)
    if tier in {"cpu", "mps"}:
        return _cpu_like_budget(runtime, tier=tier)

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
        "can_load_model": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "cpt_max_steps": cpt_cap,
        "sft_max_steps": sft_steps,
        "dpo_epochs": 3,
        "dpo_max_steps": 0,
        "dpo_max_length": 768,
        "dpo_max_prompt_length": 256,
        "a2_self_consistency_samples": 5,
        "a2_ape_variants": 5,
        "a2_react_cycles": 5,
        "a2_max_new_tokens": 200,
        "four_bit": True,
        "eight_bit_adam": eight_bit,
        "lora_cpt": False,
        "sequence_length": 1024,
        "sft_max_length": 512,
        "max_new_tokens": 40,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "optim": "adamw_bnb_8bit" if eight_bit else "adamw_torch",
        "max_memory": {0: gpu_cap, "cpu": cpu_cap},
        "reason": reason,
    }


def cpt_steps_for_dataset(n_sequences: int, budget: dict | None = None) -> int:
    """Packed-epoch length, then cap it for the device tier.

    T4 is capped in ``training_budget`` (~60 steps). Large GPUs keep up to
    one packed epoch. CPU/MPS keep the short LoRA-CPT cap so Run all finishes.
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


def wrap_lora_for_cpt(model, budget: dict | None = None):
    """On CPU/MPS, wrap the frozen backbone in LoRA so optimiser states fit."""
    budget = budget or training_budget()
    if not budget.get("lora_cpt"):
        return model
    from peft import LoraConfig, get_peft_model

    targets = resolve_lora_targets(model, list(budget.get("lora_cpt_targets") or ("q_proj", "v_proj")))
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(budget.get("lora_cpt_r") or 8),
            lora_alpha=int(budget.get("lora_cpt_alpha") or 16),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    try:
        model.print_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA CPT trainable {trainable:,} / {total:,}")
    return model


def merge_lora_if_needed(model):
    """Fold LoRA back into the base weights so the CPT folder looks like a full LM."""
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def load_causal_lm(
    model_id: str,
    revision: str | None = None,
    *,
    four_bit: bool = False,
    gradient_checkpointing: bool = True,
    runtime: dict | None = None,
):
    """Load one causal LM onto the active device.

    CUDA: map straight onto ``cuda:0`` (never park 1.5B weights in host RAM first).
    MPS: fp16 in unified memory. CPU: fp32. 4-bit is CUDA-only (bitsandbytes).
    """
    import torch
    from transformers import AutoModelForCausalLM

    disable_optional_scipy_backend()
    runtime = runtime or detect_runtime()
    budget = training_budget(runtime)
    use_4bit = bool(four_bit and runtime.get("cuda") and budget.get("four_bit", True))
    if four_bit and not use_4bit:
        kind = "fp16 on MPS" if runtime.get("mps") else "fp32 on CPU"
        print(f"4-bit load requested but this runtime has no CUDA; loading {kind} instead.")

    if runtime.get("cuda"):
        dtype = torch.bfloat16 if autocast_dtype(runtime) == "bfloat16" else torch.float16
    elif runtime.get("mps"):
        dtype = torch.float16 if runtime.get("dtype") == "float16" else torch.float32
    else:
        dtype = torch.float32
    kwargs = {
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    if revision:
        kwargs["revision"] = revision
    if runtime.get("cuda"):
        kwargs["device_map"] = {"": 0}
        if budget.get("max_memory"):
            kwargs["max_memory"] = budget["max_memory"]
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs.pop("dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if runtime.get("mps"):
        model = model.to("mps")
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
    """Hugging Face Trainer flags for the detected device."""
    runtime = runtime or detect_runtime()
    budget = budget or training_budget(runtime)
    dtype = autocast_dtype(runtime)
    cuda = bool(runtime.get("cuda"))
    kwargs = {
        "per_device_train_batch_size": budget["per_device_train_batch_size"],
        "gradient_accumulation_steps": budget["gradient_accumulation_steps"],
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "save_strategy": "no",
        "logging_steps": int(budget.get("logging_steps") or 10),
        "report_to": "none",
        "remove_unused_columns": False,
        "fp16": bool(cuda and dtype == "float16"),
        "bf16": bool(cuda and dtype == "bfloat16"),
        "optim": budget.get("optim") or (
            "adamw_bnb_8bit" if budget.get("eight_bit_adam") else "adamw_torch"
        ),
        # Warmup as a step count. Callers overwrite this once max_steps is known:
        # int(max_steps * budget["warmup_ratio"]).
        "warmup_steps": 1,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
    }
    if not cuda and not runtime.get("mps"):
        kwargs["use_cpu"] = True
    return kwargs
