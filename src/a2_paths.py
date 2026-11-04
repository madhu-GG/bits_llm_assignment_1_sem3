"""Shared paths for Assignment 2 (prompting + DPO) sitting next to Assignment 1."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
CPT_DIR = OUTPUTS / "biogpt-large-cpt"
ADAPTER_ROOT = OUTPUTS / "qlora_adapters"
INSTRUCTION_DATASET = OUTPUTS / "instruction_dataset.jsonl"
PREFERENCE_DATASET = PROJECT_ROOT / "preference_dataset.jsonl"
DPO_ADAPTER_DIR = OUTPUTS / "dpo_adapter"

BASE_MODEL_ID = "microsoft/biogpt-large"
DEFAULT_SFT_ADAPTER = "adapter_B"


def sft_adapter_path(name: str = DEFAULT_SFT_ADAPTER) -> Path:
    return ADAPTER_ROOT / name


def base_model_source() -> str:
    """Prefer the Assignment 1 CPT checkpoint when it exists."""
    return str(CPT_DIR) if (CPT_DIR / "config.json").exists() else BASE_MODEL_ID
