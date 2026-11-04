"""Path bootstrap for Assignment 2A.

Assignment 2A lives in its own folder and must not modify Assignment 1. This
module makes the Assignment 1 corpus, adapters, and `src/` modules importable
read-only from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

A2_DIR = Path(__file__).resolve().parents[1]          # .../assignment2a
A2_SRC = A2_DIR / "src"
A2_OUTPUTS = A2_DIR / "outputs"
PROJECT_ROOT = A2_DIR.parent                          # Assignment 1 project root

A1_SRC = PROJECT_ROOT / "src"
A1_CORPUS = PROJECT_ROOT / "domain_corpus"
A1_OUTPUTS = PROJECT_ROOT / "outputs"
A1_ADAPTER_ROOT = A1_OUTPUTS / "qlora_adapters"
A1_CPT_DIR = A1_OUTPUTS / "biogpt-large-cpt"
A1_INSTRUCTION_DATASET = PROJECT_ROOT / "instruction_dataset.jsonl"

PREFERENCE_DATASET = A2_DIR / "preference_dataset.jsonl"
DPO_ADAPTER_DIR = A2_OUTPUTS / "dpo_adapter"

BASE_MODEL_ID = "microsoft/biogpt-large"
# Adapter B was the balanced quality/cost configuration in Assignment 1 Part B.
DEFAULT_SFT_ADAPTER = "adapter_B"


def bootstrap() -> None:
    for path in (A2_SRC, A1_SRC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    A2_OUTPUTS.mkdir(parents=True, exist_ok=True)


def sft_adapter_path(name: str = DEFAULT_SFT_ADAPTER) -> Path:
    return A1_ADAPTER_ROOT / name


def base_model_source() -> str:
    """Prefer the Assignment 1 CPT checkpoint when it exists, else the hub model."""
    return str(A1_CPT_DIR) if A1_CPT_DIR.exists() else BASE_MODEL_ID


bootstrap()
