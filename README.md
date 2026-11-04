# Medical LLM assignment

Colab T4 target: about **15 GB GPU** and **15 GB CPU RAM**. Run one notebook at a time with **Runtime → Run all**. There are no `RUN_*` flags.

## Layout

```
Assignment1A.ipynb      CPT + QLoRA SFT (Assignment 1A, 15 marks)
src/                    shared helpers (corpus fetch, T4 memory, A2 prompting)
requirements.txt
preference_dataset.jsonl
dataset_sources.txt
```

PDFs are **not** copied by hand. `Assignment1A.ipynb` downloads `raw_pdfs.tgz` from the public dataset [`madhugg93/MedicalText`](https://huggingface.co/datasets/madhugg93/MedicalText) if `raw_pdfs/` is empty. The 12 `medical_topic_*.pdf` stubs are ignored (~0.04 MB). The real corpus is ~140 PMC papers → ~11 MB cleaned text.

Assignment 2 (prompting + DPO) helpers already live in `src/`. The combined Assignment 2 notebook is next.

## Colab

Colab's **Open notebook from GitHub** copies only the `.ipynb`, not `src/`. The first code cell clones this repo into `/content` when `src/` is missing, then `chdir`s into it. After that, Run all.

1. Runtime → T4 GPU.
2. Open `Assignment1A.ipynb` from GitHub or upload it.
3. Run all.

If the GitHub repo is **private**, add a Colab secret named `GITHUB_TOKEN` (Settings → Secrets). Do not paste a token into the notebook. If the repo is public, no token is needed.

The first cells clone the repo (if needed) and install packages. CPT length follows the GPU: about **60 steps (~1 hour)** on a Colab T4, or about **one packed epoch** on A100/L40S. Only one model is in memory at a time.

If the runtime has no GPU, extraction, cleaning, packing, and the instruction JSONL still run; training is skipped automatically.

## Memory rules (in `src/runtime.py`)

- Set env vars before importing torch.
- Load BioGPT-Large straight onto `cuda:0` (`low_cpu_mem_usage`). Do not materialise 1.5B weights on the 15 GB CPU first.
- CPT: bf16/fp16 + gradient checkpointing + 8-bit Adam.
- SFT: 4-bit NF4 QLoRA, one adapter at a time.
- Never move a finished model back to CPU — that duplicates it into system RAM.

No access tokens are stored in this repo. The Hugging Face dataset is public.
