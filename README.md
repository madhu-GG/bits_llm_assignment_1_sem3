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

1. Runtime → T4 GPU for the marked run, or CPU if that is all you have.
2. Open `Assignment1A.ipynb` from GitHub or upload it.
3. Run all.

If the GitHub repo is **private**, add a Colab secret named `GITHUB_TOKEN` (Settings → Secrets). Do not paste a token into the notebook. If the repo is public, no token is needed.

The first cells clone the repo (if needed) and install packages. CPT length follows the machine: about **60 steps (~1 hour)** on a Colab T4, about **one packed epoch** on A100/L40S, and a short **LoRA CPT** (seq 256, 8 steps) on CPU so Run all still finishes without a GPU. Only one model is in memory at a time.

A CPU (or Apple MPS) runtime **does train**. Full AdamW on 1.5B fp32 weights does not fit in ~13 GB RAM, so CPT uses LoRA, merges it back into the base weights, and SFT trains the same three adapters without 4-bit (bitsandbytes needs CUDA).

## Memory rules (in `src/runtime.py`)

- Set env vars before importing torch.
- On CUDA, load BioGPT-Large straight onto `cuda:0` (`low_cpu_mem_usage`). Do not materialise 1.5B weights on the 15 GB CPU first.
- CUDA CPT: bf16/fp16 + gradient checkpointing + 8-bit Adam. CPU/MPS CPT: LoRA on the frozen backbone (full AdamW does not fit), then merge.
- CUDA SFT: 4-bit NF4 QLoRA, one adapter at a time. CPU/MPS SFT: the same LoRA adapters on fp32 (no bitsandbytes).
- Never move a finished CUDA model back to CPU — that duplicates it into system RAM.

No access tokens are stored in this repo. The Hugging Face dataset is public.
