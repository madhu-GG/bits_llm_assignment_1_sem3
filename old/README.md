# Assignment 1A — Medical & Clinical Literature (Single Notebook)

This project is consolidated into one notebook:

- `Assignment_AllInOne.ipynb`

All Python logic for Part A and Part B is embedded directly in that notebook (no `src/` imports required).

## Colab-friendly layout

The notebook uses a simple top-level layout from the current working directory:

- `raw_pdfs/`
- `extracted_text/`
- `clean_corpus/`
- `outputs/`

This keeps folder navigation simple in Colab’s file explorer.

## Runtime and hardware behavior

- Runtime detection is generic (`cpu`, `cuda`, `mps`) and does not assume a specific accelerator.
- GPU-heavy stages are gated with flags (`RUN_CPT`, `RUN_QLORA`, etc.) and default to `False`.
- QLoRA uses 4-bit quantization when supported and falls back to full-precision LoRA when unavailable.

## How to run

1. Open `Assignment_AllInOne.ipynb` in Colab.
2. Install dependencies with:
	- `pip install -r requirements.txt`
3. Run cells from top to bottom.
4. Enable only the stages you want by switching `RUN_*` flags to `True`.

## Corpus note

For final/marked runs, add 10–50 MB of licensed or open-access medical PDFs to `raw_pdfs/`. The notebook also has a starter fallback flow when PDFs are not present.
