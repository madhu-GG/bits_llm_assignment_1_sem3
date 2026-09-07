# Assignment 1A — Medical & Clinical Literature

This project implements the supplied assignment, using the selected T4 model:

- Domain: Medical & Clinical Literature
- T4 Choice 1: BioGPT-Large, `microsoft/biogpt-large` (347M parameters)
- A100 alternatives: BioMedLM, `stanford-crfm/BioMedLM`, or Mistral-7B-v0.1, `mistralai/Mistral-7B-v0.1`

## What is included

- `Assignment_PartA.ipynb`: PDF extraction, four cleaning filters, tokenizer reuse, BOS/EOS packing to Parquet, architecture audit, baseline generation, CPT loop, loss curve, perplexity, and catastrophic-forgetting check.
- `Assignment_PartB.ipynb`: 120 instruction-response pairs, 80/20 split, QLoRA with adapters A/B/C, and a shared-prompt comparison table.
- `instruction_dataset.jsonl`: generated 120-row educational instruction dataset with `train` and `eval` labels.
- `raw_pdfs/`: 12 small educational starter PDFs for local validation.
- `domain_corpus/`: cleaned text output produced from those PDFs.
- `src/medical_pipeline.py`: reusable implementation of the assignment pipeline.
- `ARCHITECTURE.docx` and `FINAL_REPORT.docx`: project design and submission-style report.

## Important corpus note

The included PDFs are a reproducible starter corpus so that extraction and cleaning can be checked without external downloads. The assignment asks for 10–50 MB of domain-specific PDFs. For a marked run, add licensed or open-access medical literature PDFs to `raw_pdfs/` before running Step 1. The same code will process them page by page. Do not use paywalled or personally identifiable clinical documents.

## Colab execution order

1. Upload this project or clone its folder into Colab.
2. Run `!pip install -r requirements_colab.txt`.
3. Run all data-preparation cells in Part A. Confirm the cleaning-count table and Parquet metrics.
4. Set `RUN_CPT = True`, select a GPU runtime, and run CPT. For a free T4, keep the conservative defaults (`max_steps=100`, sequence length 512 or 1024, gradient accumulation enabled).
5. Set `RUN_QLORA = True` in Part B after the CPT checkpoint exists. Keep the three adapter configurations unchanged for the assignment comparison.
6. Replace the placeholder/empty model outputs in the final report with the printed outputs from the actual GPU run.

## Reproducibility and safety

The notebooks set a seed, record model IDs, tokenizer IDs, configuration values, split counts, and output paths. The medical text is educational and must not be treated as patient-specific clinical advice.
# bits_llm_assignment_1_sem3
