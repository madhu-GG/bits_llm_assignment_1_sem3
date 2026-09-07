"""Run the dependency-light local portion of Assignment 1A."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_sample_corpus import main as generate_sample_corpus
from medical_pipeline import extract_pdfs_page_by_page, clean_extracted_text, build_instruction_dataset


def main():
    raw = ROOT / "raw_pdfs"
    extracted = ROOT / "artifacts" / "extracted_text"
    clean = ROOT / "domain_corpus"
    if not list(raw.glob("*.pdf")):
        generate_sample_corpus()
    extraction = extract_pdfs_page_by_page(raw, extracted)
    counts, impact = clean_extracted_text(extracted, clean)
    dataset = build_instruction_dataset(clean, ROOT / "instruction_dataset.jsonl", min_pairs=100)
    print("PDF extraction:", {"documents": len(extraction), "pages": sum(x["pages"] for x in extraction), "characters": sum(x["characters"] for x in extraction)})
    print("Cleaning counts:", counts)
    print("Cleaning impact:", impact)
    print("Instruction dataset:", dataset)


if __name__ == "__main__":
    main()

