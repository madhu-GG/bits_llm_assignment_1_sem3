"""Build a reproducible manifest for the PDFs collected into raw_pdfs/."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RAW_PDFS = ROOT / "raw_pdfs"
EXTRACTED = ROOT / "extracted_text"
CLEAN = ROOT / "clean_corpus"
OUTPUT = ROOT / "outputs" / "pdf_manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_search_metadata() -> dict[str, dict]:
    """Load optional Europe PMC metadata cached during collection."""
    metadata: dict[str, dict] = {}
    for path in Path("/private/tmp").glob("*_page2.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for result in payload.get("resultList", {}).get("result", []):
            pmcid = result.get("pmcid")
            if pmcid:
                metadata.setdefault(pmcid, result)
    for path in Path("/private/tmp").glob("*.json"):
        if path.name.endswith("_page2.json") or path.name == "europepmc_results.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for result in payload.get("resultList", {}).get("result", []):
            pmcid = result.get("pmcid")
            if pmcid:
                metadata.setdefault(pmcid, result)
    return metadata


def build_manifest() -> dict:
    metadata = load_search_metadata()
    files = []
    for pdf_path in sorted(RAW_PDFS.glob("*.pdf")):
        extracted_path = EXTRACTED / f"{pdf_path.stem}.txt"
        clean_path = CLEAN / f"{pdf_path.stem}.txt"
        is_pmc = pdf_path.stem.isdigit()
        pmcid = f"PMC{pdf_path.stem}" if is_pmc else None
        record = {
            "local_filename": pdf_path.name,
            "local_path": str(pdf_path.relative_to(ROOT)),
            "source_repository": "Europe PMC / PMC" if is_pmc else "pre-existing starter corpus",
            "source_url": (
                f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else None
            ),
            "download_url": (
                f"https://europepmc.org/api/getPdf?pmcid={pmcid}" if pmcid else None
            ),
            "pmcid": pmcid,
            "retrieval_method": (
                "Europe PMC official PDF endpoint; query constrained to OPEN_ACCESS:Y"
                if is_pmc
                else "present before this collection run"
            ),
            "retrieved_date": str(date.today()) if is_pmc else None,
            "bytes": pdf_path.stat().st_size,
            "sha256": sha256(pdf_path),
            "pages": len(PdfReader(str(pdf_path)).pages),
            "extracted_text": str(extracted_path.relative_to(ROOT)) if extracted_path.exists() else None,
            "extracted_text_bytes": extracted_path.stat().st_size if extracted_path.exists() else None,
            "clean_text": str(clean_path.relative_to(ROOT)) if clean_path.exists() else None,
            "clean_text_bytes": clean_path.stat().st_size if clean_path.exists() else None,
        }
        if pmcid:
            result = metadata.get(pmcid, {})
            record.update(
                {
                    "title": result.get("title"),
                    "journal": result.get("journalTitle"),
                    "publication_year": result.get("pubYear"),
                }
            )
        else:
            record.update({"title": None, "journal": None, "publication_year": None})
        files.append(record)

    return {
        "manifest_version": 1,
        "generated_date": str(date.today()),
        "collection_scope": {
            "topics": ["hypertension", "diabetes", "asthma", "pneumonia"],
            "search_source_url": "https://pmc.ncbi.nlm.nih.gov/?term=open+access%5Bfilter%5D+AND+has_pdf%5Bfilter%5D+AND+(hypertension+OR+diabetes+OR+asthma+OR+pneumonia)",
            "repository_url": "https://dev.europepmc.org/downloads/openaccess",
            "api_endpoint_template": "https://europepmc.org/api/getPdf?pmcid={PMCID}",
            "access_filter": "OPEN_ACCESS:Y",
        },
        "file_count": len(files),
        "files": files,
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
