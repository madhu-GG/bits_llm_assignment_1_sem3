"""Fetch and unpack the medical PDF corpus used by Assignment 1.

The assignment needs at least 10 MB of cleaned English text. That text comes
from about 140 open-access PMC papers, not from the 12 short ``medical_topic_*.pdf``
starter notes (those are ~2.5 KB each and total ~0.04 MB).

Lookup order, no manual switches:

1. Reuse PDFs already unpacked under ``raw_pdfs/``.
2. Unpack a local ``raw_pdfs.tgz`` or ``raw_pdfs.zip`` if present.
3. Download ``raw_pdfs.tgz`` from the public Hugging Face dataset
   ``madhugg93/MedicalText``.
4. If PDFs still cannot be obtained, copy the already-cleaned files from
   ``Medical_Assignment_Deliverables.zip`` so training can continue.

All papers are open access. They are educational literature, not patient records.
The Hugging Face dataset is public, so this file never reads or stores an access token.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HF_DATASET_ID = "madhugg93/MedicalText"
HF_ARCHIVE_NAME = "raw_pdfs.tgz"
HF_ARCHIVE_URL = (
    "https://huggingface.co/datasets/madhugg93/MedicalText/resolve/main/raw_pdfs.tgz"
)
USER_AGENT = "BITS-LLM-Assignment/1A (educational; open-access corpus collection)"

# A real PMC paper in this archive is hundreds of KB. AppleDouble files are 163
# bytes; the 12 starter notes are about 2.5 KB. Both must be ignored when we
# decide whether the corpus is present.
MIN_PAPER_BYTES = 10_000
MIN_PAPER_COUNT = 100
MIN_CLEAN_BYTES = 10 * 1024 * 1024
MIN_ARCHIVE_BYTES = 50 * 1024 * 1024


def iter_source_pdfs(raw_pdfs: Path) -> list[Path]:
    """Return real paper PDFs, skipping macOS junk and the 12 starter notes."""
    papers: list[Path] = []
    if not raw_pdfs.exists():
        return papers
    for path in sorted(raw_pdfs.glob("*.pdf")):
        if path.name.startswith("._"):
            continue
        if path.name.startswith("medical_topic_"):
            continue
        if path.stat().st_size < MIN_PAPER_BYTES:
            continue
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                continue
        papers.append(path)
    return papers


def _download_to_file(url: str, dest: Path, timeout: int = 180) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _safe_extract_path(root: Path, member_name: str) -> Path | None:
    """Refuse archive members that would write outside the project root."""
    dest = (root / member_name).resolve()
    try:
        dest.relative_to(root.resolve())
    except ValueError:
        return None
    return dest


def _should_extract_pdf(member_name: str) -> bool:
    path = Path(member_name)
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    return path.suffix.lower() == ".pdf"


def _extract_tgz(archive: Path, root: Path) -> int:
    extracted = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile() or not _should_extract_pdf(member.name):
                continue
            dest = _safe_extract_path(root, member.name)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                continue
            with source, dest.open("wb") as out:
                shutil.copyfileobj(source, out)
            extracted += 1
    return extracted


def _extract_zip(archive: Path, root: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir() or not _should_extract_pdf(info.filename):
                continue
            dest = _safe_extract_path(root, info.filename)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(handle.read(info.filename))
            extracted += 1
    return extracted


def _remove_appledouble(raw_pdfs: Path) -> int:
    removed = 0
    if not raw_pdfs.exists():
        return removed
    for junk in raw_pdfs.glob("._*"):
        junk.unlink()
        removed += 1
    return removed


def unpack_local_archives(root: Path, raw_pdfs: Path) -> dict:
    """Unpack a local tarball/zip only when real papers are not already on disk."""
    raw_pdfs.mkdir(parents=True, exist_ok=True)
    report = {"tgz": False, "zip": False, "extracted_members": 0, "papers_after_unpack": 0}
    if len(iter_source_pdfs(raw_pdfs)) >= MIN_PAPER_COUNT:
        report["papers_after_unpack"] = len(iter_source_pdfs(raw_pdfs))
        return report

    tgz = root / "raw_pdfs.tgz"
    zip_path = root / "raw_pdfs.zip"
    if tgz.exists() and tgz.stat().st_size > MIN_ARCHIVE_BYTES:
        report["extracted_members"] = _extract_tgz(tgz, root)
        report["tgz"] = True
    elif zip_path.exists() and zip_path.stat().st_size > MIN_ARCHIVE_BYTES:
        report["extracted_members"] = _extract_zip(zip_path, root)
        report["zip"] = True

    nested = root / "raw_pdfs"
    if nested.exists() and nested.resolve() != raw_pdfs.resolve():
        for pdf in nested.glob("*.pdf"):
            target = raw_pdfs / pdf.name
            if not target.exists():
                target.write_bytes(pdf.read_bytes())

    report["appledouble_removed"] = _remove_appledouble(raw_pdfs)
    report["papers_after_unpack"] = len(iter_source_pdfs(raw_pdfs))
    return report


def download_hf_archive(dest: Path) -> dict:
    """Download the public PDF tarball. Never uses a hardcoded token."""
    report = {"ok": False, "source": None, "bytes": 0, "path": str(dest)}
    if dest.exists() and dest.stat().st_size > MIN_ARCHIVE_BYTES:
        report.update({"ok": True, "source": "local", "bytes": dest.stat().st_size})
        return report

    print(f"Downloading {HF_ARCHIVE_NAME} from Hugging Face dataset {HF_DATASET_ID}...")
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=HF_ARCHIVE_NAME,
                repo_type="dataset",
            )
        )
        if cached.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dest)
        report.update({"ok": True, "source": "hf_hub_download", "bytes": dest.stat().st_size})
        return report
    except Exception as exc:
        print(f"hf_hub_download unavailable ({type(exc).__name__}: {exc}); trying the direct URL.")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download_to_file(HF_ARCHIVE_URL, dest, timeout=180)
        report.update({"ok": dest.stat().st_size > MIN_ARCHIVE_BYTES, "source": "direct_url",
                       "bytes": dest.stat().st_size})
        return report
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"Direct Hugging Face download failed ({type(exc).__name__}: {exc}).")
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report


def restore_cleaned_fallback(root: Path, clean_dir: Path) -> int:
    """Last resort: reuse cleaned text from the deliverables zip (11.14 MB)."""
    zip_path = root / "Medical_Assignment_Deliverables.zip"
    if not zip_path.exists():
        return 0
    copied = 0
    clean_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if "__MACOSX" in name or not name.endswith(".txt"):
                continue
            if "/domain_corpus/" not in name:
                continue
            dest = clean_dir / Path(name).name
            if dest.exists():
                continue
            dest.write_bytes(archive.read(name))
            copied += 1
    return copied


def ensure_pdf_corpus(root: Path, raw_pdfs: Path) -> dict:
    """One-shot collection used by Assignment 1. No manual zip copy."""
    raw_pdfs.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": HF_DATASET_ID,
        "archive": HF_ARCHIVE_NAME,
        "unpack": {},
        "download": {},
        "paper_count": 0,
        "starter_notes_ignored": len(list(raw_pdfs.glob("medical_topic_*.pdf"))),
    }

    unpack = unpack_local_archives(root, raw_pdfs)
    report["unpack"] = unpack
    report["paper_count"] = unpack["papers_after_unpack"]

    if report["paper_count"] < MIN_PAPER_COUNT:
        download = download_hf_archive(root / HF_ARCHIVE_NAME)
        report["download"] = download
        if download.get("ok"):
            unpack = unpack_local_archives(root, raw_pdfs)
            report["unpack"] = unpack
            report["paper_count"] = unpack["papers_after_unpack"]

    report["starter_notes_ignored"] = len(list(raw_pdfs.glob("medical_topic_*.pdf")))
    print(json.dumps(report, indent=2))
    if report["paper_count"] < MIN_PAPER_COUNT:
        print(
            f"Warning: only {report['paper_count']} real papers are on disk "
            f"(need {MIN_PAPER_COUNT}). Cleaning will stay far below 10 MB."
        )
    else:
        print(f"Ready: {report['paper_count']} real PMC papers in {raw_pdfs}")
    return report
