"""Generate a small reproducible PDF starter corpus for local validation.

For the marked submission, replace or augment these educational PDFs with
10--50 MB of licensed/open-access medical PDFs using the download instructions
in README.md. The extraction and cleaning pipeline is identical.
"""

from pathlib import Path
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medical_data import TOPICS, topic_document


def make_pdf(topic, output):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=10)
    subheading = ParagraphStyle("Subheading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=6, spaceAfter=5)
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.7*inch, leftMargin=0.7*inch, topMargin=0.65*inch, bottomMargin=0.65*inch)
    story = []
    story.append(Paragraph("Medical and Clinical Literature — Starter Document", heading))
    story.append(Paragraph(topic["title"], subheading))
    story.append(Paragraph("Document type: Educational clinical literature summary", body))
    story.append(Paragraph("Scope: General medical and clinical concepts for language-model domain adaptation.", body))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Abstract", subheading))
    story.append(Paragraph(topic["abstract"], body))
    story.append(Paragraph("Key concepts", subheading))
    for i, point in enumerate(topic["points"], 1):
        story.append(Paragraph(f"{i}. {point}", body))
    story.append(Paragraph("Interpretation note", subheading))
    story.append(Paragraph("This educational summary supports literature-processing experiments and is not a substitute for current clinical guidelines, professional judgment, or individualized medical care.", body))
    doc.build(story)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "raw_pdfs"
    out.mkdir(parents=True, exist_ok=True)
    for idx, topic in enumerate(TOPICS, 1):
        make_pdf(topic, out / f"medical_topic_{idx:02d}.pdf")
    print(f"Generated {len(TOPICS)} PDFs in {out}")


if __name__ == "__main__":
    main()
