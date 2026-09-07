"""Create the architecture, guide, final report, and local requirements copy."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE

ROOT = Path(__file__).resolve().parent
NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
MUTED = "555555"


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_in):
    total_dxa = 9360
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(total_dxa))
    tblW.set(qn("w:type"), "dxa")
    tblInd = tblPr.find(qn("w:tblInd"))
    if tblInd is None:
        tblInd = OxmlElement("w:tblInd")
        tblPr.append(tblInd)
    tblInd.set(qn("w:w"), "120")
    tblInd.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    widths = [int(round(w * 1440)) for w in widths_in]
    scale = total_dxa / sum(widths)
    widths = [int(round(w * scale)) for w in widths]
    widths[-1] += total_dxa - sum(widths)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            cell.width = Inches(widths_in[idx])
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.append(tcW)
            tcW.set(qn("w:w"), str(widths[idx]))
            tcW.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def configure_document(doc, title):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string("222222")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        st = styles[style_name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
    if "Small" not in [s.name for s in styles]:
        small = styles.add_style("Small", WD_STYLE_TYPE.PARAGRAPH)
    else:
        small = styles["Small"]
    small.font.name = "Calibri"
    small.font.size = Pt(9)
    small.font.color.rgb = RGBColor.from_string(MUTED)
    small.paragraph_format.space_after = Pt(4)
    header = sec.header.paragraphs[0]
    header.text = title
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.name = "Calibri"
    header.runs[0].font.size = Pt(9)
    header.runs[0].font.color.rgb = RGBColor.from_string(MUTED)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Assignment 1A  •  Page ")
    run.font.name = "Calibri"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)


def add_title(doc, title, subtitle=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(24)
    r.font.color.rgb = RGBColor.from_string(NAVY)
    if subtitle:
        p2 = doc.add_paragraph()
        p2.paragraph_format.space_after = Pt(18)
        r2 = p2.add_run(subtitle)
        r2.font.name = "Calibri"
        r2.font.size = Pt(12)
        r2.font.color.rgb = RGBColor.from_string(MUTED)


def add_note(doc, label, text, fill=LIGHT_GRAY):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [6.5])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    rr = p.add_run(label + "  ")
    rr.bold = True
    rr.font.color.rgb = RGBColor.from_string(DARK_BLUE)
    p.add_run(text)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, LIGHT_BLUE)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(h))
        run.bold = True
        run.font.color.rgb = RGBColor.from_string(NAVY)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.add_run(str(value))
    for row in table.rows:
        for cell in row.cells:
            set_cell_margins(cell)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    p.add_run(text)


def number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(4)
    p.add_run(text)


def save_architecture():
    doc = Document()
    configure_document(doc, "Architecture Design")
    add_title(doc, "Architecture Design", "Assignment 1A — Medical & Clinical Literature LLM")
    add_note(doc, "Selected configuration", "T4 GPU Choice 1: BioGPT-Large (347M), microsoft/biogpt-large. The notebooks retain a configurable MODEL_ID for A100 alternatives.")
    doc.add_heading("1. Design objectives", level=1)
    doc.add_paragraph("The solution follows the assignment’s end-to-end workflow: collect literature, extract and clean text, reuse the selected tokenizer, pack sequences for continual pre-training, evaluate adaptation and forgetting, then create a supervised instruction dataset and train three QLoRA adapters.")
    doc.add_heading("2. System flow", level=1)
    add_table(doc, ["Stage", "Input", "Processing", "Output"], [
        ("1. Collection", "Medical PDFs", "Page-by-page extraction", "Raw .txt files and extraction counts"),
        ("2. Cleaning", "Raw .txt files", "Length, repetition, exact-match, English filters", "domain_corpus/*.txt and filter audit"),
        ("3. Packing", "Clean text", "BioGPT tokenizer, BOS/EOS, fixed chunks", "Parquet packed dataset"),
        ("4. CPT", "Packed Parquet", "AdamW, linear warmup, gradient checkpointing", "CPT checkpoint, tokenizer, loss history"),
        ("5. Evaluation", "Base/CPT models", "PPL and general prompts", "PPL reduction and forgetting table"),
        ("6. SFT", "Clean corpus", "120 JSONL pairs, 80/20 split", "instruction_dataset.jsonl"),
        ("7. QLoRA", "CPT/base model + JSONL", "4-bit NF4, adapters A/B/C", "Three adapter directories"),
        ("8. Comparison", "Same three prompts", "Deterministic generation", "Single adapter comparison table"),
    ], [1.05, 1.55, 2.25, 1.65])
    doc.add_heading("3. Components and responsibilities", level=1)
    add_table(doc, ["Component", "Responsibility", "Implementation"], [
        ("Data layer", "PDF extraction, cleaning, corpus metrics", "src/medical_pipeline.py"),
        ("Tokenizer layer", "Preserve vocabulary alignment and document boundaries", "AutoTokenizer.from_pretrained(MODEL_ID)"),
        ("CPT trainer", "Adapt causal LM to medical language", "Trainer + AdamW + linear warmup"),
        ("Evaluation layer", "Measure domain gain and general retention", "Cross-entropy PPL + fixed prompts"),
        ("Instruction layer", "Turn source text into supervised examples", "Deterministic 120-row JSONL generator"),
        ("Adapter layer", "Compare capacity/cost trade-offs", "PEFT LoraConfig + bitsandbytes"),
        ("Experiment layer", "Persist metrics and outputs", "outputs/ JSON and checkpoint folders"),
    ], [1.2, 2.8, 2.5])
    doc.add_heading("4. Data contracts", level=1)
    bullet(doc, "Clean corpus contract: UTF-8 text files, one source document per file, at least 50 characters, English-language content, and no repeated-paragraph ratio above 30%.")
    bullet(doc, "Packed dataset contract: each Parquet row contains input_ids and attention_mask of exactly sequence_length tokens.")
    bullet(doc, "Instruction contract: every JSONL row contains instruction, response, source, and split; split values are train or eval.")
    bullet(doc, "Evaluation contract: base and CPT models use the same held-out texts; all adapters receive the same three prompts.")
    doc.add_heading("5. Resource and risk controls", level=1)
    add_table(doc, ["Risk", "Mitigation"], [
        ("T4 memory pressure", "4-bit QLoRA, gradient checkpointing, small batch, gradient accumulation, conservative max_steps."),
        ("Tokenizer mismatch", "One MODEL_ID is reused for tokenization, model loading, CPT, and SFT."),
        ("Catastrophic forgetting", "10% held-out PPL plus unrelated prompts; reduce learning rate or max_steps if degradation appears."),
        ("Data leakage or unsafe content", "Use licensed/open-access literature only; remove PII; retain source provenance."),
        ("Library/API drift", "Notebook includes version requirements and a TRL tokenizer/processing_class compatibility branch."),
    ], [2.1, 4.4])
    doc.add_heading("6. Reproducibility", level=1)
    doc.add_paragraph("The project records model IDs, adapter configurations, split counts, metrics, and generated outputs. Training flags default to False so preprocessing can be verified before downloading weights or allocating GPU memory. Set the flags to True in Colab for the actual CPT and QLoRA runs.")
    doc.save(ROOT / "ARCHITECTURE.docx")


def save_guide():
    doc = Document()
    configure_document(doc, "Execution Guide")
    add_title(doc, "Assignment Completion Guide", "Step-by-step execution plan and marking checklist")
    add_note(doc, "Recommended path", "Use the supplied Medical & Clinical Literature data and microsoft/biogpt-large on a T4. Run preprocessing first, then enable GPU stages one at a time.")
    doc.add_heading("1. Prepare the environment", level=1)
    number(doc, "Open a Colab notebook with a T4 or A100 runtime and upload the project folder.")
    number(doc, "Install requirements_colab.txt. Restart the runtime if Colab asks.")
    number(doc, "Keep the model ID and tokenizer ID identical: microsoft/biogpt-large.")
    doc.add_heading("2. Complete Part A", level=1)
    add_table(doc, ["Assignment step", "What to run", "Evidence to save"], [
        ("Step 1", "Add 10–50 MB licensed/open-access PDFs; run extraction and cleaning.", "Counts before/after each filter and greatest-impact step."),
        ("Step 2", "Set RUN_TOKENIZATION=True.", "Total tokens, average document tokens, packed sequence count, Parquet file."),
        ("Step 3", "Set RUN_MODEL_INSPECTION=True.", "Parameter count, layers, heads, hidden size, head dimension, lm_head check, three baselines."),
        ("Step 4", "Set RUN_CPT=True.", "Checkpoint, tokenizer, cpt_loss_history.json, loss plot and plateau observation."),
        ("Step 5", "Set RUN_EVALUATION=True.", "Base/CPT PPL, percentage reduction, three general-prompt comparisons."),
    ], [1.0, 3.0, 2.5])
    doc.add_heading("3. Complete Part B", level=1)
    number(doc, "Confirm instruction_dataset.jsonl has at least 100 rows and an 80/20 train/eval split.")
    number(doc, "Use the CPT checkpoint if Step 4 completed; otherwise the notebook falls back to BioGPT-Large.")
    number(doc, "Set RUN_QLORA=True and train adapters A, B, and C with the exact configurations in the notebook.")
    number(doc, "Set RUN_ADAPTER_EVAL=True and copy the one-table comparison into the final report.")
    doc.add_heading("4. What to write in the final submission", level=1)
    bullet(doc, "State the domain and GPU/model selection with justification.")
    bullet(doc, "Include the cleaning-count table and explain which filter had the greatest impact.")
    bullet(doc, "Report tokenizer, BOS/EOS policy, sequence length, token count, average length, and packed row count.")
    bullet(doc, "Include the architecture audit and lm_head/vocabulary equality check.")
    bullet(doc, "Show the CPT loss curve, base/CPT PPL, PPL reduction, and forgetting verdicts.")
    bullet(doc, "Include JSONL counts and the three LoRA configurations.")
    bullet(doc, "Compare the same three prompts across adapters and select the most accurate adapter based on source-grounded correctness.")
    doc.add_heading("5. Acceptance checklist", level=1)
    add_table(doc, ["Check", "Status in supplied project"], [
        ("Part A notebook exists", "Complete"),
        ("Part B notebook exists", "Complete"),
        ("120 instruction pairs / 96 train / 24 eval", "Complete"),
        ("Starter PDFs and cleaned .txt files", "Complete; expand to 10–50 MB for marked run"),
        ("GPU CPT and QLoRA metrics", "Run on Colab/A100; flags are provided"),
        ("Architecture and final report", "Included"),
    ], [3.5, 3.0])
    doc.add_heading("6. Common troubleshooting", level=1)
    bullet(doc, "If GPU memory is insufficient, reduce sequence length, max_steps, or batch size; retain gradient accumulation.")
    bullet(doc, "If loss starts near 10.8, verify that AutoModelForCausalLM loaded pretrained weights rather than a fresh configuration.")
    bullet(doc, "If loss spikes, lower learning rate and increase warm-up steps.")
    bullet(doc, "If an adapter target module is not found, inspect model.named_modules() and use the model’s actual q_proj/v_proj/o_proj names.")
    doc.save(ROOT / "ASSIGNMENT_GUIDE.docx")


def save_report():
    doc = Document()
    configure_document(doc, "Final Report")
    add_title(doc, "Final Report", "Assignment 1A — Medical & Clinical Literature")
    add_note(doc, "Submission status", "The data pipeline, notebooks, starter corpus, instruction dataset, architecture, and execution guide are complete. GPU-dependent CPT/QLoRA metrics must be generated in Colab/A100 by enabling the documented run flags.")
    doc.add_heading("1. Executive summary", level=1)
    doc.add_paragraph("This project implements an end-to-end domain adaptation and instruction fine-tuning workflow for medical and clinical literature. BioGPT-Large was selected because it is the T4 Choice 1 in the supplied domain/model table and is small enough for a conservative Colab experiment. The implementation preserves tokenizer alignment throughout CPT and SFT, uses BOS/EOS boundaries for packed sequences, and compares three QLoRA adapter capacities.")
    doc.add_heading("2. Domain and model selection", level=1)
    add_table(doc, ["Item", "Selected value"], [
        ("Domain", "Medical & Clinical Literature"),
        ("T4 model", "BioGPT-Large — 347M"),
        ("Model ID", "microsoft/biogpt-large"),
        ("A100 Choice 1", "BioMedLM — 2.7B — stanford-crfm/BioMedLM"),
        ("Tokenizer rule", "AutoTokenizer paired with the selected model; no custom tokenizer"),
    ], [2.0, 4.5])
    doc.add_heading("3. Part A results", level=1)
    doc.add_heading("3.1 Data extraction and cleaning", level=2)
    add_table(doc, ["Metric", "Observed local starter-corpus result"], [
        ("PDF documents", "12"),
        ("Pages extracted", "12"),
        ("Raw extracted characters", "14,817"),
        ("Before cleaning", "12"),
        ("After length filter", "12"),
        ("After repetition filter", "12"),
        ("After exact-match deduplication", "12"),
        ("After English filter", "12"),
        ("Greatest impact", "None; no starter document was removed"),
    ], [2.7, 3.8])
    doc.add_paragraph("The included corpus is a small reproducible educational starter set for validating code paths. The assignment’s 10–50 MB corpus requirement is addressed in the notebooks and guide: add licensed/open-access medical PDFs to raw_pdfs/ before the marked run. The cleaning implementation then reports the true counts for that enlarged corpus.")
    doc.add_heading("3.2 Tokenization and packing", level=2)
    doc.add_paragraph("The implementation calls AutoTokenizer.from_pretrained('microsoft/biogpt-large'), adds a BOS/EOS boundary around every document, concatenates the token IDs, slices fixed-length chunks, and writes input_ids and attention_mask columns to Parquet. Token totals and average document length are produced at runtime because they depend on the final 10–50 MB corpus.")
    doc.add_heading("3.3 Model audit and baseline", level=2)
    doc.add_paragraph("The model-audit function reports total and trainable parameters, decoder layers, attention heads, hidden size, head dimension, vocabulary size, lm_head output dimension, and a boolean equality check. Three medical prompts are generated and saved before CPT for a fair baseline.")
    doc.add_heading("3.4 CPT and evaluation", level=2)
    doc.add_paragraph("CPT uses Hugging Face Trainer with AdamW, a linear warm-up schedule, gradient checkpointing, and a custom loss callback. The final checkpoint and tokenizer are saved for Part B. The evaluation cell computes base and CPT perplexity on the same 10% held-out text and compares general prompts for catastrophic forgetting. These values are intentionally not fabricated: they require downloading the model and running on a supported GPU.")
    doc.add_heading("4. Part B results", level=1)
    doc.add_heading("4.1 Instruction dataset", level=2)
    add_table(doc, ["Metric", "Value"], [
        ("Total pairs", "120"),
        ("Training split", "96 (80%)"),
        ("Evaluation split", "24 (20%)"),
        ("Format", "JSONL with instruction, response, source, split"),
        ("Content", "Educational, source-grounded medical concepts"),
    ], [2.3, 4.2])
    doc.add_heading("4.2 QLoRA configurations", level=2)
    add_table(doc, ["Adapter", "Rank", "Alpha", "Target modules", "Expected role"], [
        ("A", "8", "16", "q_proj, v_proj", "Fast; possible underfitting"),
        ("B", "16", "32", "q_proj, v_proj", "Balanced baseline"),
        ("C", "32", "32", "q_proj, v_proj, o_proj", "Highest capacity"),
    ], [0.9, 0.7, 0.8, 2.1, 2.0])
    doc.add_paragraph("All adapters use 4-bit NF4 quantization, the same training/evaluation split, and the same prompt set. The notebook includes compatibility handling for TRL releases that use processing_class instead of tokenizer.")
    doc.add_heading("5. Comparative evaluation plan", level=1)
    add_table(doc, ["Prompt", "Adapter A", "Adapter B", "Adapter C"], [
        ("Repeated measurements in hypertension", "Populate after GPU run", "Populate after GPU run", "Populate after GPU run"),
        ("Sensitivity versus specificity", "Populate after GPU run", "Populate after GPU run", "Populate after GPU run"),
        ("Importance of antimicrobial stewardship", "Populate after GPU run", "Populate after GPU run", "Populate after GPU run"),
    ], [2.1, 1.45, 1.45, 1.5])
    doc.add_paragraph("The final adapter verdict should be based on factual accuracy, relevance to the source corpus, completeness, and absence of unsupported clinical advice. Adapter B is the expected quality/cost baseline; Adapter C may be strongest on multi-part questions; Adapter A is the fastest and may underfit.")
    doc.add_heading("6. Conclusion", level=1)
    doc.add_paragraph("The project is complete as a reproducible implementation package. The supplied files satisfy the required structure and provide a deterministic local validation path. To finish the measured submission, enlarge the raw corpus to 10–50 MB, run the GPU flags in order, paste the generated metrics and outputs into the report table, and submit the two notebooks, JSONL file, cleaned corpus, checkpoint evidence, and this report.")
    doc.add_heading("Appendix A — Deliverables", level=1)
    add_table(doc, ["File", "Purpose"], [
        ("Assignment_PartA.ipynb", "Steps 1–5: data pipeline, CPT, PPL, forgetting"),
        ("Assignment_PartB.ipynb", "B1–B3: instruction data, three QLoRA adapters, comparison"),
        ("instruction_dataset.jsonl", "120 instruction-response pairs with split labels"),
        ("domain_corpus/*.txt", "Cleaned text files"),
        ("src/medical_pipeline.py", "Reusable implementation"),
        ("ARCHITECTURE.docx", "System design"),
        ("ASSIGNMENT_GUIDE.docx", "Execution and marking guide"),
    ], [2.4, 4.1])
    doc.save(ROOT / "FINAL_REPORT.docx")


def main():
    save_architecture()
    save_guide()
    save_report()
    source = Path("/workspace/scratch/b75e4e1cd1a5/upload/Assignment 1A - CPT and SFT(2).docx")
    if source.exists():
        (ROOT / "Assignment_Requirements.docx").write_bytes(source.read_bytes())
    print("Created architecture, guide, final report, and requirements copy.")


if __name__ == "__main__":
    main()

