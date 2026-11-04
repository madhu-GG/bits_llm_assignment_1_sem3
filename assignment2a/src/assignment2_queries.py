"""Shared query sets and schemas for Assignment 2A (Medical & Clinical Literature).

Every part of Assignment 2A reuses the same domain and the same corpus as
Assignment 1A, so the query sets live in one module to keep Parts A, B, and C
consistent with each other.
"""

from __future__ import annotations

# The five domain queries used throughout Part A and reused in Part B.
DOMAIN_QUERIES = [
    "Why are repeated blood-pressure measurements more informative than a single reading in hypertension?",
    "Explain the difference between sensitivity and specificity, and why predictive values depend on prevalence.",
    "Why does antimicrobial stewardship reduce the spread of antimicrobial resistance?",
    "What does glycated haemoglobin measure in type 2 diabetes, and how should it guide management?",
    "Why does intention-to-treat analysis preserve the benefit of randomisation in a clinical trial?",
]

# Two additional multi-step problems for Part B (require decomposition).
REASONING_PROBLEMS = [
    {
        "id": "RP1",
        "question": (
            "A 72-year-old outpatient with stable chronic kidney disease and type 2 diabetes is started on a "
            "new glucose-lowering medicine. Explain, step by step, which factors decide whether the dose is "
            "appropriate and what monitoring plan should follow."
        ),
        "topics": ["Chronic kidney disease and renal function", "Type 2 diabetes mellitus",
                   "Adverse drug reactions and medication safety"],
    },
    {
        "id": "RP2",
        "question": (
            "A new rapid test for a respiratory infection reports high sensitivity but moderate specificity. "
            "Reason step by step about how the test should be used in a low-prevalence outpatient clinic "
            "versus a high-prevalence hospital ward, and what the consequences of each error type are."
        ),
        "topics": ["Diagnostic test performance", "Community-acquired pneumonia", "Evidence-based medicine"],
    },
]

# Part C3 held-out queries. These are intentionally NOT used to build the
# preference dataset so the alignment evaluation is on unseen prompts.
HELD_OUT_QUERIES = [
    "How should a written asthma action plan help a patient recognise worsening symptoms?",
    "What should a clinician check before reporting a suspected adverse drug reaction?",
    "Why can a relative risk reduction look impressive when the absolute benefit is small?",
    "What is the purpose of medication reconciliation during a transition of care?",
    "How does high vaccination coverage protect people who cannot be vaccinated themselves?",
]

# Part A2 prompt-chain target: one complex query that needs decomposition.
CHAIN_QUERY = (
    "Using the corpus, explain how a clinician should combine diagnostic test performance and "
    "evidence quality when deciding whether to start antibiotics for a suspected community-acquired pneumonia."
)

# Part A2 function-calling schema for structured domain output.
CLINICAL_FACT_SCHEMA = {
    "name": "extract_clinical_fact",
    "description": "Extract a single structured clinical fact from the domain corpus.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Clinical concept, measurement, or intervention."},
            "value": {"type": "string", "description": "Reported value, threshold, or qualitative finding."},
            "unit": {"type": "string", "description": "Unit of measure, or 'none' when the value is qualitative."},
            "source_clause": {"type": "string", "description": "Verbatim clause from the corpus supporting the fact."},
            "confidence": {"type": "number", "description": "Model confidence between 0 and 1."},
        },
        "required": ["entity", "value", "unit", "source_clause"],
    },
}

# Part B2 PAL problems: each needs real arithmetic, not recall.
PAL_PROBLEMS = [
    {
        "id": "PAL1",
        "question": (
            "Estimate creatinine clearance with the Cockcroft-Gault equation for a female patient aged 68 years, "
            "weight 62 kg, serum creatinine 1.4 mg/dL. Report the result in mL/min to one decimal place."
        ),
        "hint": "CrCl = ((140 - age) * weight_kg) / (72 * serum_creatinine) * 0.85 for female patients.",
        "expected": 37.6,
        "unit": "mL/min",
    },
    {
        "id": "PAL2",
        "question": (
            "In a trial, the event rate was 18.0% in the control arm and 12.5% in the treatment arm. "
            "Compute the absolute risk reduction, the relative risk reduction, and the number needed to treat."
        ),
        "hint": "ARR = control - treatment; RRR = ARR / control; NNT = 1 / ARR (round up).",
        "expected": 19.0,
        "unit": "patients (NNT)",
    },
    {
        "id": "PAL3",
        "question": (
            "A test has sensitivity 0.95 and specificity 0.80. In a clinic where prevalence is 2%, "
            "compute the positive predictive value as a percentage to one decimal place."
        ),
        "hint": "PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev)).",
        "expected": 8.8,
        "unit": "% (PPV)",
    },
]

# Part B2 ReACT question: answerable only by looking facts up in the corpus.
REACT_QUESTION = (
    "Which corpus topic explains likelihood ratios, what does a negative likelihood ratio tell a clinician, "
    "and how does that interact with the predictive-value point made about prevalence?"
)

# Few-shot exemplars for Part A1. Kept short so they fit a 512-token window.
FEW_SHOT_EXAMPLES = [
    {
        "question": "What does a positive likelihood ratio indicate?",
        "answer": "A positive likelihood ratio indicates how much a positive result increases the odds of the "
                  "target condition. Larger values move the post-test odds further above the pre-test odds.",
    },
    {
        "question": "Why is a written action plan used in asthma?",
        "answer": "A written action plan explains how to recognise worsening symptoms and when to seek urgent "
                  "care, so the patient can act early instead of waiting for a scheduled review.",
    },
    {
        "question": "What does intention-to-treat analysis do?",
        "answer": "Intention-to-treat analysis keeps every participant in the group they were randomised to, "
                  "which preserves the balance of prognostic factors created by randomisation.",
    },
]

SYSTEM_ROLE = (
    "You are a careful medical-literature assistant. You answer from the supplied domain corpus, you state "
    "uncertainty explicitly, and you never give patient-specific clinical advice."
)

SAFETY_NOTE = (
    "Educational summary of published medical literature. Not patient-specific clinical advice."
)
