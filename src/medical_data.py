"""Curated educational starter content for the Medical & Clinical Literature track.

The content is intentionally general and educational. It is not clinical advice.
It is used only to make the assignment reproducible before the user supplies a
larger set of licensed/open-access medical PDFs.
"""

TOPICS = [
    {
        "title": "Hypertension and cardiovascular risk",
        "abstract": "Hypertension is persistent elevation of arterial blood pressure. Long-term pressure load can injure the heart, brain, kidneys, and blood vessels, so risk assessment considers both blood-pressure measurements and other cardiovascular risk factors.",
        "points": [
            "Repeated, correctly performed measurements are more informative than a single reading.",
            "Lifestyle measures include reducing dietary sodium, being physically active, maintaining a healthy weight, and avoiding tobacco.",
            "Antihypertensive treatment is selected according to blood pressure, overall cardiovascular risk, comorbidities, and patient preferences.",
            "Uncontrolled hypertension increases the likelihood of stroke, coronary disease, heart failure, and chronic kidney disease.",
            "Home blood-pressure monitoring can help identify white-coat or masked hypertension when used with a validated device.",
        ],
    },
    {
        "title": "Type 2 diabetes mellitus",
        "abstract": "Type 2 diabetes is a chronic metabolic disorder in which insulin resistance and progressive impairment of beta-cell function lead to elevated blood glucose. Management combines education, lifestyle measures, monitoring, and medicines when indicated.",
        "points": [
            "Glycated hemoglobin reflects average glycemia over approximately the preceding two to three months.",
            "Nutrition, physical activity, sleep, and weight management can improve metabolic health.",
            "Treatment plans should account for kidney function, cardiovascular disease, hypoglycemia risk, cost, and patient goals.",
            "Regular assessment includes eyes, feet, kidneys, blood pressure, lipids, and vaccination status.",
            "Hypoglycemia is a clinically important adverse effect of some glucose-lowering therapies and requires patient education.",
        ],
    },
    {
        "title": "Asthma and airway inflammation",
        "abstract": "Asthma is a heterogeneous respiratory disease characterized by variable respiratory symptoms and variable expiratory airflow limitation. Airway inflammation and bronchial hyperresponsiveness contribute to symptoms in many patients.",
        "points": [
            "Typical symptoms include wheeze, shortness of breath, chest tightness, and cough that vary over time.",
            "Diagnosis requires a compatible history together with objective evidence of variable expiratory airflow limitation when possible.",
            "Inhaled corticosteroids reduce airway inflammation and the risk of exacerbations for patients who need controller therapy.",
            "A written action plan explains how to recognize worsening symptoms and when to seek urgent care.",
            "Common triggers include viral infections, allergens, smoke, occupational exposures, and exercise in susceptible individuals.",
        ],
    },
    {
        "title": "Community-acquired pneumonia",
        "abstract": "Community-acquired pneumonia is an acute infection of the lung acquired outside a hospital or long-term care setting. Evaluation combines symptoms, examination, imaging when appropriate, and assessment of illness severity.",
        "points": [
            "Fever, cough, sputum production, dyspnea, pleuritic chest pain, and abnormal lung findings may occur.",
            "Severity assessment helps determine whether outpatient care, hospital admission, or critical care is appropriate.",
            "Antibiotic selection should consider likely pathogens, local resistance patterns, allergies, comorbidities, and recent antibiotic exposure.",
            "Supportive care includes assessment of oxygenation, hydration, and complications.",
            "Vaccination, hand hygiene, smoking cessation, and management of chronic disease can reduce respiratory infection risk.",
        ],
    },
    {
        "title": "Vaccination and immune protection",
        "abstract": "Vaccination exposes the immune system to an antigen or antigenic material in a controlled way so that immune memory can develop without the risks of the disease itself. Recommendations vary with age, risk, geography, and vaccine characteristics.",
        "points": [
            "Primary immune responses create memory cells that can support a faster response after later exposure.",
            "High vaccination coverage can reduce transmission and protect people who cannot receive particular vaccines.",
            "Mild local pain or fever can occur after vaccination and usually reflects immune activation.",
            "Contraindications and precautions are vaccine-specific and should be checked against current official guidance.",
            "Vaccine effectiveness and vaccine efficacy are related but are measured in different settings.",
        ],
    },
    {
        "title": "Antimicrobial resistance",
        "abstract": "Antimicrobial resistance occurs when microorganisms change or acquire mechanisms that reduce the effectiveness of medicines used against them. Resistance makes infections harder to treat and can increase illness, mortality, and healthcare costs.",
        "points": [
            "Selective pressure from antimicrobial exposure can favor survival of resistant organisms.",
            "Stewardship promotes the right drug, dose, route, and duration for a documented or suspected infection.",
            "Diagnostic testing can support targeted therapy and reduce unnecessary broad-spectrum treatment.",
            "Infection prevention, vaccination, environmental hygiene, and surveillance complement antibiotic stewardship.",
            "Resistance may spread through genetic mechanisms such as mutation and horizontal gene transfer.",
        ],
    },
    {
        "title": "Clinical trials and intervention evidence",
        "abstract": "A clinical trial prospectively evaluates an intervention in human participants using a predefined protocol. Randomization, allocation concealment, blinding when feasible, and complete follow-up can reduce bias.",
        "points": [
            "The research question, eligibility criteria, outcomes, sample size, and analysis plan should be specified before enrollment.",
            "Randomization balances measured and unmeasured prognostic factors on average between study groups.",
            "Intention-to-treat analysis preserves the benefits of the original randomized comparison.",
            "Absolute risk reduction and number needed to treat provide clinically interpretable measures of effect.",
            "External validity asks whether findings can reasonably be applied to patients outside the study population.",
        ],
    },
    {
        "title": "Diagnostic test performance",
        "abstract": "Diagnostic test evaluation describes how test results relate to a reference standard or clinical outcome. Sensitivity and specificity are properties of a test in a defined setting, while predictive values depend strongly on disease prevalence.",
        "points": [
            "Sensitivity is the proportion of people with the target condition who test positive.",
            "Specificity is the proportion of people without the target condition who test negative.",
            "A positive likelihood ratio indicates how much a positive result changes the odds of disease.",
            "A negative likelihood ratio indicates how much a negative result changes the odds of disease.",
            "Spectrum bias can occur when test performance is estimated in a study population unlike the population in practice.",
        ],
    },
    {
        "title": "Evidence-based medicine",
        "abstract": "Evidence-based medicine integrates the best available research evidence with clinical expertise and the values and circumstances of the patient. Evidence quality depends on study design, risk of bias, consistency, directness, and precision.",
        "points": [
            "A focused clinical question often separates the population, intervention, comparison, and outcome.",
            "Systematic reviews use explicit methods to identify, appraise, and synthesize relevant studies.",
            "Confidence intervals describe uncertainty around an estimated effect and are not the same as clinical importance.",
            "Relative effects can appear large even when absolute differences are small if baseline risk is low.",
            "Shared decision-making communicates benefits, harms, uncertainty, and patient priorities.",
        ],
    },
    {
        "title": "Adverse drug reactions and medication safety",
        "abstract": "An adverse drug reaction is a harmful and unintended response to a medicine at doses normally used for prevention, diagnosis, or treatment. Medication safety relies on careful prescribing, dispensing, administration, monitoring, and communication.",
        "points": [
            "A medication history should include prescription medicines, over-the-counter products, supplements, allergies, and previous reactions.",
            "Older age, kidney or liver impairment, polypharmacy, and drug interactions can increase medication-related risk.",
            "A temporal relationship is useful when evaluating causality but does not prove that a medicine caused an event.",
            "Medication reconciliation compares the medicines a person is taking with the medicines ordered during a transition of care.",
            "Serious or unexpected suspected reactions should be reported through the relevant pharmacovigilance system.",
        ],
    },
    {
        "title": "Heart failure and congestion",
        "abstract": "Heart failure is a clinical syndrome in which structural or functional cardiac abnormality causes symptoms and signs such as breathlessness, fatigue, or fluid retention. Classification can include the left-ventricular ejection fraction and the clinical stage.",
        "points": [
            "Dyspnea, orthopnea, edema, fatigue, elevated jugular venous pressure, and pulmonary crackles may be present.",
            "Echocardiography helps characterize cardiac structure and systolic function.",
            "Management addresses congestion, contributing disease, guideline-directed medicines, monitoring, and self-care education.",
            "Sudden worsening breathlessness, chest pain, confusion, or fainting requires urgent clinical assessment.",
            "Daily weight trends can help detect fluid changes when used as part of an agreed monitoring plan.",
        ],
    },
    {
        "title": "Chronic kidney disease and renal function",
        "abstract": "Chronic kidney disease is persistent abnormality of kidney structure or function that has health implications. Assessment commonly considers estimated glomerular filtration rate and markers of kidney damage such as albuminuria.",
        "points": [
            "A single abnormal kidney test does not by itself establish chronicity; persistence over time is important.",
            "Albuminuria can indicate kidney damage and is also associated with cardiovascular risk.",
            "Blood-pressure control and diabetes management can reduce the risk of kidney disease progression.",
            "Kidney function influences dosing and safety for many medicines.",
            "People with advanced disease may require coordinated planning for renal replacement options and symptom management.",
        ],
    },
]


def topic_document(topic):
    lines = [
        f"Title: {topic['title']}",
        "Document type: Educational clinical literature summary",
        "Scope: General medical and clinical concepts for language-model domain adaptation.",
        "",
        "Abstract",
        topic["abstract"],
        "",
        "Key concepts",
    ]
    lines.extend(f"{i}. {point}" for i, point in enumerate(topic["points"], 1))
    lines.extend([
        "",
        "Interpretation note",
        "This educational summary supports literature-processing experiments and is not a substitute for current clinical guidelines, professional judgment, or individualized medical care.",
    ])
    return "\n".join(lines)


def instruction_pairs():
    """Return 120 deterministic instruction/response pairs derived from TOPICS."""
    pairs = []
    for topic in TOPICS:
        title = topic["title"]
        abstract = topic["abstract"]
        points = topic["points"]
        prompts = [
            (f"What is {title}?", abstract),
            (f"Summarize the clinical importance of {title}.", abstract),
            (f"What is the main mechanism or concept described for {title}?", abstract),
            (f"List two important considerations related to {title}.", " ".join(points[:2])),
            (f"What should be assessed when studying {title}?", " ".join(points[1:3])),
            (f"What can increase risk or worsen outcomes in {title}?", points[3]),
            (f"What role does monitoring play in {title}?", points[4]),
            (f"Give an evidence-based explanation of {title} in plain language.", abstract + " " + points[0]),
            (f"What is one limitation or caution when interpreting information about {title}?", topic_document(topic).split("Interpretation note", 1)[1].strip()),
            (f"Provide a concise study note on {title}.", abstract + " " + " ".join(points[:3])),
        ]
        for instruction, response in prompts:
            pairs.append({
                "instruction": instruction,
                "response": response,
                "source": title,
            })
    return pairs
