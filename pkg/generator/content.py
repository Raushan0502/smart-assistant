"""
Synthetic case content for the test corpus.

Everything here is invented. Drug names are deliberately fictional (no real
product is named anywhere in this repository) and every patient, reporter and
event is made up, so no real patient data can leak into the corpus even by
accident.

Cases are written out explicitly rather than combinatorially generated. Random
recombination produces documents that are individually plausible but whose
ground truth is hard to state precisely -- and the ground truth is the point:
these labels are what the pipeline is scored against in Phase 6.

Detail level is varied on purpose. A real mailbox contains complete reports and
one-line messages that are missing three of the four ICSR elements, and the
classifier has to be right about both. ``expected_fields`` uses the literal
string ``Not stated`` wherever the source genuinely does not say, because
"unknown beats a guess" is a scored requirement, not a style preference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Categories, matching the four buckets in the assignment.
ICSR = "ICSR"
PQC = "PQC"
MI = "MI"
NOT_RELEVANT = "NOT_RELEVANT"

NOT_STATED = "Not stated"

# Fictional products. Suffixes echo real pharmaceutical naming conventions so
# the text reads naturally, but none of these are real medicines.
DRUGS = [
    "Cardiozan",
    "Neurolept-X",
    "Pulmoflex",
    "Dermacalm",
    "Glucostat",
    "Hepatrol",
    "Osteovance",
    "Renalix",
]


@dataclass
class Case:
    """One synthetic case and the ground truth it should produce.

    ``categories`` is a list because the buckets are multi-label: a reaction
    caused by a defective product is legitimately both ICSR and PQC.
    """

    case_id: str
    categories: list[str]
    subject: str
    body: str
    sender_name: str
    sender_email: str
    # What the extractor should produce -- always English, because that is the
    # working language of the review screen.
    expected_fields: dict[str, str] = field(default_factory=dict)
    # What the source document literally says. Set only for non-English cases,
    # where the document is in its own language and ``expected_fields`` is the
    # post-translation target. Defaults to ``expected_fields`` when unset.
    source_fields: dict[str, str] = field(default_factory=dict)
    # Set for cases that should also produce a PDF attachment.
    attachment_kind: str | None = None
    language: str = "en"
    notes: str = ""


def icsr_fields(
    age: str = NOT_STATED,
    sex: str = NOT_STATED,
    weight: str = NOT_STATED,
    history: str = NOT_STATED,
    reporter: str = NOT_STATED,
    reporter_role: str = NOT_STATED,
    country: str = NOT_STATED,
    product: str = NOT_STATED,
    dose: str = NOT_STATED,
    route: str = NOT_STATED,
    start_date: str = NOT_STATED,
    reaction: str = NOT_STATED,
    onset: str = NOT_STATED,
    outcome: str = NOT_STATED,
    serious: str = NOT_STATED,
) -> dict[str, str]:
    """Build an ICSR field map, defaulting every unmentioned field to
    ``Not stated``."""
    return {
        "patient_age": age,
        "patient_sex": sex,
        "patient_weight": weight,
        "patient_history": history,
        "reporter_name": reporter,
        "reporter_role": reporter_role,
        "reporter_country": country,
        "product_name": product,
        "product_dose": dose,
        "product_route": route,
        "product_start_date": start_date,
        "reaction": reaction,
        "reaction_onset": onset,
        "reaction_outcome": outcome,
        "seriousness": serious,
    }


# ---------------------------------------------------------------------------
# Safety reports (ICSR) -- all four elements present, in varying completeness.
# ---------------------------------------------------------------------------

ICSR_CASES: list[Case] = [
    Case(
        case_id="icsr_full_rash",
        categories=[ICSR],
        subject="Adverse reaction report - Cardiozan - severe rash",
        sender_name="Dr Amara Osei",
        sender_email="a.osei@northgate-clinic.example",
        body=(
            "Dear Safety Team,\n\n"
            "I am reporting an adverse event in one of my patients.\n\n"
            "The patient is a 54-year-old female, weight 68 kg, with a history of "
            "hypertension and type 2 diabetes. She began taking Cardiozan 20 mg "
            "orally once daily on 03 March 2026 for blood pressure control.\n\n"
            "On 11 March 2026, eight days after starting treatment, she developed a "
            "widespread erythematous rash across the trunk and upper arms, with "
            "associated pruritus. She attended our clinic the same day. Cardiozan was "
            "discontinued immediately and she was given oral antihistamines. The rash "
            "resolved over the following six days and she has now fully recovered.\n\n"
            "She was not hospitalised and there were no other serious sequelae.\n\n"
            "Please let me know if you require any further detail.\n\n"
            "Kind regards,\n"
            "Dr Amara Osei\n"
            "Northgate Clinic, Manchester, United Kingdom"
        ),
        expected_fields=icsr_fields(
            age="54 years",
            sex="Female",
            weight="68 kg",
            history="Hypertension, type 2 diabetes",
            reporter="Dr Amara Osei",
            reporter_role="Physician",
            country="United Kingdom",
            product="Cardiozan",
            dose="20 mg once daily",
            route="Oral",
            start_date="03 March 2026",
            reaction="Widespread erythematous rash with pruritus",
            onset="11 March 2026",
            outcome="Recovered",
            serious="Non-serious",
        ),
        attachment_kind="digital_form",
        notes="Complete report: all four ICSR elements plus most detail fields.",
    ),
    Case(
        case_id="icsr_full_hepatic",
        categories=[ICSR],
        subject="Serious adverse event - Hepatrol - hospitalisation",
        sender_name="Dr Liang Wei",
        sender_email="l.wei@riverside-hospital.example",
        body=(
            "To the Pharmacovigilance Department,\n\n"
            "I wish to report a serious adverse event.\n\n"
            "Patient: 67-year-old male, 81 kg, no significant prior liver disease.\n"
            "Product: Hepatrol 500 mg, taken orally twice daily, started 12 January 2026.\n\n"
            "On 02 February 2026 the patient presented with jaundice, dark urine and "
            "marked fatigue. Liver function tests were significantly deranged. He was "
            "admitted to Riverside Hospital on the same day and remained an inpatient "
            "for nine days. Hepatrol was withdrawn on admission.\n\n"
            "At the time of writing the patient is recovering but liver enzymes have "
            "not yet returned to baseline. I consider this event serious on the "
            "grounds of hospitalisation.\n\n"
            "Dr Liang Wei, Consultant Hepatologist\n"
            "Riverside Hospital, Singapore"
        ),
        expected_fields=icsr_fields(
            age="67 years",
            sex="Male",
            weight="81 kg",
            history="No significant prior liver disease",
            reporter="Dr Liang Wei",
            reporter_role="Physician (Consultant Hepatologist)",
            country="Singapore",
            product="Hepatrol",
            dose="500 mg twice daily",
            route="Oral",
            start_date="12 January 2026",
            reaction="Jaundice, dark urine, fatigue, deranged liver function tests",
            onset="02 February 2026",
            outcome="Recovering",
            serious="Serious - hospitalisation",
        ),
        attachment_kind="digital_form",
        notes="Serious case; seriousness must be picked up from hospitalisation.",
    ),
    Case(
        case_id="icsr_sparse_dizzy",
        categories=[ICSR],
        subject="my mother had a bad turn on her tablets",
        sender_name="Janet Whitfield",
        sender_email="j.whitfield88@mailbox.example",
        body=(
            "Hello,\n\n"
            "My mum has been taking Glucostat for her diabetes and since last week "
            "she keeps getting very dizzy and had a fall on Tuesday. She is 79. I am "
            "worried it is the tablets.\n\n"
            "Should she stop taking them?\n\n"
            "Janet"
        ),
        expected_fields=icsr_fields(
            age="79 years",
            sex="Female",
            reporter="Janet Whitfield",
            reporter_role="Consumer (patient's daughter)",
            product="Glucostat",
            reaction="Dizziness and a fall",
            onset="Approximately one week before the report",
            serious="Not stated",
        ),
        notes=(
            "Sparse consumer report. Four elements are present but loosely: patient "
            "(mother), reporter (daughter), drug, event. Most detail fields must "
            "come back Not stated. Also asks a question, but the reaction dominates."
        ),
    ),
    Case(
        case_id="icsr_sparse_breathless",
        categories=[ICSR],
        subject="Pulmoflex problem",
        sender_name="Tomas Berg",
        sender_email="tomas.berg@mailbox.example",
        body=(
            "I started using the Pulmoflex inhaler two weeks ago and I have been "
            "getting short of breath and my heart races after each dose. I am 41. "
            "I have stopped using it.\n\nTomas"
        ),
        expected_fields=icsr_fields(
            age="41 years",
            reporter="Tomas Berg",
            reporter_role="Consumer (patient)",
            product="Pulmoflex",
            route="Inhalation",
            reaction="Shortness of breath and palpitations",
            onset="Within two weeks of starting treatment",
            outcome="Not stated - product discontinued",
        ),
        notes="Self-reporting patient; sex and weight genuinely absent.",
    ),
    Case(
        case_id="icsr_and_pqc_combined",
        categories=[ICSR, PQC],
        subject="Discoloured Dermacalm cream caused a burning rash",
        sender_name="Priya Raman",
        sender_email="priya.raman@mailbox.example",
        body=(
            "Hello,\n\n"
            "I bought a tube of Dermacalm cream last month, batch number DC-4471-B. "
            "When I opened it the cream was brown rather than white and smelled "
            "strongly of chemicals. The seal under the cap was already broken.\n\n"
            "I used it anyway on my forearm and within an hour I had a burning "
            "sensation and the skin blistered. I am 33 years old. I saw my pharmacist "
            "who told me to stop using it and report it. The blisters are healing now.\n\n"
            "I have photographs of the tube and my arm if those are useful.\n\n"
            "Priya Raman, Leeds, UK"
        ),
        expected_fields=icsr_fields(
            age="33 years",
            reporter="Priya Raman",
            reporter_role="Consumer (patient)",
            country="United Kingdom",
            product="Dermacalm",
            route="Topical",
            reaction="Burning sensation and skin blistering",
            onset="Within one hour of application",
            outcome="Recovering",
            serious="Non-serious",
        ),
        attachment_kind="scanned_form",
        notes=(
            "Deliberately dual-label: a genuine adverse reaction AND a genuine "
            "product defect. A system that forces one label gets this wrong."
        ),
    ),
    Case(
        case_id="icsr_fatal_outcome",
        categories=[ICSR],
        subject="Fatal outcome following Renalix therapy - urgent",
        sender_name="Dr Sofia Marchetti",
        sender_email="s.marchetti@ospedale-centrale.example",
        body=(
            "Pharmacovigilance,\n\n"
            "I am reporting a fatal case. An 84-year-old male patient with chronic "
            "kidney disease stage 4 was started on Renalix 10 mg orally daily on "
            "18 December 2025.\n\n"
            "On 05 January 2026 he was admitted with severe hyperkalaemia and cardiac "
            "arrhythmia. Despite treatment he died on 07 January 2026. The treating "
            "team considers the event possibly related to Renalix.\n\n"
            "Dr Sofia Marchetti\n"
            "Ospedale Centrale, Milan, Italy"
        ),
        expected_fields=icsr_fields(
            age="84 years",
            sex="Male",
            history="Chronic kidney disease stage 4",
            reporter="Dr Sofia Marchetti",
            reporter_role="Physician",
            country="Italy",
            product="Renalix",
            dose="10 mg daily",
            route="Oral",
            start_date="18 December 2025",
            reaction="Severe hyperkalaemia and cardiac arrhythmia",
            onset="05 January 2026",
            outcome="Fatal",
            serious="Serious - death",
        ),
        attachment_kind="digital_form",
        notes="Fatal outcome; seriousness must be classified as death.",
    ),
    Case(
        case_id="icsr_pregnancy_context",
        categories=[ICSR],
        subject="Adverse event during pregnancy - Osteovance",
        sender_name="Nurse Practitioner Helen Achebe",
        sender_email="h.achebe@communitycare.example",
        body=(
            "Dear Sir or Madam,\n\n"
            "A 29-year-old female patient, 22 weeks pregnant, was taking Osteovance "
            "1000 mg orally once daily. She reports persistent severe nausea and "
            "vomiting beginning approximately four days after starting the product, "
            "leading to dehydration. She attended the day unit for intravenous fluids "
            "but was not admitted overnight.\n\n"
            "The product has been stopped and symptoms are improving.\n\n"
            "Helen Achebe, Nurse Practitioner\n"
            "Community Care Centre, Dublin, Ireland"
        ),
        expected_fields=icsr_fields(
            age="29 years",
            sex="Female",
            history="22 weeks pregnant",
            reporter="Helen Achebe",
            reporter_role="Nurse practitioner",
            country="Ireland",
            product="Osteovance",
            dose="1000 mg once daily",
            route="Oral",
            reaction="Severe nausea and vomiting leading to dehydration",
            onset="Approximately four days after starting treatment",
            outcome="Improving",
            serious="Non-serious",
        ),
        attachment_kind="digital_form",
        notes="Healthcare professional reporter who is not a physician.",
    ),
    Case(
        case_id="icsr_ambiguous_timing",
        categories=[ICSR],
        subject="Possible reaction - Neurolept-X",
        sender_name="Dr Kwame Mensah",
        sender_email="k.mensah@westside-practice.example",
        body=(
            "Reporting a possible adverse reaction.\n\n"
            "Patient is a 46-year-old male taking Neurolept-X. He describes tremor "
            "and difficulty sleeping. I cannot establish exactly when the symptoms "
            "began relative to starting the medication, and he is also taking two "
            "other products I have not been able to confirm.\n\n"
            "Reporting for completeness.\n\n"
            "Dr Kwame Mensah, Accra, Ghana"
        ),
        expected_fields=icsr_fields(
            age="46 years",
            sex="Male",
            reporter="Dr Kwame Mensah",
            reporter_role="Physician",
            country="Ghana",
            product="Neurolept-X",
            reaction="Tremor and insomnia",
            onset="Not stated",
            serious="Not stated",
        ),
        attachment_kind="digital_form",
        notes="Explicitly uncertain timing - onset must NOT be invented.",
    ),
]


# ---------------------------------------------------------------------------
# Quality complaints (PQC only) -- a defect, no patient reaction.
# ---------------------------------------------------------------------------

PQC_CASES: list[Case] = [
    Case(
        case_id="pqc_broken_seal",
        categories=[PQC],
        subject="Damaged packaging - Cardiozan batch CZ-2208-A",
        sender_name="Mark Delaney",
        sender_email="m.delaney@pharmacy-brookfield.example",
        body=(
            "Good morning,\n\n"
            "We received a delivery of Cardiozan 20 mg tablets this week, batch "
            "CZ-2208-A, expiry 09/2027. Three of the twelve cartons had crushed "
            "corners and the tamper-evident seals were broken on two of them.\n\n"
            "No product has been dispensed to any patient from these cartons and we "
            "have quarantined the stock. Please advise on return arrangements.\n\n"
            "Mark Delaney MRPharmS\n"
            "Brookfield Pharmacy, Cork, Ireland"
        ),
        expected_fields={
            "product_name": "Cardiozan",
            "batch_number": "CZ-2208-A",
            "defect_description": "Crushed cartons and broken tamper-evident seals",
            "photo_mentioned": "No",
        },
        notes="Pure PQC: explicitly states no patient exposure, so NOT an ICSR.",
    ),
    Case(
        case_id="pqc_discoloured_photo",
        categories=[PQC],
        subject="Glucostat tablets wrong colour - photo attached",
        sender_name="Ana Beltran",
        sender_email="ana.beltran@mailbox.example",
        body=(
            "Hello,\n\n"
            "I opened a new box of Glucostat this morning, lot GS-9931-C, and the "
            "tablets are pale yellow. My previous boxes were always white. Some of "
            "them also look chipped.\n\n"
            "I have not taken any of them. I have attached a photograph so you can "
            "see the difference.\n\n"
            "Ana Beltran, Valencia, Spain"
        ),
        expected_fields={
            "product_name": "Glucostat",
            "batch_number": "GS-9931-C",
            "defect_description": "Tablets pale yellow instead of white; some chipped",
            "photo_mentioned": "Yes",
        },
        attachment_kind="scanned_form",
        notes="PQC with a photo reference - photo_mentioned must be Yes.",
    ),
]


# ---------------------------------------------------------------------------
# Medical information requests (MI only) -- a question, no event, no defect.
# ---------------------------------------------------------------------------

MI_CASES: list[Case] = [
    Case(
        case_id="mi_dosing_renal",
        categories=[MI],
        subject="Dosing query - Renalix in renal impairment",
        sender_name="Dr Yuki Tanaka",
        sender_email="y.tanaka@sakura-medical.example",
        body=(
            "Dear Medical Information,\n\n"
            "Could you advise on the recommended dose adjustment for Renalix in "
            "patients with an eGFR between 30 and 45 mL/min? The prescribing "
            "information gives guidance below 30 but I cannot find a recommendation "
            "for this range.\n\n"
            "No patient has come to any harm - this is a prospective question before "
            "I prescribe.\n\n"
            "Dr Yuki Tanaka, Osaka, Japan"
        ),
        expected_fields={
            "question": (
                "What is the recommended Renalix dose adjustment for patients with "
                "eGFR 30-45 mL/min?"
            ),
            "product_or_topic": "Renalix - dosing in renal impairment",
        },
        notes="Explicitly states no harm occurred - must not be classed as ICSR.",
    ),
    Case(
        case_id="mi_interaction",
        categories=[MI],
        subject="Interaction question about Neurolept-X",
        sender_name="Fatima Al-Rashid",
        sender_email="f.alrashid@cityhealth.example",
        body=(
            "Hello,\n\n"
            "Is it safe to take Neurolept-X at the same time as a standard "
            "over-the-counter antihistamine? A patient asked me today and I would "
            "like to give an accurate answer. Also, should it be taken with food?\n\n"
            "Thank you,\n"
            "Fatima Al-Rashid, Pharmacist\n"
            "City Health Pharmacy, Dubai, UAE"
        ),
        expected_fields={
            "question": (
                "Can Neurolept-X be taken with over-the-counter antihistamines, and "
                "should it be taken with food?"
            ),
            "product_or_topic": "Neurolept-X - drug interactions and administration",
        },
        notes="Two questions in one message; both should be captured.",
    ),
]


# ---------------------------------------------------------------------------
# Not relevant -- marketing and internal admin.
# ---------------------------------------------------------------------------

IRRELEVANT_CASES: list[Case] = [
    Case(
        case_id="irrelevant_marketing",
        categories=[NOT_RELEVANT],
        subject="Boost your clinic's revenue by 40% this quarter!",
        sender_name="GrowthPartners Marketing",
        sender_email="offers@growthpartners.example",
        body=(
            "Hi there,\n\n"
            "Is your practice leaving money on the table? Our proven patient "
            "acquisition system has helped over 3,000 clinics increase quarterly "
            "revenue by an average of 40%.\n\n"
            "Book a free 15-minute discovery call this week and receive our "
            "'Practice Growth Playbook' at no charge.\n\n"
            "Unsubscribe | Manage preferences\n"
            "GrowthPartners Marketing Ltd"
        ),
        notes="Obvious marketing spam - no product, no patient, no question.",
    ),
    Case(
        case_id="irrelevant_internal",
        categories=[NOT_RELEVANT],
        subject="Reminder: office closure for the bank holiday",
        sender_name="Facilities Team",
        sender_email="facilities@internal.example",
        body=(
            "All,\n\n"
            "A reminder that the office will be closed on Monday 25 May for the bank "
            "holiday. Building access cards will not work between 18:00 Friday and "
            "07:00 Tuesday.\n\n"
            "Please make sure any samples in the third-floor fridge are logged before "
            "you leave on Friday.\n\n"
            "Facilities Team"
        ),
        notes=(
            "Internal admin chatter. Mentions 'samples' and a fridge, which is a "
            "deliberate near-miss for a keyword-matching classifier."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Non-English cases -- case-relevant content, not English.
# ---------------------------------------------------------------------------

NON_ENGLISH_CASES: list[Case] = [
    Case(
        case_id="icsr_german",
        categories=[ICSR],
        language="de",
        subject="Meldung einer Nebenwirkung - Cardiozan",
        sender_name="Dr Markus Hoffmann",
        sender_email="m.hoffmann@praxis-hoffmann.example",
        body=(
            "Sehr geehrte Damen und Herren,\n\n"
            "hiermit melde ich eine unerwuenschte Arzneimittelwirkung.\n\n"
            "Die Patientin ist 61 Jahre alt, weiblich, 74 kg. Sie nimmt seit dem "
            "05. Februar 2026 Cardiozan 20 mg einmal taeglich oral ein.\n\n"
            "Am 19. Februar 2026 entwickelte sie starke Kopfschmerzen und "
            "Schwindel. Sie wurde nicht stationaer aufgenommen. Das Praeparat "
            "wurde abgesetzt und die Beschwerden sind zurueckgegangen.\n\n"
            "Mit freundlichen Gruessen\n"
            "Dr Markus Hoffmann, Praxis Hoffmann, Muenchen, Deutschland"
        ),
        expected_fields=icsr_fields(
            age="61 years",
            sex="Female",
            weight="74 kg",
            reporter="Dr Markus Hoffmann",
            reporter_role="Physician",
            country="Germany",
            product="Cardiozan",
            dose="20 mg once daily",
            route="Oral",
            start_date="05 February 2026",
            reaction="Severe headache and dizziness",
            onset="19 February 2026",
            outcome="Recovering",
            serious="Non-serious",
        ),
        source_fields={
            "patient_age": "61 Jahre",
            "patient_sex": "Weiblich",
            "patient_weight": "74 kg",
            "patient_history": "Bluthochdruck seit vielen Jahren bekannt",
            "reporter_name": "Dr Markus Hoffmann",
            "reporter_role": "Arzt in eigener Praxis",
            "reporter_country": "Deutschland",
            "product_name": "Cardiozan",
            "product_dose": "20 mg einmal taeglich",
            "product_route": "Oral zum Einnehmen",
            "product_start_date": "05. Februar 2026",
            "reaction": "Starke Kopfschmerzen und Schwindel nach der Einnahme",
            "reaction_onset": "19. Februar 2026",
            "reaction_outcome": "Beschwerden sind zurueckgegangen",
            "seriousness": "Nicht schwerwiegend, keine Aufnahme in das Krankenhaus",
        },
        attachment_kind="non_english_pdf",
        notes="German. Fields are expected in English after translation.",
    ),
    Case(
        case_id="icsr_spanish",
        categories=[ICSR],
        language="es",
        subject="Notificacion de reaccion adversa - Pulmoflex",
        sender_name="Dra Carmen Ruiz",
        sender_email="c.ruiz@clinica-delmar.example",
        body=(
            "Estimados senores:\n\n"
            "Notifico una reaccion adversa.\n\n"
            "Paciente varon de 38 anos, 79 kg, sin antecedentes relevantes. Utiliza "
            "Pulmoflex por via inhalatoria desde el 10 de enero de 2026.\n\n"
            "El 24 de enero de 2026 presento urticaria generalizada e hinchazon de "
            "los labios. Acudio a urgencias y fue tratado con corticoides. No "
            "requirio ingreso hospitalario. El paciente se ha recuperado por "
            "completo.\n\n"
            "Atentamente,\n"
            "Dra Carmen Ruiz, Clinica del Mar, Barcelona, Espana"
        ),
        expected_fields=icsr_fields(
            age="38 years",
            sex="Male",
            weight="79 kg",
            history="No relevant history",
            reporter="Dra Carmen Ruiz",
            reporter_role="Physician",
            country="Spain",
            product="Pulmoflex",
            route="Inhalation",
            start_date="10 January 2026",
            reaction="Generalised urticaria and lip swelling",
            onset="24 January 2026",
            outcome="Recovered",
            serious="Non-serious",
        ),
        source_fields={
            "patient_age": "38 anos",
            "patient_sex": "Varon",
            "patient_weight": "79 kg",
            "patient_history": "Sin antecedentes relevantes conocidos",
            "reporter_name": "Dra Carmen Ruiz",
            "reporter_role": "Medico de la clinica",
            "reporter_country": "Espana",
            "product_name": "Pulmoflex",
            "product_dose": "Dos inhalaciones al dia",
            "product_route": "Via inhalatoria",
            "product_start_date": "10 de enero de 2026",
            "reaction": "Urticaria generalizada e hinchazon de los labios",
            "reaction_onset": "24 de enero de 2026",
            "reaction_outcome": "El paciente se ha recuperado por completo",
            "seriousness": "No grave, no requirio ingreso hospitalario",
        },
        attachment_kind="non_english_pdf",
        notes="Spanish. Fields are expected in English after translation.",
    ),
]


# ---------------------------------------------------------------------------
# Literature articles -- for the article PDF flavour and the bonus extension.
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """A synthetic journal article describing zero or more patient cases."""

    article_id: str
    title: str
    authors: str
    journal: str
    abstract: str
    sections: list[tuple[str, str]]
    # How many identifiable patient cases the article actually contains.
    case_count: int
    reportable: bool
    notes: str = ""


ARTICLES: list[Article] = [
    Article(
        article_id="art_single_case",
        title="Acute hepatic injury following Hepatrol therapy: a case report",
        authors="M. Okonkwo, R. Silva, T. Nakamura",
        journal="Journal of Clinical Pharmacovigilance, 2026;14(2):88-91",
        abstract=(
            "We describe a case of acute hepatic injury in a patient receiving "
            "Hepatrol for metabolic syndrome. The temporal relationship and "
            "resolution on withdrawal suggest a probable causal association."
        ),
        sections=[
            (
                "Introduction",
                "Drug-induced liver injury remains an important cause of acute "
                "hepatitis. Hepatrol was introduced for metabolic syndrome and "
                "hepatic adverse effects have been described only rarely.",
            ),
            (
                "Case Presentation",
                "A 58-year-old woman with a body weight of 72 kg and a history of "
                "hypothyroidism was commenced on Hepatrol 500 mg orally twice daily "
                "in November 2025. Six weeks later she presented with malaise, "
                "nausea and scleral icterus. Alanine aminotransferase was 512 U/L "
                "and total bilirubin 78 umol/L. Viral serology was negative. "
                "Hepatrol was withdrawn and liver enzymes normalised over ten weeks. "
                "The patient recovered fully without sequelae.",
            ),
            (
                "Discussion",
                "The temporal association and dechallenge response support a "
                "probable causal relationship. Clinicians should consider baseline "
                "and periodic liver function monitoring.",
            ),
            (
                "References",
                "1. Silva R, et al. Hepatic safety of metabolic agents. 2024. "
                "2. Nakamura T. Drug-induced liver injury: a review. 2023. "
                "3. Okonkwo M. Monitoring strategies in hepatology. 2025.",
            ),
        ],
        case_count=1,
        reportable=True,
        notes="One identifiable case. References must be ignored by extraction.",
    ),
    Article(
        article_id="art_two_cases",
        title="Two cases of severe cutaneous reaction associated with Dermacalm",
        authors="P. Lindqvist, A. Haddad",
        journal="European Dermatology Reports, 2026;9(1):12-16",
        abstract=(
            "Two unrelated patients developed severe cutaneous adverse reactions "
            "following topical Dermacalm use. Both recovered after withdrawal."
        ),
        sections=[
            (
                "Case 1",
                "A 24-year-old man applied Dermacalm twice daily to the forearms for "
                "eczema from 04 April 2026. After nine days he developed painful "
                "vesicles and desquamation. Treatment was withdrawn and topical "
                "corticosteroids commenced. He recovered within three weeks.",
            ),
            (
                "Case 2",
                "A 45-year-old woman used Dermacalm for contact dermatitis from "
                "12 April 2026. Within five days she developed a bullous eruption "
                "requiring hospital admission for four days. She recovered fully.",
            ),
            (
                "Discussion",
                "Severe cutaneous reactions to this agent appear uncommon but "
                "clinically significant. Both cases showed positive dechallenge.",
            ),
            (
                "References",
                "1. Haddad A. Cutaneous drug reactions. 2022. "
                "2. Lindqvist P, et al. Topical agent safety. 2025.",
            ),
        ],
        case_count=2,
        reportable=True,
        notes="Two distinct cases in one article - must be split, not merged.",
    ),
    Article(
        article_id="art_review_no_case",
        title="Mechanisms of drug-induced QT prolongation: a narrative review",
        authors="S. Fontaine, D. Oyelaran",
        journal="Cardiovascular Pharmacology Review, 2026;31(4):200-214",
        abstract=(
            "This review summarises current understanding of the mechanisms "
            "underlying drug-induced QT interval prolongation. No individual "
            "patient data are presented."
        ),
        sections=[
            (
                "Introduction",
                "QT prolongation is a well-recognised class effect of several drug "
                "families. This review synthesises the published literature.",
            ),
            (
                "Mechanisms",
                "Blockade of the rapid delayed rectifier potassium current is the "
                "predominant mechanism. Population studies indicate an incidence of "
                "between 0.1% and 3% depending on agent and comorbidity.",
            ),
            (
                "Conclusion",
                "Risk stratification before initiating therapy remains the most "
                "effective mitigation. No new patient cases are reported here.",
            ),
            (
                "References",
                "1. Oyelaran D. Cardiac electrophysiology. 2021. "
                "2. Fontaine S, et al. QT risk in practice. 2024.",
            ),
        ],
        case_count=0,
        reportable=False,
        notes="Review article with NO identifiable case - the key negative example.",
    ),
    Article(
        article_id="art_aggregate_only",
        title="Post-marketing surveillance of Glucostat: a cohort analysis",
        authors="R. Mbeki, J. Halvorsen, L. Petrova",
        journal="Pharmacoepidemiology Quarterly, 2026;18(3):145-159",
        abstract=(
            "We analysed aggregate safety data from 4,182 patients treated with "
            "Glucostat. Individual case narratives are not presented."
        ),
        sections=[
            (
                "Methods",
                "A retrospective cohort of 4,182 patients was assembled from three "
                "national registries covering 2022 to 2025.",
            ),
            (
                "Results",
                "Hypoglycaemic episodes were recorded in 6.4% of patients and "
                "gastrointestinal disturbance in 11.2%. No patient-level "
                "identifiers were available to the investigators.",
            ),
            (
                "Conclusion",
                "The aggregate safety profile is consistent with the known label. "
                "No individual case reports arise from this analysis.",
            ),
            (
                "References",
                "1. Petrova L. Registry methods. 2023. "
                "2. Mbeki R, et al. Cohort safety analysis. 2024.",
            ),
        ],
        case_count=0,
        reportable=False,
        notes=(
            "Aggregate data only. Adverse events ARE described, but no identifiable "
            "individual - a hard negative for a naive keyword classifier."
        ),
    ),
    Article(
        article_id="art_case_in_discussion",
        title="Renalix and electrolyte disturbance: clinical implications",
        authors="H. Nasser, K. Bergstrom",
        journal="Nephrology Practice, 2026;22(6):301-308",
        abstract=(
            "We discuss electrolyte disturbance associated with Renalix and "
            "illustrate the risk with an index case."
        ),
        sections=[
            (
                "Background",
                "Potassium-sparing mechanisms carry an inherent risk of "
                "hyperkalaemia, particularly in impaired renal function.",
            ),
            (
                "Illustrative Case",
                "A 71-year-old man with stage 3 chronic kidney disease received "
                "Renalix 10 mg daily from 02 February 2026. On 20 February 2026 "
                "serum potassium was 6.8 mmol/L with ECG changes. He was admitted "
                "for two days, treated medically, and recovered. Renalix was "
                "permanently discontinued.",
            ),
            (
                "Discussion",
                "Monitoring within two weeks of initiation is advisable in this "
                "population.",
            ),
            (
                "References",
                "1. Bergstrom K. Electrolyte management. 2023. "
                "2. Nasser H, et al. Renal safety. 2025.",
            ),
        ],
        case_count=1,
        reportable=True,
        notes="Case is buried inside a discussion piece, not a dedicated report.",
    ),
]


def all_email_cases() -> list[Case]:
    """Every case that should be rendered as an email in the corpus."""
    return (
        ICSR_CASES
        + NON_ENGLISH_CASES
        + PQC_CASES
        + MI_CASES
        + IRRELEVANT_CASES
    )
