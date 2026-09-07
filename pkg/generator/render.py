"""
PDF renderers for the four attachment flavours the assignment names.

Each renderer targets a different extraction problem, because the point of the
corpus is to make the ingestion pipeline prove it handles all four:

    digital form  -> a real text layer, labelled fields, and a genuine table
    article       -> two-column flow, with references that must be ignored
    scanned       -> no text layer at all; pixels only, handwriting, skewed
    non-English   -> a text layer in German or Spanish

Tables are emitted as real ReportLab ``Table`` flowables rather than aligned
text. That matters: it produces ruling lines and cell geometry in the PDF, which
is what lets the extractor recover rows and columns structurally instead of
guessing from whitespace. A table faked with spaces would make the corpus
easier than reality and the pipeline would look better than it is.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .content import NOT_STATED, Article, Case

# Windows ships these; both are genuine handwriting faces.
HANDWRITING_FONTS = [
    r"C:\Windows\Fonts\Inkfree.ttf",
    r"C:\Windows\Fonts\LHANDW.TTF",
]
FORM_FONT = r"C:\Windows\Fonts\arial.ttf"

# Lab panels attached to the fuller reports. Values are invented but the
# reference ranges are realistic, so a reader can tell which rows are abnormal.
LAB_TABLE_HEADER = ["Test", "Result", "Unit", "Reference range", "Flag"]
LAB_PANELS: dict[str, list[list[str]]] = {
    "hepatic": [
        ["Alanine aminotransferase", "512", "U/L", "10 - 40", "HIGH"],
        ["Aspartate aminotransferase", "388", "U/L", "10 - 35", "HIGH"],
        ["Total bilirubin", "78", "umol/L", "3 - 21", "HIGH"],
        ["Alkaline phosphatase", "142", "U/L", "30 - 130", "HIGH"],
        ["Albumin", "38", "g/L", "35 - 50", "Normal"],
    ],
    "renal": [
        ["Serum potassium", "6.8", "mmol/L", "3.5 - 5.0", "HIGH"],
        ["Serum sodium", "134", "mmol/L", "135 - 145", "LOW"],
        ["Creatinine", "212", "umol/L", "60 - 110", "HIGH"],
        ["eGFR", "28", "mL/min", "> 90", "LOW"],
        ["Urea", "18.4", "mmol/L", "2.5 - 7.8", "HIGH"],
    ],
    "general": [
        ["Haemoglobin", "132", "g/L", "120 - 160", "Normal"],
        ["White cell count", "11.8", "x10^9/L", "4.0 - 11.0", "HIGH"],
        ["Eosinophils", "0.9", "x10^9/L", "0.0 - 0.4", "HIGH"],
        ["C-reactive protein", "24", "mg/L", "< 5", "HIGH"],
        ["Platelets", "268", "x10^9/L", "150 - 400", "Normal"],
    ],
}


def _panel_for(case: Case) -> str:
    """Pick the lab panel that fits the case's reaction."""
    reaction = case.expected_fields.get("reaction", "").lower()
    if "hepat" in reaction or "jaundice" in reaction or "liver" in reaction:
        return "hepatic"
    if "potassium" in reaction or "hyperkal" in reaction or "renal" in reaction:
        return "renal"
    return "general"


def _styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles shared by the text-layer renderers."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FormTitle", parent=base["Heading1"], fontSize=15, spaceAfter=2 * mm
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            textColor=colors.HexColor("#1a3a5c"),
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=9.5, leading=13
        ),
        "justified": ParagraphStyle(
            "Justified",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#555555"),
        ),
    }


def _field_table(rows: list[tuple[str, str]]) -> Table:
    """Render label/value pairs as a bordered two-column table.

    Real forms put fields in ruled boxes, and keeping that structure means the
    extractor has to keep labels aligned with their values rather than reading
    a soup of adjacent strings.
    """
    data = [[label, value or NOT_STATED] for label, value in rows]
    table = Table(data, colWidths=[52 * mm, 108 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _lab_table(panel: str) -> Table:
    """Render a lab-results panel as a real table with a header row."""
    data = [LAB_TABLE_HEADER] + LAB_PANELS[panel]
    table = Table(data, colWidths=[52 * mm, 22 * mm, 22 * mm, 38 * mm, 24 * mm])
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_index, row in enumerate(LAB_PANELS[panel], start=1):
        if row[4] in {"HIGH", "LOW"}:
            style.append(
                ("TEXTCOLOR", (4, row_index), (4, row_index), colors.HexColor("#b00020"))
            )
    table.setStyle(TableStyle(style))
    return table


def render_digital_form(case: Case, path: Path, language: str = "en") -> None:
    """Render a filled-in adverse-event report form with a real text layer."""
    styles = _styles()
    labels = FORM_LABELS.get(language, FORM_LABELS["en"])
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=labels["title"],
        author=case.sender_name,
        subject=case.subject,
    )
    fields = case.expected_fields
    story = [
        Paragraph(labels["title"], styles["title"]),
        Paragraph(labels["subtitle"], styles["small"]),
        Spacer(1, 4 * mm),
        Paragraph(labels["patient"], styles["heading"]),
        _field_table(
            [
                (labels["age"], fields.get("patient_age", NOT_STATED)),
                (labels["sex"], fields.get("patient_sex", NOT_STATED)),
                (labels["weight"], fields.get("patient_weight", NOT_STATED)),
                (labels["history"], fields.get("patient_history", NOT_STATED)),
            ]
        ),
        Paragraph(labels["reporter"], styles["heading"]),
        _field_table(
            [
                (labels["reporter_name"], fields.get("reporter_name", NOT_STATED)),
                (labels["reporter_role"], fields.get("reporter_role", NOT_STATED)),
                (labels["country"], fields.get("reporter_country", NOT_STATED)),
            ]
        ),
        Paragraph(labels["product"], styles["heading"]),
        _field_table(
            [
                (labels["product_name"], fields.get("product_name", NOT_STATED)),
                (labels["dose"], fields.get("product_dose", NOT_STATED)),
                (labels["route"], fields.get("product_route", NOT_STATED)),
                (labels["start_date"], fields.get("product_start_date", NOT_STATED)),
            ]
        ),
        Paragraph(labels["reaction"], styles["heading"]),
        _field_table(
            [
                (labels["reaction_desc"], fields.get("reaction", NOT_STATED)),
                (labels["onset"], fields.get("reaction_onset", NOT_STATED)),
                (labels["outcome"], fields.get("reaction_outcome", NOT_STATED)),
                (labels["serious"], fields.get("seriousness", NOT_STATED)),
            ]
        ),
        Paragraph(labels["labs"], styles["heading"]),
        _lab_table(_panel_for(case)),
        Spacer(1, 4 * mm),
        Paragraph(labels["footer"], styles["small"]),
    ]
    doc.build(story)


def render_article(article: Article, path: Path) -> None:
    """Render a journal article across two columns on every page."""
    styles = _styles()
    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=article.title,
        author=article.authors,
    )
    full_width = doc.width
    column_width = (full_width - 6 * mm) / 2
    # The masthead spans both columns; the body flows down one column and into
    # the next, which is the layout the assignment calls out as a hazard.
    header_frame = Frame(
        doc.leftMargin,
        doc.bottomMargin + doc.height - 52 * mm,
        full_width,
        52 * mm,
        id="header",
    )
    left = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        column_width,
        doc.height - 54 * mm,
        id="left",
    )
    right = Frame(
        doc.leftMargin + column_width + 6 * mm,
        doc.bottomMargin,
        column_width,
        doc.height - 54 * mm,
        id="right",
    )
    later_left = Frame(doc.leftMargin, doc.bottomMargin, column_width, doc.height)
    later_right = Frame(
        doc.leftMargin + column_width + 6 * mm,
        doc.bottomMargin,
        column_width,
        doc.height,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="first", frames=[header_frame, left, right]),
            PageTemplate(id="later", frames=[later_left, later_right]),
        ]
    )

    story = [
        Paragraph(article.journal, styles["small"]),
        Paragraph(article.title, styles["title"]),
        Paragraph(article.authors, styles["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>Abstract.</b> {article.abstract}", styles["justified"]),
    ]
    for heading, text in article.sections:
        story.append(Paragraph(heading, styles["heading"]))
        story.append(Paragraph(text, styles["justified"]))
    doc.build(story)


FORM_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Adverse Event Report Form",
        "subtitle": "SYNTHETIC TEST DOCUMENT - no real patient data",
        "patient": "1. Patient details",
        "age": "Age",
        "sex": "Sex",
        "weight": "Weight",
        "history": "Relevant medical history",
        "reporter": "2. Reporter details",
        "reporter_name": "Reporter name",
        "reporter_role": "Role / qualification",
        "country": "Country",
        "product": "3. Suspect product",
        "product_name": "Product name",
        "dose": "Dose",
        "route": "Route of administration",
        "start_date": "Therapy start date",
        "reaction": "4. Reaction",
        "reaction_desc": "Description of reaction",
        "onset": "Date of onset",
        "outcome": "Outcome",
        "serious": "Seriousness",
        "labs": "5. Relevant laboratory results",
        "footer": "Form SYN-AE-01 | Synthetic document generated for testing purposes only.",
    },
    "de": {
        "title": "Meldeformular fuer unerwuenschte Arzneimittelwirkungen",
        "subtitle": "SYNTHETISCHES TESTDOKUMENT - keine echten Patientendaten",
        "patient": "1. Patientendaten",
        "age": "Alter",
        "sex": "Geschlecht",
        "weight": "Gewicht",
        "history": "Relevante Vorgeschichte",
        "reporter": "2. Angaben zum Melder",
        "reporter_name": "Name des Melders",
        "reporter_role": "Funktion / Qualifikation",
        "country": "Land",
        "product": "3. Verdaechtiges Arzneimittel",
        "product_name": "Arzneimittelname",
        "dose": "Dosierung",
        "route": "Art der Anwendung",
        "start_date": "Therapiebeginn",
        "reaction": "4. Reaktion",
        "reaction_desc": "Beschreibung der Reaktion",
        "onset": "Datum des Auftretens",
        "outcome": "Ausgang",
        "serious": "Schweregrad",
        "labs": "5. Relevante Laborwerte",
        "footer": "Formular SYN-AE-01 | Synthetisches Dokument, nur zu Testzwecken.",
    },
    "es": {
        "title": "Formulario de Notificacion de Reaccion Adversa",
        "subtitle": "DOCUMENTO SINTETICO DE PRUEBA - sin datos reales de pacientes",
        "patient": "1. Datos del paciente",
        "age": "Edad",
        "sex": "Sexo",
        "weight": "Peso",
        "history": "Antecedentes relevantes",
        "reporter": "2. Datos del notificador",
        "reporter_name": "Nombre del notificador",
        "reporter_role": "Funcion / cualificacion",
        "country": "Pais",
        "product": "3. Medicamento sospechoso",
        "product_name": "Nombre del medicamento",
        "dose": "Dosis",
        "route": "Via de administracion",
        "start_date": "Inicio del tratamiento",
        "reaction": "4. Reaccion",
        "reaction_desc": "Descripcion de la reaccion",
        "onset": "Fecha de inicio",
        "outcome": "Desenlace",
        "serious": "Gravedad",
        "labs": "5. Resultados de laboratorio relevantes",
        "footer": "Formulario SYN-AE-01 | Documento sintetico solo para pruebas.",
    },
}

# Values written onto the scanned forms in handwriting. Kept separate from the
# printed labels so the two can be styled independently.
SCANNED_LAYOUT: list[tuple[str, str]] = [
    ("Patient age", "patient_age"),
    ("Sex", "patient_sex"),
    ("Product", "product_name"),
    ("Route", "product_route"),
    ("Reaction", "reaction"),
    ("Onset", "reaction_onset"),
    ("Outcome", "reaction_outcome"),
    ("Reporter", "reporter_name"),
]


def render_scanned_form(case: Case, path: Path, seed: int = 0) -> None:
    """Render a hand-filled form as an image-only PDF with scan artefacts.

    There is deliberately **no text layer**: the page is drawn as pixels and
    embedded as an image, so anything that reads it has to go through OCR or a
    vision model. Skew, blur, speckle and a grey cast are applied because a
    pristine render would let a naive pipeline appear to handle scans when it
    would fail on a real one.
    """
    rng = random.Random(seed)
    width, height = 1240, 1754  # A4 at ~150 dpi
    page = Image.new("RGB", (width, height), (252, 251, 247))
    draw = ImageDraw.Draw(page)

    label_font = ImageFont.truetype(FORM_FONT, 26)
    title_font = ImageFont.truetype(FORM_FONT, 34)
    hand_path = HANDWRITING_FONTS[seed % len(HANDWRITING_FONTS)]
    hand_font = ImageFont.truetype(hand_path, 32)

    draw.text((90, 80), "ADVERSE EVENT REPORT - INTAKE FORM", font=title_font, fill=(20, 20, 20))
    draw.text((90, 128), "SYNTHETIC TEST DOCUMENT - no real patient data", font=ImageFont.truetype(FORM_FONT, 20), fill=(110, 110, 110))
    draw.line([(90, 165), (width - 90, 165)], fill=(60, 60, 60), width=3)

    y = 230
    for label, key in SCANNED_LAYOUT:
        draw.text((100, y), f"{label}:", font=label_font, fill=(40, 40, 40))
        draw.line([(430, y + 38), (width - 110, y + 38)], fill=(150, 150, 150), width=2)
        value = case.expected_fields.get(key, NOT_STATED)
        if value and value != NOT_STATED:
            # Jitter each handwritten value so lines are not machine-straight.
            draw.text(
                (445 + rng.randint(-6, 6), y - 4 + rng.randint(-5, 5)),
                value[:58],
                font=hand_font,
                fill=(28, 40, 105),
            )
        y += 118

    draw.text((100, y + 30), "Signature:", font=label_font, fill=(40, 40, 40))
    draw.text((445, y + 12), case.sender_name, font=hand_font, fill=(28, 40, 105))

    page = _apply_scan_artefacts(page, rng)
    page.save(str(path), "PDF", resolution=150.0)


def _apply_scan_artefacts(page: Image.Image, rng: random.Random) -> Image.Image:
    """Degrade a clean render so it looks like it came off a real scanner."""
    page = page.rotate(
        rng.uniform(-1.4, 1.4), resample=Image.BICUBIC, expand=False, fillcolor=(252, 251, 247)
    )
    page = page.filter(ImageFilter.GaussianBlur(radius=0.6))
    # Speckle: scattered dark pixels, the usual dust-and-sensor-noise signature.
    pixels = page.load()
    width, height = page.size
    for _ in range(int(width * height * 0.0009)):
        x, y = rng.randrange(width), rng.randrange(height)
        shade = rng.randint(90, 190)
        pixels[x, y] = (shade, shade, shade)
    # Slight grey cast, as if photocopied.
    return Image.blend(page, Image.new("RGB", page.size, (238, 236, 230)), 0.10)
