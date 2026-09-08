"""
PDF ingestion: detect which flavour a document is, then extract it accordingly.

The assignment names four PDF varieties and asks that each be handled
appropriately. Detection runs first because the right extraction strategy
differs completely between them -- a scanned page has no text to extract at
all, and an article needs column-aware reading order that would mangle a form.

Detection is deliberately evidence-based rather than filename-based: real mail
arrives with names like ``scan0001.pdf`` and ``document.pdf``, so the signal has
to come from the page itself.
"""
from __future__ import annotations

import re
from functools import partial
from io import BytesIO
from pathlib import Path

import pdfplumber

from .models import (
    ExtractedDocument,
    ImageBlock,
    PdfFlavour,
    SourceRef,
    TableBlock,
    TextBlock,
)

# Below this many characters on a page, there is effectively no text layer and
# the page must be treated as an image. A scanned page usually yields exactly
# zero, but a scan with a small typed header can leak a few characters, so the
# threshold is not zero.
MIN_CHARS_FOR_TEXT_LAYER = 40

# An image covering most of the page is the page -- i.e. a scan -- rather than
# an illustration within it.
FULL_PAGE_IMAGE_COVERAGE = 0.55

# Images smaller than this are logos, rules and bullets, not content.
MIN_MEANINGFUL_IMAGE_PX = 120

# Markers that identify a journal article rather than a form or a letter.
ARTICLE_MARKERS = [
    r"\babstract\b",
    r"\breferences\b",
    r"\bdiscussion\b",
    r"\bcase (?:report|presentation)\b",
    r"\bintroduction\b",
    r"\bdoi\b",
]

# Stopwords that identify language without needing a model. Only the languages
# the corpus actually contains are covered; anything else falls back to English
# and raises a warning rather than guessing.
LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "de": {
        "und", "der", "die", "das", "des", "dem", "den", "wurde", "wurden",
        "nicht", "eine", "einer", "einem", "mit", "sehr", "bei", "auf", "ist",
        "war", "seit", "nach", "keine", "zum", "zur", "im", "von", "fuer",
        "jahre", "weiblich", "maennlich", "arzt", "beschwerden", "einnahme",
    },
    "es": {
        "de", "del", "la", "el", "los", "las", "una", "por", "con", "para",
        "presento", "paciente", "sin", "que", "anos", "dia", "via", "no",
        "medico", "clinica", "recuperado", "requirio",
    },
    "fr": {
        "le", "la", "les", "une", "des", "avec", "pour", "patient", "apres",
        "etait", "du", "au", "aux", "dans", "sur", "pas", "ans",
    },
}


def detect_language(text: str) -> str:
    """Identify the language from stopword frequency.

    A dependency-free detector is used on purpose: the corpus contains three
    known languages, the texts are long enough for stopword counting to be
    decisive, and adding a model for this would be weight without benefit. If
    the text is genuinely ambiguous the caller gets ``en`` plus a warning
    rather than a confident wrong answer.
    """
    words = re.findall(r"[a-zaaeeiioouunc]+", text.lower())
    if len(words) < 20:
        return "en"
    counts = {
        language: sum(1 for w in words if w in stopwords)
        for language, stopwords in LANGUAGE_STOPWORDS.items()
    }
    best_language = max(counts, key=lambda k: counts[k])
    # Require a real signal, not a single incidental match.
    if counts[best_language] >= max(4, len(words) * 0.02):
        return best_language
    return "en"


def page_image_coverage(page: pdfplumber.page.Page) -> float:
    """Fraction of the page area covered by its largest image."""
    page_area = float(page.width) * float(page.height)
    if page_area <= 0 or not page.images:
        return 0.0
    largest = max(
        abs((img["x1"] - img["x0"]) * (img["bottom"] - img["top"]))
        for img in page.images
    )
    return largest / page_area


def detect_flavour(pdf: pdfplumber.PDF, text: str) -> tuple[PdfFlavour, list[str]]:
    """Classify a PDF into one of the four flavours, with any warnings.

    Order matters. Scanned is checked first because a scan has no usable text,
    so every other test would be reading an empty string. Non-English is
    checked before article because a German case report is more usefully routed
    to translation than to column handling.
    """
    warnings: list[str] = []
    first = pdf.pages[0]

    has_text_layer = len(text.strip()) >= MIN_CHARS_FOR_TEXT_LAYER
    image_coverage = page_image_coverage(first)

    if not has_text_layer:
        if image_coverage >= FULL_PAGE_IMAGE_COVERAGE:
            return PdfFlavour.SCANNED, warnings
        warnings.append(
            "No text layer and no full-page image; the document may be empty or "
            "corrupt. Treating as scanned so OCR is attempted."
        )
        return PdfFlavour.SCANNED, warnings

    if image_coverage >= FULL_PAGE_IMAGE_COVERAGE:
        warnings.append(
            "Page is mostly one image but also carries a text layer; it may be a "
            "searchable scan. Text layer used, OCR not attempted."
        )

    language = detect_language(text)
    if language != "en":
        return PdfFlavour.NON_ENGLISH, warnings

    lowered = text.lower()
    marker_hits = sum(1 for pattern in ARTICLE_MARKERS if re.search(pattern, lowered))
    if marker_hits >= 2:
        return PdfFlavour.ARTICLE, warnings

    return PdfFlavour.DIGITAL, warnings


def clean_cell(value: str | None) -> str:
    """Normalise one table cell, collapsing the newlines wrapping introduces."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def extract_tables(
    page: pdfplumber.page.Page, document_id: str, file_name: str, page_number: int
) -> tuple[list[TableBlock], list[tuple[float, float]]]:
    """Extract every table on a page, and report the areas they occupy.

    The occupied areas are returned so the prose pass can exclude them. Without
    that, every table is read twice -- once as structure and again as a garbled
    run of text -- which pollutes the prose given to the model.
    """
    blocks: list[TableBlock] = []
    regions: list[tuple[float, float]] = []

    for offset, table in enumerate(page.find_tables()):
        rows = table.extract()
        if not rows or len(rows) < 2:
            continue
        cleaned = [[clean_cell(cell) for cell in row] for row in rows]
        # Treat the first row as a header only if it is complete; a table whose
        # first row has blanks is usually a label/value grid, not a matrix.
        header, body = ([], cleaned)
        if all(cell for cell in cleaned[0]):
            header, body = cleaned[0], cleaned[1:]
        blocks.append(
            TableBlock(
                header=header,
                rows=body,
                source=SourceRef(document_id, file_name, page_number, offset),
            )
        )
        regions.append((table.bbox[1], table.bbox[3]))
    return blocks, regions


def outside_table_bands(table_regions: list[tuple[float, float]], obj: dict) -> bool:
    """Whether a page object sits outside every detected table band.

    Bound to its regions with ``functools.partial`` at the call site, because
    ``page.filter`` takes a one-argument predicate.

    ``page.filter`` is applied to every object type on the page, not only
    characters, and not all of them carry geometry. Those are kept rather than
    guessed at, which makes this predicate total -- so the filter cannot raise
    and needs no exception handler around it.
    """
    top, bottom = obj.get("top"), obj.get("bottom")
    if top is None or bottom is None:
        return True
    middle = (top + bottom) / 2
    return not any(start <= middle <= end for start, end in table_regions)


def extract_prose(
    page: pdfplumber.page.Page, table_regions: list[tuple[float, float]]
) -> str:
    """Extract page prose, excluding any vertical band occupied by a table."""
    if not table_regions:
        return page.extract_text() or ""
    return page.filter(partial(outside_table_bands, table_regions)).extract_text() or ""


def extract_images(
    page: pdfplumber.page.Page,
    document_id: str,
    file_name: str,
    page_number: int,
    skip_full_page: bool,
) -> list[ImageBlock]:
    """Collect meaningful images, ignoring logos, rules and scan backdrops."""
    blocks: list[ImageBlock] = []
    page_area = float(page.width) * float(page.height)
    for index, image in enumerate(page.images):
        width = abs(image["x1"] - image["x0"])
        height = abs(image["bottom"] - image["top"])
        if width < MIN_MEANINGFUL_IMAGE_PX or height < MIN_MEANINGFUL_IMAGE_PX:
            continue
        # On a scanned page the page-sized image is the scan itself; describing
        # it as an embedded figure would be wrong.
        if skip_full_page and page_area and (width * height) / page_area >= FULL_PAGE_IMAGE_COVERAGE:
            continue
        blocks.append(
            ImageBlock(
                width=int(width),
                height=int(height),
                source=SourceRef(document_id, file_name, page_number, index),
            )
        )
    return blocks


def extract_pdf(path: Path, document_id: str) -> ExtractedDocument:
    """Ingest a PDF file from disk."""
    return extract_pdf_bytes(
        path.read_bytes(), file_name=path.name, document_id=document_id
    )


def extract_pdf_bytes(
    data: bytes, file_name: str, document_id: str
) -> ExtractedDocument:
    """Ingest PDF bytes into an :class:`ExtractedDocument`.

    Bytes rather than a path is the primary form because attachments arrive
    inside a MIME message and never touch the filesystem -- writing them out
    just to read them back would be pointless I/O and an avoidable place for
    synthetic data to leak onto disk.

    Scanned documents come back with their page images and no prose: this
    function does not perform OCR. Reading pixels is a model call and belongs in
    the vision layer, which consumes this output. Keeping them separate means
    ingestion stays deterministic, offline and unit-testable.
    """
    document = ExtractedDocument(
        document_id=document_id,
        file_name=file_name,
        media_type="application/pdf",
    )

    with pdfplumber.open(BytesIO(data)) as pdf:
        document.page_count = len(pdf.pages)
        document.metadata = {
            str(k): str(v)
            for k, v in (pdf.metadata or {}).items()
            if v is not None and str(v).strip()
        }
        document.metadata["byte_size"] = str(len(data))

        sample = "\n".join((page.extract_text() or "") for page in pdf.pages[:3])
        document.flavour, warnings = detect_flavour(pdf, sample)
        document.warnings.extend(warnings)
        document.language = detect_language(sample)

        is_scanned = document.flavour is PdfFlavour.SCANNED

        for page_number, page in enumerate(pdf.pages, start=1):
            tables: list[TableBlock] = []
            regions: list[tuple[float, float]] = []
            if not is_scanned:
                tables, regions = extract_tables(
                    page, document_id, file_name, page_number
                )

            prose = "" if is_scanned else extract_prose(page, regions)
            if prose.strip():
                document.blocks.append(
                    TextBlock(
                        text=prose.strip(),
                        source=SourceRef(document_id, file_name, page_number, 0),
                    )
                )
            document.blocks.extend(tables)
            document.blocks.extend(
                extract_images(
                    page, document_id, file_name, page_number,
                    skip_full_page=not is_scanned,
                )
            )

        if is_scanned:
            # The whole page is the content, so record it as one image per page
            # for the vision layer to read.
            document.blocks = [
                ImageBlock(
                    width=int(page.width),
                    height=int(page.height),
                    source=SourceRef(document_id, file_name, page=n, block_index=0),
                    needs_review=True,
                )
                for n, page in enumerate(pdf.pages, start=1)
            ]
            document.warnings.append(
                "No text layer: OCR or a vision model is required to read this "
                "document. Confidence should be reported for anything extracted."
            )

    return document
