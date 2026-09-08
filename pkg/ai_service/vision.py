"""
Vision layer: read scanned pages, and describe embedded images.

The assignment is explicit about both:

    "Scanned or handwritten -- Use OCR / a vision-capable AI model to read it,
     and show a confidence score since handwriting is uncertain."

    "For meaningful images ... write a short text description and flag it for
     human review. Deep image analysis isn't required."

Two deliberate positions:

**Everything read from pixels is flagged for review, regardless of reported
confidence.** A vision model's self-reported confidence is not a measurement --
it is the model's impression of its own certainty, and it is poorly calibrated
on exactly the hard cases (faint ink, cursive, a half-ticked box). The score is
surfaced because the brief asks for it, but it is never used to auto-approve.
Calibrated per-word confidence needs a purpose-built document AI service; that
trade-off is argued in the write-up.

**A blank field is a finding, not a gap in the OCR.** A handwritten form with an
empty line means the reporter did not answer -- which must come back as
``Not stated``, not as a failure to read. The prompt says so explicitly,
because the natural failure mode is a model inventing plausible ink.
"""
from __future__ import annotations

import logging
from io import BytesIO

import pypdfium2 as pdfium

from .llm import LLMClient
from .models import ExtractedDocument, ImageBlock, PdfFlavour, SourceRef, TextBlock
from .schemas import IMAGE_DESCRIPTION_SCHEMA, OCR_SCHEMA

logger = logging.getLogger(__name__)

# Render scale for pages sent to the model. 2.0 is roughly 150 dpi: enough for
# handwriting to be legible, without paying for pixels that add no signal.
PAGE_RENDER_SCALE = 2.0

# Confidence below which a scanned page is called unreliable in its own right.
LOW_CONFIDENCE_THRESHOLD = 0.6

OCR_PROMPT = """\
Read this scanned page and transcribe it.

It is a filled-in adverse event report form. It may be handwritten, skewed,
speckled, or faint.

RULES
- Transcribe exactly what is written. Do not correct, complete or tidy it.
- Preserve the label/value structure. Put each field on its own line as
  "Label: value".
- If a field's line is BLANK, write "Label: Not stated". A blank line means the
  reporter did not answer -- it is a real finding, not something to fill in.
- If a value is partly illegible, transcribe what you can read and mark the
  rest "[illegible]". Never guess at a word you cannot read.
- Do not add any field that is not printed on the form.

ALSO RETURN
  confidence      0.0 to 1.0 -- how confident you are in this transcription
                  overall. Handwriting is uncertain; be honest rather than
                  generous. Use a low value for faint or cursive text.
  is_handwritten  true if the values are handwritten rather than typed
  notes           anything a human reviewer should know: illegible areas,
                  ambiguous characters, damage to the page
"""

IMAGE_PROMPT = """\
Describe this image in two or three sentences for a pharmacovigilance reviewer.

Say plainly what is visible. This is a good-faith description to help a human
decide whether to look closer -- it is not a clinical or forensic assessment.

RULES
- Describe only what you can actually see. Do not diagnose.
- Do not speculate about cause, severity, or whether a product is at fault.
- If the image is unclear or you cannot tell what it shows, say so.

ALSO RETURN
  shows_product_defect  true if it appears to show a damaged, discoloured or
                        otherwise defective product or its packaging
  shows_clinical_sign   true if it appears to show a person, a body part, or a
                        visible clinical sign such as a rash
  confidence            0.0 to 1.0 in your description
"""


def render_page_png(data: bytes, page_number: int, scale: float = PAGE_RENDER_SCALE) -> bytes:
    """Render one PDF page (1-based) to PNG bytes."""
    document = pdfium.PdfDocument(data)
    try:
        image = document[page_number - 1].render(scale=scale).to_pil()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        document.close()


def ocr_page(
    page_png: bytes, source: SourceRef, client: LLMClient | None = None
) -> tuple[TextBlock | None, float, list[str]]:
    """Transcribe one rendered page.

    Returns the transcribed block (or ``None`` when nothing was read), the
    reported confidence, and any warnings for the reviewer.
    """
    client = client or LLMClient()
    warnings: list[str] = []

    try:
        response = client.generate_json(OCR_PROMPT, OCR_SCHEMA, images=[page_png])
    except Exception as exc:  # noqa: BLE001 -- a failed page must not lose the document
        logger.error("OCR failed for %s: %s", source.describe(), exc)
        return None, 0.0, [f"{source.describe()}: OCR failed ({exc}). Needs manual review."]

    text = str(response.data.get("text", "") or "").strip()
    confidence = min(1.0, max(0.0, float(response.data.get("confidence", 0.0) or 0.0)))
    is_handwritten = bool(response.data.get("is_handwritten", False))
    notes = str(response.data.get("notes", "") or "").strip()

    if not text:
        warnings.append(
            f"{source.describe()}: nothing could be transcribed. Needs manual review."
        )
        return None, confidence, warnings

    if is_handwritten:
        warnings.append(
            f"{source.describe()}: handwritten content, transcription confidence "
            f"{confidence:.2f}. Verify against the original."
        )
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"{source.describe()}: low transcription confidence ({confidence:.2f}); "
            "fields read from this page are unreliable."
        )
    if notes:
        warnings.append(f"{source.describe()}: {notes}")
    if "[illegible]" in text.lower():
        warnings.append(
            f"{source.describe()}: contains illegible sections marked [illegible]."
        )

    return TextBlock(text=text, source=source), confidence, warnings


def describe_image(
    image_png: bytes, block: ImageBlock, client: LLMClient | None = None
) -> list[str]:
    """Describe one embedded image, updating the block in place."""
    client = client or LLMClient()
    try:
        response = client.generate_json(
            IMAGE_PROMPT, IMAGE_DESCRIPTION_SCHEMA, images=[image_png]
        )
    except Exception as exc:  # noqa: BLE001
        block.description = None
        block.needs_review = True
        return [f"{block.source.describe()}: image description failed ({exc})."]

    block.description = str(response.data.get("description", "") or "").strip() or None
    # Never cleared: the brief asks for a flag for human review on meaningful
    # images, and a model's own confidence is not grounds to drop it.
    block.needs_review = True

    warnings = []
    if response.data.get("shows_product_defect"):
        warnings.append(
            f"{block.source.describe()}: image may show a product defect - "
            "relevant to a quality complaint."
        )
    if response.data.get("shows_clinical_sign"):
        warnings.append(
            f"{block.source.describe()}: image may show a clinical sign - "
            "relevant to a safety report."
        )
    return warnings


def read_scanned_document(
    document: ExtractedDocument, pdf_bytes: bytes, client: LLMClient | None = None
) -> ExtractedDocument:
    """Run OCR over a scanned document, folding the text back into it.

    The transcribed text becomes ordinary :class:`TextBlock` content with the
    same provenance the page image carried, so every downstream stage --
    classification, extraction, quote verification -- works on a scanned
    document exactly as it does on a digital one. The page images are kept
    alongside, so a reviewer can still see the original.
    """
    if document.flavour is not PdfFlavour.SCANNED:
        return document

    client = client or LLMClient()
    confidences: list[float] = []
    text_blocks: list[TextBlock] = []

    for image_block in list(document.images):
        page_number = image_block.source.page or 1
        try:
            page_png = render_page_png(pdf_bytes, page_number)
        except Exception as exc:  # noqa: BLE001
            document.warnings.append(
                f"Page {page_number}: could not be rendered for OCR ({exc})."
            )
            continue

        block, confidence, warnings = ocr_page(page_png, image_block.source, client)
        document.warnings.extend(warnings)
        confidences.append(confidence)
        if block is not None:
            text_blocks.append(block)
        # The page image stays: the transcript is derived, the image is evidence.
        image_block.description = (
            block.text[:200] if block is not None else "Not transcribed"
        )
        image_block.needs_review = True

    # Text first so the document reads in a natural order, images retained after.
    document.blocks = text_blocks + document.images

    if confidences:
        mean_confidence = sum(confidences) / len(confidences)
        document.metadata["ocr_confidence"] = f"{mean_confidence:.3f}"
        document.metadata["ocr_pages"] = str(len(confidences))
    document.metadata["source_is_pixels"] = "true"
    document.warnings.append(
        "Content was read from page images. Every field derived from this "
        "document requires human verification regardless of reported confidence."
    )
    return document
