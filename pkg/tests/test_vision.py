"""
Unit tests for the vision and summarisation layers.

No model is called: a scripted fake provider stands in, so the tests assert how
the pipeline handles specific model behaviours -- including the unhelpful ones.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import ai_service.llm as llm_module
from ai_service.llm import LLMClient
from ai_service.models import ExtractedDocument, ImageBlock, PdfFlavour, SourceRef
from ai_service.pdf import extract_pdf
from ai_service.summarise import screen_article, summarise_document
from ai_service.vision import (
    LOW_CONFIDENCE_THRESHOLD,
    describe_image,
    ocr_page,
    read_scanned_document,
    render_page_png,
)

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
SCAN = SAMPLES / "scan_icsr_and_pqc_combined.pdf"


class FakeProvider:
    """Returns a scripted payload and records the prompts it was given."""

    name = "fake-vision"
    is_stub = False

    def __init__(self, payload: dict):
        self.payload = payload
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def generate_json(self, prompt, schema, images=None):
        self.prompts.append(prompt)
        self.image_counts.append(len(images or []))
        return self.payload


class FailingProvider:
    """Always raises, to test that a bad page does not lose the document."""

    name = "failing"
    is_stub = False

    def generate_json(self, prompt, schema, images=None):
        raise RuntimeError("vision service unavailable")


def setUpModule():
    """Remove retry backoff for the duration of these tests.

    The failure paths deliberately exhaust the retry budget. With the real
    backoff that costs ~6 seconds per test for no added coverage, so the delay
    is zeroed here rather than shortening the production retry policy.
    """
    global _SAVED_BACKOFF
    _SAVED_BACKOFF = llm_module.BACKOFF_SECONDS
    llm_module.BACKOFF_SECONDS = 0.0


def tearDownModule():
    """Restore the real backoff so other test modules see production values."""
    llm_module.BACKOFF_SECONDS = _SAVED_BACKOFF


def make_ref(page: int = 1) -> SourceRef:
    return SourceRef("doc", "scan.pdf", page=page, block_index=0)


def ocr_payload(**overrides) -> dict:
    payload = {
        "text": "Patient age: 33 years\nSex: Not stated\nProduct: Dermacalm",
        "confidence": 0.82,
        "is_handwritten": True,
        "notes": "",
    }
    payload.update(overrides)
    return payload


class TestPageRendering(unittest.TestCase):
    """Pages must render to PNG for the model to read."""

    @classmethod
    def setUpClass(cls):
        if not SCAN.exists():
            raise unittest.SkipTest("corpus not generated")

    def test_renders_png_bytes(self):
        png = render_page_png(SCAN.read_bytes(), 1)
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertGreater(len(png), 10_000)


class TestOcrPage(unittest.TestCase):
    """Transcription must be honest about its own reliability."""

    def test_transcribes_and_returns_confidence(self):
        client = LLMClient(FakeProvider(ocr_payload()))
        block, confidence, _ = ocr_page(b"png", make_ref(), client)
        self.assertIsNotNone(block)
        self.assertIn("Dermacalm", block.text)
        self.assertAlmostEqual(confidence, 0.82)

    def test_provenance_is_preserved(self):
        client = LLMClient(FakeProvider(ocr_payload()))
        block, _, _ = ocr_page(b"png", make_ref(page=3), client)
        self.assertEqual(block.source.page, 3)

    def test_handwriting_raises_a_warning(self):
        client = LLMClient(FakeProvider(ocr_payload(is_handwritten=True)))
        _, _, warnings = ocr_page(b"png", make_ref(), client)
        self.assertTrue(any("handwritten" in w for w in warnings))

    def test_low_confidence_raises_a_warning(self):
        payload = ocr_payload(confidence=LOW_CONFIDENCE_THRESHOLD - 0.2)
        _, _, warnings = ocr_page(b"png", make_ref(), LLMClient(FakeProvider(payload)))
        self.assertTrue(any("low transcription confidence" in w for w in warnings))

    def test_illegible_marker_raises_a_warning(self):
        payload = ocr_payload(text="Patient age: [illegible]")
        _, _, warnings = ocr_page(b"png", make_ref(), LLMClient(FakeProvider(payload)))
        self.assertTrue(any("illegible" in w for w in warnings))

    def test_empty_transcription_warns_rather_than_silently_passing(self):
        block, _, warnings = ocr_page(
            b"png", make_ref(), LLMClient(FakeProvider(ocr_payload(text="")))
        )
        self.assertIsNone(block)
        self.assertTrue(any("nothing could be transcribed" in w for w in warnings))

    def test_failure_does_not_raise(self):
        # A failed page must degrade to a warning, not kill the document.
        block, confidence, warnings = ocr_page(b"png", make_ref(), LLMClient(FailingProvider()))
        self.assertIsNone(block)
        self.assertEqual(confidence, 0.0)
        self.assertTrue(any("OCR failed" in w for w in warnings))

    def test_prompt_tells_model_blank_means_not_stated(self):
        provider = FakeProvider(ocr_payload())
        ocr_page(b"png", make_ref(), LLMClient(provider))
        prompt = provider.prompts[0]
        # The key anti-invention instruction for handwritten forms.
        self.assertIn("Not stated", prompt)
        self.assertIn("blank", prompt.lower())
        self.assertIn("[illegible]", prompt)

    def test_image_is_actually_sent(self):
        provider = FakeProvider(ocr_payload())
        ocr_page(b"png", make_ref(), LLMClient(provider))
        self.assertEqual(provider.image_counts[0], 1)


class TestImageDescription(unittest.TestCase):
    """Images get a good-faith description and always keep the review flag."""

    def _block(self) -> ImageBlock:
        return ImageBlock(width=400, height=300, source=make_ref())

    def test_description_is_attached(self):
        payload = {
            "description": "A cream tube with brown discoloured contents.",
            "shows_product_defect": True,
            "shows_clinical_sign": False,
            "confidence": 0.7,
        }
        block = self._block()
        describe_image(b"png", block, LLMClient(FakeProvider(payload)))
        self.assertIn("discoloured", block.description)

    def test_review_flag_survives_high_confidence(self):
        payload = {
            "description": "Clear photo.",
            "shows_product_defect": False,
            "shows_clinical_sign": False,
            "confidence": 0.99,
        }
        block = self._block()
        describe_image(b"png", block, LLMClient(FakeProvider(payload)))
        # A model's own confidence is never grounds to drop human review.
        self.assertTrue(block.needs_review)

    def test_defect_flag_surfaces_warning(self):
        payload = {
            "description": "Broken seal.",
            "shows_product_defect": True,
            "shows_clinical_sign": False,
            "confidence": 0.8,
        }
        warnings = describe_image(b"png", self._block(), LLMClient(FakeProvider(payload)))
        self.assertTrue(any("product defect" in w for w in warnings))

    def test_clinical_sign_flag_surfaces_warning(self):
        payload = {
            "description": "A forearm with a red rash.",
            "shows_product_defect": False,
            "shows_clinical_sign": True,
            "confidence": 0.8,
        }
        warnings = describe_image(b"png", self._block(), LLMClient(FakeProvider(payload)))
        self.assertTrue(any("clinical sign" in w for w in warnings))

    def test_failure_keeps_review_flag(self):
        block = self._block()
        warnings = describe_image(b"png", block, LLMClient(FailingProvider()))
        self.assertIsNone(block.description)
        self.assertTrue(block.needs_review)
        self.assertTrue(warnings)

    def test_prompt_forbids_diagnosis(self):
        provider = FakeProvider(
            {
                "description": "x",
                "shows_product_defect": False,
                "shows_clinical_sign": False,
                "confidence": 0.5,
            }
        )
        describe_image(b"png", self._block(), LLMClient(provider))
        self.assertIn("Do not diagnose", provider.prompts[0])


class TestReadScannedDocument(unittest.TestCase):
    """OCR output must rejoin the normal document structure."""

    @classmethod
    def setUpClass(cls):
        if not SCAN.exists():
            raise unittest.SkipTest("corpus not generated")
        cls.pdf_bytes = SCAN.read_bytes()

    def _scanned_doc(self) -> ExtractedDocument:
        return extract_pdf(SCAN, "scan")

    def test_scanned_doc_starts_with_no_text(self):
        self.assertEqual(self._scanned_doc().full_text(), "")

    def test_ocr_text_becomes_text_blocks(self):
        document = self._scanned_doc()
        client = LLMClient(FakeProvider(ocr_payload()))
        read_scanned_document(document, self.pdf_bytes, client)
        # After OCR the document behaves like any other for downstream stages.
        self.assertIn("Dermacalm", document.full_text())
        self.assertEqual(len(document.text_blocks), 1)

    def test_page_images_are_retained_as_evidence(self):
        document = self._scanned_doc()
        read_scanned_document(document, self.pdf_bytes, LLMClient(FakeProvider(ocr_payload())))
        self.assertEqual(len(document.images), 1)

    def test_confidence_recorded_in_metadata(self):
        document = self._scanned_doc()
        read_scanned_document(document, self.pdf_bytes, LLMClient(FakeProvider(ocr_payload())))
        self.assertIn("ocr_confidence", document.metadata)
        self.assertEqual(document.metadata["source_is_pixels"], "true")

    def test_always_warns_that_verification_is_required(self):
        document = self._scanned_doc()
        read_scanned_document(document, self.pdf_bytes, LLMClient(FakeProvider(ocr_payload())))
        self.assertTrue(
            any("requires human verification" in w for w in document.warnings)
        )

    def test_non_scanned_document_is_untouched(self):
        digital = extract_pdf(SAMPLES / "form_icsr_full_hepatic.pdf", "form")
        before = len(digital.blocks)
        read_scanned_document(digital, b"", LLMClient(FailingProvider()))
        self.assertEqual(len(digital.blocks), before)
        self.assertIsNot(digital.flavour, PdfFlavour.SCANNED)


class TestSummarisation(unittest.TestCase):
    """Summaries carry a separate, structured relevance judgement."""

    @classmethod
    def setUpClass(cls):
        if not SAMPLES.exists() or not any(SAMPLES.glob("*.pdf")):
            raise unittest.SkipTest("corpus not generated")

    def _payload(self, **overrides) -> dict:
        payload = {
            "summary": "A case report. " * 12,
            "looks_relevant": True,
            "relevance_reason": "Describes an adverse reaction in a named patient.",
            "confidence": 0.85,
        }
        payload.update(overrides)
        return payload

    def test_summary_is_parsed(self):
        document = extract_pdf(SAMPLES / "art_single_case.pdf", "art")
        result = summarise_document(document, LLMClient(FakeProvider(self._payload())))
        self.assertTrue(result.looks_relevant)
        self.assertEqual(result.document_id, "art")
        self.assertAlmostEqual(result.confidence, 0.85)

    def test_summary_prompt_asks_for_length_and_gaps(self):
        provider = FakeProvider(self._payload())
        document = extract_pdf(SAMPLES / "art_single_case.pdf", "art")
        summarise_document(document, LLMClient(provider))
        prompt = provider.prompts[0]
        self.assertIn("10 TO 15 SENTENCES", prompt)
        self.assertIn("not stated", prompt.lower())

    def test_screening_prompt_excludes_aggregate_data(self):
        provider = FakeProvider(self._payload(looks_relevant=False))
        document = extract_pdf(SAMPLES / "art_aggregate_only.pdf", "agg")
        screen_article(document, LLMClient(provider))
        prompt = provider.prompts[0]
        # The distinction a keyword classifier gets wrong. Matched on a single
        # line of the prompt, since the sentence wraps.
        self.assertIn("Aggregate numbers are not an", prompt)
        self.assertIn("Cohort or registry studies", prompt)
        self.assertIn("MORE THAN ONE case", prompt)

    def test_screening_returns_negative_verdict(self):
        document = extract_pdf(SAMPLES / "art_review_no_case.pdf", "rev")
        result = screen_article(
            document,
            LLMClient(
                FakeProvider(
                    self._payload(
                        looks_relevant=False,
                        relevance_reason="Review article, no individual patient.",
                    )
                )
            ),
        )
        self.assertFalse(result.looks_relevant)
        self.assertIn("no individual patient", result.relevance_reason)


if __name__ == "__main__":
    unittest.main()
