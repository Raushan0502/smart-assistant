"""
End-to-end tests: a real corpus document in, a complete analysis out.

The other suites test units in isolation with fakes. This one runs the whole
chain -- parse, preprocess, OCR, classify, extract, summarise -- over the
*actual generated corpus*, and asserts on properties that must hold no matter
which model is behind it.

That distinction matters. Asserting "the model said 54 years" would test the
model, and would break the moment a provider changes. These tests assert the
things that must be true of the *pipeline*:

  * structure survives the whole journey (a table is still a table at the end)
  * provenance survives (every fact still names its page)
  * the honesty guarantees hold (no invented value, no unsourced claim)
  * every stage is timed and audited, including the ones that failed

They run against the offline stub, so they need no API key and no network.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from ai_service.llm import LLMClient, StubProvider
from ai_service.models import PdfFlavour
from ai_service.pdf import extract_pdf
from ai_service.pipeline import process_message, screen_literature
from ai_service.preprocess import preprocess_document
from ai_service.schemas import NOT_STATED, Category

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


def stub_client() -> LLMClient:
    """A client pinned to the offline stub, so tests never hit the network."""
    return LLMClient(StubProvider())


def run(name: str):
    """Process one corpus message end to end through the stub."""
    return process_message((SAMPLES / name).read_bytes(), client=stub_client())


class EndToEndBase(unittest.TestCase):
    """Skips the whole module if the corpus has not been generated."""

    @classmethod
    def setUpClass(cls):
        if not (SAMPLES / "icsr_full_rash.eml").exists():
            raise unittest.SkipTest(
                "corpus not generated; run: cd pkg && python -m generator.generate"
            )


class TestFullPipeline(EndToEndBase):
    """One complete message, start to finish."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = run("icsr_full_rash.eml")
        cls.payload = cls.result.to_dict()

    def test_message_identity_survives(self):
        message = self.result.message
        self.assertTrue(message.message_id)
        self.assertIn("Cardiozan", message.subject)
        self.assertIn("osei", message.sender.lower())

    def test_body_and_attachments_are_all_present(self):
        # One body, one PDF form, one CSV that must be logged but not parsed.
        self.assertEqual(len(self.result.message.attachments), 2)
        names = {a.file_name for a in self.result.message.attachments}
        self.assertEqual(names, {"form_icsr_full_rash.pdf", "ward_roster.csv"})

    def test_unsupported_attachment_logged_not_processed(self):
        csv = next(
            a for a in self.result.message.attachments if a.file_name.endswith(".csv")
        )
        self.assertFalse(csv.processed)
        self.assertTrue(csv.warnings)
        # It must not reach the model, but must remain visible to a reviewer.
        self.assertNotIn(csv, self.result.message.documents)

    def test_table_structure_survives_the_whole_journey(self):
        # The single most important structural guarantee: a lab value, its unit
        # and its reference range are still associated at the end. Looked up by
        # test name rather than row position, so reordering the panel does not
        # silently change what this asserts.
        tables = [t for d in self.result.message.documents for t in d.tables]
        lab = next(t for t in tables if t.header and t.header[0] == "Test")
        by_test = {row[0]: row for row in lab.rows}

        self.assertIn("Haemoglobin", by_test)
        self.assertEqual(by_test["Haemoglobin"][1:4], ["132", "g/L", "120 - 160"])
        # An abnormal row keeps its flag alongside its value.
        self.assertEqual(by_test["White cell count"][1:5],
                         ["11.8", "x10^9/L", "4.0 - 11.0", "HIGH"])

    def test_every_block_still_carries_provenance(self):
        for document in self.result.message.documents:
            for block in document.blocks:
                with self.subTest(doc=document.file_name):
                    self.assertTrue(block.source.document_id)
                    self.assertTrue(block.source.file_name)

    def test_all_four_categories_are_judged(self):
        self.assertEqual(len(self.result.classification.verdicts), 4)
        judged = {v.category for v in self.result.classification.verdicts}
        self.assertEqual(judged, set(Category))

    def test_every_stage_is_timed(self):
        for stage in ("parse", "preprocess", "classify", "summarise"):
            self.assertIn(stage, self.result.stage_ms)
        self.assertGreater(self.result.total_ms, 0)

    def test_audit_event_recorded_per_model_call(self):
        kinds = [e.event_type for e in self.result.audit_events]
        self.assertIn("CLASSIFY", kinds)
        self.assertIn("SUMMARY", kinds)
        for event in self.result.audit_events:
            self.assertTrue(event.model_name)

    def test_result_serialises_completely(self):
        for key in ("message", "classification", "extractions", "summaries",
                    "warnings", "audit_events", "timings"):
            self.assertIn(key, self.payload)


class TestHonestyGuarantees(EndToEndBase):
    """The properties that must hold whatever model is behind the pipeline."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = [
            run(name) for name in
            ["icsr_full_rash.eml", "icsr_full_hepatic.eml", "pqc_broken_seal.eml"]
        ]

    def test_no_field_is_stated_without_a_source(self):
        # The core audit requirement: a value with no provenance is worse than
        # no value, because it cannot be checked.
        for result in self.results:
            for extraction in result.extractions:
                for field in extraction.fields:
                    if field.is_stated:
                        with self.subTest(field=field.name):
                            self.assertTrue(
                                field.source or field.quote,
                                f"{field.name} stated with neither source nor quote",
                            )

    def test_unstated_fields_carry_zero_confidence(self):
        for result in self.results:
            for extraction in result.extractions:
                for field in extraction.fields:
                    if not field.is_stated:
                        with self.subTest(field=field.name):
                            self.assertEqual(field.value, NOT_STATED)
                            self.assertEqual(field.confidence, 0.0)

    def test_confidence_always_in_range(self):
        for result in self.results:
            for verdict in result.classification.verdicts:
                self.assertGreaterEqual(verdict.confidence, 0.0)
                self.assertLessEqual(verdict.confidence, 1.0)
            for extraction in result.extractions:
                for field in extraction.fields:
                    self.assertGreaterEqual(field.confidence, 0.0)
                    self.assertLessEqual(field.confidence, 1.0)

    def test_every_expected_field_is_present(self):
        # A missing key would read as "not asked"; an explicit Not stated reads
        # as "asked, and the source did not say". Only the second is honest.
        from ai_service.schemas import FIELDS_BY_CATEGORY

        for result in self.results:
            for extraction in result.extractions:
                expected = set(FIELDS_BY_CATEGORY[extraction.category])
                self.assertEqual({f.name for f in extraction.fields}, expected)

    def test_stub_output_is_always_marked(self):
        # Placeholder output must never be mistaken for a model's judgement.
        for result in self.results:
            for verdict in result.classification.verdicts:
                self.assertIn("[offline stub]", verdict.reason)


class TestFlavourRouting(EndToEndBase):
    """Each PDF flavour must take its own path through the pipeline."""

    def test_scanned_attachment_is_routed_to_ocr(self):
        result = run("icsr_and_pqc_combined.eml")
        scan = next(
            a for a in result.message.attachments if a.flavour is PdfFlavour.SCANNED
        )
        self.assertEqual(scan.metadata.get("source_is_pixels"), "true")
        # Anything read from pixels needs a human, whatever the model reported.
        self.assertTrue(any("human verification" in w for w in scan.warnings))
        # An OCR attempt must be audited even when the stub reads nothing.
        self.assertIn("OCR", [e.event_type for e in result.audit_events])

    def test_non_english_attachment_is_detected(self):
        result = run("icsr_german.eml")
        form = next(a for a in result.message.attachments if a.processed)
        self.assertIs(form.flavour, PdfFlavour.NON_ENGLISH)
        self.assertEqual(form.language, "de")

    def test_digital_attachment_yields_tables_and_prose(self):
        result = run("icsr_full_hepatic.eml")
        form = next(a for a in result.message.attachments if a.processed)
        self.assertIs(form.flavour, PdfFlavour.DIGITAL)
        self.assertTrue(form.tables)
        self.assertTrue(form.full_text())

    def test_message_with_no_attachment_still_completes(self):
        result = run("icsr_sparse_dizzy.eml")
        self.assertEqual(result.message.attachments, [])
        self.assertTrue(result.message.body.full_text())
        self.assertEqual(len(result.classification.verdicts), 4)


class TestLiteratureScreening(EndToEndBase):
    """The bonus path: article in, reportable-case verdict out."""

    def test_article_screens_and_is_audited(self):
        document = extract_pdf(SAMPLES / "art_single_case.pdf", "art")
        preprocess_document(document)
        summary, event = screen_literature(document, stub_client())

        self.assertEqual(summary.document_id, "art")
        self.assertEqual(event.event_type, "SCREEN")
        self.assertTrue(event.succeeded)

    def test_every_corpus_article_is_screenable(self):
        for path in sorted(SAMPLES.glob("art_*.pdf")):
            with self.subTest(article=path.name):
                document = extract_pdf(path, path.stem)
                preprocess_document(document)
                self.assertIs(document.flavour, PdfFlavour.ARTICLE)
                summary, event = screen_literature(document, stub_client())
                self.assertTrue(event.succeeded)
                self.assertIsInstance(summary.looks_relevant, bool)


class TestWholeCorpus(EndToEndBase):
    """Every message in the corpus must survive the pipeline."""

    def test_no_message_raises(self):
        for path in sorted(SAMPLES.glob("*.eml")):
            with self.subTest(message=path.name):
                result = run(path.name)
                self.assertTrue(result.message.message_id)
                self.assertEqual(len(result.classification.verdicts), 4)
                self.assertGreater(result.total_ms, 0)

    def test_corpus_produces_expected_flavour_spread(self):
        # Guards against a regression that quietly routes everything one way.
        flavours = [
            a.flavour
            for path in sorted(SAMPLES.glob("*.eml"))
            for a in run(path.name).message.attachments
            if a.processed
        ]
        self.assertGreaterEqual(flavours.count(PdfFlavour.DIGITAL), 5)
        self.assertGreaterEqual(flavours.count(PdfFlavour.SCANNED), 2)
        self.assertGreaterEqual(flavours.count(PdfFlavour.NON_ENGLISH), 2)


if __name__ == "__main__":
    unittest.main()
