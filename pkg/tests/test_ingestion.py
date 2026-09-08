"""
Unit tests for ingestion and preprocessing.

These run offline against the generated corpus -- no network, no API key, no
model. Anything requiring a model call belongs in the extraction layer and is
tested separately with stubs.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai_service.mail import parse_message, parse_message_file
from ai_service.models import (
    ExtractedDocument,
    ImageBlock,
    PdfFlavour,
    SourceRef,
    TableBlock,
    TextBlock,
)
from ai_service.pdf import detect_language, extract_pdf
from ai_service.preprocess import (
    normalise_text,
    preprocess_document,
    preprocess_message,
    strip_page_furniture,
)

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


def make_ref(page: int = 1, index: int = 0) -> SourceRef:
    return SourceRef("doc", "file.pdf", page=page, block_index=index)


class TestSourceRef(unittest.TestCase):
    """Provenance must be present and immutable."""

    def test_describe_includes_page_for_pdf(self):
        self.assertEqual(make_ref(page=3).describe(), "file.pdf p.3")

    def test_describe_omits_page_when_absent(self):
        ref = SourceRef("doc", "(email body)", page=None)
        self.assertEqual(ref.describe(), "(email body)")

    def test_source_ref_is_frozen(self):
        # A reference must not be editable after a fact has cited it.
        with self.assertRaises(Exception):
            make_ref().page = 9


class TestTableBlock(unittest.TestCase):
    """Tables must survive as rows and columns, not prose."""

    def test_markdown_preserves_columns(self):
        table = TableBlock(
            header=["Test", "Result", "Unit"],
            rows=[["Potassium", "6.8", "mmol/L"]],
            source=make_ref(),
        )
        rendered = table.to_prompt_text()
        self.assertIn("| Test | Result | Unit |", rendered)
        self.assertIn("| Potassium | 6.8 | mmol/L |", rendered)
        # The source must travel with the table into the prompt.
        self.assertIn("file.pdf p.1", rendered)

    def test_handles_missing_cells(self):
        table = TableBlock(header=["A", "B"], rows=[["x", None]], source=make_ref())
        self.assertIn("| x |  |", table.to_prompt_text())


class TestNormalisation(unittest.TestCase):
    """Normalisation must fix artefacts without eating content."""

    def test_rejoins_hyphenated_linebreak(self):
        self.assertEqual(normalise_text("hospital-\nisation"), "hospitalisation")

    def test_keeps_real_hyphen_before_capital(self):
        # Not hyphenation: the continuation is not lowercase.
        self.assertIn("-", normalise_text("Neurolept-\nX"))

    def test_maps_typographic_punctuation(self):
        result = normalise_text("‘quoted’ — dash…")
        self.assertNotIn("‘", result)
        self.assertNotIn("…", result)
        self.assertIn("'quoted'", result)

    def test_non_breaking_hyphen_is_mapped(self):
        # U+2011 is rewritten by NFKC before a narrow map would see it, which
        # is why the whole dash range is mapped.
        self.assertNotIn("‑", normalise_text("co‑administered"))

    def test_strips_bare_page_numbers(self):
        self.assertEqual(normalise_text("Body text\nPage 3 of 12\nMore"), "Body text\nMore")

    def test_strips_dot_leaders(self):
        self.assertNotIn("....", normalise_text("Introduction ......... 4"))

    def test_empty_input_is_safe(self):
        self.assertEqual(normalise_text(""), "")

    def test_does_not_delete_short_real_content(self):
        # "None." is the actual content of some form fields and must survive.
        self.assertEqual(normalise_text("None."), "None.")


class TestPageFurniture(unittest.TestCase):
    """Repeated headers go; repeated content stays."""

    def _doc(self, pages: list[str]) -> ExtractedDocument:
        doc = ExtractedDocument("d", "f.pdf", "application/pdf")
        doc.blocks = [
            TextBlock(text=text, source=make_ref(page=n))
            for n, text in enumerate(pages, start=1)
        ]
        return doc

    def test_removes_repeated_header_with_varying_page_number(self):
        doc = self._doc(
            [
                "Clinevo Report Page 1\nReal content one",
                "Clinevo Report Page 2\nReal content two",
                "Clinevo Report Page 3\nReal content three",
            ]
        )
        removed = strip_page_furniture(doc)
        self.assertEqual(removed, 3)
        for block in doc.text_blocks:
            self.assertNotIn("Clinevo Report", block.text)
            self.assertIn("Real content", block.text)

    def test_keeps_long_repeated_sentences(self):
        # Long lines are content even when repeated -- the word guard protects
        # body text that recurs across pages.
        sentence = (
            "The patient was advised to discontinue the product immediately and "
            "to attend the clinic for review within seven days."
        )
        doc = self._doc([sentence, sentence, sentence])
        self.assertEqual(strip_page_furniture(doc), 0)

    def test_no_furniture_removal_on_single_page(self):
        doc = self._doc(["Header\nContent"])
        self.assertEqual(strip_page_furniture(doc), 0)


class TestPreprocessPreservesStructure(unittest.TestCase):
    """The central guarantee: preprocessing must not touch tables."""

    def test_tables_are_untouched(self):
        table = TableBlock(
            header=["Test", "Reference range"],
            rows=[["Potassium", "3.5 - 5.0"]],
            source=make_ref(),
        )
        doc = ExtractedDocument("d", "f.pdf", "application/pdf")
        doc.blocks = [TextBlock(text="some  prose", source=make_ref()), table]

        preprocess_document(doc)

        # Whitespace inside a reference range is load-bearing and must survive.
        self.assertEqual(doc.tables[0].rows[0], ["Potassium", "3.5 - 5.0"])
        self.assertEqual(doc.tables[0].header, ["Test", "Reference range"])

    def test_images_are_preserved(self):
        image = ImageBlock(width=400, height=300, source=make_ref())
        doc = ExtractedDocument("d", "f.pdf", "application/pdf")
        doc.blocks = [image]
        preprocess_document(doc)
        self.assertEqual(len(doc.images), 1)
        self.assertTrue(doc.images[0].needs_review)

    def test_provenance_survives_preprocessing(self):
        block = TextBlock(text="text  with   spaces", source=make_ref(page=4, index=2))
        doc = ExtractedDocument("d", "f.pdf", "application/pdf")
        doc.blocks = [block]
        preprocess_document(doc)
        self.assertEqual(doc.text_blocks[0].source.page, 4)
        self.assertEqual(doc.text_blocks[0].source.block_index, 2)


class TestLanguageDetection(unittest.TestCase):
    """Language detection must be decisive but not reckless."""

    def test_detects_german(self):
        text = (
            "Die Patientin wurde nicht stationaer aufgenommen und das Praeparat "
            "wurde abgesetzt. Der Arzt hat die Beschwerden nach der Einnahme "
            "dokumentiert und seit dem Absetzen sind keine Beschwerden mehr "
            "aufgetreten bei der Patientin."
        )
        self.assertEqual(detect_language(text), "de")

    def test_detects_spanish(self):
        text = (
            "El paciente presento urticaria generalizada por la via inhalatoria. "
            "La paciente no requirio ingreso hospitalario y se ha recuperado por "
            "completo sin antecedentes relevantes para el medico de la clinica."
        )
        self.assertEqual(detect_language(text), "es")

    def test_short_text_defaults_to_english(self):
        # Too little evidence to be decisive -- must not guess.
        self.assertEqual(detect_language("der die das"), "en")

    def test_english_stays_english(self):
        text = (
            "The patient developed a widespread rash across the trunk and upper "
            "arms with associated itching after starting the medication and was "
            "advised to stop taking it immediately by the treating physician."
        )
        self.assertEqual(detect_language(text), "en")


class TestCorpusIngestion(unittest.TestCase):
    """End-to-end against the generated corpus."""

    @classmethod
    def setUpClass(cls):
        if not SAMPLES.exists() or not any(SAMPLES.glob("*.eml")):
            raise unittest.SkipTest(
                "corpus not generated; run: python -m generator.generate"
            )

    def test_every_email_parses(self):
        for path in sorted(SAMPLES.glob("*.eml")):
            with self.subTest(email=path.name):
                message = parse_message_file(path)
                self.assertTrue(message.message_id)
                self.assertTrue(message.subject)
                self.assertTrue(message.sender)
                self.assertTrue(message.body.full_text())

    def test_flavour_detection_matches_ground_truth(self):

        truth = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))
        expected = {}
        for doc in truth["documents"]:
            if doc["kind"] == "article":
                expected[doc["file"]] = "article"
            for attachment in doc.get("attachments", []):
                if attachment["pdf_flavour"]:
                    expected[attachment["file"]] = attachment["pdf_flavour"]

        for file_name, want in sorted(expected.items()):
            with self.subTest(pdf=file_name):
                document = extract_pdf(SAMPLES / file_name, file_name)
                self.assertEqual(str(document.flavour), want)

    def test_non_pdf_attachment_is_logged_not_processed(self):
        message = parse_message_file(SAMPLES / "icsr_full_rash.eml")
        others = [a for a in message.attachments if not a.file_name.endswith(".pdf")]
        self.assertEqual(len(others), 1)
        self.assertFalse(others[0].processed)
        self.assertTrue(others[0].warnings)
        # It must still be visible, not silently dropped.
        self.assertIn(others[0], message.attachments)
        self.assertNotIn(others[0], message.documents)

    def test_scanned_pdf_has_no_text_and_warns(self):
        document = extract_pdf(
            SAMPLES / "scan_icsr_and_pqc_combined.pdf", "scan"
        )
        self.assertIs(document.flavour, PdfFlavour.SCANNED)
        self.assertEqual(document.full_text(), "")
        self.assertEqual(len(document.images), 1)
        self.assertTrue(any("OCR" in w for w in document.warnings))

    def test_lab_table_survives_as_rows(self):
        document = extract_pdf(SAMPLES / "form_icsr_full_hepatic.pdf", "form")
        lab = [t for t in document.tables if t.header and t.header[0] == "Test"]
        self.assertEqual(len(lab), 1)
        row = lab[0].rows[0]
        self.assertEqual(row[0], "Alanine aminotransferase")
        self.assertEqual(row[1], "512")
        self.assertEqual(row[2], "U/L")
        self.assertEqual(row[3], "10 - 40")

    def test_preprocessing_keeps_tables_across_whole_corpus(self):
        for path in sorted(SAMPLES.glob("*.eml")):
            with self.subTest(email=path.name):
                message = parse_message_file(path)
                before = [
                    (t.header, [list(r) for r in t.rows])
                    for d in message.documents
                    for t in d.tables
                ]
                preprocess_message(message)
                after = [
                    (t.header, [list(r) for r in t.rows])
                    for d in message.documents
                    for t in d.tables
                ]
                self.assertEqual(before, after)

    def test_message_serialises(self):
        message = parse_message_file(SAMPLES / "icsr_full_hepatic.eml")
        payload = message.to_dict()
        self.assertIn("body", payload)
        self.assertIn("attachments", payload)
        self.assertEqual(payload["body"]["counts"]["text_blocks"], 1)


class TestMalformedInput(unittest.TestCase):
    """Bad input must degrade, not crash."""

    def test_message_without_date_or_id_still_parses(self):
        raw = b"From: a@b.example\r\nSubject: no date\r\n\r\nBody here.\r\n"
        message = parse_message(raw)
        # Falls back to a content hash so ingestion stays idempotent.
        self.assertTrue(message.message_id.startswith("sha256:"))
        self.assertTrue(any("Date" in w for w in message.warnings))

    def test_same_bytes_produce_same_id(self):
        raw = b"From: a@b.example\r\nSubject: x\r\n\r\nBody.\r\n"
        self.assertEqual(parse_message(raw).message_id, parse_message(raw).message_id)

    def test_unparseable_pdf_attachment_is_recorded(self):
        raw = (
            b"From: a@b.example\r\nSubject: bad pdf\r\n"
            b"MIME-Version: 1.0\r\n"
            b'Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
            b"--B\r\nContent-Type: text/plain\r\n\r\nSee attached.\r\n"
            b"--B\r\nContent-Type: application/pdf\r\n"
            b'Content-Disposition: attachment; filename="broken.pdf"\r\n\r\n'
            b"not a pdf at all\r\n"
            b"--B--\r\n"
        )
        message = parse_message(raw)
        self.assertEqual(len(message.attachments), 1)
        self.assertFalse(message.attachments[0].processed)
        self.assertTrue(message.attachments[0].warnings)


if __name__ == "__main__":
    unittest.main()
