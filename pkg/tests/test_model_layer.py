"""
Unit tests for classification and extraction.

A scripted fake provider is used rather than the offline stub, so the tests can
assert what happens for *specific* model outputs -- including the dishonest ones
the validation layer exists to catch.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from ai_service.classify import build_prompt, classify_message, parse_verdicts
from ai_service.extract import extract_fields, parse_fields, verify_quote
from ai_service.llm import (
    AllProvidersExhausted,
    LLMClient,
    LLMError,
    QuotaExhausted,
    StubProvider,
    build_provider_chain,
    extract_json,
    rate_limit_delay,
)
from ai_service.mail import parse_message_file
from ai_service.schemas import (
    CLASSIFICATION_SCHEMA,
    ICSR_FIELDS,
    Category,
    CategoryVerdict,
    Classification,
    ExtractedField,
    extraction_schema,
)

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


class FakeProvider:
    """Returns a scripted payload, so a specific model behaviour can be tested."""

    name = "fake-model"
    is_stub = False

    def __init__(self, payload: dict):
        self.payload = payload
        self.prompts: list[str] = []

    def generate_json(self, prompt, schema, images=None):
        self.prompts.append(prompt)
        return self.payload


class TestJsonRecovery(unittest.TestCase):
    """Model output is not always clean JSON."""

    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_with_preamble(self):
        self.assertEqual(extract_json('Sure, here it is:\n{"a": 1}'), {"a": 1})

    def test_no_json_raises(self):
        # Must raise rather than return {} -- an empty result would read as
        # "the document said nothing", which is the dangerous false negative.
        with self.assertRaises(LLMError):
            extract_json("I could not do that.")


class TestRateLimitHandling(unittest.TestCase):
    """Throttling and exhaustion are different conditions."""

    def test_short_window_is_worth_waiting_for(self):
        self.assertAlmostEqual(
            rate_limit_delay("429 quota. Please retry in 2.24s"), 3.24
        )

    def test_exhausted_quota_raises_rather_than_waiting(self):
        # Retrying a spent allowance burned the caller timeout and surfaced a
        # clear "quota exhausted" in the UI as an opaque "read timed out".
        with self.assertRaises(QuotaExhausted):
            rate_limit_delay("429 RESOURCE_EXHAUSTED quota exceeded, limit: 20")

    def test_implausibly_long_wait_is_treated_as_exhaustion(self):
        with self.assertRaises(QuotaExhausted):
            rate_limit_delay("429 quota. Please retry in 3600s")

    def test_non_rate_limit_errors_are_not_treated_as_quota(self):
        self.assertIsNone(rate_limit_delay("500 internal server error"))


class TestClassificationParsing(unittest.TestCase):
    """Verdict parsing must be complete and self-consistent."""

    def test_missing_categories_are_filled_in(self):
        result = parse_verdicts(
            {"verdicts": [{"category": "ICSR", "applies": True, "confidence": 0.9, "reason": "r"}]},
            model="m",
        )
        self.assertEqual(len(result.verdicts), 4)

    def test_multi_label_is_preserved(self):
        result = parse_verdicts(
            {
                "verdicts": [
                    {"category": "ICSR", "applies": True, "confidence": 0.9, "reason": "reaction"},
                    {"category": "PQC", "applies": True, "confidence": 0.8, "reason": "defect"},
                ]
            },
            model="m",
        )
        self.assertIn(Category.ICSR, result.labels)
        self.assertIn(Category.PQC, result.labels)

    def test_not_relevant_cleared_when_substantive_label_present(self):
        result = parse_verdicts(
            {
                "verdicts": [
                    {"category": "ICSR", "applies": True, "confidence": 0.9, "reason": "reaction"},
                    {"category": "NOT_RELEVANT", "applies": True, "confidence": 0.6, "reason": "spam"},
                ]
            },
            model="m",
        )
        self.assertNotIn(Category.NOT_RELEVANT, result.labels)

    def test_not_relevant_kept_when_alone(self):
        result = parse_verdicts(
            {"verdicts": [{"category": "NOT_RELEVANT", "applies": True, "confidence": 0.9, "reason": "marketing"}]},
            model="m",
        )
        self.assertEqual(result.labels, [Category.NOT_RELEVANT])

    def test_unknown_category_is_ignored(self):
        result = parse_verdicts(
            {"verdicts": [{"category": "NONSENSE", "applies": True, "confidence": 1.0, "reason": "x"}]},
            model="m",
        )
        self.assertEqual(result.labels, [])

    def test_confidence_is_clamped(self):
        result = parse_verdicts(
            {"verdicts": [{"category": "MI", "applies": True, "confidence": 7.5, "reason": "x"}]},
            model="m",
        )
        mi = next(v for v in result.verdicts if v.category is Category.MI)
        self.assertEqual(mi.confidence, 1.0)

    def test_labels_ordered_by_confidence(self):
        result = parse_verdicts(
            {
                "verdicts": [
                    {"category": "PQC", "applies": True, "confidence": 0.5, "reason": "a"},
                    {"category": "ICSR", "applies": True, "confidence": 0.95, "reason": "b"},
                ]
            },
            model="m",
        )
        self.assertEqual(result.labels[0], Category.ICSR)


class TestNotStatedEnforcement(unittest.TestCase):
    """'Unknown beats a guess' must be enforced, not merely requested."""

    def test_empty_value_becomes_not_stated(self):
        field = ExtractedField(name="patient_age", value="", confidence=0.9).normalised()
        self.assertEqual(field.value, "Not stated")
        self.assertEqual(field.confidence, 0.0)

    def test_unknown_spellings_collapse(self):
        for spelling in ["unknown", "N/A", "n/a", "Not Stated", "none stated", "  "]:
            with self.subTest(spelling=spelling):
                field = ExtractedField(
                    name="x", value=spelling, confidence=0.95
                ).normalised()
                self.assertEqual(field.value, "Not stated")
                self.assertEqual(field.confidence, 0.0)

    def test_real_value_is_kept(self):
        field = ExtractedField(
            name="patient_age", value=" 54 years ", confidence=0.9
        ).normalised()
        self.assertEqual(field.value, "54 years")
        self.assertEqual(field.confidence, 0.9)

    def test_missing_fields_are_added_as_not_stated(self):
        fields, _ = parse_fields({"fields": []}, Category.MI, "source text")
        self.assertEqual(len(fields), 2)
        self.assertTrue(all(f.value == "Not stated" for f in fields))


class TestQuoteVerification(unittest.TestCase):
    """The check that catches confident fabrication."""

    SOURCE = (
        "[form.pdf p.1]\nThe patient is a 54-year-old female, weight 68 kg, "
        "with a history of hypertension."
    )

    def test_exact_quote_verifies(self):
        self.assertTrue(verify_quote("54-year-old female", self.SOURCE))

    def test_quote_across_line_break_verifies(self):
        self.assertTrue(verify_quote("a 54-year-old\nfemale, weight 68 kg", self.SOURCE))

    def test_case_insensitive(self):
        self.assertTrue(verify_quote("WEIGHT 68 KG", self.SOURCE))

    def test_invented_quote_fails(self):
        self.assertFalse(verify_quote("the patient weighed 92 kg", self.SOURCE))

    def test_empty_quote_fails(self):
        self.assertFalse(verify_quote("", self.SOURCE))

    def test_fabricated_field_is_demoted_and_flagged(self):
        payload = {
            "fields": [
                {
                    "name": "patient_weight",
                    "value": "92 kg",
                    "confidence": 0.95,
                    "source": "form.pdf p.1",
                    "quote": "the patient weighed 92 kg",
                }
            ],
            "narrative": "",
        }
        fields, warnings = parse_fields(payload, Category.ICSR, self.SOURCE)
        weight = next(f for f in fields if f.name == "patient_weight")
        # Confidence must be cut, and the discrepancy surfaced to a reviewer.
        self.assertLess(weight.confidence, 0.95)
        self.assertTrue(any("quote not found" in w for w in warnings))

    def test_value_without_quote_is_capped(self):
        payload = {
            "fields": [
                {
                    "name": "patient_age",
                    "value": "54 years",
                    "confidence": 0.99,
                    "source": "form.pdf p.1",
                    "quote": "",
                }
            ],
            "narrative": "",
        }
        fields, warnings = parse_fields(payload, Category.ICSR, self.SOURCE)
        age = next(f for f in fields if f.name == "patient_age")
        self.assertLessEqual(age.confidence, 0.5)
        self.assertTrue(any("without a supporting quote" in w for w in warnings))

    def test_verified_field_keeps_confidence(self):
        payload = {
            "fields": [
                {
                    "name": "patient_age",
                    "value": "54 years",
                    "confidence": 0.93,
                    "source": "form.pdf p.1",
                    "quote": "a 54-year-old female",
                }
            ],
            "narrative": "",
        }
        fields, warnings = parse_fields(payload, Category.ICSR, self.SOURCE)
        age = next(f for f in fields if f.name == "patient_age")
        self.assertEqual(age.confidence, 0.93)
        self.assertEqual(warnings, [])


class TestStubProvider(unittest.TestCase):
    """The stub must be safe, not clever."""

    def test_stub_extracts_nothing(self):

        payload = StubProvider().generate_json("anything", extraction_schema(ICSR_FIELDS))
        self.assertTrue(all(f["value"] == "Not stated" for f in payload["fields"]))
        self.assertTrue(all(f["confidence"] == 0.0 for f in payload["fields"]))

    def test_stub_confidence_is_capped_low(self):

        payload = StubProvider().generate_json(
            "rash adverse reaction hospital", CLASSIFICATION_SCHEMA
        )
        self.assertTrue(all(v["confidence"] <= 0.45 for v in payload["verdicts"]))

    def test_stub_output_is_marked(self):

        payload = StubProvider().generate_json("x", CLASSIFICATION_SCHEMA)
        self.assertTrue(all("[offline stub]" in v["reason"] for v in payload["verdicts"]))

    def test_stub_scores_only_the_message_not_the_instructions(self):
        """Regression: the stub must ignore the prompt's own wording.

        The classification prompt names every signal word in its definitions
        ("rash", "broken seal", "dosing"...). Scoring the whole prompt matched
        all four categories on every message -- including obvious marketing.
        """

        message = parse_message_file(SAMPLES / "irrelevant_marketing.eml")
        payload = StubProvider().generate_json(
            build_prompt(message), CLASSIFICATION_SCHEMA
        )
        applied = [v["category"] for v in payload["verdicts"] if v["applies"]]
        self.assertEqual(applied, ["NOT_RELEVANT"])

    def test_stub_finds_dual_label_case(self):

        message = parse_message_file(SAMPLES / "icsr_and_pqc_combined.eml")
        payload = StubProvider().generate_json(
            build_prompt(message), CLASSIFICATION_SCHEMA
        )
        applied = {v["category"] for v in payload["verdicts"] if v["applies"]}
        self.assertEqual(applied, {"ICSR", "PQC"})

    def test_client_falls_back_to_stub_without_any_key(self):
        """With no provider configured, the chain is the stub alone."""
        keys = ["GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"]
        saved = {key: os.environ.pop(key, None) for key in keys}
        try:
            client = LLMClient()
            self.assertTrue(client.is_stub)
            self.assertEqual(len(client.providers), 1)
        finally:
            for key, value in saved.items():
                if value:
                    os.environ[key] = value

    def test_stub_always_terminates_the_chain(self):
        """There must always be something left to call."""
        self.assertTrue(build_provider_chain()[-1].is_stub)


class TestProviderFailover(unittest.TestCase):
    """An exhausted provider must hand off, not fail the request."""

    class Exhausted:
        """A provider whose allowance is spent."""

        name = "exhausted"
        is_stub = False
        supports_vision = True

        def generate_json(self, prompt, schema, images=None):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded, limit: 20")

    class Working:
        """A provider that answers normally."""

        name = "working"
        is_stub = False
        supports_vision = True

        def generate_json(self, prompt, schema, images=None):
            return {"ok": True}

    class TextOnly:
        """A provider that cannot read images."""

        name = "text-only"
        is_stub = False
        supports_vision = False

        def generate_json(self, prompt, schema, images=None):
            return {"ok": True}

    def client_with(self, *providers) -> LLMClient:
        """Build a client over an explicit provider chain."""
        client = LLMClient(StubProvider())
        client.providers = list(providers)
        client.provider = client.providers[0]
        return client

    def test_exhausted_provider_hands_off_to_the_next(self):
        client = self.client_with(self.Exhausted(), self.Working())
        response = client.generate_json("x", {"type": "object"})
        self.assertEqual(response.data, {"ok": True})
        self.assertEqual(response.model, "working")

    def test_all_exhausted_raises_a_readable_error(self):
        client = self.client_with(self.Exhausted(), self.Exhausted())
        with self.assertRaises(AllProvidersExhausted) as ctx:
            client.generate_json("x", {"type": "object"})
        # This message reaches a reviewer's screen, so it must say what to do
        # about it, not only what broke.
        self.assertIn("AI usage limit reached", str(ctx.exception))
        self.assertIn("upgrade to a paid plan", str(ctx.exception))

    def test_text_only_provider_is_skipped_for_images(self):
        # Failing a scanned page over to a provider that cannot see would
        # return an empty transcription and look like a blank document.
        client = self.client_with(self.TextOnly(), self.Working())
        response = client.generate_json("x", {"type": "object"}, images=[b"png"])
        self.assertEqual(response.model, "working")


class TestPromptContent(unittest.TestCase):
    """The prompt must carry the rules the score depends on."""

    @classmethod
    def setUpClass(cls):
        if not (SAMPLES / "icsr_full_rash.eml").exists():
            raise unittest.SkipTest("corpus not generated")
        cls.message = parse_message_file(SAMPLES / "icsr_full_rash.eml")

    def test_classification_prompt_states_multi_label(self):
        provider = FakeProvider({"verdicts": []})
        classify_message(self.message, LLMClient(provider))
        prompt = provider.prompts[0]
        self.assertIn("NOT mutually exclusive", prompt)
        self.assertIn("BOTH ICSR and PQC", prompt)

    def test_extraction_prompt_states_not_stated_rule(self):
        provider = FakeProvider({"fields": [], "narrative": ""})
        extract_fields(self.message, Category.ICSR, LLMClient(provider))
        prompt = provider.prompts[0]
        self.assertIn("Not stated", prompt)
        self.assertIn("Do NOT infer", prompt)

    def test_prompt_carries_source_tags(self):
        provider = FakeProvider({"fields": [], "narrative": ""})
        extract_fields(self.message, Category.ICSR, LLMClient(provider))
        # Source tags must reach the model or it cannot cite them back.
        self.assertIn("(email body)", provider.prompts[0])

    def test_attachment_tables_reach_the_prompt(self):
        message = parse_message_file(SAMPLES / "icsr_full_hepatic.eml")
        provider = FakeProvider({"fields": [], "narrative": ""})
        extract_fields(message, Category.ICSR, LLMClient(provider))
        prompt = provider.prompts[0]
        # The lab table must arrive as a table, not as flattened prose.
        self.assertIn("| Alanine aminotransferase | 512 | U/L |", prompt)


class TestEndToEndWithStub(unittest.TestCase):
    """The whole path must run offline without a key."""

    @classmethod
    def setUpClass(cls):
        if not (SAMPLES / "irrelevant_marketing.eml").exists():
            raise unittest.SkipTest("corpus not generated")

    def test_classification_runs_offline(self):
        message = parse_message_file(SAMPLES / "irrelevant_marketing.eml")
        result = classify_message(message, LLMClient(StubProvider()))
        self.assertEqual(len(result.verdicts), 4)
        self.assertTrue(result.to_dict()["model"], "offline-stub")

    def test_extraction_runs_offline_and_states_nothing(self):
        message = parse_message_file(SAMPLES / "icsr_full_rash.eml")
        extraction, _ = extract_fields(message, Category.ICSR, LLMClient(StubProvider()))
        self.assertEqual(extraction.completeness, 0.0)
        self.assertEqual(len(extraction.fields), 15)


if __name__ == "__main__":
    unittest.main()
