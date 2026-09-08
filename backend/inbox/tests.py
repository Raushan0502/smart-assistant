"""
Tests for the backend: persistence, the mailbox guard, the queue and the API.

Run against SQLite so no container is needed:

    USE_SQLITE=true python manage.py test inbox

The models use no Oracle-specific column types, so the schema is equivalent.
The two behaviours that *are* Oracle-specific -- the PL/SQL package and the
NCLOB constraints that shaped the model -- are exercised by
``db/apply_plsql.py --verify`` against the real database instead, because
testing them against SQLite would prove nothing.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from django.db import connection
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.test.utils import CaptureQueriesContext

from .ai_client import AIServiceError
from .mailbox import SYNTHETIC_ONLY_SEARCH, MailboxError, poll_imap
from .models import (
    AuditEvent,
    Classification,
    Document,
    ExtractedField,
    Extraction,
    Message,
    ProcessingStatus,
    ReviewAction,
    ReviewStatus,
)
from .pipeline import mark_failed, persist_analysis
from .queue import Job, ProcessingQueue
from .serializers import MessageListSerializer


def analysis_fixture(message_id: str = "msg-1", **overrides) -> dict:
    """A representative AI-service payload, shaped like the real thing."""
    payload = {
        "message": {
            "message_id": message_id,
            "subject": "Adverse reaction report - Cardiozan",
            "sender": "Dr Amara Osei <a.osei@example.test>",
            "recipient": "pv-intake@example.test",
            "sent_at": "2026-03-16T09:00:00+00:00",
            "metadata": {"X-Synthetic-Case-Id": "icsr_full_rash"},
            "body": {
                "document_id": f"{message_id}::body",
                "file_name": "(email body)",
                "media_type": "text/plain",
                "flavour": "unknown",
                "language": "en",
                "page_count": 0,
                "processed": True,
                "metadata": {},
                "warnings": [],
                "blocks": [
                    {
                        "kind": "text",
                        "text": "The patient is a 54-year-old female.",
                        "source": {"page": None},
                    }
                ],
            },
            "attachments": [
                {
                    "document_id": f"{message_id}::att0",
                    "file_name": "form.pdf",
                    "media_type": "application/pdf",
                    "flavour": "digital",
                    "language": "en",
                    "page_count": 1,
                    "processed": True,
                    "metadata": {"byte_size": "3305"},
                    "warnings": [],
                    "blocks": [
                        {
                            "kind": "table",
                            "header": ["Test", "Result", "Unit"],
                            "rows": [["Potassium", "6.8", "mmol/L"]],
                            "source": {"page": 1},
                        }
                    ],
                },
                {
                    "document_id": f"{message_id}::att1",
                    "file_name": "roster.csv",
                    "media_type": "text/csv",
                    "flavour": "unknown",
                    "language": "en",
                    "page_count": 0,
                    "processed": False,
                    "metadata": {},
                    "warnings": ["Attachment type 'text/csv' is not processed."],
                    "blocks": [],
                },
            ],
        },
        "classification": {
            "model": "test-model",
            "verdicts": [
                {"category": "ICSR", "applies": True, "confidence": 0.92, "reason": "reaction"},
                {"category": "PQC", "applies": False, "confidence": 0.05, "reason": "no defect"},
                {"category": "MI", "applies": False, "confidence": 0.02, "reason": "no question"},
                {"category": "NOT_RELEVANT", "applies": False, "confidence": 0.01, "reason": "-"},
            ],
        },
        "extractions": [
            {
                "category": "ICSR",
                "narrative": "A 54-year-old developed a rash.",
                "model": "test-model",
                "completeness": 0.5,
                "fields": {
                    "patient_age": {
                        "value": "54 years",
                        "confidence": 0.95,
                        "source": "form.pdf p.1",
                        "quote": "a 54-year-old female",
                    },
                    "patient_weight": {
                        "value": "Not stated",
                        "confidence": 0.0,
                        "source": "",
                        "quote": "",
                    },
                    "product_name": {
                        "value": "Cardiozan",
                        "confidence": 0.9,
                        "source": "(email body)",
                        "quote": "",
                    },
                },
            }
        ],
        "summaries": [
            {
                "document_id": f"{message_id}::att0",
                "summary": "A completed report form.",
                "looks_relevant": True,
                "relevance_reason": "Describes an adverse reaction.",
                "confidence": 0.8,
                "model": "test-model",
            }
        ],
        "warnings": ["one warning"],
        "audit_events": [
            {
                "event_type": "CLASSIFY",
                "model_name": "test-model",
                "is_stub": False,
                "prompt_sha256": "abc123",
                "prompt_chars": 900,
                "latency_ms": 812.5,
                "succeeded": True,
                "error": "",
                "document_id": "",
            }
        ],
        "timings": {"stages_ms": {"parse": 12.0}, "total_ms": 980.0},
    }
    payload.update(overrides)
    return payload


class TestMailboxGuard(TestCase):
    """The guard that stops real personal mail being ingested."""

    def test_default_search_is_not_all(self):
        # An ALL search against a mailbox containing real correspondence would
        # send it to a cloud LLM. The default must be restrictive.
        self.assertNotEqual(SYNTHETIC_ONLY_SEARCH.strip().upper(), "ALL")

    def test_default_search_matches_only_generated_messages(self):
        # Every message this project sends carries this header; nothing else
        # in an ordinary mailbox does.
        self.assertIn("X-Synthetic-Case-Id", SYNTHETIC_ONLY_SEARCH)
        self.assertIn("HEADER", SYNTHETIC_ONLY_SEARCH.upper())

    @override_settings(MAILBOX={"offline": False, "user": "", "password": "", "host": "h",
                               "port": 993, "folder": "INBOX", "poll_seconds": 60, "search": ""})
    def test_missing_credentials_raise_a_clear_error(self):

        with self.assertRaises(MailboxError) as ctx:
            poll_imap()
        self.assertIn("IMAP_USER", str(ctx.exception))


class TestPersistence(TestCase):
    """persist_analysis is the only path AI output takes into the database."""

    def test_persists_message_and_documents(self):
        message = persist_analysis(analysis_fixture())
        self.assertEqual(message.processing_status, ProcessingStatus.READY)
        self.assertEqual(message.processing_ms, 980)
        self.assertEqual(message.documents.count(), 3)

    def test_all_four_verdicts_stored_including_negatives(self):
        # Keeping the negatives makes the decision auditable: a reviewer can
        # see PQC was considered and rejected, with the reason.
        message = persist_analysis(analysis_fixture())
        self.assertEqual(message.classifications.count(), 4)
        self.assertEqual(message.categories, ["ICSR"])
        pqc = message.classifications.get(category="PQC")
        self.assertFalse(pqc.applies)
        self.assertEqual(pqc.reason, "no defect")

    def test_field_provenance_is_persisted(self):
        persist_analysis(analysis_fixture())
        age = ExtractedField.objects.get(name="patient_age")
        self.assertEqual(age.value, "54 years")
        self.assertEqual(age.source_document, "form.pdf p.1")
        self.assertEqual(age.source_page, 1)
        self.assertEqual(age.source_quote, "a 54-year-old female")
        self.assertTrue(age.quote_verified)

    def test_stated_field_without_quote_is_marked_unverified(self):
        persist_analysis(analysis_fixture())
        product = ExtractedField.objects.get(name="product_name")
        self.assertTrue(product.is_stated)
        # No quote means the value cannot be traced back, so it needs review.
        self.assertFalse(product.quote_verified)

    def test_not_stated_field_is_not_flagged_as_unverified(self):
        persist_analysis(analysis_fixture())
        weight = ExtractedField.objects.get(name="patient_weight")
        self.assertFalse(weight.is_stated)
        self.assertFalse(weight.quote_verified)

    def test_tables_stay_structured(self):
        persist_analysis(analysis_fixture())
        attachment = Document.objects.get(file_name="form.pdf")
        self.assertEqual(attachment.tables[0]["header"], ["Test", "Result", "Unit"])
        self.assertEqual(attachment.tables[0]["rows"][0], ["Potassium", "6.8", "mmol/L"])
        self.assertEqual(attachment.tables[0]["page"], 1)

    def test_unprocessed_attachment_is_kept_not_dropped(self):
        persist_analysis(analysis_fixture())
        csv = Document.objects.get(file_name="roster.csv")
        self.assertFalse(csv.processed)
        self.assertTrue(csv.warnings)

    def test_summary_attaches_to_the_right_document(self):
        persist_analysis(analysis_fixture())
        attachment = Document.objects.get(file_name="form.pdf")
        self.assertEqual(attachment.summary, "A completed report form.")
        self.assertTrue(attachment.looks_relevant)

    def test_audit_event_recorded(self):
        message = persist_analysis(analysis_fixture())
        event = message.audit_events.get()
        self.assertEqual(event.event_type, "CLASSIFY")
        self.assertEqual(event.prompt_sha256, "abc123")
        self.assertAlmostEqual(event.latency_ms, 812.5)

    def test_reprocessing_is_idempotent(self):
        # Polling the same mailbox twice must not duplicate records.
        persist_analysis(analysis_fixture())
        persist_analysis(analysis_fixture())
        self.assertEqual(Message.objects.count(), 1)
        self.assertEqual(Classification.objects.count(), 4)
        self.assertEqual(Extraction.objects.count(), 1)
        self.assertEqual(ExtractedField.objects.count(), 3)

    def test_overlong_value_is_truncated_not_fatal(self):
        payload = analysis_fixture()
        payload["extractions"][0]["fields"]["patient_age"]["value"] = "x" * 5000
        persist_analysis(payload)
        self.assertEqual(len(ExtractedField.objects.get(name="patient_age").value), 2000)

    def test_failure_is_recorded_rather_than_dropped(self):
        # The mail arrived; a reviewer must know it exists even if unreadable.
        message = mark_failed("bad-1", "Broken message", "AI service unreachable")
        self.assertEqual(message.processing_status, ProcessingStatus.FAILED)
        self.assertIn("unreachable", message.error)
        self.assertEqual(AuditEvent.objects.filter(succeeded=False).count(), 1)


class TestQueue(TransactionTestCase):
    """Submitting work must return immediately and never lose a message.

    TransactionTestCase rather than TestCase: the workers are real threads, and
    TestCase wraps each test in a transaction its own connection never commits,
    so a worker thread would never see the data it wrote.
    """

    def test_processes_and_persists(self):
        queue = ProcessingQueue(workers=1)
        queue.start()
        try:
            with patch("inbox.queue.ai_client.process_message", return_value=analysis_fixture()):
                queue.submit(Job(raw=b"raw", file_name="m.eml"))
                self.assertTrue(queue.join(timeout=20))
        finally:
            queue.stop()
        self.assertEqual(Message.objects.count(), 1)
        self.assertEqual(queue.stats().processed, 1)

    def test_ai_failure_marks_message_failed_and_keeps_worker_alive(self):
        queue = ProcessingQueue(workers=1)
        queue.start()
        try:
            with patch(
                "inbox.queue.ai_client.process_message", side_effect=RuntimeError("service down")
            ):
                queue.submit(Job(raw=b"raw", file_name="bad.eml", message_id="bad-1"))
                self.assertTrue(queue.join(timeout=20))
            # The worker must survive to process the next job.
            with patch("inbox.queue.ai_client.process_message", return_value=analysis_fixture()):
                queue.submit(Job(raw=b"raw", file_name="ok.eml"))
                self.assertTrue(queue.join(timeout=20))
        finally:
            queue.stop()

        self.assertEqual(queue.stats().failed, 1)
        self.assertEqual(queue.stats().processed, 1)
        self.assertEqual(
            Message.objects.filter(processing_status=ProcessingStatus.FAILED).count(), 1
        )


class TestSerializers(TestCase):
    """The queue row must flag what a reviewer should not skim past."""

    def test_needs_attention_when_field_unverified(self):
        message = persist_analysis(analysis_fixture())
        data = MessageListSerializer(message).data
        # product_name is stated with no quote, so it cannot be traced back.
        self.assertTrue(data["needs_attention"])

    def test_needs_attention_when_processing_failed(self):
        message = mark_failed("bad-2", "s", "boom")
        self.assertTrue(MessageListSerializer(message).data["needs_attention"])

    def test_top_confidence_is_the_applying_category(self):
        message = persist_analysis(analysis_fixture())
        self.assertAlmostEqual(MessageListSerializer(message).data["top_confidence"], 0.92)


@override_settings(ALLOWED_HOSTS=["testserver"])
class TestApi(TestCase):
    """The endpoints the reviewer screen depends on."""

    def setUp(self):
        self.message = persist_analysis(analysis_fixture())
        self.client = Client()

    def test_queue_lists_messages(self):
        response = self.client.get("/api/messages/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)

    def test_queue_query_count_does_not_grow_with_rows(self):
        """The queue endpoint must cost a constant number of queries.

        It was 49 for 16 rows: the serializer counted documents and unverified
        fields per row, and Message.categories called .filter() on a prefetched
        relation, which issues a fresh query. The UI polls this endpoint, so the
        cost recurred constantly.
        """
        for index in range(8):
            persist_analysis(analysis_fixture(message_id=f"bulk-{index}"))

        with CaptureQueriesContext(connection) as ctx:
            response = self.client.get("/api/messages/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 9)
        # Page, count, and the classifications prefetch. Nothing per row.
        self.assertLessEqual(len(ctx.captured_queries), 5)

    def test_category_filter(self):
        self.assertEqual(self.client.get("/api/messages/?category=ICSR").json()["count"], 1)
        self.assertEqual(self.client.get("/api/messages/?category=PQC").json()["count"], 0)

    def test_detail_includes_provenance_and_audit(self):
        data = self.client.get(f"/api/messages/{self.message.id}/").json()
        field = next(
            f for f in data["extractions"][0]["fields_data"] if f["name"] == "patient_age"
        )
        self.assertEqual(field["source_quote"], "a 54-year-old female")
        self.assertEqual(len(data["audit_events"]), 1)

    def test_accept_records_a_timestamped_action(self):
        response = self.client.post(f"/api/messages/{self.message.id}/accept/", {})
        self.assertEqual(response.status_code, 200)
        self.message.refresh_from_db()
        self.assertEqual(self.message.review_status, ReviewStatus.ACCEPTED)
        action = ReviewAction.objects.get()
        self.assertEqual(action.action, ReviewAction.Action.ACCEPT)
        self.assertIsNotNone(action.created_at)

    def test_override_preserves_the_original_value(self):
        response = self.client.post(
            f"/api/messages/{self.message.id}/override_field/",
            data=json.dumps(
                {"field_name": "patient_weight", "new_value": "68 kg", "reviewer": "raushan"}
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        action = ReviewAction.objects.get(action=ReviewAction.Action.OVERRIDE_FIELD)
        # The AI's value must survive alongside the human's, not be overwritten.
        self.assertEqual(action.original_value, "Not stated")
        self.assertEqual(action.new_value, "68 kg")
        self.assertEqual(action.reviewer, "raushan")

        field = ExtractedField.objects.get(name="patient_weight")
        self.assertEqual(field.value, "68 kg")
        self.assertEqual(field.confidence, 1.0)
        self.assertTrue(field.quote_verified)

    def test_override_unknown_field_is_404(self):
        response = self.client.post(
            f"/api/messages/{self.message.id}/override_field/",
            data=json.dumps({"field_name": "nope", "new_value": "x"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)

    def test_override_category(self):
        self.client.post(
            f"/api/messages/{self.message.id}/override_category/",
            data=json.dumps({"category": "PQC", "applies": True}),
            content_type="application/json",
        )
        self.message.refresh_from_db()
        self.assertIn("PQC", self.message.categories)
        self.assertEqual(
            ReviewAction.objects.filter(action=ReviewAction.Action.OVERRIDE_CATEGORY).count(), 1
        )

    def test_audit_endpoint_returns_both_trails(self):
        self.client.post(f"/api/messages/{self.message.id}/accept/", {})
        data = self.client.get(f"/api/messages/{self.message.id}/audit/").json()
        self.assertEqual(len(data["ai_calls"]), 1)
        self.assertEqual(len(data["review_actions"]), 1)

    def test_status_reports_ai_service_unreachable_without_crashing(self):
        with patch(
            "inbox.views.ai_client.health",
            side_effect=AIServiceError("down"),
        ):
            data = self.client.get("/api/status/").json()
        self.assertEqual(data["ai_service"]["status"], "unreachable")
        self.assertEqual(data["messages"]["total"], 1)

    def test_ingest_reports_mailbox_failure_as_503(self):

        with patch("inbox.views.mailbox.ingest", side_effect=MailboxError("no credentials")):
            response = self.client.post("/api/ingest/")
        self.assertEqual(response.status_code, 503)
        self.assertIn("no credentials", response.json()["detail"])
