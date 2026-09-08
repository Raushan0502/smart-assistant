"""
DRF serializers for the reviewer API.

The queue list and the detail view are deliberately different shapes. The queue
is polled repeatedly and renders many rows, so it carries only what the list
displays; the detail view carries everything, including provenance and the
audit trail. Serving the detail shape to a list endpoint is the usual reason
these screens get slow.
"""
from __future__ import annotations

from rest_framework import serializers

from .models import (
    AuditEvent,
    Classification,
    Document,
    ExtractedField,
    Extraction,
    Message,
    ReviewAction,
)


class ClassificationSerializer(serializers.ModelSerializer):
    """One category verdict, including the ones that did not apply."""

    class Meta:
        model = Classification
        fields = ["category", "applies", "confidence", "reason", "model_name"]


class ExtractedFieldSerializer(serializers.ModelSerializer):
    """One fact with its confidence and provenance."""

    is_stated = serializers.BooleanField(read_only=True)

    class Meta:
        model = ExtractedField
        fields = [
            "name",
            "value",
            "confidence",
            "is_stated",
            "source_document",
            "source_page",
            "source_quote",
            "quote_verified",
        ]


class ExtractionSerializer(serializers.ModelSerializer):
    """A category's extracted fields plus the AI narrative."""

    fields_data = ExtractedFieldSerializer(source="fields", many=True, read_only=True)

    class Meta:
        model = Extraction
        fields = ["category", "narrative", "completeness", "model_name", "fields_data"]


class DocumentSerializer(serializers.ModelSerializer):
    """One document, with its tables kept structured."""

    class Meta:
        model = Document
        fields = [
            "document_id",
            "file_name",
            "media_type",
            "flavour",
            "language",
            "page_count",
            "processed",
            "extracted_text",
            "tables",
            "images",
            "doc_metadata",
            "warnings",
            "summary",
            "looks_relevant",
            "relevance_reason",
            "summary_confidence",
            "ocr_confidence",
        ]


class ReviewActionSerializer(serializers.ModelSerializer):
    """One timestamped human decision."""

    class Meta:
        model = ReviewAction
        fields = [
            "id",
            "action",
            "reviewer",
            "created_at",
            "field_name",
            "original_value",
            "new_value",
            "category",
            "note",
        ]
        read_only_fields = ["id", "created_at"]


class AuditEventSerializer(serializers.ModelSerializer):
    """One recorded AI call."""

    class Meta:
        model = AuditEvent
        fields = [
            "event_type",
            "model_name",
            "is_stub",
            "prompt_sha256",
            "prompt_chars",
            "latency_ms",
            "succeeded",
            "error",
            "created_at",
        ]


class MessageListSerializer(serializers.ModelSerializer):
    """The lightweight shape the review queue renders."""

    categories = serializers.ListField(child=serializers.CharField(), read_only=True)
    top_confidence = serializers.SerializerMethodField()
    document_count = serializers.SerializerMethodField()
    needs_attention = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = [
            "id",
            "message_id",
            "subject",
            "sender",
            "sent_at",
            "received_at",
            "processing_status",
            "review_status",
            "processing_ms",
            "categories",
            "top_confidence",
            "document_count",
            "needs_attention",
        ]

    def get_top_confidence(self, message: Message) -> float | None:
        """Confidence of the most confident applying category."""
        applied = [c.confidence for c in message.classifications.all() if c.applies]
        return round(max(applied), 3) if applied else None

    def get_document_count(self, message: Message) -> int:
        """How many documents arrived with this message.

        Reads the annotation the view attaches, so listing a page of messages
        costs one query rather than one per row.
        """
        return getattr(message, "document_total", 0)

    def get_needs_attention(self, message: Message) -> bool:
        """Whether a reviewer should prioritise this message.

        True when processing failed, when any warning was raised, or when a
        field was stated without a verifiable quote -- the three cases where
        the AI's output should not be taken at face value.

        The unverified count is an annotation from the view for the same
        reason as above: computing it per row turned a 16-row page into 49
        queries.
        """
        if message.processing_status == "FAILED" or message.warnings:
            return True
        return getattr(message, "unverified_total", 0) > 0


class MessageDetailSerializer(serializers.ModelSerializer):
    """The full record, including provenance and the audit trail."""

    documents = DocumentSerializer(many=True, read_only=True)
    classifications = ClassificationSerializer(many=True, read_only=True)
    extractions = ExtractionSerializer(many=True, read_only=True)
    review_actions = ReviewActionSerializer(many=True, read_only=True)
    audit_events = AuditEventSerializer(many=True, read_only=True)
    categories = serializers.ListField(child=serializers.CharField(), read_only=True)

    class Meta:
        model = Message
        fields = [
            "id",
            "message_id",
            "subject",
            "sender",
            "recipient",
            "sent_at",
            "received_at",
            "body_text",
            "language",
            "processing_status",
            "review_status",
            "processing_ms",
            "error",
            "warnings",
            "headers",
            "categories",
            "documents",
            "classifications",
            "extractions",
            "review_actions",
            "audit_events",
        ]
