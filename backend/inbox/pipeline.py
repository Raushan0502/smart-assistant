"""
Persist what the AI service returns.

This module owns the translation from the AI service's JSON into the relational
model, and it is the only place that writes AI output to the database. Keeping
that in one function means the audit guarantees are enforceable: every write
path goes through :func:`persist_analysis`, so there is no route by which a
fact reaches the database without its provenance.

Writes are wrapped in a single transaction per message. A half-persisted
message -- classified but with no fields, or fields with no audit trail -- would
be worse than one that failed cleanly and can be retried.
"""
from __future__ import annotations

import logging
from datetime import datetime

from django.db import transaction
from django.utils.dateparse import parse_datetime

from .models import (
    AuditEvent,
    Classification,
    Document,
    ExtractedField,
    Extraction,
    Message,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)

# Maps the AI service's flavour strings onto the Document choices.
FLAVOUR_MAP = {
    "digital": Document.Flavour.DIGITAL,
    "scanned": Document.Flavour.SCANNED,
    "article": Document.Flavour.ARTICLE,
    "non_english": Document.Flavour.NON_ENGLISH,
    "unknown": Document.Flavour.UNKNOWN,
}


def _parse_sent_at(value: str) -> datetime | None:
    """Parse the message date, tolerating whatever the sender put there."""
    if not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def _tables_from(document_payload: dict) -> list[dict]:
    """Pull table blocks out of a document payload, keeping them structured."""
    return [
        {
            "header": block.get("header", []),
            "rows": block.get("rows", []),
            "page": (block.get("source") or {}).get("page"),
        }
        for block in document_payload.get("blocks", [])
        if block.get("kind") == "table"
    ]


def _images_from(document_payload: dict) -> list[dict]:
    """Pull image blocks out of a document payload."""
    return [
        {
            "width": block.get("width"),
            "height": block.get("height"),
            "description": block.get("description"),
            "needs_review": block.get("needs_review", True),
            "page": (block.get("source") or {}).get("page"),
        }
        for block in document_payload.get("blocks", [])
        if block.get("kind") == "image"
    ]


def _text_from(document_payload: dict) -> str:
    """Concatenate the prose blocks of a document payload."""
    return "\n\n".join(
        block.get("text", "")
        for block in document_payload.get("blocks", [])
        if block.get("kind") == "text"
    )


def _save_document(
    message: Message, payload: dict, summaries: dict[str, dict], is_body: bool
) -> Document:
    """Persist one document (body or attachment) with its structure."""
    summary = summaries.get(payload.get("document_id", ""), {})
    flavour = (
        Document.Flavour.EMAIL_BODY
        if is_body
        else FLAVOUR_MAP.get(payload.get("flavour", "unknown"), Document.Flavour.UNKNOWN)
    )
    metadata = payload.get("metadata", {}) or {}
    ocr_confidence = metadata.get("ocr_confidence")

    document, _ = Document.objects.update_or_create(
        message=message,
        document_id=payload.get("document_id", ""),
        defaults={
            "file_name": payload.get("file_name", ""),
            "media_type": payload.get("media_type", ""),
            "flavour": flavour,
            "language": payload.get("language", "en"),
            "page_count": payload.get("page_count", 0),
            "processed": payload.get("processed", True),
            "extracted_text": _text_from(payload),
            "tables": _tables_from(payload),
            "images": _images_from(payload),
            "doc_metadata": metadata,
            "warnings": payload.get("warnings", []),
            "summary": summary.get("summary", ""),
            "looks_relevant": summary.get("looks_relevant"),
            "relevance_reason": summary.get("relevance_reason", ""),
            "summary_confidence": summary.get("confidence"),
            "ocr_confidence": float(ocr_confidence) if ocr_confidence else None,
        },
    )
    return document


def _save_fields(extraction: Extraction, fields_payload: dict) -> None:
    """Persist extracted fields with their provenance.

    ``quote_verified`` is derived rather than trusted: the AI service reduces
    confidence when a quote cannot be found in the source, so a stated field
    arriving with an empty quote is recorded as unverified here too.
    """
    for name, field in fields_payload.items():
        value = field.get("value", "Not stated")
        quote = field.get("quote", "") or ""
        source = field.get("source", "") or ""

        page = None
        if " p." in source:
            tail = source.rsplit(" p.", 1)[-1].strip()
            if tail.isdigit():
                page = int(tail)

        is_stated = value.strip().lower() not in {"", "not stated", "unknown", "n/a"}
        ExtractedField.objects.update_or_create(
            extraction=extraction,
            name=name,
            defaults={
                # Guard the column bound; a pathological value must not fail
                # the whole message.
                "value": value[:2000],
                "confidence": float(field.get("confidence", 0.0) or 0.0),
                "source_document": source[:500],
                "source_page": page,
                "source_quote": quote,
                "quote_verified": bool(quote) and is_stated,
            },
        )


@transaction.atomic
def persist_analysis(analysis: dict) -> Message:
    """Persist a complete AI analysis, returning the saved message.

    Idempotent on ``message_id``: re-processing the same email updates the
    existing record rather than creating a duplicate, so a mailbox can be
    re-polled safely.
    """
    payload = analysis["message"]
    timings = analysis.get("timings", {})

    message, _ = Message.objects.update_or_create(
        message_id=payload["message_id"],
        defaults={
            "subject": payload.get("subject", "")[:1000],
            "sender": payload.get("sender", "")[:400],
            "recipient": payload.get("recipient", "")[:400],
            "sent_at": _parse_sent_at(payload.get("sent_at", "")),
            "body_text": _text_from(payload.get("body", {})),
            "language": payload.get("body", {}).get("language", "en"),
            "processing_status": ProcessingStatus.READY,
            "processing_ms": int(timings.get("total_ms", 0)),
            "warnings": analysis.get("warnings", []),
            "headers": payload.get("metadata", {}),
            "error": "",
        },
    )

    summaries = {s["document_id"]: s for s in analysis.get("summaries", [])}
    _save_document(message, payload.get("body", {}), summaries, is_body=True)
    for attachment in payload.get("attachments", []):
        _save_document(message, attachment, summaries, is_body=False)

    for verdict in analysis.get("classification", {}).get("verdicts", []):
        Classification.objects.update_or_create(
            message=message,
            category=verdict["category"],
            defaults={
                "applies": verdict.get("applies", False),
                "confidence": float(verdict.get("confidence", 0.0) or 0.0),
                "reason": verdict.get("reason", ""),
                "model_name": analysis.get("classification", {}).get("model", ""),
            },
        )

    for item in analysis.get("extractions", []):
        extraction, _ = Extraction.objects.update_or_create(
            message=message,
            category=item["category"],
            defaults={
                "narrative": item.get("narrative", ""),
                "model_name": item.get("model", ""),
                "completeness": float(item.get("completeness", 0.0) or 0.0),
            },
        )
        _save_fields(extraction, item.get("fields", {}))

    for event in analysis.get("audit_events", []):
        AuditEvent.objects.create(
            message=message,
            event_type=event.get("event_type", "CLASSIFY"),
            model_name=event.get("model_name", ""),
            is_stub=event.get("is_stub", False),
            prompt_sha256=event.get("prompt_sha256", ""),
            prompt_chars=event.get("prompt_chars", 0),
            latency_ms=event.get("latency_ms", 0.0),
            succeeded=event.get("succeeded", True),
            error=event.get("error", ""),
            response_summary={"document_id": event.get("document_id", "")},
        )

    logger.info(
        "Persisted %s: %s in %sms",
        message.message_id,
        message.categories or ["no labels"],
        message.processing_ms,
    )
    return message


def mark_failed(message_id: str, subject: str, error: str) -> Message:
    """Record that a message could not be processed.

    A failure is stored rather than dropped: the mail arrived, and a reviewer
    needs to know it exists even though the pipeline could not read it.
    """
    message, _ = Message.objects.update_or_create(
        message_id=message_id,
        defaults={
            "subject": subject[:1000],
            "processing_status": ProcessingStatus.FAILED,
            "error": error[:4000],
        },
    )
    AuditEvent.objects.create(
        message=message,
        event_type=AuditEvent.EventType.CLASSIFY,
        succeeded=False,
        error=error[:4000],
    )
    return message
