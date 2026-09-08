"""
End-to-end orchestration for one message.

    parse -> preprocess -> OCR any scans -> classify -> extract -> summarise

Kept separate from the HTTP layer in :mod:`ai_service.api` so the whole
pipeline is callable and testable in-process, without a running server.

Two things are recorded as first-class output rather than logged and forgotten:

**Per-stage timings.** The assignment asks how long each document takes. Timing
each stage separately is what makes that answer useful -- "12 seconds" is not
actionable, "11 of those 12 seconds were OCR" is.

**An audit event per model call.** Every call the pipeline makes is returned in
``audit_events``, including failures, so the backend can persist a complete
trace of how a record came to look the way it does.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from .classify import classify_message
from .extract import extract_all
from .llm import LLMClient
from .mail import parse_message
from .models import ExtractedDocument, IngestedMessage, PdfFlavour
from .preprocess import preprocess_message
from .schemas import Category, Classification, DocumentSummary, Extraction
from .summarise import screen_article, summarise_document
from .vision import read_scanned_document

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """One recorded model call or pipeline stage."""

    event_type: str
    model_name: str = ""
    is_stub: bool = False
    prompt_sha256: str = ""
    prompt_chars: int = 0
    latency_ms: float = 0.0
    succeeded: bool = True
    error: str = ""
    document_id: str = ""

    def to_dict(self) -> dict:
        """Serialise for the backend's audit table."""
        return {
            "event_type": self.event_type,
            "model_name": self.model_name,
            "is_stub": self.is_stub,
            "prompt_sha256": self.prompt_sha256,
            "prompt_chars": self.prompt_chars,
            "latency_ms": round(self.latency_ms, 1),
            "succeeded": self.succeeded,
            "error": self.error,
            "document_id": self.document_id,
        }


@dataclass
class ProcessedMessage:
    """Everything the pipeline produced for one message."""

    message: IngestedMessage
    classification: Classification
    extractions: list[Extraction] = field(default_factory=list)
    summaries: list[DocumentSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialise the whole result for the backend."""
        return {
            "message": self.message.to_dict(),
            "classification": self.classification.to_dict(),
            "extractions": [e.to_dict() for e in self.extractions],
            "summaries": [s.to_dict() for s in self.summaries],
            "warnings": self.warnings,
            "audit_events": [a.to_dict() for a in self.audit_events],
            "timings": {
                "stages_ms": {k: round(v, 1) for k, v in self.stage_ms.items()},
                "total_ms": round(self.total_ms, 1),
            },
        }


def elapsed_ms(started: float) -> float:
    """Milliseconds since a ``time.perf_counter()`` reading.

    The assignment asks how long each document takes, and a per-stage breakdown
    is what makes that answer useful: "12 seconds" is not actionable, "11 of
    those 12 were OCR" is.
    """
    return (time.perf_counter() - started) * 1000


def _ocr_scanned_attachments(
    message: IngestedMessage,
    attachment_bytes: dict[str, bytes],
    client: LLMClient,
    events: list[AuditEvent],
) -> list[str]:
    """Run OCR over every scanned attachment, in place."""
    warnings: list[str] = []
    for document in message.attachments:
        if not document.processed or document.flavour is not PdfFlavour.SCANNED:
            continue
        data = attachment_bytes.get(document.file_name)
        if data is None:
            warnings.append(
                f"{document.file_name}: scanned document could not be re-read for OCR."
            )
            continue

        started = time.perf_counter()
        before = len(document.warnings)
        read_scanned_document(document, data, client)
        events.append(
            AuditEvent(
                event_type="OCR",
                model_name=client.provider.name,
                is_stub=client.is_stub,
                latency_ms=(time.perf_counter() - started) * 1000,
                succeeded=bool(document.full_text()),
                document_id=document.document_id,
            )
        )
        warnings.extend(document.warnings[before:])
    return warnings


def _summarise_attachments(
    message: IngestedMessage, client: LLMClient, events: list[AuditEvent]
) -> tuple[list[DocumentSummary], list[str]]:
    """Summarise every processed attachment for the review queue."""
    summaries: list[DocumentSummary] = []
    warnings: list[str] = []
    for document in message.attachments:
        if not document.processed:
            continue
        started = time.perf_counter()
        try:
            summaries.append(summarise_document(document, client))
            succeeded, error = True, ""
        except Exception as exc:  # noqa: BLE001 -- a failed summary is not fatal
            succeeded, error = False, str(exc)
            warnings.append(f"{document.file_name}: summary failed ({exc}).")
        events.append(
            AuditEvent(
                event_type="SUMMARY",
                model_name=client.provider.name,
                is_stub=client.is_stub,
                latency_ms=(time.perf_counter() - started) * 1000,
                succeeded=succeeded,
                error=error,
                document_id=document.document_id,
            )
        )
    return summaries, warnings


def process_message(
    raw: bytes,
    attachment_bytes: dict[str, bytes] | None = None,
    client: LLMClient | None = None,
) -> ProcessedMessage:
    """Run the full pipeline over one raw email.

    ``attachment_bytes`` supplies the original PDF bytes by filename, needed to
    render pages for OCR. When omitted they are recovered from the raw message,
    so callers normally do not pass it.
    """
    client = client or LLMClient()
    stage_ms: dict[str, float] = {}
    events: list[AuditEvent] = []
    warnings: list[str] = []
    overall_started = time.perf_counter()

    started = time.perf_counter()
    message = parse_message(raw)
    if attachment_bytes is None:
        attachment_bytes = _recover_attachment_bytes(raw)
    stage_ms["parse"] = elapsed_ms(started)
    warnings.extend(message.warnings)

    started = time.perf_counter()
    preprocess_message(message)
    stage_ms["preprocess"] = elapsed_ms(started)

    started = time.perf_counter()
    warnings.extend(_ocr_scanned_attachments(message, attachment_bytes, client, events))
    stage_ms["ocr"] = elapsed_ms(started)

    started = time.perf_counter()
    prompt_len = len(message.to_prompt_text())
    try:
        classification = classify_message(message, client)
        succeeded, error = True, ""
    except Exception as exc:  # noqa: BLE001
        logger.error("Classification failed: %s", exc)
        classification = Classification(verdicts=[], model="")
        succeeded, error = False, str(exc)
        warnings.append(f"Classification failed: {exc}")
    stage_ms["classify"] = elapsed_ms(started)
    events.append(
        AuditEvent(
            event_type="CLASSIFY",
            model_name=classification.model or client.provider.name,
            is_stub=client.is_stub,
            # The prompt is identified by hash rather than stored: the full
            # text is large, and the hash still proves which prompt produced
            # this output.
            prompt_sha256=hashlib.sha256(
                message.to_prompt_text().encode("utf-8", errors="replace")
            ).hexdigest(),
            prompt_chars=prompt_len,
            latency_ms=stage_ms["classify"],
            succeeded=succeeded,
            error=error,
        )
    )

    started = time.perf_counter()
    extractions: list[Extraction] = []
    if classification.labels:
        try:
            extractions, extraction_warnings = extract_all(
                message, classification.labels, client
            )
            warnings.extend(extraction_warnings)
            succeeded, error = True, ""
        except Exception as exc:  # noqa: BLE001
            logger.error("Extraction failed: %s", exc)
            succeeded, error = False, str(exc)
            warnings.append(f"Extraction failed: {exc}")
        stage_ms["extract"] = elapsed_ms(started)
        events.append(
            AuditEvent(
                event_type="EXTRACT",
                model_name=client.provider.name,
                is_stub=client.is_stub,
                prompt_chars=prompt_len,
                latency_ms=stage_ms.get("extract", 0.0),
                succeeded=succeeded,
                error=error,
            )
        )
    else:
        stage_ms["extract"] = elapsed_ms(started)

    started = time.perf_counter()
    summaries, summary_warnings = _summarise_attachments(message, client, events)
    warnings.extend(summary_warnings)
    stage_ms["summarise"] = elapsed_ms(started)

    return ProcessedMessage(
        message=message,
        classification=classification,
        extractions=extractions,
        summaries=summaries,
        warnings=warnings,
        audit_events=events,
        stage_ms=stage_ms,
        total_ms=(time.perf_counter() - overall_started) * 1000,
    )


def _recover_attachment_bytes(raw: bytes) -> dict[str, bytes]:
    """Pull attachment payloads out of a raw message, keyed by filename."""
    import email
    from email import policy

    parsed = email.message_from_bytes(raw, policy=policy.default)
    recovered: dict[str, bytes] = {}
    for part in parsed.iter_attachments():
        name = part.get_filename()
        if name:
            recovered[name] = part.get_payload(decode=True) or b""
    return recovered


def screen_literature(
    document: ExtractedDocument, client: LLMClient | None = None
) -> tuple[DocumentSummary, AuditEvent]:
    """Screen one article for a reportable patient case (bonus extension)."""
    client = client or LLMClient()
    started = time.perf_counter()
    try:
        summary = screen_article(document, client)
        succeeded, error = True, ""
    except Exception as exc:  # noqa: BLE001
        summary = DocumentSummary(
            document_id=document.document_id,
            summary="",
            looks_relevant=False,
            relevance_reason=f"Screening failed: {exc}",
            confidence=0.0,
        )
        succeeded, error = False, str(exc)

    event = AuditEvent(
        event_type="SCREEN",
        model_name=client.provider.name,
        is_stub=client.is_stub,
        latency_ms=(time.perf_counter() - started) * 1000,
        succeeded=succeeded,
        error=error,
        document_id=document.document_id,
    )
    return summary, event
