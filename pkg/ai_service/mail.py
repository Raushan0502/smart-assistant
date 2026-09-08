"""
Email ingestion: parse a MIME message into a message plus its documents.

The assignment asks for sender, subject, date and body, every PDF attachment
processed, and every other file type *logged rather than processed*. That last
rule is honoured literally here: a non-PDF attachment still produces an
:class:`ExtractedDocument` so it appears in the audit trail and on the review
screen, but with ``processed=False`` and a warning saying why. Silently
dropping it would leave a reviewer unaware that something arrived.
"""
from __future__ import annotations

import email
import hashlib
import re
from email import policy
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path

from .models import ExtractedDocument, IngestedMessage, SourceRef, TextBlock
from .pdf import detect_language, extract_pdf_bytes

PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}

# Headers worth keeping for audit. The full header set is large and mostly
# routing noise; these are the ones a reviewer or an auditor would ask about.
AUDIT_HEADERS = [
    "Message-ID",
    "Date",
    "From",
    "To",
    "Cc",
    "Subject",
    "Return-Path",
    "X-Synthetic-Case-Id",
]


def _stable_id(raw: bytes, message_id: str | None) -> str:
    """Derive a stable identifier for a message.

    ``Message-ID`` is used when present, since it is the mail system's own
    identity for the message. When it is missing or malformed -- which happens
    with some senders -- a content hash keeps ingestion idempotent, so polling
    the same mailbox twice cannot create duplicate records.
    """
    if message_id:
        cleaned = message_id.strip().strip("<>")
        if cleaned:
            return cleaned
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:32]


def _body_text(message: EmailMessage) -> tuple[str, list[str]]:
    """Return the plain-text body, falling back to HTML when necessary."""
    warnings: list[str] = []
    part = message.get_body(preferencelist=("plain",))
    if part is not None:
        return part.get_content().strip(), warnings

    part = message.get_body(preferencelist=("html",))
    if part is None:
        warnings.append("Message has no readable text or HTML body.")
        return "", warnings

    # Crude but adequate: strip tags rather than add an HTML parser dependency
    # for a fallback path. Recorded as a warning so the degradation is visible.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", part.get_content(), flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    warnings.append("No plain-text body; HTML was stripped, formatting may be lost.")
    return text, warnings


def _attachment_document(part: EmailMessage, message_id: str, index: int) -> ExtractedDocument:
    """Ingest one attachment, or record why it was not processed."""
    file_name = part.get_filename() or f"attachment_{index}"
    content_type = (part.get_content_type() or "").lower()
    data = part.get_payload(decode=True) or b""
    document_id = f"{message_id}::att{index}"

    is_pdf = content_type in PDF_CONTENT_TYPES or file_name.lower().endswith(".pdf")
    if not is_pdf:
        return ExtractedDocument(
            document_id=document_id,
            file_name=file_name,
            media_type=content_type or "application/octet-stream",
            processed=False,
            metadata={"byte_size": str(len(data))},
            warnings=[
                f"Attachment type '{content_type or 'unknown'}' is not processed; "
                "logged for audit only."
            ],
        )

    try:
        return extract_pdf_bytes(data, file_name=file_name, document_id=document_id)
    except Exception as exc:  # noqa: BLE001 -- one bad PDF must not lose the message
        return ExtractedDocument(
            document_id=document_id,
            file_name=file_name,
            media_type="application/pdf",
            processed=False,
            metadata={"byte_size": str(len(data))},
            warnings=[f"PDF could not be parsed ({type(exc).__name__}: {exc})."],
        )


def parse_message(raw: bytes) -> IngestedMessage:
    """Parse raw MIME bytes into an :class:`IngestedMessage`."""
    message: EmailMessage = email.message_from_bytes(raw, policy=policy.default)
    message_id = _stable_id(raw, message.get("Message-ID"))

    sent_at = ""
    warnings: list[str] = []
    if message.get("Date"):
        try:
            sent_at = parsedate_to_datetime(message["Date"]).isoformat()
        except (TypeError, ValueError):
            warnings.append(f"Unparseable Date header: {message['Date']!r}")
    else:
        warnings.append("Message has no Date header.")

    body_text, body_warnings = _body_text(message)
    warnings.extend(body_warnings)

    body = ExtractedDocument(
        document_id=f"{message_id}::body",
        file_name="(email body)",
        media_type="text/plain",
        language=detect_language(body_text),
        page_count=0,
        metadata={"char_count": str(len(body_text))},
    )
    if body_text:
        body.blocks.append(
            TextBlock(
                text=body_text,
                source=SourceRef(
                    document_id=body.document_id,
                    file_name="(email body)",
                    page=None,
                    block_index=0,
                ),
            )
        )

    attachments = [
        _attachment_document(part, message_id, index)
        for index, part in enumerate(message.iter_attachments())
    ]

    metadata = {
        header: str(message[header])
        for header in AUDIT_HEADERS
        if message.get(header)
    }
    metadata["raw_byte_size"] = str(len(raw))
    metadata["attachment_count"] = str(len(attachments))

    return IngestedMessage(
        message_id=message_id,
        subject=str(message.get("Subject", "")),
        sender=str(message.get("From", "")),
        recipient=str(message.get("To", "")),
        sent_at=sent_at,
        body=body,
        attachments=attachments,
        metadata=metadata,
        warnings=warnings,
    )


def parse_message_file(path: Path) -> IngestedMessage:
    """Parse a ``.eml`` file from disk."""
    return parse_message(path.read_bytes())
