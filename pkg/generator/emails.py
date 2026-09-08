"""
Assemble synthetic cases into RFC 5322 ``.eml`` files.

Real ``.eml`` files are written rather than a convenient JSON shape, because
the ingestion pipeline has to parse actual MIME: multipart bodies, encoded
headers, base64 attachments and non-ASCII subject lines. A simplified format
would let the parser pass here and fail on the live mailbox.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path

from .content import Case

MAILBOX = "pv-intake@clinevo-demo.example"

# Fixed base date so a regenerated corpus is byte-comparable run to run.
BASE_DATE = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)


def build_email(
    case: Case,
    attachments: list[Path],
    index: int,
    extra_files: list[Path] | None = None,
) -> EmailMessage:
    """Build one email message for a case, attaching any generated PDFs.

    ``extra_files`` carries non-PDF attachments, which the assignment says
    should be logged rather than processed -- so at least one message needs to
    contain one for that path to be exercised.
    """
    message = EmailMessage()
    message["From"] = f"{case.sender_name} <{case.sender_email}>"
    message["To"] = f"Pharmacovigilance Intake <{MAILBOX}>"
    message["Subject"] = case.subject
    message["Date"] = format_datetime(BASE_DATE + timedelta(hours=6 * index))
    message["Message-ID"] = make_msgid(domain="clinevo-demo.example")
    # Carried through to the corpus so a reviewer can trace a processed record
    # back to the generator case that produced it.
    message["X-Synthetic-Case-Id"] = case.case_id
    message.set_content(case.body)

    for path in attachments:
        message.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=path.name,
        )
    for path in extra_files or []:
        message.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="octet-stream",
            filename=path.name,
        )
    return message


def make_unsupported_attachment(directory: Path, rng: random.Random) -> Path:
    """Create a small non-PDF attachment.

    The assignment requires non-PDF attachments to be logged rather than
    processed. Producing one here means that branch is covered by the corpus
    instead of being asserted in the write-up and never exercised.
    """
    path = directory / "ward_roster.csv"
    rows = ["ward,shift,staff_on_duty"]
    for ward in ("A", "B", "C"):
        rows.append(f"Ward {ward},night,{rng.randint(2, 6)}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
