"""
Mailbox intake: IMAP polling, plus an offline folder source.

The assignment asks the app to "connect to a real test mailbox", which
:func:`poll_imap` does over IMAP against a throwaway Gmail account. The offline
source exists alongside it for two practical reasons: the batch run the
assignment requires must be reproducible without depending on what happens to
be sitting in an inbox, and a live walkthrough should not fail because of a
network or credential problem.

Polling is **read-only and idempotent**. Messages are fetched with ``BODY.PEEK``
so they are not marked as seen, nothing is ever deleted, and each message is
keyed by its ``Message-ID`` -- so polling the same mailbox repeatedly re-queues
nothing that has already been stored.
"""
from __future__ import annotations

import email
import hashlib
import imaplib
import logging
from email import policy
from pathlib import Path

from django.conf import settings

from .models import Message
from .queue import Job, get_queue

logger = logging.getLogger(__name__)

# Cap per poll so a large mailbox cannot flood the queue in one pass.
MAX_PER_POLL = 50

# Default IMAP search. Deliberately NOT "ALL".
#
# Every message this project generates carries an X-Synthetic-Case-Id header,
# and this search matches only those. That matters because a test account is
# often an ordinary mailbox with real personal mail in it: an "ALL" search
# would send a stranger's genuine correspondence to a cloud LLM and store it in
# the database. Restricting the search by header makes it impossible to ingest
# anything this project did not itself create.
#
# Override with IMAP_SEARCH in .env -- e.g. 'ALL' for a genuinely empty
# throwaway account, or 'SUBJECT "PV-TEST"' for a different marker.
SYNTHETIC_ONLY_SEARCH = 'HEADER X-Synthetic-Case-Id ""'


class MailboxError(RuntimeError):
    """The mailbox could not be reached or read."""


def already_stored(message_id: str) -> bool:
    """Whether this message has already been ingested."""
    return Message.objects.filter(message_id=message_id).exists()


def message_identity(raw: bytes) -> tuple[str, str]:
    """Return the Message-ID and subject without fully parsing the message."""
    parsed = email.message_from_bytes(raw, policy=policy.default)
    message_id = (parsed.get("Message-ID") or "").strip().strip("<>")
    if not message_id:
        message_id = "sha256:" + hashlib.sha256(raw).hexdigest()[:32]
    return message_id, str(parsed.get("Subject", ""))


def poll_imap(limit: int = MAX_PER_POLL) -> int:
    """Poll the configured mailbox and queue anything new.

    Returns the number of messages queued. Raises :class:`MailboxError` if the
    mailbox cannot be reached, so the caller can distinguish "no new mail" from
    "could not check".
    """
    config = settings.MAILBOX
    if not config["user"] or not config["password"]:
        raise MailboxError(
            "IMAP_USER and IMAP_PASSWORD are not set in .env. Set them, or use "
            "MAIL_OFFLINE_MODE=true to read from data/samples instead."
        )

    queued = 0
    try:
        with imaplib.IMAP4_SSL(config["host"], config["port"]) as connection:
            connection.login(config["user"], config["password"])
            # Read-only: the assistant must never mutate the shared mailbox.
            connection.select(config["folder"], readonly=True)

            search = config.get("search") or SYNTHETIC_ONLY_SEARCH
            status, data = connection.search(None, search)
            if status != "OK":
                raise MailboxError(f"IMAP search {search!r} failed: {status}")

            ids = data[0].split()[-limit:]
            logger.info(
                "IMAP search %r matched %d message(s) in a folder of unknown size",
                search,
                len(data[0].split()),
            )
            for raw_id in ids:
                status, payload = connection.fetch(raw_id, "(BODY.PEEK[])")
                if status != "OK" or not payload or not isinstance(payload[0], tuple):
                    logger.warning("Could not fetch message %s", raw_id)
                    continue

                raw = payload[0][1]
                message_id, subject = message_identity(raw)
                if already_stored(message_id):
                    continue

                get_queue().submit(
                    Job(
                        raw=raw,
                        file_name=f"{message_id}.eml",
                        message_id=message_id,
                        subject=subject,
                    )
                )
                queued += 1
    except imaplib.IMAP4.error as exc:
        raise MailboxError(f"IMAP error: {exc}") from exc
    except OSError as exc:
        raise MailboxError(f"Could not reach {config['host']}: {exc}") from exc

    logger.info("IMAP poll queued %d new message(s)", queued)
    return queued


def load_offline(directory: Path | None = None, limit: int = MAX_PER_POLL) -> int:
    """Queue every ``.eml`` file in a folder, skipping ones already stored."""
    directory = directory or settings.SAMPLES_DIR
    if not directory.exists():
        raise MailboxError(f"Sample directory not found: {directory}")

    queued = 0
    for path in sorted(directory.glob("*.eml"))[:limit]:
        raw = path.read_bytes()
        message_id, subject = message_identity(raw)
        if already_stored(message_id):
            continue
        get_queue().submit(
            Job(raw=raw, file_name=path.name, message_id=message_id, subject=subject)
        )
        queued += 1

    logger.info("Offline load queued %d message(s) from %s", queued, directory)
    return queued


def ingest() -> int:
    """Fetch new mail from whichever source is configured."""
    if settings.MAILBOX["offline"]:
        return load_offline()
    return poll_imap()
