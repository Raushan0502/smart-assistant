"""
Send the synthetic corpus to the test mailbox over SMTP.

    python manage.py send_samples              # send all 16, with attachments
    python manage.py send_samples --limit 5    # send a few
    python manage.py send_samples --dry-run    # show what would be sent

This exists so the *real* IMAP path can be exercised end to end. Reading
``.eml`` files off disk proves the parser works; it does not prove the mailbox
connection, the MIME that Gmail actually delivers, or that attachments survive
the round trip. Those only get tested by sending real mail and fetching it back.

What Gmail does to a message in transit is the point: it rewrites headers,
assigns its own ``Message-ID``, may re-encode parts, and delivers base64
attachments that must decode back to byte-identical PDFs. This command makes
that round trip observable.

Safety: every message is addressed to the mailbox in ``.env`` and to nowhere
else, and the corpus is entirely synthetic. Nothing is sent to a real person.
"""
from __future__ import annotations

import smtplib
import ssl
import time
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

# Gmail's submission port. STARTTLS on 587 rather than implicit TLS on 465:
# both work, 587 is the modern default and gives a clearer failure when the
# app password is wrong.
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587

# Providers rate-limit bursts; a small gap keeps a 16-message send well clear.
SEND_GAP_SECONDS = 1.5


class Command(BaseCommand):
    help = "Send the synthetic sample corpus to the configured test mailbox."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=0, help="send at most N messages")
        parser.add_argument("--dry-run", action="store_true", help="list without sending")
        parser.add_argument("--host", default=DEFAULT_SMTP_HOST)
        parser.add_argument("--port", type=int, default=DEFAULT_SMTP_PORT)
        parser.add_argument(
            "--to",
            default="",
            help="recipient; defaults to IMAP_USER so mail lands in your own inbox",
        )

    def handle(self, *args, **options):
        config = settings.MAILBOX
        user, password = config["user"], config["password"]
        if not user or not password:
            raise CommandError(
                "IMAP_USER / IMAP_PASSWORD are not set in .env.\n"
                "Create a throwaway Gmail, enable 2-Step Verification, generate an\n"
                "App Password for Mail, and put both in .env."
            )

        recipient = options["to"] or user
        samples = sorted(Path(settings.SAMPLES_DIR).glob("*.eml"))
        if not samples:
            raise CommandError(
                f"No .eml files in {settings.SAMPLES_DIR}. "
                "Generate the corpus first: cd pkg && python -m generator.generate"
            )
        if options["limit"]:
            samples = samples[: options["limit"]]

        self.stdout.write(f"{len(samples)} message(s) -> {recipient}\n")

        if options["dry_run"]:
            for path in samples:
                message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
                attachments = [a.get_filename() for a in message.iter_attachments()]
                self.stdout.write(
                    f"  {path.stem:<26} {message['Subject'][:40]:<42} "
                    f"{attachments if attachments else 'no attachments'}"
                )
            self.stdout.write(self.style.WARNING("\nDry run - nothing sent."))
            return

        sent = failed = 0
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(options["host"], options["port"], timeout=30) as server:
                server.starttls(context=context)
                server.login(user, password)

                for index, path in enumerate(samples, start=1):
                    original = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
                    outgoing = self._rebuild(original, sender=user, recipient=recipient)
                    try:
                        server.send_message(outgoing)
                        sent += 1
                        attachments = len(list(outgoing.iter_attachments()))
                        self.stdout.write(
                            f"  [{index:>2}/{len(samples)}] sent {path.stem:<26} "
                            f"({attachments} attachment(s))"
                        )
                    except smtplib.SMTPException as exc:
                        failed += 1
                        self.stderr.write(self.style.ERROR(f"  failed {path.stem}: {exc}"))
                    time.sleep(SEND_GAP_SECONDS)

        except smtplib.SMTPAuthenticationError as exc:
            raise CommandError(
                f"SMTP authentication failed: {exc}\n\n"
                "Almost always one of:\n"
                "  - the value in IMAP_PASSWORD is the account password, not an\n"
                "    App Password (App Passwords are 16 characters, no spaces)\n"
                "  - 2-Step Verification is not enabled on the account\n"
                "  - the App Password was revoked"
            ) from exc
        except OSError as exc:
            raise CommandError(f"Could not reach {options['host']}:{options['port']}: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(f"\nSent {sent}, failed {failed}.")
        )
        self.stdout.write(
            "\nNow fetch them back through the real IMAP path:\n"
            "  python manage.py poll_mailbox --once\n"
            "Delivery usually takes a few seconds."
        )

    def _rebuild(self, original: EmailMessage, sender: str, recipient: str) -> EmailMessage:
        """Rebuild a stored message for sending, preserving body and attachments.

        The From/To are rewritten to the test account because a provider will
        not relay mail claiming to be from an address it does not own. The
        original sender is preserved in ``X-Original-From`` so the synthetic
        reporter identity is still visible after the round trip, and the case id
        header is carried through so results can be scored against ground truth.
        """
        outgoing = EmailMessage()
        outgoing["From"] = sender
        outgoing["To"] = recipient
        outgoing["Subject"] = original["Subject"] or "(no subject)"
        outgoing["X-Original-From"] = original["From"] or ""
        if original["X-Synthetic-Case-Id"]:
            outgoing["X-Synthetic-Case-Id"] = original["X-Synthetic-Case-Id"]

        body = original.get_body(preferencelist=("plain",))
        text = body.get_content() if body else ""
        # Keep the original sender in the body too: the classifier reads the
        # reporter from the message, and rewriting From would otherwise lose it.
        outgoing.set_content(
            f"[Synthetic test message - originally from {original['From']}]\n\n{text}"
        )

        for part in original.iter_attachments():
            data = part.get_payload(decode=True) or b""
            maintype, _, subtype = (part.get_content_type() or "application/octet-stream").partition("/")
            outgoing.add_attachment(
                data,
                maintype=maintype,
                subtype=subtype or "octet-stream",
                filename=part.get_filename() or "attachment",
            )
        return outgoing
