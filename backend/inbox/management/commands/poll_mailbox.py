"""
Poll the test mailbox for new messages and queue them.

    python manage.py poll_mailbox --once      # one pass, then exit
    python manage.py poll_mailbox             # poll continuously
    python manage.py poll_mailbox --check     # test the connection only

Runs as its own process rather than a thread inside the web server. Django's
autoreloader would otherwise start two pollers, and a poller tied to the
request/response cycle stops when the server is idle -- neither is what a
mailbox watcher should do.

Polling is read-only and idempotent: messages are fetched with ``BODY.PEEK`` so
they are never marked as seen, nothing is deleted, and each is keyed by its
``Message-ID`` -- so running this repeatedly re-queues nothing already stored.
"""
from __future__ import annotations

import imaplib
import signal
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from inbox import mailbox
from inbox.models import Message
from inbox.queue import get_queue


class Command(BaseCommand):
    help = "Poll the configured mailbox and queue new messages for processing."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stopping = False

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="single pass, then exit")
        parser.add_argument("--check", action="store_true", help="test the connection and exit")
        parser.add_argument(
            "--interval", type=int, default=0, help="seconds between polls (default from .env)"
        )
        parser.add_argument(
            "--wait", type=int, default=0, help="with --once, wait N seconds for the queue to drain"
        )

    def handle(self, *args, **options):
        config = settings.MAILBOX
        interval = options["interval"] or config["poll_seconds"]

        if options["check"]:
            self._check(config)
            return

        if config["offline"]:
            self.stdout.write(
                self.style.WARNING(
                    "MAIL_OFFLINE_MODE=true - reading data/samples instead of IMAP.\n"
                    "Set it to false in .env to poll the real mailbox."
                )
            )

        # Ctrl-C should finish the current pass rather than tear the process
        # down mid-fetch.
        signal.signal(signal.SIGINT, self._request_stop)

        if options["once"]:
            self._poll_once()
            if options["wait"]:
                self.stdout.write(f"waiting up to {options['wait']}s for the queue to drain...")
                get_queue().join(timeout=options["wait"])
                self.stdout.write(f"queue: {get_queue().stats().to_dict()}")
            return

        self.stdout.write(
            f"polling {config['host']} every {interval}s as {config['user'] or '(offline)'}. "
            "Ctrl-C to stop."
        )
        while not self._stopping:
            self._poll_once()
            for _ in range(interval):
                if self._stopping:
                    break
                time.sleep(1)
        self.stdout.write("\nstopped.")

    def _request_stop(self, *_args) -> None:
        self._stopping = True

    def _poll_once(self) -> None:
        """One poll pass, reporting what changed."""
        before = Message.objects.count()
        try:
            queued = mailbox.ingest()
        except mailbox.MailboxError as exc:
            # A transient mailbox failure must not kill a long-running poller.
            self.stderr.write(self.style.ERROR(f"poll failed: {exc}"))
            return

        stats = get_queue().stats().to_dict()
        if queued:
            self.stdout.write(
                self.style.SUCCESS(
                    f"queued {queued} new message(s) "
                    f"(stored {before} -> pending {stats['pending']})"
                )
            )
        else:
            self.stdout.write(f"no new mail ({before} stored)")

    def _check(self, config: dict) -> None:
        """Verify the mailbox credentials and report what is in there."""
        if not config["user"] or not config["password"]:
            raise CommandError(
                "IMAP_USER / IMAP_PASSWORD are not set in .env.\n\n"
                "Setup:\n"
                "  1. Create a throwaway Gmail account\n"
                "  2. Google Account -> Security -> turn ON 2-Step Verification\n"
                "  3. Security -> App passwords -> generate one for 'Mail'\n"
                "  4. Put the address and the 16-character password in .env"
            )

        self.stdout.write(f"connecting to {config['host']}:{config['port']} as {config['user']}...")
        try:
            with imaplib.IMAP4_SSL(config["host"], config["port"]) as connection:
                connection.login(config["user"], config["password"])
                status, data = connection.select(config["folder"], readonly=True)
                if status != "OK":
                    raise CommandError(f"Could not open folder {config['folder']!r}: {data}")
                total = int(data[0])
                status, unseen = connection.search(None, "UNSEEN")
                unseen_count = len(unseen[0].split()) if status == "OK" else 0

                self.stdout.write(self.style.SUCCESS("connection OK"))
                self.stdout.write(f"  folder   {config['folder']}")
                self.stdout.write(f"  messages {total}")
                self.stdout.write(f"  unseen   {unseen_count}")
                self.stdout.write(f"  stored   {Message.objects.count()} already ingested")
        except imaplib.IMAP4.error as exc:
            raise CommandError(
                f"IMAP login failed: {exc}\n\n"
                "Almost always one of:\n"
                "  - IMAP_PASSWORD holds the account password rather than an\n"
                "    App Password (App Passwords are 16 characters, no spaces)\n"
                "  - 2-Step Verification is not enabled, so App Passwords are hidden\n"
                "  - IMAP access is disabled in Gmail settings"
            ) from exc
        except OSError as exc:
            raise CommandError(f"Could not reach {config['host']}: {exc}") from exc
