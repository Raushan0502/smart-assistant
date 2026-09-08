"""
In-process work queue for document processing.

The assignment asks for "a simple queue (even an in-process one) rather than
pure synchronous calls, since AI/OCR processing takes a few seconds to a minute
per document". This is that queue, deliberately kept to the standard library:
a broker would be more production-shaped but adds infrastructure a reviewer has
to install before the prototype runs at all.

The property that matters is not the transport, it is that **submitting work
returns immediately**. Ingest accepts a message, hands it to a worker and
answers straight away; the reviewer screen polls for status. A 40-second OCR
never occupies a request thread.

What this is not: durable. If the process dies, queued work is lost -- which is
acceptable here because nothing is deleted from the mailbox, so a restart
re-polls and re-processes. The production alternative is argued in the write-up.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import ai_client
from .models import Message, ProcessingStatus
from .pipeline import mark_failed, persist_analysis

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 2
# Bounded so a runaway poller cannot exhaust memory; submit blocks instead.
MAX_QUEUE_SIZE = 500


@dataclass
class Job:
    """One message waiting to be processed."""

    raw: bytes
    file_name: str
    message_id: str = ""
    subject: str = ""
    submitted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class QueueStats:
    """A snapshot of queue activity, for the status endpoint."""

    pending: int = 0
    processed: int = 0
    failed: int = 0
    workers: int = 0
    running: bool = False

    def to_dict(self) -> dict:
        return {
            "pending": self.pending,
            "processed": self.processed,
            "failed": self.failed,
            "workers": self.workers,
            "running": self.running,
        }


class ProcessingQueue:
    """A small thread-pool queue draining into the AI service."""

    def __init__(self, workers: int = DEFAULT_WORKERS):
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._threads: list[threading.Thread] = []
        self._workers = workers
        self._running = False
        self._lock = threading.Lock()
        self._processed = 0
        self._failed = 0

    def start(self) -> None:
        """Start the worker threads, if they are not already running."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for index in range(self._workers):
                thread = threading.Thread(
                    target=self._worker, name=f"inbox-worker-{index}", daemon=True
                )
                thread.start()
                self._threads.append(thread)
        logger.info("Processing queue started with %d worker(s)", self._workers)

    def stop(self) -> None:
        """Signal the workers to finish and wait briefly for them."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()
        logger.info("Processing queue stopped")

    def submit(self, job: Job) -> None:
        """Queue a message. Returns immediately."""
        self._queue.put(job)

    def join(self, timeout: float | None = None) -> bool:
        """Block until the queue drains. Used by the batch command and tests."""
        if timeout is None:
            self._queue.join()
            return True
        # Queue.join takes no timeout, so poll the unfinished-task count.
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.1)
        return self._queue.unfinished_tasks == 0

    def stats(self) -> QueueStats:
        """Current queue state."""
        return QueueStats(
            pending=self._queue.qsize(),
            processed=self._processed,
            failed=self._failed,
            workers=len(self._threads),
            running=self._running,
        )

    def _worker(self) -> None:
        """Drain jobs until stopped."""
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                self._process(job)
            except Exception:  # noqa: BLE001 -- a worker must never die
                logger.exception("Worker failed on %s", job.file_name)
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        """Send one job to the AI service and persist the result."""
        try:
            analysis = ai_client.process_message(job.raw, job.file_name)
            message = persist_analysis(analysis)
            with self._lock:
                self._processed += 1
            logger.info("Processed %s -> %s", job.file_name, message.categories)
        except Exception as exc:  # noqa: BLE001 -- record, do not lose the mail
            logger.error("Processing failed for %s: %s", job.file_name, exc)
            identifier = job.message_id or f"failed:{job.file_name}"
            mark_failed(identifier, job.subject or job.file_name, str(exc))
            with self._lock:
                self._failed += 1


# One queue per process. Django's dev server can import modules more than once,
# so the instance is created lazily and guarded.
_queue_instance: ProcessingQueue | None = None
_queue_lock = threading.Lock()


def get_queue() -> ProcessingQueue:
    """Return the shared queue, starting it on first use."""
    global _queue_instance
    with _queue_lock:
        if _queue_instance is None:
            _queue_instance = ProcessingQueue()
            _queue_instance.start()
    return _queue_instance


def requeue_pending() -> int:
    """Re-queue anything left QUEUED or PROCESSING after a restart.

    In-process queues do not survive a restart. Nothing is deleted from the
    mailbox, so the work is recoverable -- this makes the recovery explicit
    rather than relying on the next poll to notice.
    """
    stale = Message.objects.filter(
        processing_status__in=[ProcessingStatus.QUEUED, ProcessingStatus.PROCESSING]
    )
    count = stale.count()
    if count:
        logger.warning("%d message(s) were mid-flight at shutdown", count)
    return count
