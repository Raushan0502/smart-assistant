"""
Process the sample corpus end to end and report timings and accuracy.

    python manage.py run_batch                # process everything, then report
    python manage.py run_batch --report-only  # report on what is already stored
    python manage.py run_batch --reset        # clear first, then process

The assignment asks for two things this produces:

    "Process at least 10-15 sample documents automatically and report how long
     each one takes."

    "Extracted JSON per test document."

Accuracy is scored against ``data/samples/ground_truth.json`` -- the labels the
generator emitted alongside the corpus. Because those labels were written
before any model saw the documents, this is a genuine held-out comparison
rather than a self-assessment.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from inbox import mailbox
from inbox.models import Classification, ExtractedField, Message
from inbox.queue import get_queue
from inbox.serializers import MessageDetailSerializer

CATEGORIES = ["ICSR", "PQC", "MI", "NOT_RELEVANT"]


class Command(BaseCommand):
    help = "Run the sample corpus through the pipeline and report timings and accuracy."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="delete stored messages first")
        parser.add_argument("--report-only", action="store_true", help="skip processing")
        parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for the queue")
        parser.add_argument(
            "--out",
            type=Path,
            default=Path(settings.REPO_ROOT) / "results",
            help="directory for per-document JSON and the summary",
        )

    def handle(self, *args, **options):
        out_dir: Path = options["out"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if not options["report_only"]:
            if options["reset"]:
                deleted = Message.objects.count()
                Message.objects.all().delete()
                self.stdout.write(f"cleared {deleted} existing message(s)")

            settings.MAILBOX["offline"] = True
            started = time.perf_counter()
            queued = mailbox.ingest()
            self.stdout.write(f"queued {queued} message(s); waiting for workers...")

            queue = get_queue()
            if not queue.join(timeout=options["timeout"]):
                self.stderr.write(self.style.WARNING("Queue did not drain within the timeout."))
            wall = time.perf_counter() - started
            self.stdout.write(f"batch wall time: {wall:.1f}s\n")

        messages = list(
            Message.objects.prefetch_related("classifications", "extractions__fields", "documents")
        )
        if not messages:
            self.stderr.write(self.style.ERROR("No messages stored. Run without --report-only."))
            return

        self._write_documents(messages, out_dir)
        timings = self._report_timings(messages)
        accuracy = self._report_accuracy(messages)
        integrity = self._report_integrity(messages)

        summary = {
            "messages": len(messages),
            "timings": timings,
            "accuracy": accuracy,
            "integrity": integrity,
        }
        (out_dir / "batch_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self.stdout.write(self.style.SUCCESS(f"\nWrote {out_dir / 'batch_summary.json'}"))

    # ---- output ---------------------------------------------------------

    def _write_documents(self, messages: list[Message], out_dir: Path) -> None:
        """Write the extracted JSON for each document, as the brief requires."""
        per_doc = out_dir / "documents"
        per_doc.mkdir(exist_ok=True)
        for message in messages:
            payload = MessageDetailSerializer(message).data
            name = message.message_id.split("@")[0].replace(":", "_")[:60]
            (per_doc / f"{name}.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        self.stdout.write(f"wrote {len(messages)} per-document JSON file(s) to {per_doc}")

    def _report_timings(self, messages: list[Message]) -> dict:
        """Per-document processing time, which the assignment asks for."""
        times = [m.processing_ms for m in messages if m.processing_ms]
        self.stdout.write(self.style.MIGRATE_HEADING("\nPer-document processing time"))
        self.stdout.write(f"  {'message':<46} {'ms':>7}")
        for message in sorted(messages, key=lambda m: -(m.processing_ms or 0)):
            self.stdout.write(f"  {message.subject[:44]:<46} {message.processing_ms or 0:>7}")

        if not times:
            return {}
        stats = {
            "count": len(times),
            "mean_ms": round(statistics.mean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "min_ms": min(times),
            "max_ms": max(times),
            "total_ms": sum(times),
        }
        self.stdout.write(
            f"\n  mean {stats['mean_ms']}ms | median {stats['median_ms']}ms | "
            f"min {stats['min_ms']}ms | max {stats['max_ms']}ms"
        )
        return stats

    def _report_accuracy(self, messages: list[Message]) -> dict:
        """Score classification against the generator's ground-truth labels."""
        truth_path = Path(settings.SAMPLES_DIR) / "ground_truth.json"
        if not truth_path.exists():
            self.stderr.write("No ground_truth.json; skipping accuracy.")
            return {}

        truth_docs = json.loads(truth_path.read_text(encoding="utf-8"))["documents"]
        # Ground truth is keyed by the generator's case id, which travels in a
        # custom header, so processed records can be matched back to it.
        expected = {
            d["doc_id"]: set(d["categories"]) for d in truth_docs if d["kind"] == "email"
        }

        exact = partial = 0
        per_category = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
        rows = []

        for message in messages:
            case_id = message.headers.get("X-Synthetic-Case-Id", "")
            if case_id not in expected:
                continue
            want = expected[case_id]
            got = set(message.categories)
            if got == want:
                exact += 1
            if got & want:
                partial += 1
            for category in CATEGORIES:
                if category in got and category in want:
                    per_category[category]["tp"] += 1
                elif category in got:
                    per_category[category]["fp"] += 1
                elif category in want:
                    per_category[category]["fn"] += 1
            rows.append((case_id, sorted(want), sorted(got), got == want))

        self.stdout.write(self.style.MIGRATE_HEADING("\nClassification vs ground truth"))
        for case_id, want, got, ok in sorted(rows):
            mark = "ok " if ok else "MISS"
            self.stdout.write(f"  [{mark}] {case_id:<26} expected {want} got {got}")

        total = len(rows)
        result = {
            "scored": total,
            "exact_match": round(exact / total, 3) if total else 0,
            "any_overlap": round(partial / total, 3) if total else 0,
            "per_category": {},
        }
        self.stdout.write(f"\n  {'category':<14} {'prec':>6} {'recall':>7} {'F1':>6}")
        for category, counts in per_category.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            result["per_category"][category] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                **counts,
            }
            self.stdout.write(
                f"  {category:<14} {precision:>6.2f} {recall:>7.2f} {f1:>6.2f}"
            )
        self.stdout.write(
            f"\n  exact-match {result['exact_match']:.1%} | "
            f"any-overlap {result['any_overlap']:.1%} (n={total})"
        )
        return result

    def _report_integrity(self, messages: list[Message]) -> dict:
        """Report how honest the extraction was about what it did not know."""
        fields = ExtractedField.objects.filter(extraction__message__in=messages)
        total = fields.count()
        stated = sum(1 for f in fields if f.is_stated)
        unverified = sum(1 for f in fields if f.is_stated and not f.quote_verified)

        self.stdout.write(self.style.MIGRATE_HEADING("\nExtraction integrity"))
        self.stdout.write(f"  fields extracted        {total}")
        self.stdout.write(
            f"  stated by the source    {stated}"
            f" ({stated / total:.1%})" if total else "  stated 0"
        )
        self.stdout.write(
            f"  honest 'Not stated'     {total - stated}"
            f" ({(total - stated) / total:.1%})" if total else ""
        )
        self.stdout.write(
            f"  stated but unverified   {unverified}"
            + ("  <-- needs human review" if unverified else "")
        )
        return {
            "fields": total,
            "stated": stated,
            "not_stated": total - stated,
            "stated_unverified": unverified,
        }
