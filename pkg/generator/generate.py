"""
Generate the full synthetic corpus and its ground-truth labels.

    python -m generator.generate            # writes to data/samples
    python -m generator.generate --out DIR  # writes elsewhere

Output is deterministic: same inputs, same bytes. That matters because the
ground-truth file is the evaluation set, and a corpus that shifted between runs
would make accuracy numbers incomparable.

``ground_truth.json`` is committed to the repository while the rendered
documents are not -- the labels are the durable artifact, the PDFs are
reproducible from this script.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .content import ARTICLES, Article, Case, all_email_cases
from .emails import build_email, make_unsupported_attachment
from .render import render_article, render_digital_form, render_scanned_form

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "samples"
SEED = 20260907

# The one case that also gets a non-PDF attachment, to exercise the
# "log it, don't process it" branch.
UNSUPPORTED_ATTACHMENT_CASE = "icsr_full_rash"


def render_attachment(case: Case, out_dir: Path, index: int) -> tuple[Path, str]:
    """Render a case's attachment and return its path and PDF flavour."""
    if case.attachment_kind == "digital_form":
        path = out_dir / f"form_{case.case_id}.pdf"
        render_digital_form(case, path)
        return path, "digital"
    if case.attachment_kind == "scanned_form":
        path = out_dir / f"scan_{case.case_id}.pdf"
        render_scanned_form(case, path, seed=index)
        return path, "scanned"
    if case.attachment_kind == "non_english_pdf":
        path = out_dir / f"form_{case.case_id}.pdf"
        render_digital_form(case, path, language=case.language)
        return path, "non_english"
    raise ValueError(f"unknown attachment kind: {case.attachment_kind}")


def email_record(
    case: Case,
    eml_path: Path,
    attachment: tuple[Path, str] | None,
    unsupported: Path | None,
) -> dict:
    """Build the ground-truth record for one email."""
    attachments = []
    if attachment is not None:
        path, flavour = attachment
        attachments.append(
            {
                "file": path.name,
                "pdf_flavour": flavour,
                "language": case.language,
                "processed": True,
            }
        )
    if unsupported is not None:
        attachments.append(
            {
                "file": unsupported.name,
                "pdf_flavour": None,
                "language": None,
                # Non-PDF: the pipeline must log it and move on.
                "processed": False,
            }
        )
    return {
        "doc_id": case.case_id,
        "kind": "email",
        "file": eml_path.name,
        "subject": case.subject,
        "sender": case.sender_email,
        "language": case.language,
        "categories": sorted(case.categories),
        "attachments": attachments,
        "expected_fields": case.expected_fields,
        "notes": case.notes,
    }


def article_record(article: Article, path: Path) -> dict:
    """Build the ground-truth record for one article PDF."""
    return {
        "doc_id": article.article_id,
        "kind": "article",
        "file": path.name,
        "title": article.title,
        "language": "en",
        "pdf_flavour": "article",
        # Articles arrive by upload, not through the mailbox, so they carry no
        # mailbox category. Their label is the literature-screening decision.
        "categories": [],
        "case_count": article.case_count,
        "reportable": article.reportable,
        "notes": article.notes,
    }


def generate(out_dir: Path) -> dict:
    """Render the whole corpus into ``out_dir`` and return the ground truth."""
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear the contents rather than the directory itself: on Windows a synced
    # folder (OneDrive) keeps a handle on the directory and rmdir fails, and
    # this also preserves the committed .gitkeep.
    for existing in out_dir.iterdir():
        if existing.name == ".gitkeep":
            continue
        if existing.is_dir():
            shutil.rmtree(existing, ignore_errors=True)
        else:
            existing.unlink()

    records: list[dict] = []

    for index, case in enumerate(all_email_cases()):
        attachment = None
        if case.attachment_kind:
            attachment = render_attachment(case, out_dir, index)

        unsupported = None
        if case.case_id == UNSUPPORTED_ATTACHMENT_CASE:
            unsupported = make_unsupported_attachment(out_dir, rng)

        message = build_email(
            case,
            attachments=[attachment[0]] if attachment else [],
            index=index,
            extra_files=[unsupported] if unsupported else [],
        )
        eml_path = out_dir / f"{case.case_id}.eml"
        eml_path.write_bytes(bytes(message))
        records.append(email_record(case, eml_path, attachment, unsupported))

    for article in ARTICLES:
        path = out_dir / f"{article.article_id}.pdf"
        render_article(article, path)
        records.append(article_record(article, path))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "note": (
            "Entirely synthetic. No real patient data, and every product name is "
            "fictional. Regenerate with: python -m generator.generate"
        ),
        "documents": records,
    }


def summarise(truth: dict) -> str:
    """Render a human-readable coverage summary of what was generated."""
    docs = truth["documents"]
    emails = [d for d in docs if d["kind"] == "email"]
    articles = [d for d in docs if d["kind"] == "article"]
    flavours: dict[str, int] = {}
    for doc in emails:
        for attachment in doc["attachments"]:
            key = attachment["pdf_flavour"] or "non-pdf (logged only)"
            flavours[key] = flavours.get(key, 0) + 1
    categories: dict[str, int] = {}
    for doc in emails:
        for category in doc["categories"]:
            categories[category] = categories.get(category, 0) + 1

    lines = [
        f"emails            {len(emails)}",
        f"article PDFs      {len(articles)}  "
        f"({sum(a['reportable'] for a in articles)} reportable, "
        f"{sum(a['case_count'] for a in articles)} patient cases)",
        "categories        " + ", ".join(f"{k}={v}" for k, v in sorted(categories.items())),
        "attachments       " + ", ".join(f"{k}={v}" for k, v in sorted(flavours.items())),
    ]
    return "\n".join(lines)


def count_flavour(emails: list[dict], flavour: str) -> int:
    """How many attachments across all emails have the given PDF flavour."""
    return sum(
        1 for d in emails for a in d["attachments"] if a["pdf_flavour"] == flavour
    )


def count_single_category(emails: list[dict], category: str) -> int:
    """How many emails carry exactly one category, and it is this one.

    The assignment asks for messages that are *only* a quality complaint or
    *only* an info request, so a dual-labelled message must not count.
    """
    return sum(1 for d in emails if d["categories"] == [category])


def check_requirements(truth: dict) -> list[str]:
    """Check the corpus against the assignment's stated minimums.

    Returned as a list of failures rather than an assertion so the caller can
    report every shortfall at once instead of stopping at the first.
    """
    docs = truth["documents"]
    emails = [d for d in docs if d["kind"] == "email"]
    articles = [d for d in docs if d["kind"] == "article"]

    checks = [
        (">=10 emails about a reaction", sum(1 for d in emails if "ICSR" in d["categories"]), 10),
        (">=5 digital PDFs", count_flavour(emails, "digital"), 5),
        (">=2 scanned/handwritten PDFs", count_flavour(emails, "scanned"), 2),
        (">=5 article PDFs", len(articles), 5),
        (">=2 non-English PDFs", count_flavour(emails, "non_english"), 2),
        (">=2 quality-complaint-only", count_single_category(emails, "PQC"), 2),
        (">=2 info-request-only", count_single_category(emails, "MI"), 2),
        (">=1 clearly irrelevant", count_single_category(emails, "NOT_RELEVANT"), 1),
    ]
    return [
        f"{label}: have {actual}, need {needed}"
        for label, actual, needed in checks
        if actual < needed
    ]


def main() -> int:
    """Generate the corpus, write ground truth, and verify coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    truth = generate(args.out)
    truth_path = args.out / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    print(summarise(truth))
    print(f"\nwrote {len(list(args.out.glob('*')))} files -> {args.out}")

    failures = check_requirements(truth)
    if failures:
        print("\nCOVERAGE FAILURES (assignment section 6):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll assignment section 6 minimums satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
