"""
Text normalisation for extracted documents.

PDF text extraction produces artefacts that are invisible to a human reading
the page but expensive for a model: ligatures, non-breaking spaces, words split
across a line break by hyphenation, and page furniture repeated on every page.
Each of these either wastes tokens or actively corrupts a value.

**Tables are never normalised.** Normalisation collapses whitespace, and
whitespace inside a cell can be load-bearing -- a reference range of ``3.5 -
5.0`` and a dose of ``20 mg`` both depend on it. Tables arrive already
structured from the extractor, so there is nothing for this module to fix and
real risk in trying. Only :class:`TextBlock` content is touched.

Order matters and is fixed:

    1. Unicode normalise, then map punctuation to ASCII
    2. Re-join hyphenated line breaks
    3. Strip dot leaders and bare page numbers
    4. Remove page furniture repeated across pages
    5. Collapse remaining whitespace

De-hyphenation runs before whitespace collapse because it depends on the line
break still being there.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .models import ExtractedDocument, IngestedMessage, TextBlock

# Typographic characters mapped to ASCII equivalents. The dash range is mapped
# wholesale because NFKC rewrites some dashes into others before any narrower
# mapping would run, which previously let U+2011 survive normalisation.
PUNCTUATION_MAP = {
    **{chr(c): "-" for c in range(0x2010, 0x2016)},
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "…": "...", " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "", "­": "",
}

# A hyphen at end of line followed by a lowercase continuation is almost always
# hyphenation, not a real hyphen. Requiring lowercase protects "Cardiozan-\n20"
# style constructions and proper nouns.
HYPHEN_LINEBREAK = re.compile(r"(\w)-\n([a-z])")

# "Chapter 3 .......... 42" -- table-of-contents leaders carry no information.
DOT_LEADER = re.compile(r"\.{4,}\s*\d*")

# A line that is nothing but a page number.
BARE_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*(?:of\s+\d{1,4})?\s*$", re.I)

# Guards for page-furniture detection. A repeated line is only furniture when it
# is short and appears on several pages: without the length guard, a repeated
# body sentence differing only by a number would be deleted as boilerplate.
MIN_FURNITURE_LEN = 10
MAX_FURNITURE_LEN = 90
MAX_FURNITURE_WORDS = 12
MIN_FURNITURE_REPEATS = 3

DIGITS = re.compile(r"\d+")


def normalise_text(text: str) -> str:
    """Apply the ordered normalisation passes to a single run of prose."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans(PUNCTUATION_MAP))
    text = HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = DOT_LEADER.sub(" ", text)

    lines = [line for line in text.split("\n") if not BARE_PAGE_NUMBER.match(line)]
    text = "\n".join(lines)

    # Collapse runs of spaces/tabs, and runs of blank lines down to one break,
    # without destroying paragraph structure entirely.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def furniture_lines(blocks: list[TextBlock]) -> set[str]:
    """Find lines repeated across pages that are page furniture.

    Lines are compared with digits masked, so ``Page 3 of 12`` and ``Page 7 of
    12`` count as the same piece of furniture. The length and word-count guards
    stop real content that happens to recur from being treated as a header.
    """
    if len(blocks) < MIN_FURNITURE_REPEATS:
        return set()

    counts: Counter[str] = Counter()
    for block in blocks:
        # Count each distinct line once per page, so a line repeated within one
        # page does not look like it spans pages.
        seen = set()
        for raw in block.text.split("\n"):
            line = raw.strip()
            if not (MIN_FURNITURE_LEN <= len(line) <= MAX_FURNITURE_LEN):
                continue
            if len(line.split()) > MAX_FURNITURE_WORDS:
                continue
            seen.add(DIGITS.sub("#", line))
        counts.update(seen)

    return {
        masked
        for masked, count in counts.items()
        if count >= min(MIN_FURNITURE_REPEATS, len(blocks))
    }


def strip_page_furniture(document: ExtractedDocument) -> int:
    """Remove repeated headers and footers from a document's prose.

    Returns the number of lines removed, which is recorded as a warning by
    :func:`preprocess_document` so the reviewer can see that content was
    dropped rather than having it disappear silently.
    """
    blocks = document.text_blocks
    furniture = furniture_lines(blocks)
    if not furniture:
        return 0

    removed = 0
    for block in blocks:
        kept = []
        for raw in block.text.split("\n"):
            if DIGITS.sub("#", raw.strip()) in furniture:
                removed += 1
                continue
            kept.append(raw)
        block.text = "\n".join(kept).strip()
    return removed


def preprocess_document(document: ExtractedDocument) -> ExtractedDocument:
    """Normalise a document's prose in place, leaving tables and images alone.

    Mutates and returns the same object so provenance is preserved: every block
    keeps the :class:`~.models.SourceRef` it was created with, and a normalised
    block still points at the page it came from.
    """
    removed = strip_page_furniture(document)
    if removed:
        document.warnings.append(
            f"Removed {removed} repeated header/footer line(s) during preprocessing."
        )

    empties = 0
    for block in document.text_blocks:
        block.text = normalise_text(block.text)
        if not block.text:
            empties += 1

    if empties:
        document.blocks = [
            b for b in document.blocks if not (isinstance(b, TextBlock) and not b.text)
        ]
        document.warnings.append(
            f"Dropped {empties} text block(s) that were empty after normalisation."
        )
    return document


def preprocess_message(message: IngestedMessage) -> IngestedMessage:
    """Normalise the body and every processed attachment of a message."""
    preprocess_document(message.body)
    for attachment in message.attachments:
        if attachment.processed:
            preprocess_document(attachment)
    return message
