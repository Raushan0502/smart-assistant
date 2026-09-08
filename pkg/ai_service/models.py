"""
The document model produced by ingestion and consumed by extraction.

Two decisions are baked in here rather than left to callers, because both are
scored requirements and both are expensive to retrofit:

**Provenance is mandatory.** Every block carries a :class:`SourceRef` naming the
document and page it came from. There is no way to construct a block without
one. When a fact is later extracted from a block, the block's ``SourceRef``
travels with it, so "where did this come from?" is always answerable by
construction rather than by convention.

**Tables stay tabular.** A table is header plus rows, not a string. Flattening
"Serum potassium | 6.8 | mmol/L | 3.5 - 5.0 | HIGH" into prose destroys the
association between a value, its unit and its reference range -- exactly the
structure that makes a lab result meaningful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class PdfFlavour(StrEnum):
    """The four PDF varieties the assignment requires handling."""

    DIGITAL = "digital"
    SCANNED = "scanned"
    ARTICLE = "article"
    NON_ENGLISH = "non_english"
    UNKNOWN = "unknown"


class BlockKind(StrEnum):
    """What a block of content is."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(frozen=True)
class SourceRef:
    """Where a piece of content came from.

    Frozen because a reference must not be mutated after the fact it describes
    has been extracted -- that would silently break the audit trail.
    """

    document_id: str
    file_name: str
    # 1-based, matching what a human sees in a PDF viewer. None for email bodies.
    page: int | None = None
    block_index: int | None = None

    def describe(self) -> str:
        """Human-readable citation, e.g. ``form_x.pdf p.2``."""
        if self.page is None:
            return self.file_name
        return f"{self.file_name} p.{self.page}"


@dataclass
class TextBlock:
    """A run of prose."""

    text: str
    source: SourceRef
    kind: BlockKind = BlockKind.TEXT

    def to_prompt_text(self) -> str:
        """Render for inclusion in an LLM prompt, tagged with its source."""
        return f"[{self.source.describe()}]\n{self.text}"


@dataclass
class TableBlock:
    """A table, kept as header and rows."""

    header: list[str]
    rows: list[list[str]]
    source: SourceRef
    caption: str | None = None
    kind: BlockKind = BlockKind.TABLE

    def to_prompt_text(self) -> str:
        """Render as Markdown so the model sees columns, not a flat string.

        Markdown is used rather than JSON because it survives tokenisation more
        compactly and models read it reliably, while still preserving the
        row/column relationships that matter for lab values.
        """
        lines = [f"[{self.source.describe()}]"]
        if self.caption:
            lines.append(self.caption)
        if self.header:
            lines.append("| " + " | ".join(self.header) + " |")
            lines.append("| " + " | ".join("---" for _ in self.header) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(cell or "" for cell in row) + " |")
        return "\n".join(lines)


@dataclass
class ImageBlock:
    """An embedded image, with a description once one has been produced.

    ``needs_review`` starts true for every meaningful image: the assignment
    asks for a good-faith description and a human review flag, not confident
    image analysis.
    """

    width: int
    height: int
    source: SourceRef
    description: str | None = None
    needs_review: bool = True
    kind: BlockKind = BlockKind.IMAGE

    def to_prompt_text(self) -> str:
        """Render the description, or a placeholder if none was produced."""
        body = self.description or "(image not yet described)"
        return f"[{self.source.describe()} - image {self.width}x{self.height}]\n{body}"


Block = TextBlock | TableBlock | ImageBlock


@dataclass
class ExtractedDocument:
    """One ingested document: its content, its structure and its metadata."""

    document_id: str
    file_name: str
    media_type: str
    blocks: list[Block] = field(default_factory=list)
    flavour: PdfFlavour = PdfFlavour.UNKNOWN
    language: str = "en"
    page_count: int = 0
    # Container metadata: PDF title/author, email headers, byte size, and so on.
    metadata: dict[str, str] = field(default_factory=dict)
    # Non-fatal problems worth surfacing to a reviewer rather than swallowing.
    warnings: list[str] = field(default_factory=list)
    # False for attachments that are logged but deliberately not parsed.
    processed: bool = True

    @property
    def tables(self) -> list[TableBlock]:
        """Every table in the document."""
        return [b for b in self.blocks if isinstance(b, TableBlock)]

    @property
    def images(self) -> list[ImageBlock]:
        """Every image in the document."""
        return [b for b in self.blocks if isinstance(b, ImageBlock)]

    @property
    def text_blocks(self) -> list[TextBlock]:
        """Every prose block in the document."""
        return [b for b in self.blocks if isinstance(b, TextBlock)]

    def full_text(self) -> str:
        """All prose concatenated, without tables or images."""
        return "\n\n".join(b.text for b in self.text_blocks)

    def to_prompt_text(self) -> str:
        """Render the whole document for an LLM, preserving block structure.

        Blocks stay in document order and keep their source tags, so a model
        asked to cite where a fact came from has the reference in front of it.
        """
        return "\n\n".join(block.to_prompt_text() for block in self.blocks)

    def to_dict(self) -> dict:
        """Serialise for the API boundary and for saved sample outputs."""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "media_type": self.media_type,
            "flavour": str(self.flavour),
            "language": self.language,
            "page_count": self.page_count,
            "processed": self.processed,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "counts": {
                "text_blocks": len(self.text_blocks),
                "tables": len(self.tables),
                "images": len(self.images),
            },
            "blocks": [
                {"kind": str(b.kind), **asdict(b), "source": asdict(b.source)}
                for b in self.blocks
            ],
        }


@dataclass
class IngestedMessage:
    """An email plus every document that came with it."""

    message_id: str
    subject: str
    sender: str
    recipient: str
    sent_at: str
    body: ExtractedDocument
    attachments: list[ExtractedDocument] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def documents(self) -> list[ExtractedDocument]:
        """The body and every processed attachment, in order."""
        return [self.body] + [a for a in self.attachments if a.processed]

    def to_prompt_text(self) -> str:
        """Render the whole message for classification or extraction."""
        parts = [
            f"FROM: {self.sender}",
            f"SUBJECT: {self.subject}",
            f"DATE: {self.sent_at}",
            "",
            self.body.to_prompt_text(),
        ]
        for attachment in self.attachments:
            if not attachment.processed:
                continue
            parts.append(f"\n--- ATTACHMENT: {attachment.file_name} ---")
            parts.append(attachment.to_prompt_text())
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialise the message and all its documents."""
        return {
            "message_id": self.message_id,
            "subject": self.subject,
            "sender": self.sender,
            "recipient": self.recipient,
            "sent_at": self.sent_at,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "body": self.body.to_dict(),
            "attachments": [a.to_dict() for a in self.attachments],
        }
