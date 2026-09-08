"""
Output contracts for every model call.

The assignment scores "structured outputs, sensible confidence scoring, saying
unknown instead of guessing". All three are enforced here rather than hoped for
in a prompt: the model is given a JSON schema it must satisfy, and the parsed
result is then validated again on our side, because a model that returns valid
JSON can still return a confident value for a field the document never
mentioned.

Every extracted fact carries four things:

    value        what was found, or the literal string "Not stated"
    confidence   0.0-1.0, and forced to 0.0 when the value is "Not stated"
    source       which document and page it came from
    quote        the span of source text supporting it

The quote is what makes the source verifiable rather than merely asserted. A
reviewer can search for it in the original; a fabricated quote is detectable,
whereas a fabricated page number is not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

NOT_STATED = "Not stated"


class Category(StrEnum):
    """The four buckets from the assignment."""

    ICSR = "ICSR"
    PQC = "PQC"
    MI = "MI"
    NOT_RELEVANT = "NOT_RELEVANT"


# Field groups per category, matching the assignment's Section 3D tables.
ICSR_FIELDS = [
    "patient_age",
    "patient_sex",
    "patient_weight",
    "patient_history",
    "reporter_name",
    "reporter_role",
    "reporter_country",
    "product_name",
    "product_dose",
    "product_route",
    "product_start_date",
    "reaction",
    "reaction_onset",
    "reaction_outcome",
    "seriousness",
]

PQC_FIELDS = [
    "product_name",
    "batch_number",
    "defect_description",
    "photo_mentioned",
]

MI_FIELDS = [
    "question",
    "product_or_topic",
]

FIELDS_BY_CATEGORY: dict[Category, list[str]] = {
    Category.ICSR: ICSR_FIELDS,
    Category.PQC: PQC_FIELDS,
    Category.MI: MI_FIELDS,
}


@dataclass
class ExtractedField:
    """One extracted fact with its confidence and its provenance."""

    name: str
    value: str = NOT_STATED
    confidence: float = 0.0
    source: str = ""
    quote: str = ""

    @property
    def is_stated(self) -> bool:
        """Whether the source actually provided this value."""
        return self.value.strip().lower() not in {"", "not stated", "unknown", "n/a", "none stated"}

    def normalised(self) -> "ExtractedField":
        """Return a copy with the "unknown beats a guess" rules enforced.

        A model asked for a missing field will sometimes return an empty
        string, ``null``, ``N/A`` or a plausible invention with high
        confidence. All the honest-gap spellings collapse to the canonical
        ``Not stated`` with zero confidence, and confidence is clamped into
        range, so downstream code and the reviewer see one consistent signal.
        """
        if not self.is_stated:
            return ExtractedField(name=self.name, value=NOT_STATED, confidence=0.0)
        return ExtractedField(
            name=self.name,
            value=self.value.strip(),
            confidence=min(1.0, max(0.0, float(self.confidence))),
            source=self.source.strip(),
            quote=self.quote.strip(),
        )

    def to_dict(self) -> dict:
        """Serialise for the API and for saved sample outputs."""
        return asdict(self)


@dataclass
class CategoryVerdict:
    """The model's judgement on one category."""

    category: Category
    applies: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "category": str(self.category),
            "applies": self.applies,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }


@dataclass
class Classification:
    """A full multi-label classification of one message."""

    verdicts: list[CategoryVerdict] = field(default_factory=list)
    model: str = ""

    @property
    def labels(self) -> list[Category]:
        """Categories that apply, most confident first."""
        applied = [v for v in self.verdicts if v.applies]
        applied.sort(key=lambda v: v.confidence, reverse=True)
        return [v.category for v in applied]

    def resolved(self) -> "Classification":
        """Apply the one rule the four buckets impose on each other.

        ``NOT_RELEVANT`` is defined as "anything else", so it cannot coexist
        with a substantive label. If the model marks both, the substantive
        labels win and NOT_RELEVANT is cleared -- a message containing a real
        safety report is not also irrelevant.
        """
        substantive = [
            v for v in self.verdicts if v.applies and v.category is not Category.NOT_RELEVANT
        ]
        if not substantive:
            return self
        verdicts = []
        for verdict in self.verdicts:
            if verdict.category is Category.NOT_RELEVANT and verdict.applies:
                verdict = CategoryVerdict(
                    category=verdict.category,
                    applies=False,
                    confidence=verdict.confidence,
                    reason=(
                        "Cleared: message carries a substantive label "
                        f"({', '.join(str(v.category) for v in substantive)})."
                    ),
                )
            verdicts.append(verdict)
        return Classification(verdicts=verdicts, model=self.model)

    def to_dict(self) -> dict:
        return {
            "labels": [str(c) for c in self.labels],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "model": self.model,
        }


@dataclass
class Extraction:
    """The fields extracted for one category of one message."""

    category: Category
    fields: list[ExtractedField] = field(default_factory=list)
    narrative: str = ""
    model: str = ""

    @property
    def completeness(self) -> float:
        """Fraction of expected fields the source actually stated.

        Reported alongside the fields because a low value is informative --
        it means the source was thin, not that extraction failed.
        """
        if not self.fields:
            return 0.0
        return sum(1 for f in self.fields if f.is_stated) / len(self.fields)

    def to_dict(self) -> dict:
        return {
            "category": str(self.category),
            "completeness": round(self.completeness, 3),
            "narrative": self.narrative,
            "model": self.model,
            "fields": {f.name: f.to_dict() for f in self.fields},
        }


@dataclass
class DocumentSummary:
    """A reviewer-facing summary of one PDF."""

    document_id: str
    summary: str
    looks_relevant: bool
    relevance_reason: str
    confidence: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "summary": self.summary,
            "looks_relevant": self.looks_relevant,
            "relevance_reason": self.relevance_reason,
            "confidence": round(float(self.confidence), 3),
            "model": self.model,
        }


# --------------------------------------------------------------------------
# JSON schemas handed to the model. Kept adjacent to the dataclasses they
# populate so the two cannot drift apart unnoticed.
# --------------------------------------------------------------------------

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in Category],
                    },
                    "applies": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["category", "applies", "confidence", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


def extraction_schema(field_names: list[str]) -> dict:
    """Build the response schema for a given category's field list."""
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": field_names},
                        "value": {"type": "string"},
                        "confidence": {"type": "number"},
                        "source": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["name", "value", "confidence", "source", "quote"],
                },
            },
            "narrative": {"type": "string"},
        },
        "required": ["fields", "narrative"],
    }


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "looks_relevant": {"type": "boolean"},
        "relevance_reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "looks_relevant", "relevance_reason", "confidence"],
}


OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "confidence": {"type": "number"},
        "is_handwritten": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["text", "confidence", "is_handwritten", "notes"],
}


IMAGE_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "shows_product_defect": {"type": "boolean"},
        "shows_clinical_sign": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": [
        "description",
        "shows_product_defect",
        "shows_clinical_sign",
        "confidence",
    ],
}
