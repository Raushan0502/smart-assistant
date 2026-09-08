"""
Persistence model for the review queue and its audit trail.

Three of the assignment's ground rules are enforced by the schema rather than
by application code, because a rule that lives only in code is one refactor
away from being untrue:

**Every extracted fact links to its source.** :class:`ExtractedField` carries
``source_document``, ``source_page`` and ``source_quote``, and is meaningless
without them. There is no way to persist a fact with no provenance.

**Every AI decision is traceable to the input that produced it.**
:class:`AuditEvent` records the model, the prompt hash, latency, and the
document the call was about -- written for every model call, not only the
successful ones.

**Every reviewer action is timestamped.** :class:`ReviewAction` is
append-only: an override does not overwrite the AI's value, it records a new
row beside it. The original stays visible, which is the point of an audit
trail.
"""
from __future__ import annotations

from django.db import models

NOT_STATED = "Not stated"


class Category(models.TextChoices):
    """The four buckets from the assignment."""

    ICSR = "ICSR", "Safety Report"
    PQC = "PQC", "Quality Complaint"
    MI = "MI", "Info Request"
    NOT_RELEVANT = "NOT_RELEVANT", "Not Relevant"


class ProcessingStatus(models.TextChoices):
    """Where a message is in the pipeline."""

    QUEUED = "QUEUED", "Queued"
    PROCESSING = "PROCESSING", "Processing"
    READY = "READY", "Ready for review"
    FAILED = "FAILED", "Failed"


class ReviewStatus(models.TextChoices):
    """Where a message is in human review."""

    PENDING = "PENDING", "Pending review"
    ACCEPTED = "ACCEPTED", "Accepted"
    OVERRIDDEN = "OVERRIDDEN", "Overridden"


class Message(models.Model):
    """One email received in the shared mailbox."""

    # The mail system's own identity where available, else a content hash, so
    # re-polling the same mailbox cannot create duplicates.
    message_id = models.CharField(max_length=400, unique=True, db_index=True)
    subject = models.CharField(max_length=1000, blank=True)
    sender = models.CharField(max_length=400, blank=True)
    recipient = models.CharField(max_length=400, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    received_at = models.DateTimeField(auto_now_add=True)

    body_text = models.TextField(blank=True)
    language = models.CharField(max_length=8, default="en")

    processing_status = models.CharField(
        max_length=20, choices=ProcessingStatus, default=ProcessingStatus.QUEUED,
        db_index=True,
    )
    review_status = models.CharField(
        max_length=20, choices=ReviewStatus, default=ReviewStatus.PENDING,
        db_index=True,
    )

    # Wall-clock cost of the whole pipeline for this message. The assignment
    # asks how long each document takes, so it is stored rather than inferred
    # from log timestamps after the fact.
    processing_ms = models.IntegerField(null=True, blank=True)
    error = models.TextField(blank=True)
    warnings = models.JSONField(default=list, blank=True)
    headers = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-received_at"]
        indexes = [models.Index(fields=["processing_status", "review_status"])]

    def __str__(self) -> str:
        return f"{self.message_id}: {self.subject[:60]}"

    @property
    def categories(self) -> list[str]:
        """Categories the AI decided apply, most confident first."""
        return [
            c.category
            for c in self.classifications.filter(applies=True).order_by("-confidence")
        ]


class Document(models.Model):
    """The email body, or one attachment."""

    class Flavour(models.TextChoices):
        DIGITAL = "digital", "Digital PDF"
        SCANNED = "scanned", "Scanned / handwritten"
        ARTICLE = "article", "Published article"
        NON_ENGLISH = "non_english", "Non-English"
        EMAIL_BODY = "email_body", "Email body"
        UNKNOWN = "unknown", "Unknown"

    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="documents"
    )
    document_id = models.CharField(max_length=400, db_index=True)
    file_name = models.CharField(max_length=500)
    media_type = models.CharField(max_length=200, blank=True)
    flavour = models.CharField(max_length=20, choices=Flavour, default=Flavour.UNKNOWN)
    language = models.CharField(max_length=8, default="en")
    page_count = models.IntegerField(default=0)

    # False for attachment types the pipeline deliberately does not parse. Kept
    # rather than discarded so a reviewer can see something arrived.
    processed = models.BooleanField(default=True)

    extracted_text = models.TextField(blank=True)
    # Tables are stored as structured rows, never flattened: the link between a
    # value, its unit and its reference range is the information.
    tables = models.JSONField(default=list, blank=True)
    images = models.JSONField(default=list, blank=True)
    doc_metadata = models.JSONField(default=dict, blank=True)
    warnings = models.JSONField(default=list, blank=True)

    # Reviewer-facing summary and the AI's relevance judgement.
    summary = models.TextField(blank=True)
    looks_relevant = models.BooleanField(null=True, blank=True)
    relevance_reason = models.TextField(blank=True)
    summary_confidence = models.FloatField(null=True, blank=True)

    # Set when content came from OCR rather than a text layer, so the UI can
    # warn that the values were read from pixels.
    ocr_confidence = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = [("message", "document_id")]

    def __str__(self) -> str:
        return f"{self.file_name} ({self.flavour})"


class Classification(models.Model):
    """The AI's verdict on one category for one message.

    One row per category per message -- all four are always stored, including
    the ones that do not apply. Keeping the negatives makes the decision
    auditable: a reviewer can see the model considered PQC and rejected it,
    with its reason, rather than inferring that from an absence.
    """

    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="classifications"
    )
    category = models.CharField(max_length=20, choices=Category)
    applies = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    reason = models.TextField(blank=True)
    model_name = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("message", "category")]
        ordering = ["-confidence"]

    def __str__(self) -> str:
        return f"{self.category}={self.applies} ({self.confidence:.2f})"


class Extraction(models.Model):
    """The set of fields extracted for one category of one message."""

    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="extractions"
    )
    category = models.CharField(max_length=20, choices=Category)
    narrative = models.TextField(blank=True)
    model_name = models.CharField(max_length=100, blank=True)
    # Fraction of expected fields the source actually stated. Low is
    # informative -- it means a thin source, not a failed extraction.
    completeness = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("message", "category")]

    def __str__(self) -> str:
        return f"{self.category} extraction for {self.message_id}"


class ExtractedField(models.Model):
    """One extracted fact, with its confidence and its provenance.

    ``source_document``, ``source_page`` and ``source_quote`` are what make the
    fact auditable. The quote in particular is verifiable: a reviewer can search
    for it in the original, and a fabricated quote is detectable in a way a
    fabricated page number is not.
    """

    extraction = models.ForeignKey(
        Extraction, on_delete=models.CASCADE, related_name="fields"
    )
    name = models.CharField(max_length=100, db_index=True)
    # Bounded rather than a TextField: extracted values are short by nature
    # (an age, a dose, a date), and on Oracle a TextField becomes NCLOB, which
    # cannot be used as a comparison key -- so the PL/SQL that counts stated
    # versus "Not stated" fields could not read it.
    value = models.CharField(max_length=2000, default=NOT_STATED)
    confidence = models.FloatField(default=0.0)

    source_document = models.CharField(max_length=500, blank=True)
    source_page = models.IntegerField(null=True, blank=True)
    source_quote = models.TextField(blank=True)
    # False when the supporting quote could not be found in the source. Such a
    # field has had its confidence reduced and needs human attention.
    quote_verified = models.BooleanField(default=False)

    class Meta:
        unique_together = [("extraction", "name")]
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name}={self.value[:40]}"

    @property
    def is_stated(self) -> bool:
        """Whether the source actually provided this value."""
        return self.value.strip().lower() not in {"", "not stated", "unknown", "n/a"}


class ReviewAction(models.Model):
    """A human decision, recorded rather than applied destructively.

    Append-only by design. An override stores the reviewer's value alongside
    the AI's original instead of replacing it, so the audit trail shows what
    the system proposed, what the human decided, and when.
    """

    class Action(models.TextChoices):
        ACCEPT = "ACCEPT", "Accepted AI output"
        OVERRIDE_FIELD = "OVERRIDE_FIELD", "Overrode a field value"
        OVERRIDE_CATEGORY = "OVERRIDE_CATEGORY", "Overrode a category"
        REJECT = "REJECT", "Rejected as not relevant"
        COMMENT = "COMMENT", "Added a comment"

    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="review_actions"
    )
    action = models.CharField(max_length=30, choices=Action)
    reviewer = models.CharField(max_length=200, default="demo-reviewer")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    # Populated for field-level overrides.
    field_name = models.CharField(max_length=100, blank=True)
    original_value = models.TextField(blank=True)
    new_value = models.TextField(blank=True)
    # Populated for category overrides.
    category = models.CharField(max_length=20, blank=True)
    note = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.action} by {self.reviewer} at {self.created_at:%Y-%m-%d %H:%M}"


class AuditEvent(models.Model):
    """One AI call, recorded for traceability.

    Written for every model call including failures, because a failed call is
    part of how a record came to look the way it does. The prompt is stored as
    a hash plus its length rather than in full: the full text can be large and
    would bloat the table, while the hash still proves which prompt produced a
    given output.
    """

    class EventType(models.TextChoices):
        CLASSIFY = "CLASSIFY", "Classification"
        EXTRACT = "EXTRACT", "Field extraction"
        OCR = "OCR", "Scanned page transcription"
        IMAGE = "IMAGE", "Image description"
        SUMMARY = "SUMMARY", "Document summary"
        SCREEN = "SCREEN", "Literature screening"

    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="audit_events",
        null=True, blank=True,
    )
    document = models.ForeignKey(
        Document, on_delete=models.SET_NULL, related_name="audit_events",
        null=True, blank=True,
    )
    event_type = models.CharField(max_length=20, choices=EventType, db_index=True)
    model_name = models.CharField(max_length=100, blank=True)
    is_stub = models.BooleanField(default=False)

    prompt_sha256 = models.CharField(max_length=64, blank=True)
    prompt_chars = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0.0)

    succeeded = models.BooleanField(default=True)
    error = models.TextField(blank=True)
    # A trimmed copy of what came back, enough to reconstruct the decision.
    response_summary = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["event_type", "created_at"])]

    def __str__(self) -> str:
        outcome = "ok" if self.succeeded else "FAILED"
        return f"{self.event_type} {outcome} via {self.model_name}"
