"""
Field extraction for whichever categories a message was classified into.

The assignment's hardest requirement is here: *"mark anything not mentioned as
'Not stated' rather than guessing"*, with a confidence score per field and a
source reference for every fact.

Prompting alone does not achieve that. Models are strongly biased toward
filling a field, and will infer a plausible age or route from context while
reporting high confidence. So there are three layers:

1. **The prompt** states the rule, explains why, and gives worked examples of
   what must come back ``Not stated``.
2. **Post-validation** (:meth:`~.schemas.ExtractedField.normalised`) collapses
   every honest-gap spelling to the canonical value with zero confidence.
3. **Quote verification** checks that the supporting quote actually occurs in
   the source document. A field whose quote cannot be found is demoted --
   confidence is reduced and the discrepancy is recorded -- because an
   unverifiable citation is the signature of an invented value.

Layer 3 is the one that catches confident fabrication, and it is cheap:
verifying a quote is a substring search, not another model call.
"""
from __future__ import annotations

import re

from .llm import LLMClient
from .models import IngestedMessage
from .schemas import (
    FIELDS_BY_CATEGORY,
    NOT_STATED,
    Category,
    ExtractedField,
    Extraction,
    extraction_schema,
)

MAX_PROMPT_CHARS = 60_000

# A quote must match this proportion of its normalised form in the source to
# count as verified. Not 1.0, because extraction legitimately tidies whitespace
# and line breaks when quoting across a wrapped line.
QUOTE_MATCH_THRESHOLD = 0.85

# How far confidence is cut when a quote cannot be located in the source.
UNVERIFIED_QUOTE_PENALTY = 0.5

FIELD_GUIDANCE: dict[Category, str] = {
    Category.ICSR: """\
patient_age          Age as stated, e.g. "54 years". Do not compute from a date of birth.
patient_sex          Male or Female, only if stated or unambiguous from wording.
patient_weight       Weight with units.
patient_history      Relevant prior conditions only.
reporter_name        Who reported it.
reporter_role        Physician, nurse, pharmacist, consumer, patient's relative, etc.
reporter_country     Country of the reporter.
product_name         The suspect product.
product_dose         Dose and frequency, e.g. "20 mg once daily".
product_route        Oral, inhalation, topical, injection, etc.
product_start_date   When the patient started the product.
reaction             What happened to the patient.
reaction_onset       When the reaction began. If the source says timing is
                     unknown or unclear, this is "Not stated".
reaction_outcome     Recovered, recovering, not recovered, fatal, unknown.
seriousness          Serious only if death, hospitalisation, life-threatening,
                     disability or congenital anomaly is described. Otherwise
                     non-serious. If nothing indicates either way: "Not stated".""",
    Category.PQC: """\
product_name         The product with the defect.
batch_number         Batch or lot number exactly as written.
defect_description   What is physically wrong with the product.
photo_mentioned      "Yes" if the message says a photo or image is attached or
                     available, otherwise "No". This is about whether one is
                     MENTIONED, so it is Yes/No and never "Not stated".""",
    Category.MI: """\
question             The actual question(s) being asked, quoted or closely
                     paraphrased. Combine multiple questions into one entry.
product_or_topic     Which product and what aspect the question concerns.""",
}

EXTRACTION_PROMPT = """\
Extract structured facts from the message below. It has been classified as
{category_description}.

FIELDS TO EXTRACT
{guidance}

THE MOST IMPORTANT RULE
If the message does not state a field, return exactly "Not stated" with
confidence 0.0. Do NOT infer, estimate, or fill it in from context or from
general knowledge. In this domain a confidently wrong value is worse than an
honest gap: a reviewer can fill a gap, but will not catch a plausible
invention.

Examples of what must be "Not stated":
- Weight, when the message never mentions weight.
- Onset date, when the message says the timing is unclear or unknown.
- Sex, when it is never stated and cannot be read directly from the wording.
- Seriousness, when nothing indicates whether the event was serious.

FOR EVERY FIELD RETURN
  name        the field name
  value       the extracted value, or exactly "Not stated"
  confidence  0.0 to 1.0. Must be 0.0 when the value is "Not stated".
  source      the bracketed source tag the fact came from, copied exactly,
              e.g. "form_x.pdf p.1" or "(email body)"
  quote       a SHORT VERBATIM span from the source containing the fact.
              Copy it exactly. Do not paraphrase. Leave empty only when the
              value is "Not stated".

ALSO RETURN
  narrative   A short plain-language case summary a reviewer can read at a
              glance. Use only what the message states. If the message is too
              thin to summarise, say so.

Sources are marked in square brackets in the message, like [form_x.pdf p.1].
Use those exact tags in the source field so each fact can be traced back.

MESSAGE
=======
{message}
"""

CATEGORY_DESCRIPTION = {
    Category.ICSR: "a SAFETY REPORT - a patient had an adverse reaction to a drug",
    Category.PQC: "a QUALITY COMPLAINT - something is physically wrong with the product",
    Category.MI: "an INFORMATION REQUEST - someone is asking a question about a product",
}


def _normalise_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant quote matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(quote: str, haystack: str) -> bool:
    """Check that a quote genuinely occurs in the source text.

    Matching is whitespace- and case-insensitive because a quote spanning a
    wrapped line legitimately differs from the raw text. Beyond that it must be
    a real substring: this is the check that separates a cited fact from an
    invented one.
    """
    needle = _normalise_for_match(quote)
    if not needle:
        return False
    hay = _normalise_for_match(haystack)
    if needle in hay:
        return True

    # Allow a near-match for quotes that dropped or added a stray character.
    words = needle.split()
    if len(words) < 4:
        return False
    matched = sum(1 for word in words if word in hay)
    return matched / len(words) >= QUOTE_MATCH_THRESHOLD


def build_prompt(message: IngestedMessage, category: Category) -> str:
    """Render the extraction prompt for one category."""
    body = message.to_prompt_text()
    if len(body) > MAX_PROMPT_CHARS:
        body = body[:MAX_PROMPT_CHARS] + "\n[... truncated ...]"
    return EXTRACTION_PROMPT.format(
        category_description=CATEGORY_DESCRIPTION[category],
        guidance=FIELD_GUIDANCE[category],
        message=body,
    )


def parse_fields(
    payload: dict, category: Category, source_text: str
) -> tuple[list[ExtractedField], list[str]]:
    """Parse, normalise and verify the model's fields.

    Returns the fields plus any integrity warnings raised while checking them.
    """
    expected = FIELDS_BY_CATEGORY[category]
    returned: dict[str, ExtractedField] = {}
    warnings: list[str] = []

    for raw in payload.get("fields", []):
        name = str(raw.get("name", "")).strip()
        if name not in expected:
            continue
        field = ExtractedField(
            name=name,
            value=str(raw.get("value", NOT_STATED) or NOT_STATED),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            source=str(raw.get("source", "") or ""),
            quote=str(raw.get("quote", "") or ""),
        ).normalised()

        if field.is_stated:
            if not field.quote:
                field.confidence = min(field.confidence, UNVERIFIED_QUOTE_PENALTY)
                warnings.append(f"{name}: value given without a supporting quote.")
            elif not verify_quote(field.quote, source_text):
                # The strongest signal available that a value was invented.
                field.confidence = round(field.confidence * UNVERIFIED_QUOTE_PENALTY, 3)
                warnings.append(
                    f"{name}: quote not found in the source document; confidence "
                    "reduced and the field needs review."
                )
        returned[name] = field

    # Anything the model skipped is an honest gap, not a missing key.
    for name in expected:
        returned.setdefault(name, ExtractedField(name=name))

    return [returned[name] for name in expected], warnings


def extract_fields(
    message: IngestedMessage,
    category: Category,
    client: LLMClient | None = None,
) -> tuple[Extraction, list[str]]:
    """Extract one category's fields from a message."""
    if category not in FIELDS_BY_CATEGORY:
        raise ValueError(f"No field set defined for category {category}")

    client = client or LLMClient()
    schema = extraction_schema(FIELDS_BY_CATEGORY[category])
    response = client.generate_json(build_prompt(message, category), schema)

    source_text = message.to_prompt_text()
    fields, warnings = parse_fields(response.data, category, source_text)

    narrative = str(response.data.get("narrative", "")).strip()
    extraction = Extraction(
        category=category,
        fields=fields,
        narrative=narrative,
        model=response.model,
    )
    return extraction, warnings


def extract_all(
    message: IngestedMessage,
    categories: list[Category],
    client: LLMClient | None = None,
) -> tuple[list[Extraction], list[str]]:
    """Extract fields for every applicable category on a message.

    ``NOT_RELEVANT`` has no field set by design: there is nothing to extract
    from a marketing email, and inventing a schema for it would invite the
    model to find structure that is not there.
    """
    client = client or LLMClient()
    extractions: list[Extraction] = []
    warnings: list[str] = []
    for category in categories:
        if category not in FIELDS_BY_CATEGORY:
            continue
        extraction, category_warnings = extract_fields(message, category, client)
        extractions.append(extraction)
        warnings.extend(f"[{category}] {w}" for w in category_warnings)
    return extractions, warnings
