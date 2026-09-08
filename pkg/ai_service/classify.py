"""
Multi-label classification of a message into the four buckets.

Prompt design notes, since this is the scored part:

**All four categories are always judged.** The model returns a verdict for
each, not just the ones it likes. Forcing a decision on every bucket is what
makes multi-label work -- asking "which category is this?" produces a single
answer and silently loses the reaction-caused-by-a-defect case, which is
legitimately both ICSR and PQC.

**Definitions come from the assignment, not from pharmacovigilance training.**
The four buckets are described in the same plain English the brief uses, with
the specific decision rules that separate them. A model given "classify this
pharmacovigilance email" leans on prior knowledge and drifts; a model given the
operative definitions applies those.

**The reason is required and must cite evidence.** It is one line, and it has
to point at what in the message drove the call. That is what a reviewer reads
first, and requiring it discourages label-by-vibe.
"""
from __future__ import annotations

from .llm import LLMClient
from .models import IngestedMessage
from .schemas import CLASSIFICATION_SCHEMA, Category, CategoryVerdict, Classification

# Truncation guard. Long articles can exceed a sensible prompt budget, and the
# classification signal is almost always in the opening; extraction reads the
# whole document separately.
MAX_PROMPT_CHARS = 24_000

CLASSIFICATION_PROMPT = """\
You are triaging incoming mail for a pharmaceutical company's safety mailbox.
Sort the message below into one or more of four categories.

DEFINITIONS

1. ICSR - Safety Report. A patient had a bad reaction to a drug.
   Requires ALL FOUR of these to be present, even loosely or informally:
     - an identifiable patient (may be "my mother", "a 54-year-old female")
     - an identifiable reporter (the sender counts)
     - a specific product
     - an adverse outcome or bad reaction
   If any one of the four is genuinely absent, it is NOT an ICSR.

2. PQC - Quality Complaint. Something is physically wrong with the product
   itself: broken seal, wrong colour, contamination, damaged packaging,
   counterfeit, chipped or crushed product.
   A defect with NO patient reaction is PQC only.

3. MI - Info Request. Someone is asking a question about a product: dosing,
   administration, interactions, storage.
   A question with NO bad reaction and NO defect is MI only.

4. NOT_RELEVANT - Anything else: marketing, spam, internal admin chatter.

CRITICAL RULES
- Categories are NOT mutually exclusive. A bad reaction caused by a defective
  product is BOTH ICSR and PQC. Judge each category independently.
- NOT_RELEVANT applies only when none of the other three do.
- A message can mention a product without being about a safety issue.
- A question asked alongside a real reaction is still an ICSR; add MI only if a
  genuinely separate product question is also being asked.
- Do not infer a reaction that is not described. Worry, risk or a prospective
  question is not an adverse event.

For EACH of the four categories return:
  applies    - true or false
  confidence - 0.0 to 1.0, your confidence in that specific verdict
  reason     - ONE line citing the specific evidence in the message

Be honest with confidence. Use a low value when the message is ambiguous or
thin. A confident wrong label is worse than an uncertain right one.

MESSAGE
=======
{message}
"""


def build_prompt(message: IngestedMessage) -> str:
    """Render the classification prompt for a message."""
    body = message.to_prompt_text()
    if len(body) > MAX_PROMPT_CHARS:
        body = body[:MAX_PROMPT_CHARS] + "\n[... truncated for classification ...]"
    return CLASSIFICATION_PROMPT.format(message=body)


def parse_verdicts(payload: dict, model: str) -> Classification:
    """Turn raw model output into a :class:`Classification`.

    Any category the model omitted is added back as a non-applying verdict, so
    downstream code can always rely on all four being present rather than
    testing for absence.
    """
    seen: dict[Category, CategoryVerdict] = {}
    for raw in payload.get("verdicts", []):
        try:
            category = Category(str(raw.get("category", "")).strip().upper())
        except ValueError:
            continue
        seen[category] = CategoryVerdict(
            category=category,
            applies=bool(raw.get("applies", False)),
            confidence=min(1.0, max(0.0, float(raw.get("confidence", 0.0) or 0.0))),
            reason=str(raw.get("reason", "")).strip(),
        )

    for category in Category:
        seen.setdefault(
            category,
            CategoryVerdict(
                category=category,
                applies=False,
                confidence=0.0,
                reason="Model returned no verdict for this category.",
            ),
        )

    ordered = [seen[category] for category in Category]
    return Classification(verdicts=ordered, model=model).resolved()


def classify_message(message: IngestedMessage, client: LLMClient | None = None) -> Classification:
    """Classify one message into the four buckets."""
    client = client or LLMClient()
    response = client.generate_json(build_prompt(message), CLASSIFICATION_SCHEMA)
    classification = parse_verdicts(response.data, response.model)

    if response.is_stub:
        for verdict in classification.verdicts:
            if not verdict.reason.startswith("[offline stub]"):
                verdict.reason = f"[offline stub] {verdict.reason}"
    return classification
