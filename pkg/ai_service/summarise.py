"""
Reviewer-facing summaries of PDF attachments.

The assignment asks for "a short AI-generated summary (10-15 sentences) of each
PDF for a human reviewer, saying whether it looks relevant and why".

The audience shapes the prompt. This is not a summary for a search index or for
another model -- it is read by a person deciding within a few seconds whether
to open the document. So it leads with the decision-relevant facts, and the
relevance judgement is a separate structured field rather than something buried
in the prose, because the review screen sorts and filters on it.

The same module serves the bonus literature-screening extension: deciding
whether an article describes a real identifiable patient case is the same
question in a different costume, and :func:`screen_article` reuses the
machinery with a stricter rubric.
"""
from __future__ import annotations

from .llm import LLMClient
from .models import ExtractedDocument
from .schemas import SUMMARY_SCHEMA, DocumentSummary

MAX_PROMPT_CHARS = 60_000

SUMMARY_PROMPT = """\
Summarise this document for a pharmacovigilance reviewer who will spend about
ten seconds deciding whether to open it.

WRITE 10 TO 15 SENTENCES covering, where the document states them:
- what kind of document this is
- who the patient is
- who is reporting
- which product is involved
- what happened, and when
- the outcome and how serious it was
- anything notable in tables or figures
- anything missing that a reviewer would expect to see

RULES
- Use ONLY what the document states. Do not add background knowledge.
- If something is not stated, say it is not stated rather than omitting it
  silently -- a reviewer needs to know what is missing.
- Plain language. No hedging padding.

ALSO RETURN
  looks_relevant     true if this document appears to concern a safety report,
                     product quality complaint, or medical information request
  relevance_reason   ONE line explaining that judgement
  confidence         0.0 to 1.0 in your relevance judgement

DOCUMENT
========
{document}
"""

SCREENING_PROMPT = """\
Decide whether this article describes a real, identifiable patient case that
would be worth reporting to a safety database.

WHAT COUNTS AS AN IDENTIFIABLE CASE
An individual patient whose experience is described specifically enough to
report: some patient detail (age, sex), a product, and an adverse event that
happened to THAT person.

WHAT DOES NOT COUNT
- Review articles that discuss adverse effects in general.
- Cohort or registry studies reporting rates across a population, even when
  adverse events are described in detail. Aggregate numbers are not an
  identifiable individual.
- Background, mechanism or discussion sections with no specific patient.
- The reference list.

An article may contain MORE THAN ONE case. Count them separately: two patients
described in one paper are two cases, not one.

RETURN
  summary            10 to 15 sentences describing what the article reports and,
                     if there are cases, who they concern and what happened.
  looks_relevant     true only if at least one identifiable patient case is
                     described
  relevance_reason   ONE line. If false, say specifically why -- for example
                     "aggregate cohort data, no individual patient described".
  confidence         0.0 to 1.0 in this judgement

ARTICLE
=======
{document}
"""


def _render(document: ExtractedDocument) -> str:
    """Render a document for a prompt, truncating very long ones."""
    body = document.to_prompt_text()
    if len(body) > MAX_PROMPT_CHARS:
        body = body[:MAX_PROMPT_CHARS] + "\n[... truncated ...]"
    return body


def _build(document: ExtractedDocument, payload: dict, model: str) -> DocumentSummary:
    """Turn a raw model payload into a :class:`DocumentSummary`."""
    return DocumentSummary(
        document_id=document.document_id,
        summary=str(payload.get("summary", "") or "").strip(),
        looks_relevant=bool(payload.get("looks_relevant", False)),
        relevance_reason=str(payload.get("relevance_reason", "") or "").strip(),
        confidence=min(1.0, max(0.0, float(payload.get("confidence", 0.0) or 0.0))),
        model=model,
    )


def summarise_document(
    document: ExtractedDocument, client: LLMClient | None = None
) -> DocumentSummary:
    """Summarise one attachment for the review queue."""
    client = client or LLMClient()
    response = client.generate_json(
        SUMMARY_PROMPT.format(document=_render(document)), SUMMARY_SCHEMA
    )
    return _build(document, response.data, response.model)


def screen_article(
    document: ExtractedDocument, client: LLMClient | None = None
) -> DocumentSummary:
    """Judge whether an article contains a reportable patient case.

    Used by the bonus literature-screening extension. The rubric is stricter
    than the general summary: an article can describe adverse events at length
    and still contain no reportable case, which is the distinction a keyword
    approach gets wrong.
    """
    client = client or LLMClient()
    response = client.generate_json(
        SCREENING_PROMPT.format(document=_render(document)), SUMMARY_SCHEMA
    )
    return _build(document, response.data, response.model)
