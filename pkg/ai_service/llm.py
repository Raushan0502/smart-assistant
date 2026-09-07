"""
Model access: one interface, a real provider and an offline stub behind it.

Gemini is the default provider because this pipeline needs vision (scanned
pages, handwriting, product photos) and text in the same model, at a price and
free-tier that suit a prototype.

**The stub is not a test fixture bolted on afterwards.** It is a first-class
provider, selected automatically when no API key is present, and it makes the
whole system runnable and demonstrable offline. That matters for three
practical reasons: the test suite must not require a key or a network, a live
walkthrough must not fail because a free-tier quota was exhausted mid-demo, and
development continues when the key is missing. It returns deterministic,
schema-valid, deliberately *low-confidence* output clearly marked as stubbed,
so stub output can never be mistaken for a real model's judgement.

Data handling, as the assignment asks: prompts are sent to Google's API when a
real key is configured. Everything in this repository is synthetic, so no
patient data leaves the machine. In production this is the decision point --
either a self-hosted model or a provider under a data-processing agreement with
no training retention. That trade-off is argued in the write-up.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "offline-stub"
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


@dataclass
class LLMResponse:
    """One model response, with the metadata needed for the audit log."""

    data: dict[str, Any]
    model: str
    latency_ms: float
    is_stub: bool = False
    raw: str = ""


class LLMError(RuntimeError):
    """Raised when a model call cannot be completed or parsed."""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Even in JSON mode, models occasionally wrap output in a fenced code block
    or add a sentence before it. Rather than fail the document, the first
    balanced object is recovered. A parse that still fails raises, because a
    silently empty result would look like "the document said nothing" -- which
    is exactly the false negative this system must not produce.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object in model response: {text[:200]!r}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Malformed JSON in model response: {exc}") from exc
    raise LLMError("Unterminated JSON object in model response")


class StubProvider:
    """Deterministic offline provider.

    Returns schema-valid output derived from simple keyword heuristics. It
    exists so the pipeline runs without a key -- not to be good at the task.
    Confidence is capped low and every reason is prefixed so stub results are
    obvious wherever they surface.
    """

    name = STUB_MODEL_NAME
    is_stub = True

    # Signals for the four buckets. Crude on purpose: this is a fallback, and
    # making it cleverer would blur the line between stub and real output.
    SIGNALS: dict[str, list[str]] = {
        "ICSR": [
            "reaction", "adverse", "rash", "dizzy", "dizziness", "nausea",
            "vomiting", "hospital", "admitted", "died", "death", "side effect",
            "blister", "swelling", "urticaria", "tremor", "breath",
        ],
        "PQC": [
            "broken seal", "damaged", "discolour", "discolor", "wrong colour",
            "wrong color", "contamination", "counterfeit", "batch", "lot ",
            "crushed", "chipped", "packaging", "defect",
        ],
        "MI": [
            "could you advise", "is it safe", "question", "dosing", "dose adjustment",
            "interaction", "how should", "recommend", "?",
        ],
        "NOT_RELEVANT": [
            "unsubscribe", "marketing", "revenue", "discovery call", "office will be closed",
            "bank holiday", "newsletter", "webinar",
        ],
    }

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> dict[str, Any]:
        """Produce schema-valid stub output for whichever call was made."""
        properties = schema.get("properties", {})
        if "verdicts" in properties:
            return self._classify(prompt)
        if "fields" in properties:
            return self._extract(prompt, schema)
        if "summary" in properties:
            return {
                "summary": (
                    "[offline stub] No model was available, so this document has "
                    "not been summarised. It requires manual review."
                ),
                "looks_relevant": True,
                "relevance_reason": "[offline stub] Relevance not assessed.",
                "confidence": 0.1,
            }
        if "text" in properties:
            return {
                "text": "",
                "confidence": 0.0,
                "is_handwritten": True,
                "notes": (
                    "[offline stub] No vision model available; this page was not "
                    "read and needs manual transcription."
                ),
            }
        if "description" in properties:
            return {
                "description": "[offline stub] Image not analysed; needs human review.",
                "shows_product_defect": False,
                "shows_clinical_sign": False,
                "confidence": 0.0,
            }
        raise LLMError("Stub provider does not recognise the requested schema")

    def _classify(self, prompt: str) -> dict[str, Any]:
        """Keyword-count classification, deliberately low confidence."""
        lowered = prompt.lower()
        hits = {
            category: sum(1 for token in tokens if token in lowered)
            for category, tokens in self.SIGNALS.items()
        }
        substantive = {k: v for k, v in hits.items() if k != "NOT_RELEVANT"}
        any_substantive = any(v >= 2 for v in substantive.values())

        verdicts = []
        for category, count in hits.items():
            if category == "NOT_RELEVANT":
                applies = not any_substantive
            else:
                applies = count >= 2
            verdicts.append(
                {
                    "category": category,
                    "applies": applies,
                    # Capped well below anything a real model would report, so
                    # stub output never looks authoritative.
                    "confidence": round(min(0.45, 0.15 + 0.05 * count), 3),
                    "reason": f"[offline stub] {count} keyword signal(s) matched.",
                }
            )
        return {"verdicts": verdicts}

    def _extract(self, prompt: str, schema: dict) -> dict[str, Any]:
        """Return every requested field as Not stated.

        The stub deliberately extracts nothing. Guessing values from a keyword
        pass would produce exactly the confident-but-wrong output the whole
        design is trying to avoid.
        """
        names = (
            schema["properties"]["fields"]["items"]["properties"]["name"].get("enum", [])
        )
        return {
            "fields": [
                {
                    "name": name,
                    "value": "Not stated",
                    "confidence": 0.0,
                    "source": "",
                    "quote": "",
                }
                for name in names
            ],
            "narrative": (
                "[offline stub] No model available; no narrative was generated."
            ),
        }


class GeminiProvider:
    """Google Gemini, in JSON mode, with vision support."""

    is_stub = False

    def __init__(self, api_key: str, model: str, vision_model: str | None = None):
        from google import genai

        self._genai = genai
        # The client is created once and reused: constructing one per call
        # leaves the shared transport to be closed by the garbage collector,
        # which surfaces later as "client has been closed" mid-run.
        self._client = genai.Client(api_key=api_key)
        self.name = model
        self.vision_model = vision_model or model

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> dict[str, Any]:
        """Call the model in JSON mode and return the parsed object."""
        from google.genai import types

        contents: list[Any] = [prompt]
        for image in images or []:
            contents.append(types.Part.from_bytes(data=image, mime_type="image/png"))

        model = self.vision_model if images else self.name
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            # Deterministic output: this is extraction, not composition, and
            # reproducibility matters more than variety for an audit trail.
            temperature=0.0,
        )
        response = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        return _extract_json(response.text or "")


class LLMClient:
    """The interface the rest of the service uses.

    Chooses a provider at construction: Gemini when a key is configured,
    otherwise the offline stub. Callers never branch on which is active; they
    read ``is_stub`` on the response if they need to surface it.
    """

    def __init__(self, provider: Any | None = None):
        self.provider = provider or self._select_provider()

    @staticmethod
    def _select_provider() -> Any:
        """Pick Gemini if configured, else fall back to the offline stub."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not set - using the offline stub. Output will "
                "be low-confidence placeholders, not real extraction."
            )
            return StubProvider()
        try:
            return GeminiProvider(
                api_key=api_key,
                model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                vision_model=os.getenv("VISION_MODEL", "gemini-2.0-flash"),
            )
        except Exception as exc:  # noqa: BLE001 -- never let setup kill the run
            logger.error("Gemini unavailable (%s); falling back to stub.", exc)
            return StubProvider()

    @property
    def is_stub(self) -> bool:
        """Whether the active provider is the offline stub."""
        return bool(getattr(self.provider, "is_stub", False))

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> LLMResponse:
        """Call the model, retrying transient failures with backoff."""
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                data = self.provider.generate_json(prompt, schema, images=images)
                return LLMResponse(
                    data=data,
                    model=self.provider.name,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    is_stub=self.is_stub,
                )
            except Exception as exc:  # noqa: BLE001 -- retry, then report honestly
                last_error = exc
                logger.warning(
                    "Model call failed (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(BACKOFF_SECONDS * attempt)

        raise LLMError(f"Model call failed after {MAX_ATTEMPTS} attempts: {last_error}")
