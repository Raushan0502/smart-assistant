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

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "offline-stub"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0

# Free-tier quotas are windowed, so the ordinary 2s/4s backoff retries straight
# back into the same closed window and exhausts the budget in seconds. Measured:
# 17 of 34 calls failed this way on a first live run, and every one left its
# message unclassified.
#
# The API usually states how long to wait ("Please retry in 2.24s"), which is
# authoritative -- a fixed guess is either wasteful or too short. This is only
# the fallback when no delay is given.
RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota")
RETRY_DELAY_PATTERN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)

# A per-minute window rolls over in seconds, so waiting is worth it. A quoted
# delay longer than this means the window is not the problem -- the allowance
# is spent -- and retrying just burns the caller's timeout for a result that
# cannot arrive. Waiting is only sensible when there is something to wait for.
MAX_USEFUL_WAIT_SECONDS = 20.0


class QuotaExhausted(RuntimeError):
    """The provider's allowance is spent; retrying cannot succeed."""


def rate_limit_delay(error: str) -> float | None:
    """Seconds to wait before retrying a rate-limited call.

    Returns None when the error is not a rate limit at all. Raises
    :class:`QuotaExhausted` when the provider signals an allowance that will
    not refill in a useful timeframe -- which is a different condition from
    being briefly throttled, and must not be retried.

    This distinction is not academic. Retrying an exhausted daily quota three
    times at 60s each consumed 180s per call, and with several calls per
    message that reliably exceeded the caller's 300s timeout -- turning a clear
    "quota spent" into an opaque "read timed out".
    """
    if not any(marker.lower() in error.lower() for marker in RATE_LIMIT_MARKERS):
        return None

    match = RETRY_DELAY_PATTERN.search(error)
    if not match:
        raise QuotaExhausted(
            "Provider reported a quota limit with no retry delay; the "
            "allowance appears to be spent."
        )

    delay = float(match.group(1)) + 1.0
    if delay > MAX_USEFUL_WAIT_SECONDS:
        raise QuotaExhausted(
            f"Provider asked to retry in {delay:.0f}s, beyond the "
            f"{MAX_USEFUL_WAIT_SECONDS:.0f}s worth waiting for; the allowance "
            "appears to be spent."
        )
    return delay


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


class AllProvidersExhausted(LLMError):
    """Every configured provider refused the request.

    Carries a message written for a reviewer looking at the screen, not for a
    log file. The underlying provider errors are kept in ``details`` for the
    audit trail, because the person debugging it needs both.
    """

    USER_MESSAGE = (
        "AI usage limit reached. All configured providers have exhausted their "
        "free-tier allowance. Wait for the quota to reset, add another provider "
        "key, or upgrade to a paid plan to continue processing."
    )

    def __init__(self, details: dict[str, str]):
        self.details = details
        tried = ", ".join(f"{name} ({reason})" for name, reason in details.items())
        super().__init__(f"{self.USER_MESSAGE} Tried: {tried}")


def extract_json(text: str) -> dict[str, Any]:
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

    # Prompts embed the message after a marker. The stub must score only the
    # message: scoring the whole prompt matches the instructions themselves --
    # which name every signal word - and labels every message with every
    # category, including obvious marketing.
    MESSAGE_MARKERS = ("MESSAGE\n=======", "ARTICLE\n=======", "DOCUMENT\n========")

    def _message_body(self, prompt: str) -> str:
        """Return just the embedded message, excluding the instructions."""
        for marker in self.MESSAGE_MARKERS:
            if marker in prompt:
                return prompt.split(marker, 1)[1]
        return prompt

    def _classify(self, prompt: str) -> dict[str, Any]:
        """Keyword-count classification, deliberately low confidence."""
        lowered = self._message_body(prompt).lower()
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
        """Create one reusable client for this process."""
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
        return extract_json(response.text or "")


def build_provider_chain() -> list[Any]:
    """Build the ordered provider chain from whichever keys are configured.

    Order is deliberate. Gemini leads because it is the only one wired for
    vision here, so scanned pages and images work on the primary path. Groq and
    Mistral follow as text-only fallbacks: when Gemini's allowance is spent,
    classification and extraction keep working even though OCR cannot.

    A provider whose SDK or key is unusable is skipped with a warning rather
    than failing construction -- one broken key must not disable the others.
    The offline stub always terminates the chain, so the pipeline never has
    nothing to call.
    """
    chain: list[Any] = []
    candidates = [
        ("Gemini", "GEMINI_API_KEY", GeminiProvider,
         os.getenv("LLM_MODEL", "gemini-3.5-flash")),
        ("Groq", "GROQ_API_KEY", GroqProvider,
         os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
        ("Mistral", "MISTRAL_API_KEY", MistralProvider,
         os.getenv("MISTRAL_MODEL", "mistral-large-latest")),
    ]

    for label, env_var, factory, model in candidates:
        api_key = os.getenv(env_var, "").strip()
        if not api_key:
            continue
        try:
            if factory is GeminiProvider:
                chain.append(factory(
                    api_key=api_key,
                    model=model,
                    vision_model=os.getenv("VISION_MODEL", model),
                ))
            else:
                chain.append(factory(api_key=api_key, model=model))
        except Exception as exc:  # noqa: BLE001 -- one bad key must not disable the rest
            logger.error("%s unavailable, skipping it in the chain: %s", label, exc)

    if not chain:
        logger.warning(
            "No usable model API key found - using the offline stub. Output will "
            "be low-confidence placeholders, not real extraction."
        )
    # The stub always terminates the chain so there is always something to call.
    chain.append(StubProvider())
    return chain


class GroqProvider:
    """Groq, in JSON mode.

    Text only in this pipeline. Groq does host vision models, but keeping the
    fallback text-only means a scanned page fails over to a provider that
    genuinely cannot read it -- better to surface that than to silently return
    an empty transcription.
    """

    is_stub = False
    supports_vision = False

    def __init__(self, api_key: str, model: str):
        """Create one reusable client for this process."""
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self.name = model

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> dict[str, Any]:
        """Call the model in JSON mode and return the parsed object."""
        if images:
            raise LLMError("Groq fallback does not handle images in this pipeline")
        response = self._client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You return only JSON matching this schema, with no "
                        f"commentary: {json.dumps(schema)}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return extract_json(response.choices[0].message.content or "")


class MistralProvider:
    """Mistral, in JSON mode. Text only, for the same reason as Groq."""

    is_stub = False
    supports_vision = False

    def __init__(self, api_key: str, model: str):
        """Create one reusable client for this process."""
        # The SDK moved: in mistralai 2.x the top-level package is a namespace
        # with no __init__, and the client lives in mistralai.client. Both
        # spellings are tried so either version of the SDK works.
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral

        self._client = Mistral(api_key=api_key)
        self.name = model

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> dict[str, Any]:
        """Call the model in JSON mode and return the parsed object."""
        if images:
            raise LLMError("Mistral fallback does not handle images in this pipeline")
        response = self._client.chat.complete(
            model=self.name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You return only JSON matching this schema, with no "
                        f"commentary: {json.dumps(schema)}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return extract_json(response.choices[0].message.content or "")


class LLMClient:
    """The interface the rest of the service uses.

    Chooses a provider at construction: Gemini when a key is configured,
    otherwise the offline stub. Callers never branch on which is active; they
    read ``is_stub`` on the response if they need to surface it.
    """

    def __init__(self, provider: Any | None = None):
        """Build the provider chain, or wrap a single provider for tests."""
        self.providers = [provider] if provider else build_provider_chain()
        # The provider currently in use. Callers read this for the audit trail.
        self.provider = self.providers[0]

    @property
    def is_stub(self) -> bool:
        """Whether the active provider is the offline stub."""
        return bool(getattr(self.provider, "is_stub", False))

    def generate_json(
        self, prompt: str, schema: dict, images: list[bytes] | None = None
    ) -> LLMResponse:
        """Call the model, retrying transient failures and failing over.

        Two distinct recovery strategies, because the failures are different:

        * **Throttled** -- a per-minute window. Wait the delay the provider
          quotes and retry the *same* provider; the allowance is still there.
        * **Exhausted, or persistently failing** -- move to the next provider.
          Retrying a spent allowance cannot succeed however long it waits.

        Only when every provider has refused does this raise, and the error it
        raises carries a message written for the reviewer looking at the screen.
        """
        started = time.perf_counter()
        failures: dict[str, str] = {}

        for provider in self.providers:
            if images and not getattr(provider, "supports_vision", True):
                failures[provider.name] = "cannot read images"
                continue

            outcome = self._try_provider(provider, prompt, schema, images)
            if isinstance(outcome, dict):
                self.provider = provider
                return LLMResponse(
                    data=outcome,
                    model=provider.name,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    is_stub=bool(getattr(provider, "is_stub", False)),
                )
            failures[provider.name] = outcome
            logger.warning("Provider %s unusable (%s); trying the next one.",
                           provider.name, outcome)

        raise AllProvidersExhausted(failures)

    def _try_provider(
        self, provider: Any, prompt: str, schema: dict, images: list[bytes] | None
    ) -> dict[str, Any] | str:
        """Attempt one provider, returning its data or why it could not serve."""
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return provider.generate_json(prompt, schema, images=images)
            except Exception as exc:  # noqa: BLE001 -- classify, then retry or move on
                last_error = str(exc)[:200]
                try:
                    quota_delay = rate_limit_delay(last_error)
                except QuotaExhausted as spent:
                    return f"quota exhausted: {spent}"

                if attempt == MAX_ATTEMPTS:
                    break
                logger.warning(
                    "%s failed (attempt %d/%d)%s: %s",
                    provider.name, attempt, MAX_ATTEMPTS,
                    f" [throttled, waiting {quota_delay:.0f}s]" if quota_delay else "",
                    last_error,
                )
                time.sleep(quota_delay or BACKOFF_SECONDS * attempt)
        return f"failed after {MAX_ATTEMPTS} attempts: {last_error}"
