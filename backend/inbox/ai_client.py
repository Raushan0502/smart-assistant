"""
HTTP client for the AI service.

The AI tier runs as a separate process, so this is a real network call rather
than a library import. Two consequences are handled here rather than left to
callers:

**Timeouts are generous but finite.** OCR plus classification plus extraction
on a multi-page scan legitimately takes tens of seconds, so a default 30-second
timeout would fail healthy work. But no timeout at all means one wedged request
occupies a worker forever, so the ceiling is explicit and configurable.

**Unavailability is a normal condition, not a crash.** If the AI service is
down the message is marked failed with the reason recorded, and stays in the
queue for a retry. A dead dependency must not lose mail.
"""
from __future__ import annotations

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Long enough for OCR plus several model calls on a multi-page document,
# including short throttling waits. Quota exhaustion now fails fast rather
# than backing off into this timeout, so this bounds real work only.
PROCESS_TIMEOUT_SECONDS = 600
HEALTH_TIMEOUT_SECONDS = 5


class AIServiceError(RuntimeError):
    """The AI service could not be reached, or returned an error."""


def base_url() -> str:
    return settings.AI_SERVICE_URL.rstrip("/")


def health() -> dict:
    """Ask the AI service for its status and which model it is using."""
    try:
        response = requests.get(f"{base_url()}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise AIServiceError(f"AI service unreachable at {base_url()}: {exc}") from exc


def process_message(raw: bytes, file_name: str = "message.eml") -> dict:
    """Send one raw email to the AI service and return its analysis."""
    try:
        response = requests.post(
            f"{base_url()}/process-message",
            files={"file": (file_name, raw, "message/rfc822")},
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AIServiceError(f"AI service call failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text[:500]
        raise AIServiceError(f"AI service returned {response.status_code}: {detail}")
    return response.json()


def screen_article(data: bytes, file_name: str = "article.pdf") -> dict:
    """Send one article PDF for literature screening."""
    try:
        response = requests.post(
            f"{base_url()}/screen-article",
            files={"file": (file_name, data, "application/pdf")},
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AIServiceError(f"AI service call failed: {exc}") from exc

    if response.status_code != 200:
        raise AIServiceError(
            f"AI service returned {response.status_code}: {response.text[:500]}"
        )
    return response.json()
