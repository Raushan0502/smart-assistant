"""
AI service package: ingestion, OCR, classification and extraction.

Environment is loaded here, at package import, so every entry point picks it up
-- the FastAPI app, the test suite, and anything importing the pipeline
directly. Without this the service reads ``GEMINI_API_KEY`` from a process
environment that nobody sets, silently falls back to the offline stub, and
reports "no API key configured" while a perfectly good key sits in ``.env``.

The repository-root ``.env`` is shared by all three tiers, so the AI service
and the Django backend cannot drift apart on configuration.

``override=False`` means a variable already exported in the shell wins over the
file, which is what a deployment expects.
"""
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env", override=False)
