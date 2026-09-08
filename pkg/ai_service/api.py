"""
FastAPI service exposing the AI pipeline over REST.

    uvicorn ai_service.api:app --port 8000

This is the boundary the assignment's architecture describes: the AI work runs
as its own process behind an HTTP API, not as library calls inside the web
application. Keeping it a real network boundary means the model layer can be
scaled, restarted or swapped without touching the backend -- and, practically,
that a hung OCR call cannot take a request thread of the web app with it.

The service is intentionally stateless. It owns no database and stores nothing
between requests; persistence and the audit trail are the backend's job. That
keeps this process horizontally scalable and keeps one system of record.
"""
from __future__ import annotations

import logging

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .llm import AllProvidersExhausted, LLMClient
from .pdf import extract_pdf_bytes
from .pipeline import process_message, screen_literature
from .preprocess import preprocess_document

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Inbox Assistant - AI service",
    description=(
        "Document understanding for pharmacovigilance intake: ingestion, OCR, "
        "classification, field extraction and literature screening."
    ),
    version="1.0.0",
)

# One client for the process. Constructing a provider per request would rebuild
# the SDK transport every time, which is both slow and a known source of
# closed-client errors under load.
_client: LLMClient | None = None

# Uploads are bounded because both endpoints read the whole file into memory
# before parsing. A pharmacovigilance attachment is a form or a paper, not a
# dataset; anything past this is a mistake or an attack, and either way should
# be refused rather than allowed to exhaust the process.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def read_upload(file: UploadFile, description: str) -> bytes:
    """Read an upload, rejecting empty or oversized files."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty {description}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{description} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )
    return data


def get_client() -> LLMClient:
    """Return the shared model client, creating it on first use."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


class HealthResponse(BaseModel):
    """Service and model status."""

    status: str
    model: str
    is_stub: bool
    note: str
    providers: list[str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether a real model is configured.

    ``is_stub`` is surfaced deliberately: the backend shows it in the UI so a
    reviewer is never shown placeholder output believing it came from a model.
    """
    client = get_client()
    return HealthResponse(
        status="ok",
        model=client.provider.name,
        providers=[p.name for p in client.providers],
        is_stub=client.is_stub,
        note=(
            "No API key configured - running the offline stub. Output is "
            "low-confidence placeholder text, not real extraction."
            if client.is_stub
            else "Model configured."
        ),
    )


@app.post("/process-message")
def process_message_endpoint(file: UploadFile = File(...)) -> dict:
    """Run the full pipeline over one uploaded ``.eml`` message."""
    raw = read_upload(file, "message file")
    try:
        return process_message(raw, client=get_client()).to_dict()
    except AllProvidersExhausted as exc:
        # 429, not 500: nothing is broken, the allowance is spent. The message
        # is written for the reviewer who will see it on screen.
        logger.error("All providers exhausted for %s", file.filename)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- surface the reason, not a 500 page
        logger.exception("Pipeline failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.post("/screen-article")
def screen_article_endpoint(file: UploadFile = File(...)) -> dict:
    """Screen one uploaded article PDF for a reportable patient case.

    This is the bonus literature-screening path: articles arrive by direct
    upload rather than through the mailbox.
    """
    data = read_upload(file, "PDF")
    try:
        document = extract_pdf_bytes(
            data, file_name=file.filename or "article.pdf", document_id="upload"
        )
        preprocess_document(document)
        summary, event = screen_literature(document, get_client())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Screening failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Screening failed: {exc}") from exc

    return {
        "document": document.to_dict(),
        "screening": summary.to_dict(),
        "audit_event": event.to_dict(),
    }


@app.post("/extract-pdf")
def extract_pdf_endpoint(file: UploadFile = File(...)) -> dict:
    """Ingest a PDF and return its structure without calling any model.

    Useful for debugging flavour detection and table extraction in isolation --
    it is fully deterministic and costs nothing.
    """
    data = read_upload(file, "PDF")
    document = extract_pdf_bytes(
        data, file_name=file.filename or "document.pdf", document_id="debug"
    )
    preprocess_document(document)
    return document.to_dict()
