"""
REST API for the reviewer screen.

Endpoints are shaped around what the UI actually does: list a queue, open one
item, accept or override it, and trigger ingestion. Nothing here calls a model
directly -- work is handed to the queue, which calls the AI service, so no
request thread ever waits on OCR.
"""
from __future__ import annotations

import logging

from django.db.models import Prefetch
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from . import ai_client, mailbox
from .models import (
    Classification,
    ExtractedField,
    Extraction,
    Message,
    ReviewAction,
    ReviewStatus,
)
from .queue import get_queue
from .serializers import (
    MessageDetailSerializer,
    MessageListSerializer,
    ReviewActionSerializer,
)

logger = logging.getLogger(__name__)


class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    """The review queue and message detail."""

    def get_queryset(self):
        """Build the queryset, prefetching what each shape needs.

        The list view prefetches only classifications, which is all it renders.
        The detail view prefetches the whole graph, so opening one message is a
        handful of queries rather than one per field.
        """
        queryset = Message.objects.all()
        if self.action == "retrieve":
            return queryset.prefetch_related(
                "documents",
                "classifications",
                "review_actions",
                "audit_events",
                Prefetch(
                    "extractions",
                    queryset=Extraction.objects.prefetch_related("fields"),
                ),
            )
        queryset = queryset.prefetch_related("classifications")

        # Filters the queue screen uses.
        params = self.request.query_params
        if review_status := params.get("review_status"):
            queryset = queryset.filter(review_status=review_status)
        if processing_status := params.get("processing_status"):
            queryset = queryset.filter(processing_status=processing_status)
        if category := params.get("category"):
            # A subquery rather than a join + .distinct(): Message.body_text is
            # a TextField, which Oracle stores as NCLOB, and Oracle cannot
            # SELECT DISTINCT over an NCLOB column (ORA-22848). This also
            # avoids the duplicate rows the join would produce.
            queryset = queryset.filter(
                id__in=Classification.objects.filter(
                    category=category, applies=True
                ).values("message_id")
            )
        return queryset

    def get_serializer_class(self):
        if self.action == "retrieve":
            return MessageDetailSerializer
        return MessageListSerializer

    @action(detail=True, methods=["post"])
    def accept(self, request, pk=None):
        """Accept the AI's output as it stands."""
        message = self.get_object()
        message.review_status = ReviewStatus.ACCEPTED
        message.save(update_fields=["review_status"])
        ReviewAction.objects.create(
            message=message,
            action=ReviewAction.Action.ACCEPT,
            reviewer=request.data.get("reviewer", "demo-reviewer"),
            note=request.data.get("note", ""),
        )
        return Response(MessageDetailSerializer(message).data)

    @action(detail=True, methods=["post"])
    def override_field(self, request, pk=None):
        """Override one extracted field.

        The AI's value is recorded in the action before the field is updated,
        so the audit trail shows what was proposed as well as what was decided.
        """
        message = self.get_object()
        name = request.data.get("field_name")
        new_value = request.data.get("new_value", "")
        if not name:
            return Response(
                {"detail": "field_name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        field = ExtractedField.objects.filter(
            extraction__message=message, name=name
        ).first()
        if field is None:
            return Response(
                {"detail": f"No extracted field named {name!r} on this message"},
                status=status.HTTP_404_NOT_FOUND,
            )

        ReviewAction.objects.create(
            message=message,
            action=ReviewAction.Action.OVERRIDE_FIELD,
            reviewer=request.data.get("reviewer", "demo-reviewer"),
            field_name=name,
            original_value=field.value,
            new_value=new_value,
            note=request.data.get("note", ""),
        )
        field.value = new_value[:2000]
        # A human-supplied value is certain and needs no quote to support it.
        field.confidence = 1.0
        field.quote_verified = True
        field.save(update_fields=["value", "confidence", "quote_verified"])

        message.review_status = ReviewStatus.OVERRIDDEN
        message.save(update_fields=["review_status"])
        return Response(MessageDetailSerializer(message).data)

    @action(detail=True, methods=["post"])
    def override_category(self, request, pk=None):
        """Add or remove a category the AI got wrong."""
        message = self.get_object()
        category = request.data.get("category")
        applies = bool(request.data.get("applies", True))
        if not category:
            return Response(
                {"detail": "category is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        verdict, _ = Classification.objects.get_or_create(
            message=message, category=category
        )
        ReviewAction.objects.create(
            message=message,
            action=ReviewAction.Action.OVERRIDE_CATEGORY,
            reviewer=request.data.get("reviewer", "demo-reviewer"),
            category=category,
            original_value=str(verdict.applies),
            new_value=str(applies),
            note=request.data.get("note", ""),
        )
        verdict.applies = applies
        verdict.confidence = 1.0
        verdict.reason = f"Set by reviewer: {request.data.get('note', 'no note given')}"
        verdict.save(update_fields=["applies", "confidence", "reason"])

        message.review_status = ReviewStatus.OVERRIDDEN
        message.save(update_fields=["review_status"])
        return Response(MessageDetailSerializer(message).data)

    @action(detail=True, methods=["get"])
    def audit(self, request, pk=None):
        """The full audit trail for one message: AI calls and human actions."""
        message = self.get_object()
        return Response(
            {
                "message_id": message.message_id,
                "ai_calls": [
                    {
                        "event_type": e.event_type,
                        "model": e.model_name,
                        "is_stub": e.is_stub,
                        "prompt_sha256": e.prompt_sha256,
                        "latency_ms": e.latency_ms,
                        "succeeded": e.succeeded,
                        "error": e.error,
                        "at": e.created_at,
                    }
                    for e in message.audit_events.all()
                ],
                "review_actions": ReviewActionSerializer(
                    message.review_actions.all(), many=True
                ).data,
            }
        )


@api_view(["POST"])
def ingest_now(request):
    """Fetch new mail and queue it. Returns immediately."""
    try:
        queued = mailbox.ingest()
    except mailbox.MailboxError as exc:
        return Response(
            {"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return Response({"queued": queued, "queue": get_queue().stats().to_dict()})


@api_view(["GET"])
def system_status(request):
    """Health of the queue, the AI service and the corpus."""
    try:
        ai_health = ai_client.health()
    except ai_client.AIServiceError as exc:
        ai_health = {"status": "unreachable", "detail": str(exc)}

    return Response(
        {
            "queue": get_queue().stats().to_dict(),
            "ai_service": ai_health,
            "messages": {
                "total": Message.objects.count(),
                "pending_review": Message.objects.filter(
                    review_status=ReviewStatus.PENDING
                ).count(),
                "failed": Message.objects.filter(processing_status="FAILED").count(),
            },
        }
    )


@api_view(["POST"])
def screen_article(request):
    """Screen an uploaded article PDF for a reportable case (bonus path)."""
    uploaded = request.FILES.get("file")
    if uploaded is None:
        return Response(
            {"detail": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST
        )
    try:
        result = ai_client.screen_article(uploaded.read(), uploaded.name)
    except ai_client.AIServiceError as exc:
        return Response(
            {"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return Response(result)
