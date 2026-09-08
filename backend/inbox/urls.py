"""URL routing for the inbox API."""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"messages", views.MessageViewSet, basename="message")

urlpatterns = [
    path("", include(router.urls)),
    path("ingest/", views.ingest_now, name="ingest"),
    path("status/", views.system_status, name="status"),
    path("screen-article/", views.screen_article, name="screen-article"),
]
