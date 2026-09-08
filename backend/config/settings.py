"""
Django settings for the Smart Inbox Assistant backend.

Configuration comes from the repository-root ``.env`` so all four tiers read
one file. Nothing secret has a usable default: the Oracle password and the mail
app-password must be supplied, and ``DJANGO_SECRET_KEY`` falls back only to an
obviously-marked development value.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# One .env at the repository root, shared by every tier.
load_dotenv(REPO_ROOT / ".env")

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY", "dev-only-insecure-key-change-for-any-real-deployment"
)
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "inbox",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
# Oracle, as the assignment specifies. SQLite is used only when
# USE_SQLITE=true, which exists so the test suite can run without a container
# -- never as a production path. The models use no Oracle-specific column
# types, so the two stay compatible.
if os.getenv("USE_SQLITE", "false").lower() == "true":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "test-db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.oracle",
            "NAME": (
                f"{os.getenv('ORACLE_HOST', 'localhost')}:"
                f"{os.getenv('ORACLE_PORT', '1521')}/"
                f"{os.getenv('ORACLE_SERVICE', 'FREEPDB1')}"
            ),
            "USER": os.getenv("ORACLE_USER", "smartinbox"),
            "PASSWORD": os.getenv("ORACLE_PASSWORD", ""),
            # python-oracledb runs in thin mode by default, so no Oracle
            # Instant Client install is needed. Django passes OPTIONS straight
            # to oracledb.connect(), so only real connect kwargs belong here.
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}

# The Angular dev server. Kept to an explicit allow-list rather than
# CORS_ALLOW_ALL, even in development.
CORS_ALLOWED_ORIGINS = os.getenv(
    "FRONTEND_URL", "http://localhost:4200"
).split(",")

# --------------------------------------------------------------------------
# Application settings
# --------------------------------------------------------------------------
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")

MAILBOX = {
    "host": os.getenv("IMAP_HOST", "imap.gmail.com"),
    "port": int(os.getenv("IMAP_PORT", "993")),
    "user": os.getenv("IMAP_USER", ""),
    "password": os.getenv("IMAP_PASSWORD", ""),
    "folder": os.getenv("IMAP_FOLDER", "INBOX"),
    # IMAP search expression. Defaults (in inbox.mailbox) to matching only
    # messages this project generated, so pointing at a mailbox containing real
    # personal mail cannot ingest it.
    "search": os.getenv("IMAP_SEARCH", ""),
    "poll_seconds": int(os.getenv("IMAP_POLL_SECONDS", "60")),
    # When true the poller is skipped and documents are read from data/samples,
    # so the whole system can be demonstrated with no mailbox at all.
    "offline": os.getenv("MAIL_OFFLINE_MODE", "false").lower() == "true",
}

SAMPLES_DIR = REPO_ROOT / "data" / "samples"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s %(levelname)-7s %(name)s | %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "standard"}
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
}
