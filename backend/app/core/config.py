"""Application settings, read from environment variables.

Qdrant connection precedence (see app.db.qdrant.create_qdrant_client):
    QDRANT_LOCATION  >  QDRANT_URL  >  QDRANT_HOST / QDRANT_PORT

``QDRANT_LOCATION`` / ``QDRANT_URL`` may hold either a remote URL
(``http://localhost:6333``) or a *local* target: ``:memory:`` or a folder path
(``./qdrant_db``, ``../db``, ``/var/lib/qdrant``, ``C:\\qdrant``, ``path:./db``).
Local targets are always routed to ``qdrant_location`` and never to ``qdrant_url``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

MEMORY_LOCATION = ":memory:"

# A value starting with one of these is a filesystem path, not a URL.
LOCAL_PREFIXES = ("./", "../", "/", ".\\", "..\\", "~", "path:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")  # C:\data or C:/data


def is_local_location(value: str | None) -> bool:
    """True for ``:memory:`` or a filesystem path (incl. ``path:`` prefix, Windows paths)."""
    if not value:
        return False
    v = value.strip()
    return v == MEMORY_LOCATION or v.startswith(LOCAL_PREFIXES) or bool(_WINDOWS_DRIVE.match(v))


def normalize_local_location(value: str) -> str:
    """Strip whitespace and any ``path:`` prefix (``path:./db`` -> ``./db``)."""
    v = value.strip()
    if v.startswith("path:"):
        v = v[len("path:"):].strip()
    return v


@dataclass(frozen=True)
class Settings:
    # Qdrant connection
    qdrant_location: str | None = None  # ":memory:" or a local folder path
    qdrant_url: str | None = None  # remote only, e.g. https://xyz.cloud.qdrant.io:6333
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    # Collection
    qdrant_collection: str = "sonar_pulse_audio"
    embedding_dim: int = 768  # AST pooled output size

    # API
    max_upload_mb: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process (call ``get_settings.cache_clear()`` in tests)."""
    url = (os.getenv("QDRANT_URL") or "").strip() or None
    location = (os.getenv("QDRANT_LOCATION") or "").strip() or None

    # A local path / ":memory:" supplied via QDRANT_URL is really a location.
    if url and is_local_location(url):
        location = location or normalize_local_location(url)
        url = None
    if location and is_local_location(location):
        location = normalize_local_location(location)

    return Settings(
        qdrant_location=location,
        qdrant_url=url,
        qdrant_host=(os.getenv("QDRANT_HOST") or "").strip() or "localhost",
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "sonar_pulse_audio"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
    )