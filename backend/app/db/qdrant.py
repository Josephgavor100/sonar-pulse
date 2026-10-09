"""Qdrant client creation and FastAPI dependency."""
from __future__ import annotations

import logging
import socket
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.core.config import (
    MEMORY_LOCATION,
    Settings,
    get_settings,
    is_local_location,
    normalize_local_location,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_PATH = "./qdrant_db"
DOCKER_SERVICE_HOST = "qdrant"  # hostname that only exists inside the compose network

_client: AsyncQdrantClient | None = None


def _host_resolves(host: str) -> bool:
    """True if ``host`` can be resolved by DNS (one quick, blocking lookup)."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


def _local_client(location: str) -> AsyncQdrantClient:
    """Open an in-process Qdrant: in-memory, or persisted to a local folder.

    Note: ``AsyncQdrantClient(location=...)`` only understands ``":memory:"``; any
    other string is treated as a URL. Folder storage must be opened with ``path=``.
    A folder can be opened by one process at a time (e.g. the CLI *or* the API).
    """
    if location.strip() == MEMORY_LOCATION:
        return AsyncQdrantClient(location=MEMORY_LOCATION)
    path = Path(normalize_local_location(location)).expanduser()
    logger.info("Using local Qdrant storage at %s", path.resolve())
    return AsyncQdrantClient(path=str(path))


def create_qdrant_client(settings: Settings | None = None) -> AsyncQdrantClient:
    """Build a client from settings (no network call is made yet).

    Resolution order:
      1. ``qdrant_location``: ``:memory:`` or a folder path -> local mode.
      2. ``qdrant_url``: a path / ``:memory:`` is treated as local; otherwise remote.
      3. ``qdrant_host``/``qdrant_port`` -> remote. The one exception: the Docker-only
         hostname ``"qdrant"`` that does not resolve (no compose network) falls back
         to local storage at ``./qdrant_db`` instead of failing with getaddrinfo.
    """
    s = settings or get_settings()

    for candidate in (s.qdrant_location, s.qdrant_url):
        value = (candidate or "").strip()
        if not value:
            continue
        if is_local_location(value):
            return _local_client(value)
        return AsyncQdrantClient(url=value, api_key=s.qdrant_api_key)

    host = (s.qdrant_host or "").strip() or "localhost"
    if host == DOCKER_SERVICE_HOST and not _host_resolves(host):
        logger.warning(
            "Host '%s' does not resolve (Docker not running?); falling back to local storage at %s",
            host,
            DEFAULT_LOCAL_PATH,
        )
        return _local_client(DEFAULT_LOCAL_PATH)
    return AsyncQdrantClient(host=host, port=s.qdrant_port, api_key=s.qdrant_api_key)


async def get_qdrant_client() -> AsyncQdrantClient:
    """FastAPI dependency returning the process-wide client (created lazily).

    A single shared instance matters for ``:memory:`` mode, where each client
    would otherwise get its own empty database, and for folder storage, which
    only allows one open client per process.
    """
    global _client
    if _client is None:
        _client = create_qdrant_client()
        logger.info("Qdrant client created")
    return _client


async def close_qdrant_client() -> None:
    """Close the shared client (call from the FastAPI shutdown/lifespan hook)."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None