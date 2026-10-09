"""Qdrant client creation and FastAPI dependency."""
from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def create_qdrant_client(settings: Settings | None = None) -> AsyncQdrantClient:
    """Build a new async client from settings (no network call is made yet)."""
    s = settings or get_settings()
    if s.qdrant_location:  # e.g. ":memory:"
        return AsyncQdrantClient(location=s.qdrant_location)
    if s.qdrant_url:
        return AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return AsyncQdrantClient(host=s.qdrant_host, port=s.qdrant_port, api_key=s.qdrant_api_key)


async def get_qdrant_client() -> AsyncQdrantClient:
    """FastAPI dependency returning the process-wide client (created lazily).

    A single shared instance matters for ``:memory:`` mode, where each client
    would otherwise get its own empty database.
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