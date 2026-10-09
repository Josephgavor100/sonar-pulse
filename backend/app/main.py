"""Sonar Pulse API entrypoint.

Run:  uvicorn app.main:app --reload
Qdrant must be reachable (e.g. ``docker run -p 6333:6333 qdrant/qdrant``),
or set ``QDRANT_LOCATION=:memory:`` for a throwaway in-process database.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.db.qdrant import close_qdrant_client, get_qdrant_client
from app.services.embedder import get_embedder
from app.services.qdrant_service import QdrantService

logger = logging.getLogger("sonar_pulse")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Fail fast if Qdrant is unreachable; preload the model so the first request is fast."""
    client = await get_qdrant_client()
    await QdrantService(client).ensure_collection_exists()
    await asyncio.to_thread(get_embedder)
    logger.info("Sonar Pulse ready")
    yield
    await close_qdrant_client()


app = FastAPI(title="Sonar Pulse API", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}