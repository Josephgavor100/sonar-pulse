"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends

from app.services.embedder import get_embedder
from app.services.matcher import AudioMatcher
from app.services.qdrant_service import QdrantService, get_qdrant_service


async def get_matcher(qdrant: QdrantService = Depends(get_qdrant_service)) -> AudioMatcher:
    """Build a matcher around the shared embedder singleton and Qdrant service."""
    return AudioMatcher(embedder=get_embedder(), qdrant=qdrant)