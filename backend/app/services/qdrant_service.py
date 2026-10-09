"""Qdrant vector-store service for Sonar Pulse.

Design notes (roadmap-friendly):
* Vectors are *named* (default ``"ast"``) so other embedding spaces, such as
  CLAP text/audio vectors, can live beside AST vectors in the same points.
* Payloads are free-form JSON. Known fields: ``track_id``, ``title``,
  ``artist``, ``vibe_tags``, ``ai_generated_score``, ``timestamp`` (window
  start offset in seconds), ``window_index``. Extra keys (e.g. a future
  ``stem`` field) pass through untouched.
* Point IDs are deterministic UUIDs from (track_id, window_index), so
  re-ingesting a track overwrites instead of duplicating.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Depends
from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.db.qdrant import get_qdrant_client

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_NAME = "ast"
UPSERT_BATCH_SIZE = 256


class QdrantServiceError(Exception):
    """Raised when a Qdrant operation fails."""


class QdrantService:
    """Thin async wrapper around Qdrant for audio-window embeddings."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str | None = None,
        vector_size: int | None = None,
        vector_name: str = DEFAULT_VECTOR_NAME,
    ) -> None:
        settings = get_settings()
        self.client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_dim
        self.vector_name = vector_name

    # ------------------------------------------------------------------ setup
    async def ensure_collection_exists(self) -> None:
        """Create the collection (and payload indexes) if it doesn't exist."""
        try:
            if await self.client.collection_exists(self.collection_name):
                return
            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.vector_name: models.VectorParams(
                            size=self.vector_size, distance=models.Distance.COSINE
                        )
                    },
                )
            except Exception:
                # Another worker may have created it between check and create.
                if not await self.client.collection_exists(self.collection_name):
                    raise
                return

            indexes = {
                "track_id": models.PayloadSchemaType.KEYWORD,
                "vibe_tags": models.PayloadSchemaType.KEYWORD,
                "ai_generated_score": models.PayloadSchemaType.FLOAT,
                "window_index": models.PayloadSchemaType.INTEGER,
            }
            for field, schema in indexes.items():
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            logger.info("Created Qdrant collection %s", self.collection_name)
        except Exception as exc:
            raise QdrantServiceError(f"Could not ensure collection: {exc}") from exc

    # ----------------------------------------------------------------- upsert
    @staticmethod
    def make_point_id(track_id: str, window_index: int) -> str:
        """Deterministic UUID for a (track, window) pair."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{track_id}:{window_index}"))

    def _check_vector(self, vector: list[float]) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Vector has {len(vector)} dimensions, expected {self.vector_size}."
            )

    async def upsert_audio_embeddings(self, points: list[dict[str, Any]]) -> bool:
        """Upsert audio-window embeddings.

        Args:
            points: Dicts shaped ``{"vector": list[float], "payload": {...},
                "id": optional}``. The payload must include ``track_id`` and
                ``window_index`` unless an explicit ``id`` is given.

        Returns:
            True if Qdrant acknowledged every batch.

        Raises:
            ValueError: On malformed points (wrong dimension, missing keys).
            QdrantServiceError: If the Qdrant request fails.
        """
        if not points:
            return True

        structs: list[models.PointStruct] = []
        for p in points:
            if "vector" not in p:
                raise ValueError("Each point needs a 'vector'.")
            vector = list(p["vector"])
            self._check_vector(vector)
            payload = dict(p.get("payload") or {})

            point_id = p.get("id")
            if point_id is None:
                if "track_id" not in payload or "window_index" not in payload:
                    raise ValueError(
                        "Payload must include 'track_id' and 'window_index' when no 'id' is given."
                    )
                point_id = self.make_point_id(str(payload["track_id"]), int(payload["window_index"]))

            structs.append(
                models.PointStruct(
                    id=point_id, vector={self.vector_name: vector}, payload=payload
                )
            )

        try:
            for i in range(0, len(structs), UPSERT_BATCH_SIZE):
                result = await self.client.upsert(
                    collection_name=self.collection_name,
                    points=structs[i : i + UPSERT_BATCH_SIZE],
                    wait=True,
                )
                if result.status != models.UpdateStatus.COMPLETED:
                    return False
            return True
        except Exception as exc:
            raise QdrantServiceError(f"Upsert failed: {exc}") from exc

    # ----------------------------------------------------------------- search
    async def search_similar_audio(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        vibe_tags: list[str] | None = None,
        max_ai_generated_score: float | None = None,
        track_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search over stored windows.

        Args:
            query_vector: Embedding of the query audio window.
            limit: Maximum number of matches.
            score_threshold: Minimum cosine similarity to return.
            vibe_tags: Keep only points having at least one of these tags.
            max_ai_generated_score: Keep only points scored at or below this
                value (points without a score are excluded).
            track_id: Restrict the search to a single track.

        Returns:
            ``[{"id": str, "score": float, "payload": dict}, ...]``, best first.
        """
        self._check_vector(query_vector)

        conditions: list[models.Condition] = []
        if vibe_tags:
            conditions.append(
                models.FieldCondition(key="vibe_tags", match=models.MatchAny(any=vibe_tags))
            )
        if max_ai_generated_score is not None:
            conditions.append(
                models.FieldCondition(
                    key="ai_generated_score", range=models.Range(lte=max_ai_generated_score)
                )
            )
        if track_id is not None:
            conditions.append(
                models.FieldCondition(key="track_id", match=models.MatchValue(value=track_id))
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.vector_name,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantServiceError(f"Search failed: {exc}") from exc

        return [
            {"id": str(hit.id), "score": float(hit.score), "payload": dict(hit.payload or {})}
            for hit in response.points
        ]

    # ----------------------------------------------------------------- delete
    async def delete_points_by_track_id(self, track_id: str) -> bool:
        """Delete every window belonging to ``track_id``.

        Returns:
            True if Qdrant acknowledged the deletion (also when nothing matched).
        """
        try:
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="track_id", match=models.MatchValue(value=track_id)
                            )
                        ]
                    )
                ),
                wait=True,
            )
            return result.status == models.UpdateStatus.COMPLETED
        except Exception as exc:
            raise QdrantServiceError(f"Delete failed: {exc}") from exc


async def get_qdrant_service(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> QdrantService:
    """FastAPI dependency: ``service: QdrantService = Depends(get_qdrant_service)``."""
    return QdrantService(client)