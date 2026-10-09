"""Audio matching engine: ties together decoding, embedding and Qdrant.

Ingest:    audio -> 16 kHz mono tensor -> 10 s windows (2 s hop) -> embeddings
           -> Qdrant points (deterministic IDs, one per window).
Recognize: query audio -> windows -> one Qdrant search per window -> group
           hits by track -> offset-consistency scoring -> ranked tracks.

Offset consistency: if a query clip really comes from a track, then for every
query window ``i`` the matching track window sits at ``query_start_i + P``,
where ``P`` is where the clip begins inside the track. So the quantity
``delta = track_timestamp - query_start`` is (nearly) the same for all of a
track's true hits. Random similar-sounding hits scatter across many deltas.
We therefore pick, per track, the delta cluster that collects the most score
and ignore hits outside it.
"""
from __future__ import annotations

import asyncio
import io
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol, Sequence, Union

import torch

from app.services.audio_processor import TARGET_SAMPLE_RATE, load_audio
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

AudioInput = Union[str, Path, bytes, bytearray, BinaryIO]

INGEST_HOP_SECONDS = 2.0
QUERY_HOP_SECONDS = 5.0
PER_WINDOW_LIMIT = 50  # hits fetched per query window before aggregation

_RESERVED_PAYLOAD_KEYS = frozenset(
    {"track_id", "title", "artist", "vibe_tags", "ai_generated_score", "timestamp", "window_index"}
)


class MatcherError(Exception):
    """Raised when the matching pipeline cannot complete."""


class EmbedderLike(Protocol):
    """What the matcher needs from an embedder (AudioEmbedder satisfies this)."""

    def embed_windows(
        self, waveform: torch.Tensor, hop_seconds: float | None = None
    ) -> tuple[torch.Tensor, list[int]]: ...


# --------------------------------------------------------------------- helpers
def _load_waveform(audio: AudioInput) -> torch.Tensor:
    """Decode a path, bytes or file-like object to a 16 kHz mono tensor."""
    if isinstance(audio, (bytes, bytearray)):
        audio = io.BytesIO(bytes(audio))
    return load_audio(audio)


def _validate_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Check required fields and normalize optional ones."""
    clean: dict[str, Any] = {}
    for key in ("track_id", "title", "artist"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"track_metadata['{key}'] must be a non-empty string.")
        clean[key] = value.strip()

    tags = meta.get("vibe_tags") or []
    if not isinstance(tags, (list, tuple)) or not all(isinstance(t, str) for t in tags):
        raise ValueError("track_metadata['vibe_tags'] must be a list of strings.")
    clean["vibe_tags"] = list(tags)

    score = meta.get("ai_generated_score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise ValueError("track_metadata['ai_generated_score'] must be a number in [0, 1].")
        score = float(score)
    clean["ai_generated_score"] = score

    # Extra keys (e.g. a future "stem" field) pass through to the payload.
    for key, value in meta.items():
        if key not in _RESERVED_PAYLOAD_KEYS:
            clean[key] = value
    return clean


def score_track_matches(
    window_hits: Sequence[Sequence[Mapping[str, Any]]],
    query_starts: Sequence[float],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Group per-window hits by track and score them by offset consistency.

    Args:
        window_hits: For each query window, the hits returned by
            ``QdrantService.search_similar_audio`` (``{"score", "payload"}``).
        query_starts: Start time (seconds) of each query window in the query.
        tolerance: Max spread (seconds) of ``delta`` values counted as the
            same alignment. Should be about the ingest hop, since stored
            windows sit on that grid.

    Returns:
        Matches sorted by ``confidence`` (descending). ``confidence`` is the
        mean, over *all* query windows, of the best score from a window inside
        the winning alignment cluster (windows without a consistent hit
        contribute 0), so scattered hits score low.
    """
    n_windows = len(query_starts)
    if n_windows == 0:
        return []

    # track_id -> [(query_window, delta, score, payload)]
    by_track: dict[str, list[tuple[int, float, float, Mapping[str, Any]]]] = defaultdict(list)
    for qi, hits in enumerate(window_hits):
        for hit in hits:
            payload = hit["payload"]
            track_id, ts = payload.get("track_id"), payload.get("timestamp")
            if track_id is None or ts is None:
                continue
            by_track[str(track_id)].append((qi, float(ts) - query_starts[qi], float(hit["score"]), payload))

    results: list[dict[str, Any]] = []
    for track_id, hits in by_track.items():
        best_total, best_cluster = -1.0, {}
        for centre in sorted({round(h[1], 6) for h in hits}):
            cluster: dict[int, tuple[int, float, float, Mapping[str, Any]]] = {}
            for h in hits:
                if abs(h[1] - centre) <= tolerance:
                    current = cluster.get(h[0])
                    if current is None or h[2] > current[2]:
                        cluster[h[0]] = h  # best hit per query window
            total = sum(h[2] for h in cluster.values())
            if total > best_total:
                best_total, best_cluster = total, cluster

        chosen = list(best_cluster.values())
        top = max(chosen, key=lambda h: h[2])
        payload = top[3]
        results.append(
            {
                "track_id": track_id,
                "title": payload.get("title"),
                "artist": payload.get("artist"),
                "vibe_tags": list(payload.get("vibe_tags") or []),
                "ai_generated_score": payload.get("ai_generated_score"),
                "confidence": best_total / n_windows,
                "best_window_score": top[2],
                "matched_windows": len(chosen),
                "query_windows": n_windows,
                # Where the query clip starts inside the track.
                "offset_seconds": max(0.0, statistics.median(h[1] for h in chosen)),
            }
        )

    results.sort(key=lambda m: (m["confidence"], m["best_window_score"]), reverse=True)
    return results


# ---------------------------------------------------------------------- matcher
class AudioMatcher:
    """Unified ingest/recognize engine over an embedder and a Qdrant service."""

    def __init__(
        self,
        embedder: EmbedderLike,
        qdrant: QdrantService,
        ingest_hop_seconds: float = INGEST_HOP_SECONDS,
        query_hop_seconds: float = QUERY_HOP_SECONDS,
        per_window_limit: int = PER_WINDOW_LIMIT,
        offset_tolerance_seconds: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.ingest_hop = ingest_hop_seconds
        self.query_hop = query_hop_seconds
        self.per_window_limit = per_window_limit
        self.offset_tolerance = (
            ingest_hop_seconds if offset_tolerance_seconds is None else offset_tolerance_seconds
        )

    async def ingest_track(
        self, file_path_or_bytes: AudioInput, track_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Index a track: decode, window, embed and upsert into Qdrant.

        Re-ingesting an existing ``track_id`` replaces its previous windows.

        Args:
            file_path_or_bytes: Path, raw bytes, or binary file-like object.
            track_metadata: ``track_id``, ``title``, ``artist`` (required);
                ``vibe_tags`` and ``ai_generated_score`` (optional).

        Returns:
            ``{"track_id", "title", "artist", "windows_indexed", "duration_seconds"}``

        Raises:
            ValueError: Invalid metadata.
            AudioProcessingError: The audio cannot be decoded.
            QdrantServiceError / MatcherError: Storage failed.
        """
        meta = _validate_metadata(track_metadata)

        # Decoding and model inference are CPU-bound; keep the event loop free.
        waveform = await asyncio.to_thread(_load_waveform, file_path_or_bytes)
        embeddings, starts = await asyncio.to_thread(
            self.embedder.embed_windows, waveform, self.ingest_hop
        )

        points = []
        for window_index, (start, vector) in enumerate(zip(starts, embeddings)):
            payload = {
                **{k: v for k, v in meta.items()},
                "timestamp": start / TARGET_SAMPLE_RATE,
                "window_index": window_index,
            }
            points.append(
                {
                    "id": self.qdrant.make_point_id(meta["track_id"], window_index),
                    "vector": vector.tolist(),
                    "payload": payload,
                }
            )

        await self.qdrant.delete_points_by_track_id(meta["track_id"])  # drop stale windows
        if not await self.qdrant.upsert_audio_embeddings(points):
            raise MatcherError(f"Qdrant did not acknowledge ingest of '{meta['track_id']}'.")

        logger.info("Ingested %s: %d windows", meta["track_id"], len(points))
        return {
            "track_id": meta["track_id"],
            "title": meta["title"],
            "artist": meta["artist"],
            "windows_indexed": len(points),
            "duration_seconds": waveform.numel() / TARGET_SAMPLE_RATE,
        }

    async def recognize_audio(
        self,
        file_path_or_bytes: AudioInput,
        limit: int = 5,
        score_threshold: float = 0.5,
        max_ai_generated_score: float | None = None,
        vibe_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Identify an audio clip against the indexed tracks.

        Args:
            file_path_or_bytes: Path, raw bytes, or binary file-like object.
            limit: Maximum number of tracks to return.
            score_threshold: Minimum per-window cosine similarity for a hit.
            max_ai_generated_score: Exclude tracks scored above this value.
            vibe_tags: Keep only tracks with at least one of these tags.

        Returns:
            ``{"matches": [...], "query_windows": int, "query_duration_seconds": float}``
            where each match holds track metadata, ``confidence``,
            ``best_window_score``, ``matched_windows`` and ``offset_seconds``.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1.")

        waveform = await asyncio.to_thread(_load_waveform, file_path_or_bytes)
        embeddings, starts = await asyncio.to_thread(
            self.embedder.embed_windows, waveform, self.query_hop
        )
        query_starts = [s / TARGET_SAMPLE_RATE for s in starts]

        window_hits = await asyncio.gather(
            *(
                self.qdrant.search_similar_audio(
                    vector.tolist(),
                    limit=self.per_window_limit,
                    score_threshold=score_threshold,
                    vibe_tags=vibe_tags,
                    max_ai_generated_score=max_ai_generated_score,
                )
                for vector in embeddings
            )
        )

        matches = score_track_matches(window_hits, query_starts, self.offset_tolerance)
        return {
            "matches": matches[:limit],
            "query_windows": len(query_starts),
            "query_duration_seconds": waveform.numel() / TARGET_SAMPLE_RATE,
        }