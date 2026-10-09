"""Audio endpoints: ingest reference tracks and recognize audio clips."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_matcher
from app.core.config import get_settings
from app.models.schemas import IngestResponse, RecognizeResponse
from app.services.audio_processor import AudioProcessingError
from app.services.matcher import AudioMatcher, MatcherError
from app.services.qdrant_service import QdrantServiceError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audio"])


def parse_vibe_tags(raw: str | None) -> list[str] | None:
    """Parse a JSON list (``'["chill","lofi"]'``) or comma-separated string."""
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "vibe_tags is not valid JSON.") from exc
        if not isinstance(parsed, list):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "vibe_tags must be a list.")
        tags = [str(t).strip() for t in parsed]
    else:
        tags = [t.strip() for t in raw.split(",")]
    return [t for t in tags if t] or None


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload, enforcing non-empty content and the size limit."""
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {get_settings().max_upload_mb} MB limit.",
        )
    return data


def to_http_error(exc: Exception) -> HTTPException:
    """Map pipeline exceptions to HTTP errors (re-raises unknown ones)."""
    if isinstance(exc, AudioProcessingError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unreadable audio: {exc}")
    if isinstance(exc, ValueError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    if isinstance(exc, (QdrantServiceError, MatcherError)):
        logger.exception("Vector store failure")
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Vector database unavailable.")
    raise exc


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    file: UploadFile = File(..., description="Reference audio file"),
    track_id: str = Form(..., min_length=1, max_length=200),
    title: str = Form(..., min_length=1),
    artist: str = Form(..., min_length=1),
    vibe_tags: str | None = Form(None, description='JSON list or comma-separated tags'),
    ai_generated_score: float | None = Form(None, ge=0.0, le=1.0),
    matcher: AudioMatcher = Depends(get_matcher),
) -> IngestResponse:
    """Index a reference track so it can be recognized later."""
    data = await read_upload(file)
    metadata = {
        "track_id": track_id,
        "title": title,
        "artist": artist,
        "vibe_tags": parse_vibe_tags(vibe_tags) or [],
        "ai_generated_score": ai_generated_score,
    }
    try:
        result = await matcher.ingest_track(data, metadata)
    except Exception as exc:
        raise to_http_error(exc) from exc
    return IngestResponse(**result)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(..., description="Audio sample to identify"),
    max_ai_generated_score: float | None = Form(None, ge=0.0, le=1.0),
    vibe_tags: str | None = Form(None, description='JSON list or comma-separated tags'),
    limit: int = Form(5, ge=1, le=50),
    score_threshold: float = Form(0.5, ge=-1.0, le=1.0),
    matcher: AudioMatcher = Depends(get_matcher),
) -> RecognizeResponse:
    """Identify an audio sample against the indexed tracks."""
    data = await read_upload(file)
    try:
        result = await matcher.recognize_audio(
            data,
            limit=limit,
            score_threshold=score_threshold,
            max_ai_generated_score=max_ai_generated_score,
            vibe_tags=parse_vibe_tags(vibe_tags),
        )
    except Exception as exc:
        raise to_http_error(exc) from exc
    return RecognizeResponse(**result)