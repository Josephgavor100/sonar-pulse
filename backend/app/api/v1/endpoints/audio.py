"""Audio endpoints: ingest reference tracks and recognize audio clips."""
from __future__ import annotations

import json
import logging
import struct

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.api.deps import get_matcher
from app.core.config import get_settings
from app.models.schemas import IngestResponse, RecognizeResponse
from app.services.audio_processor import AudioProcessingError
from app.services.matcher import AudioMatcher, MatcherError
from app.services.qdrant_service import QdrantServiceError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audio"])


class TrackUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    artist: str = Field(min_length=1, max_length=500)
    vibe_tags: list[str] = Field(default_factory=list)
    ai_generated_score: float | None = Field(default=None, ge=0, le=1)


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


@router.get("/library", tags=["library"])
async def list_library(matcher: AudioMatcher = Depends(get_matcher)) -> list[dict]:
    """List indexed tracks and their preview URLs."""
    try:
        tracks = await matcher.qdrant.list_tracks()
    except Exception as exc:
        raise to_http_error(exc) from exc
    return [{**track, "preview_url": f"/api/v1/library/{track['track_id']}/preview"} for track in tracks]


@router.put("/library/{track_id}", tags=["library"])
async def update_library_track(track_id: str, update: TrackUpdate, matcher: AudioMatcher = Depends(get_matcher)) -> dict:
    try:
        track = await matcher.update_track_metadata(track_id, update.model_dump())
    except Exception as exc:
        raise to_http_error(exc) from exc
    if track is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Track not found.")
    return track


@router.delete("/library/{track_id}", tags=["library"])
async def delete_library_track(track_id: str, matcher: AudioMatcher = Depends(get_matcher)) -> dict:
    try:
        if await matcher.qdrant.get_track(track_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Track not found.")
        await matcher.qdrant.delete_points_by_track_id(track_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise to_http_error(exc) from exc
    return {"deleted": True, "track_id": track_id}


@router.websocket("/stream")
async def stream_audio(websocket: WebSocket, matcher: AudioMatcher = Depends(get_matcher)) -> None:
    """Receive WAV audio chunks and send incremental recognition results."""
    await websocket.accept()
    audio = bytearray()
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    try:
        options = await websocket.receive_json()
        if options.get("type") != "start":
            await websocket.send_json({"type": "error", "detail": "First message must be a start options object."})
            await websocket.close(code=1008)
            return

        limit = max(1, min(int(options.get("limit", 5)), 50))
        threshold = max(-1.0, min(float(options.get("score_threshold", 0.5)), 1.0))
        max_ai = options.get("max_ai_generated_score")
        if max_ai is not None:
            max_ai = max(0.0, min(float(max_ai), 1.0))
        tags = parse_vibe_tags(options.get("vibe_tags"))
        last_recognized_size = 0

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            chunk = message.get("bytes")
            if chunk is None:
                text = message.get("text")
                if text == "stop":
                    break
                await websocket.send_json({"type": "error", "detail": "Expected binary WAV audio chunks or 'stop'."})
                continue
            if not chunk:
                continue
            audio.extend(chunk)
            if len(audio) > max_bytes:
                await websocket.send_json({"type": "error", "detail": "Audio stream exceeded the upload size limit."})
                await websocket.close(code=1009)
                return

            # Chunks are raw 16 kHz mono signed 16-bit PCM; wrap for the decoder.
            if len(audio) < 5 * 16_000 * 2 or len(audio) - last_recognized_size < 5 * 16_000 * 2:
                continue
            pcm_size = len(audio)
            wav = b"RIFF" + struct.pack("<I", 36 + pcm_size) + b"WAVEfmt "
            wav += struct.pack("<IHHIIHH", 16, 1, 1, 16_000, 32_000, 2, 16)
            wav += b"data" + struct.pack("<I", pcm_size) + bytes(audio)
            try:
                result = await matcher.recognize_audio(
                    wav, limit=limit, score_threshold=threshold,
                    max_ai_generated_score=max_ai, vibe_tags=tags,
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": str(to_http_error(exc).detail)})
                continue
            last_recognized_size = len(audio)
            await websocket.send_json({"type": "result", **result})
    except WebSocketDisconnect:
        pass
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        try:
            await websocket.send_json({"type": "error", "detail": f"Invalid stream options: {exc}"})
            await websocket.close(code=1008)
        except RuntimeError:
            pass


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