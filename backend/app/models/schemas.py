"""Pydantic response models for the Sonar Pulse API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    track_id: str
    title: str
    artist: str
    windows_indexed: int
    duration_seconds: float


class MatchResult(BaseModel):
    track_id: str
    title: str | None = None
    artist: str | None = None
    vibe_tags: list[str] = Field(default_factory=list)
    ai_generated_score: float | None = None
    confidence: float = Field(description="Mean best-window score over all query windows (0-1).")
    best_window_score: float
    matched_windows: int
    query_windows: int
    offset_seconds: float = Field(description="Where the query clip starts inside the track.")


class RecognizeResponse(BaseModel):
    matches: list[MatchResult]
    query_windows: int
    query_duration_seconds: float