"""Application settings, read from environment variables.

Qdrant connection precedence: QDRANT_LOCATION (e.g. ":memory:") >
QDRANT_URL > QDRANT_HOST/QDRANT_PORT.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # Qdrant connection
    qdrant_location: str | None = None  # ":memory:" for tests / local dev
    qdrant_url: str | None = None  # e.g. https://xyz.cloud.qdrant.io:6333
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
    return Settings(
        qdrant_location=os.getenv("QDRANT_LOCATION") or None,
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "sonar_pulse_audio"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
    )