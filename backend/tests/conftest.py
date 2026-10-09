"""Shared fixtures: synthetic audio, a fast fake embedder, in-memory Qdrant."""
import io

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf
import torch
from qdrant_client import AsyncQdrantClient

from app.services.audio_processor import TARGET_SAMPLE_RATE as SR
from app.services.embedder import window_starts
from app.services.matcher import AudioMatcher
from app.services.qdrant_service import QdrantService

DIM = 768
COLLECTION = "test_matcher"

# Disjoint frequency ranges so two synthetic songs never share spectral bins.
SONG_A_FREQS = [220 + 97 * i for i in range(15)]  # 220 .. 1578 Hz
SONG_B_FREQS = [2000 + 83 * i for i in range(15)]  # 2000 .. 3162 Hz


def make_song(freqs: list[float], seg_seconds: float = 2.0) -> torch.Tensor:
    """A 'song' = consecutive sine segments, so each time position sounds different."""
    n = int(seg_seconds * SR)
    t = torch.arange(n) / SR
    return torch.cat([0.5 * torch.sin(2 * torch.pi * f * t) for f in freqs]).float()


def crop(wave: torch.Tensor, start_s: float, end_s: float) -> torch.Tensor:
    return wave[int(start_s * SR) : int(end_s * SR)]


def wav_bytes(wave: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wave.numpy(), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeEmbedder:
    """Fast stand-in for AudioEmbedder: a normalized, pooled magnitude spectrum.

    Different tones/time contents give different vectors and identical audio
    gives identical vectors, which is all the matching logic needs.
    """

    dim = DIM

    def __init__(self) -> None:
        self.window = 10 * SR

    def embed_windows(self, waveform, hop_seconds=None):
        hop = int((hop_seconds or 5.0) * SR)
        starts = window_starts(waveform.numel(), self.window, hop)
        rows = []
        for s in starts:
            chunk = waveform[s : s + self.window]
            spec = torch.fft.rfft(chunk * torch.hann_window(chunk.numel())).abs()
            rows.append(torch.nn.functional.adaptive_avg_pool1d(spec[None, None], DIM)[0, 0])
        return torch.nn.functional.normalize(torch.stack(rows), dim=1), starts


@pytest.fixture
def song_a() -> torch.Tensor:
    return make_song(SONG_A_FREQS)  # 30 s


@pytest.fixture
def song_b() -> torch.Tensor:
    return make_song(SONG_B_FREQS)  # 30 s


@pytest_asyncio.fixture
async def qdrant_client():
    c = AsyncQdrantClient(location=":memory:")
    yield c
    await c.close()


@pytest_asyncio.fixture
async def qdrant_service(qdrant_client):
    svc = QdrantService(qdrant_client, collection_name=COLLECTION)
    await svc.ensure_collection_exists()
    return svc


@pytest_asyncio.fixture
async def matcher(qdrant_service):
    return AudioMatcher(embedder=FakeEmbedder(), qdrant=qdrant_service)


def meta(track_id: str, **extra) -> dict:
    return {"track_id": track_id, "title": f"Title {track_id}", "artist": "Test Artist", **extra}