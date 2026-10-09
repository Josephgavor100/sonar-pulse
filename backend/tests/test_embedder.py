"""Tests for app.services.embedder.

The model tests download the AST weights (~350 MB) on first run.
    python -m pytest tests/test_embedder.py -v
"""
import pytest
import torch

from app.services.audio_processor import TARGET_SAMPLE_RATE as SR
from app.services.embedder import AudioEmbedder, window_starts


def test_window_starts_short_clip_single_window():
    assert window_starts(SR * 3, SR * 10, SR * 5) == [0]


def test_window_starts_covers_tail():
    starts = window_starts(SR * 22, SR * 10, SR * 5)
    assert starts == [0, SR * 5, SR * 10, SR * 12]  # last aligned to the end
    assert starts[-1] + SR * 10 == SR * 22


@pytest.fixture(scope="module")
def embedder():
    return AudioEmbedder()


def sine(seconds, freq=440.0):
    t = torch.arange(int(SR * seconds)) / SR
    return torch.sin(2 * torch.pi * freq * t).float()


def test_embed_shape_and_unit_norm(embedder):
    # 12 s -> 2 windows (0 s and tail-aligned 2 s)
    emb = embedder.embed(sine(12))
    assert emb.shape == (2, embedder.dim)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4)


def test_embed_is_deterministic(embedder):
    wave = sine(5)
    assert torch.allclose(embedder.embed(
        wave), embedder.embed(wave), atol=1e-5)


def test_different_sounds_give_different_embeddings(embedder):
    a = embedder.embed_mean(sine(5, 440.0))
    b = embedder.embed_mean(torch.randn(SR * 5).clamp(-1, 1))
    assert torch.dot(a, b).item() < 0.99


def test_rejects_bad_input(embedder):
    with pytest.raises(ValueError):
        embedder.embed(torch.zeros(2, 100))
