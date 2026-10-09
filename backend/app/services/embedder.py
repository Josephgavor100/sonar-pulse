"""Audio embedding service for Sonar Pulse.

Turns a 16 kHz mono float32 waveform (from audio_processor.load_audio) into
L2-normalized embedding vectors, one per sliding window, ready for Qdrant
cosine search.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import torch
from transformers import AutoFeatureExtractor, AutoModel

from app.services.audio_processor import TARGET_SAMPLE_RATE

MODEL_NAME = os.getenv("EMBEDDER_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")
WINDOW_SECONDS = 10.0  # AST's native input is ~10.24 s
HOP_SECONDS = 5.0
BATCH_SIZE = 8


def window_starts(num_samples: int, window: int, hop: int) -> List[int]:
    """Start offsets of sliding windows covering the whole waveform.

    Clips shorter than one window yield a single window (the feature
    extractor pads it). A trailing remainder gets one extra window aligned
    to the end so the tail of the audio is never dropped.
    """
    if num_samples <= window:
        return [0]
    starts = list(range(0, num_samples - window + 1, hop))
    if starts[-1] + window < num_samples:
        starts.append(num_samples - window)
    return starts


class AudioEmbedder:
    """Loads a pretrained audio model once and embeds waveforms."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window = int(window_seconds * TARGET_SAMPLE_RATE)
        self.hop = int(hop_seconds * TARGET_SAMPLE_RATE)
        self.batch_size = batch_size

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim: int = self.model.config.hidden_size

    @torch.inference_mode()
    def embed_windows(
        self, waveform: torch.Tensor, hop_seconds: float | None = None
    ) -> tuple[torch.Tensor, list[int]]:
        """Embed a waveform into one vector per sliding window.

        Args:
            waveform: float32 tensor of shape ``(num_samples,)`` at 16 kHz.
            hop_seconds: Override the default hop for this call.

        Returns:
            ``(embeddings, starts)``: embeddings of shape ``(num_windows, dim)``
            (rows L2-normalized, on CPU) and each window's start in samples.
        """
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("Expected a non-empty 1-D waveform tensor.")

        hop = self.hop if hop_seconds is None else max(1, int(hop_seconds * TARGET_SAMPLE_RATE))
        starts = window_starts(waveform.numel(), self.window, hop)
        chunks = [waveform[s : s + self.window].cpu().numpy() for s in starts]

        outputs = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            inputs = self.feature_extractor(
                batch, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
            ).to(self.device)
            pooled = self.model(**inputs).pooler_output  # (batch, dim)
            outputs.append(pooled.cpu())

        embeddings = torch.cat(outputs, dim=0)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1), starts

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """Embeddings only, shape ``(num_windows, dim)``, using the default hop."""
        return self.embed_windows(waveform)[0]

    def embed_mean(self, waveform: torch.Tensor) -> torch.Tensor:
        """Single re-normalized vector of shape ``(dim,)`` for the whole clip."""
        mean = self.embed(waveform).mean(dim=0)
        return torch.nn.functional.normalize(mean, p=2, dim=0)


@lru_cache(maxsize=1)
def get_embedder() -> AudioEmbedder:
    """Process-wide singleton so the model loads once (e.g. at FastAPI startup)."""
    return AudioEmbedder()