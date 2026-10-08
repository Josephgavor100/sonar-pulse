"""Audio preprocessing for Sonar Pulse.

Loads an audio file and converts it into a model-ready tensor:
16 kHz, mono, float32, peak-normalized to [-1, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import torch
import torchaudio
import torchaudio.functional as F

TARGET_SAMPLE_RATE = 16_000
_EPS = 1e-9  # guards against division by zero on silent audio

AudioSource = Union[str, Path, BinaryIO]


class AudioProcessingError(Exception):
    """Raised when an audio file cannot be loaded or processed."""


def load_audio(
    source: AudioSource,
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an audio file as a 16 kHz mono float32 tensor.

    Args:
        source: File path or binary file-like object.
        target_sr: Output sample rate in Hz (default 16000).
        normalize: Scale so the peak absolute amplitude is 1.0.

    Returns:
        Tensor of shape ``(num_samples,)``, dtype ``torch.float32``.

    Raises:
        AudioProcessingError: If the file is unreadable or empty.
    """
    try:
        waveform, sr = torchaudio.load(source)  # (channels, samples), float32
    except Exception as exc:
        raise AudioProcessingError(f"Could not decode audio: {exc}") from exc

    if waveform.numel() == 0:
        raise AudioProcessingError("Audio file contains no samples.")

    # Downmix to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)

    waveform = waveform.squeeze(0).to(torch.float32)

    # Peak normalization
    if normalize:
        peak = waveform.abs().max()
        if peak > _EPS:
            waveform = waveform / peak

    return waveform.contiguous()