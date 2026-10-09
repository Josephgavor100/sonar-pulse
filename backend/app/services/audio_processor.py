"""Audio preprocessing for Sonar Pulse.

Loads an audio file and converts it into a model-ready tensor:
16 kHz, mono, float32, peak-normalized to [-1, 1].

Decoding uses soundfile (WAV/FLAC/OGG/MP3 via libsndfile), with librosa
(audioread/FFmpeg) as a fallback for formats like M4A/AAC. torchaudio is
only used for resampling, so TorchCodec is not required.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

TARGET_SAMPLE_RATE = 16_000
_EPS = 1e-9  # guards against division by zero on silent audio

AudioSource = Union[str, Path, BinaryIO]


class AudioProcessingError(Exception):
    """Raised when an audio file cannot be loaded or processed."""


def _decode(source: AudioSource) -> Tuple[np.ndarray, int]:
    """Decode audio to a float32 array of shape (channels, samples) + sample rate."""
    # Normalize the input to either a filesystem path or raw bytes.
    if isinstance(source, (str, Path)):
        path, raw = str(source), None
    else:
        path, raw = None, source.read()

    # 1) soundfile: fast, no FFmpeg needed
    try:
        src = path if path is not None else io.BytesIO(raw)
        data, sr = sf.read(src, dtype="float32", always_2d=True)  # (samples, channels)
        return data.T, sr
    except Exception as sf_err:
        first_error = sf_err

    # 2) librosa fallback (uses audioread/FFmpeg for M4A, AAC, etc.)
    tmp_path = None
    try:
        if path is None:
            # audioread needs a real file, so spill the bytes to disk.
            fd, tmp_path = tempfile.mkstemp()
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            path = tmp_path
        data, sr = librosa.load(path, sr=None, mono=False)  # keep native rate/channels
        return np.atleast_2d(data).astype(np.float32, copy=False), int(sr)
    except Exception as lib_err:
        raise AudioProcessingError(
            f"Could not decode audio (soundfile: {first_error}; librosa: {lib_err})"
        ) from lib_err
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    data, sr = _decode(source)

    if data.size == 0:
        raise AudioProcessingError("Audio file contains no samples.")

    waveform = torch.from_numpy(np.ascontiguousarray(data))  # (channels, samples)

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