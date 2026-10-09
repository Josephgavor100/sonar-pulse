"""Tests for app.services.audio_processor.

Run from the backend/ directory:
    pip install pytest
    python -m pytest tests/test_audio_processor.py -v
"""
import io

import numpy as np
import pytest
import soundfile as sf  # installed as a librosa dependency
import torch

from app.services.audio_processor import (
    TARGET_SAMPLE_RATE,
    AudioProcessingError,
    load_audio,
)


def make_sine(path, sr=44_100, seconds=1.0, channels=1, freq=440.0, amp=0.5):
    """Write a sine-wave WAV file and return its path."""
    t = np.arange(int(sr * seconds)) / sr
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
    sf.write(str(path), data, sr)
    return path


def test_output_format_and_resample(tmp_path):
    f = make_sine(tmp_path / "tone.wav", sr=44_100, seconds=1.0)
    out = load_audio(f)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.ndim == 1  # mono, (num_samples,)
    # 1 second at 44.1 kHz -> ~16000 samples at 16 kHz
    assert abs(out.numel() - TARGET_SAMPLE_RATE) <= 2


def test_stereo_is_downmixed_to_mono(tmp_path):
    f = make_sine(tmp_path / "stereo.wav", channels=2)
    out = load_audio(f)
    assert out.ndim == 1


def test_peak_is_normalized_to_one(tmp_path):
    f = make_sine(tmp_path / "quiet.wav", amp=0.1)
    out = load_audio(f)
    assert out.abs().max().item() == pytest.approx(1.0, abs=1e-5)


def test_normalize_false_keeps_original_level(tmp_path):
    f = make_sine(tmp_path / "quiet.wav", amp=0.1)
    out = load_audio(f, normalize=False)
    assert out.abs().max().item() == pytest.approx(0.1, abs=0.01)


def test_sine_frequency_survives_resampling(tmp_path):
    """The dominant FFT bin of a 440 Hz tone should still be ~440 Hz at 16 kHz."""
    f = make_sine(tmp_path / "a440.wav", sr=48_000, seconds=1.0, freq=440.0)
    out = load_audio(f)
    spectrum = torch.fft.rfft(out).abs()
    freqs = torch.fft.rfftfreq(out.numel(), d=1 / TARGET_SAMPLE_RATE)
    assert freqs[spectrum.argmax()].item() == pytest.approx(440.0, abs=2.0)


def test_silent_audio_has_no_nans(tmp_path):
    f = make_sine(tmp_path / "silence.wav", amp=0.0)
    out = load_audio(f)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() == 0.0


def test_accepts_file_like_object(tmp_path):
    f = make_sine(tmp_path / "tone.wav")
    with open(f, "rb") as fh:
        out = load_audio(io.BytesIO(fh.read()))
    assert out.dtype == torch.float32 and out.numel() > 0


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioProcessingError):
        load_audio(bad)
