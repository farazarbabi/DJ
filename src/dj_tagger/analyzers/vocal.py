"""Vocal presence detection (V / NV) with confidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from ..audio import TrackAudio
from ..constants import (
    VOCAL_ENERGY_RATIO,
    VOCAL_FLATNESS_MAX,
    VOCAL_FRAME_THRESHOLD,
    VOCAL_FREQ_HIGH,
    VOCAL_FREQ_LOW,
)

logger = logging.getLogger(__name__)


@dataclass
class VocalResult:
    has_vocals: bool
    vocal_ratio: float
    confidence: float = 1.0  # 0=uncertain, 1=confident


def analyze_vocal(track_audio: TrackAudio) -> VocalResult:
    """Multi-stage vocal detection with confidence.

    Stage 1: Spectral energy ratio in vocal band (300-3000 Hz)
    Stage 2: Spectral flatness (tonal vs noise)
    Stage 3: Temporal consistency (sustained vocal-like frames)
    """
    y_h = track_audio.y_harmonic
    sr = track_audio.sr

    S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    vocal_mask = (freqs >= VOCAL_FREQ_LOW) & (freqs <= VOCAL_FREQ_HIGH)

    # Stage 1: Energy ratio per frame
    vocal_energy = np.mean(S[vocal_mask, :] ** 2, axis=0)
    total_energy = np.mean(S ** 2, axis=0)
    vocal_ratio_per_frame = vocal_energy / (total_energy + 1e-8)

    # Stage 2: Spectral flatness in vocal band per frame
    vocal_S = S[vocal_mask, :]
    log_mean = np.mean(np.log(vocal_S + 1e-8), axis=0)
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(vocal_S, axis=0)
    flatness_per_frame = geo_mean / (arith_mean + 1e-8)

    # Stage 3: Harmonic structure in vocal band
    # Use high-frequency content (>5kHz) as reference instead of sub-bass,
    # since electronic music has massive sub-bass that dwarfs the vocal band.
    hi_mask = freqs > 5000
    hi_energy = np.mean(S[hi_mask, :] ** 2, axis=0) if np.any(hi_mask) else np.ones_like(vocal_energy)
    vocal_vs_hi = vocal_energy / (hi_energy + 1e-8)
    harmonic_check = vocal_vs_hi > 1.5  # vocal band should dominate over high noise

    # Combined vocal-like detection
    vocal_frames = (
        (vocal_ratio_per_frame > VOCAL_ENERGY_RATIO)
        & (flatness_per_frame < VOCAL_FLATNESS_MAX)
        & harmonic_check
    )

    vocal_fraction = float(np.mean(vocal_frames))

    # Stage 4: Temporal consistency — vocal frames should cluster in time
    # Count runs of consecutive vocal frames
    if np.any(vocal_frames):
        diffs = np.diff(vocal_frames.astype(int))
        n_runs = max(1, np.sum(diffs == 1))
        avg_run_len = np.sum(vocal_frames) / n_runs
        # Long runs = real vocals; short bursts = likely transients/FX
        temporal_bonus = min(1.0, avg_run_len / 20.0)  # 20 frames ~= 0.5s
    else:
        temporal_bonus = 0.0

    # Adjusted vocal score combines fraction and temporal consistency
    vocal_score = vocal_fraction * (0.6 + 0.4 * temporal_bonus)

    has_vocals = vocal_score > VOCAL_FRAME_THRESHOLD

    # Confidence: how far from the decision boundary
    # High confidence when clearly above or clearly below threshold
    distance_from_threshold = abs(vocal_score - VOCAL_FRAME_THRESHOLD)
    confidence = min(1.0, distance_from_threshold / 0.15)

    logger.debug(
        "Vocal: %s (score=%.3f, fraction=%.3f, temporal=%.2f, conf=%.2f)",
        "V" if has_vocals else "NV", vocal_score, vocal_fraction, temporal_bonus, confidence,
    )
    return VocalResult(has_vocals=has_vocals, vocal_ratio=vocal_fraction, confidence=confidence)
