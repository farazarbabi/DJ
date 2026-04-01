"""Vibe classification into one of 8 categories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from ..audio import TrackAudio

logger = logging.getLogger(__name__)


@dataclass
class VibeResult:
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # margin between top two scores


def analyze_vibe(track_audio: TrackAudio) -> VibeResult:
    """Classify the track's vibe using spectral heuristics.

    Labels: HYPN, DRK, RAW, DEEP, TRIB, MEL, ACID, ATM
    """
    y, sr = track_audio.y, track_audio.sr
    y_h = track_audio.y_harmonic
    y_p = track_audio.y_percussive

    S = np.abs(librosa.stft(y))

    # --- Precompute features ---
    rms = float(np.mean(librosa.feature.rms(y=y)[0]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_var = float(np.var(centroid)) / (centroid_mean ** 2 + 1e-8)

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma_strength = float(np.mean(np.max(chroma, axis=0)))

    spectral_stability = 1.0 - min(1.0, centroid_var * 10)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_density = float(np.mean(onset_env))
    onset_var = float(np.var(onset_env)) / (float(np.mean(onset_env)) ** 2 + 1e-8)

    harmonic_energy = float(np.mean(y_h ** 2))
    percussive_energy = float(np.mean(y_p ** 2))
    perc_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-8)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs < 150
    S_power = S ** 2
    low_ratio = float(np.mean(S_power[low_mask, :])) / (float(np.mean(S_power)) + 1e-8)

    flux_raw = np.diff(S, axis=1)
    flux = float(np.mean(np.sqrt(np.mean(flux_raw ** 2, axis=0))))

    spec_avg = np.mean(S, axis=1)
    peakiness = float(np.max(spec_avg)) / (float(np.mean(spec_avg)) + 1e-8)

    # --- Score each vibe ---
    scores: dict[str, float] = {}

    # HYPN: repetitive, stable, trance-inducing — works at ANY energy level
    # Key: spectral stability + low onset variance + groove consistency
    # No RMS penalty — trance and hypnotic techno are loud
    scores["HYPN"] = (
        0.40 * spectral_stability
        + 0.30 * (1.0 - min(1.0, onset_var / 2.0))
        + 0.15 * (1.0 - min(1.0, chroma_strength * 1.2))  # less melodic = more hypnotic
        + 0.15 * min(1.0, onset_density / 4.0)             # driving rhythm
    )

    # DRK: dark, heavy, low-frequency — low brightness, bass-heavy
    scores["DRK"] = (
        0.30 * (1.0 - min(1.0, centroid_mean / 3000))      # dark = low centroid
        + 0.25 * min(1.0, low_ratio * 3)                    # heavy bass
        + 0.25 * min(1.0, flux / 3.0)                       # spectral aggression
        + 0.20 * min(1.0, rms * 8)                          # loud
    )

    # RAW: industrial, harsh, aggressive — high energy + noise + brightness
    scores["RAW"] = (
        0.25 * min(1.0, flatness * 10)                      # noisy
        + 0.25 * min(1.0, rms * 10)                         # loud
        + 0.20 * min(1.0, flux / 2.5)                       # harsh spectral change
        + 0.15 * min(1.0, centroid_mean / 3000)             # bright/harsh
        + 0.15 * min(1.0, onset_density / 4.0)              # dense transients
    )

    # DEEP: warm, subby, lower energy — low centroid, bass-heavy, NOT loud
    scores["DEEP"] = (
        0.30 * min(1.0, low_ratio * 3)                      # bass-heavy
        + 0.30 * (1.0 - min(1.0, centroid_mean / 2500))     # warm/dark
        + 0.20 * (1.0 - min(1.0, rms * 10))                 # not aggressive
        + 0.20 * (1.0 - min(1.0, onset_density / 4.0))      # sparse
    )

    # TRIB: percussive, polyrhythmic
    scores["TRIB"] = (
        0.40 * perc_ratio
        + 0.35 * min(1.0, onset_density / 5.0)
        + 0.25 * (1.0 - min(1.0, harmonic_energy * 50))
    )

    # MEL: melodic — requires genuine melodic MOVEMENT, not just tonal content
    # High bar: needs chroma variation over time AND harmonic dominance
    chroma_var = float(np.mean(np.var(chroma, axis=1)))
    melodic_movement = min(1.0, chroma_var * 30)  # stricter threshold
    scores["MEL"] = (
        0.35 * melodic_movement                              # actual melodic progression
        + 0.25 * chroma_strength                             # tonal content
        + 0.20 * (1.0 - perc_ratio)                          # harmonic-dominant
        + 0.20 * (1.0 - min(1.0, flatness * 10))            # not noisy
    )

    # ACID: filter sweeps, resonant peaks
    scores["ACID"] = (
        0.40 * min(1.0, centroid_var * 3)
        + 0.35 * min(1.0, peakiness / 10)
        + 0.25 * (1.0 - chroma_strength)
    )

    # ATM: atmospheric, ambient, spatial — quiet, wide, sparse
    scores["ATM"] = (
        0.30 * (1.0 - min(1.0, onset_density / 3.0))        # sparse
        + 0.30 * min(1.0, bandwidth / 3000)                  # wide spectrum
        + 0.25 * (1.0 - min(1.0, rms * 10))                 # quiet
        + 0.15 * (1.0 - min(1.0, flux / 2.0))               # smooth, not harsh
    )

    best_label = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Confidence: margin between top two scores
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0

    logger.debug("Vibe: %s (conf=%.2f) scores=%s", best_label, confidence, {k: f"{v:.3f}" for k, v in scores.items()})
    return VibeResult(label=best_label, scores=scores, confidence=confidence)
