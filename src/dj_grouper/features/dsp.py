"""Engineered DSP feature extraction for grouping."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from numpy.typing import NDArray

from dj_tagger.audio import TrackAudio

logger = logging.getLogger(__name__)


def extract_dsp_features(track_audio: TrackAudio) -> dict[str, float]:
    """Extract ~45-dimension DSP feature vector from precomputed audio."""
    y, sr = track_audio.y, track_audio.sr
    y_h = track_audio.y_harmonic
    y_p = track_audio.y_percussive

    features: dict[str, float] = {}

    # --- Rhythm / groove ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_density"] = float(np.mean(onset_env))
    features["onset_variance"] = float(np.var(onset_env))

    beat_strength = librosa.onset.onset_strength(y=y_p, sr=sr)
    features["beat_strength"] = float(np.mean(beat_strength))

    harmonic_e = float(np.mean(y_h ** 2))
    percussive_e = float(np.mean(y_p ** 2))
    total_e = harmonic_e + percussive_e + 1e-8
    features["perc_harmonic_ratio"] = percussive_e / total_e
    features["harmonic_energy"] = harmonic_e
    features["percussive_energy"] = percussive_e

    # --- Timbre / texture ---
    S = np.abs(librosa.stft(y))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_var"] = float(np.var(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["rolloff_mean"] = float(np.mean(rolloff))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["flatness_mean"] = float(np.mean(flatness))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["bandwidth_mean"] = float(np.mean(bandwidth))

    # MFCCs (13 coefficients, mean + variance = 26 dims)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_var"] = float(np.var(mfcc[i]))

    # --- Bass / energy shape ---
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_var"] = float(np.var(rms))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs < 150
    S_power = S ** 2
    features["low_freq_ratio"] = float(np.mean(S_power[low_mask, :])) / (float(np.mean(S_power)) + 1e-8)

    # Energy trend (slope of RMS over time)
    if len(rms) > 1:
        features["energy_trend"] = float(np.polyfit(np.arange(len(rms)), rms, 1)[0])
    else:
        features["energy_trend"] = 0.0

    # Spectral flux
    diff = np.diff(S, axis=1)
    flux = np.sqrt(np.mean(diff ** 2, axis=0))
    features["flux_mean"] = float(np.mean(flux))

    # --- Harmonic / melodic ---
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    features["chroma_strength"] = float(np.mean(np.max(chroma, axis=0)))
    features["chroma_var"] = float(np.mean(np.var(chroma, axis=1)))

    # Tonal stability: how consistent the strongest pitch class is over time
    dominant_chroma = np.argmax(chroma, axis=0)
    if len(dominant_chroma) > 0:
        from scipy import stats
        mode_result = stats.mode(dominant_chroma, keepdims=False)
        mode_count = mode_result.count if hasattr(mode_result, 'count') else np.sum(dominant_chroma == mode_result.mode)
        features["tonal_stability"] = float(mode_count) / len(dominant_chroma)
    else:
        features["tonal_stability"] = 0.0

    return features


DSP_FEATURE_NAMES: list[str] = [
    "onset_density", "onset_variance", "beat_strength",
    "perc_harmonic_ratio", "harmonic_energy", "percussive_energy",
    "centroid_mean", "centroid_var", "rolloff_mean",
    "flatness_mean", "bandwidth_mean",
] + [f"mfcc_{i}_mean" for i in range(13)] + [f"mfcc_{i}_var" for i in range(13)] + [
    "rms_mean", "rms_var", "low_freq_ratio", "energy_trend", "flux_mean",
    "chroma_strength", "chroma_var", "tonal_stability",
]

# Curated subset: 10 DJ-relevant features (no MFCCs, no noise dims)
DSP_CURATED_NAMES: list[str] = [
    "onset_density",        # groove density
    "beat_strength",        # kick presence
    "perc_harmonic_ratio",  # percussive vs melodic balance
    "centroid_mean",        # brightness / darkness
    "flatness_mean",        # noisiness / rawness
    "bandwidth_mean",       # spectral width
    "low_freq_ratio",       # bass weight
    "rms_mean",             # loudness / energy
    "chroma_strength",      # tonality
    "tonal_stability",      # harmonic consistency
]


def extract_section_dsp(
    track_audio: TrackAudio,
    section_map,
) -> dict[str, dict[str, float]]:
    """Extract curated DSP features per section (intro, groove).

    Returns dict keyed by section label with DSP feature dicts.
    """
    sr = track_audio.sr
    tempo = track_audio.tempo if track_audio.tempo > 0 else 120.0

    # Convert bars to samples
    beat_duration = 60.0 / tempo
    bar_duration = beat_duration * 4
    results: dict[str, dict[str, float]] = {}

    for section in (section_map.sections if section_map else []):
        if section.label not in ("intro", "groove", "peak"):
            continue

        start_sample = int(section.start_bar * bar_duration * sr)
        end_sample = int(section.end_bar * bar_duration * sr)
        end_sample = min(end_sample, len(track_audio.y))

        if end_sample - start_sample < sr:  # skip sections shorter than 1 second
            continue

        # Create a mini TrackAudio for this section
        y_sec = track_audio.y[start_sample:end_sample]
        y_h_sec = track_audio.y_harmonic[start_sample:end_sample]
        y_p_sec = track_audio.y_percussive[start_sample:end_sample]

        from dj_tagger.audio import TrackAudio as TA
        sec_audio = TA(
            path=track_audio.path,
            y=y_sec, sr=sr,
            y_harmonic=y_h_sec, y_percussive=y_p_sec,
            tempo=tempo,
            beat_frames=track_audio.beat_frames,  # not perfectly accurate but OK for features
            duration=len(y_sec) / sr,
        )

        feats = extract_dsp_features(sec_audio)
        # Keep only curated features
        results[section.label] = {k: feats.get(k, 0.0) for k in DSP_CURATED_NAMES}

    return results
