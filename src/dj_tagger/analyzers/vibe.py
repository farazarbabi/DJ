"""Vibe classification into one of 8 categories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from ..audio import TrackAudio
from ..settings import get_section

logger = logging.getLogger(__name__)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _scored(raw: float, params: list) -> float:
    """Compute weight * clamp01((raw - offset) / scale) from [weight, offset, scale]."""
    weight, offset, scale = params
    return weight * _clip01((raw - offset) / scale)


def _scored_inv(raw: float, params: list) -> float:
    """Inverted: weight * clamp01((offset - raw) / scale)."""
    weight, center, scale = params
    return weight * _clip01((center - raw) / scale)


@dataclass
class VibeResult:
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # margin between top two scores


def analyze_vibe(
    track_audio: TrackAudio,
    audio_features: dict[str, float] | None = None,
) -> VibeResult:
    """Classify the track's vibe using spectral heuristics + optional API features.

    Labels: HYPN, DRK, RAW, DEEP, TRIB, MEL, ACID, ATM

    audio_features: optional dict with keys like 'valence', 'instrumentalness',
    'liveness', 'energy', 'acousticness' (0-1 scale). When provided, these
    contribute additional scoring signals per vibe via [vibe.songstats] settings.
    """
    s = get_section("vibe")
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

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma_var = float(np.mean(np.var(chroma, axis=1)))

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

    # --- Score each vibe (all params from settings.toml) ---
    scores: dict[str, float] = {}

    # MEL
    mel_base = s.get("mel_base", 0.0)
    mel_chroma = min(1.0, chroma_var / s.get("mel_chroma_scale", 0.10))
    mel_lowflat = _clip01((s.get("mel_lowflat_center", 0.025) - flatness) / s.get("mel_lowflat_scale", 0.015))
    scores["MEL"] = mel_base + s.get("mel_chroma_weight", 0.55) * mel_chroma + s.get("mel_lowflat_weight", 0.20) * mel_lowflat

    # DRK
    scores["DRK"] = (
        _scored(onset_var, s.get("drk_onset_var", [0.25, 1.5, 2.0]))
        + _scored(flatness, s.get("drk_flatness", [0.30, 0.010, 0.030]))
        + _scored(low_ratio, s.get("drk_low_ratio", [0.25, 45, 25]))
        + _scored(flux, s.get("drk_flux", [0.20, 2.0, 2.5]))
    )

    # TRIB
    scores["TRIB"] = (
        _scored(perc_ratio, s.get("trib_perc_ratio", [0.70, 0.40, 0.15]))
        + _scored_inv(flatness, s.get("trib_clean", [0.30, 0.03, 0.02]))
    )

    # RAW
    scores["RAW"] = (
        _scored_inv(low_ratio, s.get("raw_nobass", [0.45, 42, 12]))
        + _scored_inv(onset_density, s.get("raw_sparse", [0.30, 1.5, 0.8]))
        + _scored_inv(onset_var, s.get("raw_lowvar", [0.25, 1.5, 1.0]))
    )

    # HYPN
    hypn_stab_w = s.get("hypn_stability_weight", 0.50)
    hypn_onset = s.get("hypn_onset_var", [0.50, 2.0, 1.5])
    scores["HYPN"] = hypn_stab_w * spectral_stability + _scored_inv(onset_var, hypn_onset)

    # DEEP
    scores["DEEP"] = (
        _scored_inv(centroid_mean, s.get("deep_centroid", [0.35, 2000, 1000]))
        + _scored_inv(rms, s.get("deep_rms", [0.35, 0.22, 0.10]))
        + _scored(low_ratio, s.get("deep_low", [0.30, 50, 20]))
    )

    # ACID
    scores["ACID"] = _scored(centroid_var, s.get("acid_centroid_var", [1.0, 0.75, 0.40]))

    # ATM
    scores["ATM"] = (
        _scored_inv(onset_density, s.get("atm_sparse", [0.50, 1.5, 1.0]))
        + _scored_inv(rms, s.get("atm_quiet", [0.50, 0.20, 0.08]))
    )

    # --- Add Songstats audio feature contributions (when available) ---
    if audio_features:
        ss = s.get("songstats", {})

        af_valence = audio_features.get("valence")
        af_energy = audio_features.get("energy")
        af_instrumentalness = audio_features.get("instrumentalness")
        af_liveness = audio_features.get("liveness")
        af_acousticness = audio_features.get("acousticness")

        if af_valence is not None:
            scores["DRK"] += _scored_inv(af_valence, ss.get("drk_valence", [0.15, 0.35, 0.25]))
        if af_energy is not None:
            scores["DRK"] += _scored(af_energy, ss.get("drk_energy", [0.10, 0.60, 0.30]))

        if af_instrumentalness is not None:
            scores["HYPN"] += _scored(af_instrumentalness, ss.get("hypn_instrumentalness", [0.15, 0.70, 0.25]))
        if af_valence is not None:
            scores["HYPN"] += _scored_inv(af_valence, ss.get("hypn_valence", [0.10, 0.40, 0.30]))

        if af_instrumentalness is not None:
            scores["TRIB"] += _scored(af_instrumentalness, ss.get("trib_instrumentalness", [0.10, 0.70, 0.25]))
        if af_energy is not None:
            scores["TRIB"] += _scored(af_energy, ss.get("trib_energy", [0.10, 0.60, 0.30]))

        if af_energy is not None:
            scores["DEEP"] += _scored_inv(af_energy, ss.get("deep_energy", [0.15, 0.40, 0.25]))
        if af_valence is not None:
            scores["DEEP"] += _scored_inv(af_valence, ss.get("deep_valence", [0.10, 0.35, 0.25]))

        if af_energy is not None:
            scores["ATM"] += _scored_inv(af_energy, ss.get("atm_energy", [0.15, 0.35, 0.25]))
        if af_acousticness is not None:
            scores["ATM"] += _scored(af_acousticness, ss.get("atm_acousticness", [0.10, 0.40, 0.30]))

        if af_liveness is not None:
            scores["RAW"] += _scored(af_liveness, ss.get("raw_liveness", [0.15, 0.40, 0.30]))

        if af_valence is not None:
            scores["MEL"] += _scored(af_valence, ss.get("mel_valence", [0.15, 0.50, 0.30]))

    # Clamp all scores to [0, 1]
    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    best_label = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Confidence: margin between top two scores
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0

    logger.debug("Vibe: %s (conf=%.2f) scores=%s", best_label, confidence, {k: f"{v:.3f}" for k, v in scores.items()})
    return VibeResult(label=best_label, scores=scores, confidence=confidence)
