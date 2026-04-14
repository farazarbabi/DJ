"""Recompute derived analysis values from cached raw features.

This module lets you change classification logic (vibe formulas, energy
thresholds, vocal decision boundary, structure flow rules) and recompute
results instantly from cached raw data — no audio loading needed.

Raw inputs:
  - dsp: dict with ~45 DSP features (from grouper extraction)
  - raw_analysis: dict with additional features not in DSP
    (bar_energies, vocal_ratio, vocal_temporal_bonus, onset_rate, segment_chroma)

Derived outputs:
  - energy: level (E1-E5), confidence
  - vibe: label, scores (8 floats), confidence
  - vocal: has_vocals (V/NV), confidence
  - structure: intro_bars, flow_type, formatted, confidence
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ── Vibe derivation ──────────────────────────────────────────────────────────
# Mirrors the formulas in analyzers/vibe.py.
# Change these formulas → bump DERIVED_VERSIONS["tagger"] → auto-recomputes.


def derive_vibe(dsp: dict) -> dict:
    """Recompute vibe from cached DSP features.

    Returns dict with: vibe, vibe_scores, vibe_confidence
    """
    rms = dsp.get("rms_mean", 0.0)
    centroid_mean = dsp.get("centroid_mean", 0.0)
    centroid_var = dsp.get("centroid_var", 0.0)
    flatness = dsp.get("flatness_mean", 0.0)
    onset_density = dsp.get("onset_density", 0.0)
    onset_variance = dsp.get("onset_variance", 0.0)
    perc_ratio = dsp.get("perc_harmonic_ratio", 0.0)
    low_ratio = dsp.get("low_freq_ratio", 0.0)
    flux = dsp.get("flux_mean", 0.0)
    chroma_var = dsp.get("chroma_var", 0.0)

    # Normalize onset_var and centroid_var to coefficient of variation
    onset_var = onset_variance / (onset_density ** 2 + 1e-8)
    centroid_cv = centroid_var / (centroid_mean ** 2 + 1e-8)
    spectral_stability = 1.0 - min(1.0, centroid_cv * 10)

    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    scores: dict[str, float] = {}

    # MEL: baseline
    mel_chroma = min(1.0, chroma_var / 0.065)
    mel_lowflat = _clip01((0.025 - flatness) / 0.015)
    scores["MEL"] = 0.38 + 0.25 * mel_chroma + 0.12 * mel_lowflat

    # DRK: aggressive + noisy + bass-heavy
    scores["DRK"] = (
        0.25 * _clip01((onset_var - 1.5) / 2.0)
        + 0.30 * _clip01((flatness - 0.010) / 0.030)
        + 0.25 * _clip01((low_ratio - 45) / 25)
        + 0.20 * _clip01((flux - 2.0) / 2.5)
    )

    # TRIB: very high percussion ratio, clean
    scores["TRIB"] = (
        0.70 * _clip01((perc_ratio - 0.40) / 0.15)
        + 0.30 * _clip01((0.03 - flatness) / 0.02)
    )

    # RAW: non-electronic (sparse, low bass)
    scores["RAW"] = (
        0.45 * _clip01((42 - low_ratio) / 12)
        + 0.30 * _clip01((1.5 - onset_density) / 0.8)
        + 0.25 * _clip01((1.5 - onset_var) / 1.0)
    )

    # HYPN: high stability, low onset variance
    scores["HYPN"] = (
        0.50 * spectral_stability
        + 0.50 * _clip01((2.0 - onset_var) / 1.5)
    )

    # DEEP: low centroid + quiet + bass-heavy
    scores["DEEP"] = (
        0.35 * _clip01((2000 - centroid_mean) / 1000)
        + 0.35 * _clip01((0.22 - rms) / 0.10)
        + 0.30 * _clip01((low_ratio - 50) / 20)
    )

    # ACID: extreme spectral movement
    scores["ACID"] = _clip01((centroid_cv - 0.75) / 0.40)

    # ATM: sparse + quiet
    scores["ATM"] = (
        0.50 * _clip01((1.5 - onset_density) / 1.0)
        + 0.50 * _clip01((0.20 - rms) / 0.08)
    )

    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0

    return {"vibe": label, "vibe_scores": scores, "vibe_confidence": confidence}


# ── Energy derivation ────────────────────────────────────────────────────────
# Mirrors _compute_track_energy in features/builder.py.

_ENERGY_FEATURES = {
    "beat_strength": (1.5, 2.5, 0.35),
    "onset_density": (1.0, 2.0, 0.30),
    "centroid_mean": (1000, 2000, 0.20),
    "rms_mean": (0.15, 0.20, 0.15),
}
_ENERGY_THRESHOLDS = [0.25, 0.42, 0.58, 0.75]


def derive_energy(dsp: dict) -> dict:
    """Recompute energy level from cached DSP features.

    Returns dict with: energy, energy_confidence
    """
    composite = 0.0
    for key, (lo, rng, weight) in _ENERGY_FEATURES.items():
        raw = dsp.get(key, 0.0)
        normalized = max(0.0, min(1.0, (raw - lo) / rng))
        composite += weight * normalized

    level = 1
    for threshold in _ENERGY_THRESHOLDS:
        if composite >= threshold:
            level += 1
    level = min(level, 5)

    # Confidence: distance from nearest threshold
    distances = [abs(composite - t) for t in _ENERGY_THRESHOLDS]
    confidence = min(1.0, min(distances) / 0.10) if distances else 1.0

    return {"energy": level, "energy_confidence": confidence}


# ── Vocal derivation ─────────────────────────────────────────────────────────
# Mirrors analyze_vocal decision logic.

VOCAL_FRAME_THRESHOLD = 0.20


def derive_vocal(raw_analysis: dict) -> dict:
    """Recompute vocal V/NV from cached raw analysis features.

    Returns dict with: vocal, vocal_confidence
    """
    vocal_ratio = raw_analysis.get("vocal_ratio", 0.0)
    temporal_bonus = raw_analysis.get("vocal_temporal_bonus", 0.0)
    vocal_score = vocal_ratio * (0.6 + 0.4 * temporal_bonus)

    has_vocals = vocal_score > VOCAL_FRAME_THRESHOLD
    distance_from_threshold = abs(vocal_score - VOCAL_FRAME_THRESHOLD)
    confidence = min(1.0, distance_from_threshold / 0.15)

    return {
        "vocal": "V" if has_vocals else "NV",
        "has_vocals": has_vocals,
        "vocal_ratio": vocal_ratio,
        "vocal_confidence": confidence,
    }


# ── Structure derivation ─────────────────────────────────────────────────────
# Mirrors analyze_structure logic.

from .constants import (
    FLOW_JUMP_DROP,
    FLOW_JUMP_HYPNOTIC,
    FLOW_PLATEAU_MIN,
    FLOW_TREND_BUILDER,
    FLOW_TREND_GROOVE,
    FLOW_VARIANCE_HYPNOTIC,
    FLOW_VARIANCE_LINEAR,
    INTRO_ENERGY_RATIO,
    INTRO_SUSTAIN_BARS,
    STANDARD_INTRO_BARS,
)

N_SEGMENTS = 8


def derive_structure(raw_analysis: dict) -> dict:
    """Recompute structure from cached bar energies.

    Returns dict with: structure, intro_bars, flow_type, structure_confidence
    """
    bar_energies = raw_analysis.get("bar_energies", [])
    if not bar_energies or len(bar_energies) == 0:
        return {
            "structure": "16H", "intro_bars": 16, "flow_type": "H",
            "structure_confidence": 0.2,
        }

    bar_e = np.array(bar_energies)

    # Detect intro
    intro_bars = STANDARD_INTRO_BARS[0]
    intro_conf = 0.3
    if len(bar_e) >= INTRO_SUSTAIN_BARS:
        median_energy = float(np.median(bar_e))
        threshold = INTRO_ENERGY_RATIO * median_energy
        intro_end = 0
        for i in range(len(bar_e) - INTRO_SUSTAIN_BARS + 1):
            window = bar_e[i: i + INTRO_SUSTAIN_BARS]
            if np.all(window > threshold):
                intro_end = i
                break
        intro_bars = min(STANDARD_INTRO_BARS, key=lambda x: abs(x - intro_end))
        snap_error = abs(intro_end - intro_bars)
        intro_conf = max(0.2, 1.0 - snap_error / 16.0)

    # Detect flow type
    flow_type = "H"
    if len(bar_e) >= N_SEGMENTS:
        seg_len = len(bar_e) // N_SEGMENTS
        seg_energies = np.array([
            float(np.mean(bar_e[i * seg_len: (i + 1) * seg_len]))
            for i in range(N_SEGMENTS)
        ])
        seg_max = float(np.max(seg_energies))
        seg_norm = seg_energies / seg_max if seg_max > 0 else seg_energies

        variance = float(np.var(seg_norm))
        energy_diff = np.diff(seg_norm)
        max_jump = float(np.max(np.abs(energy_diff))) if len(energy_diff) > 0 else 0.0
        trend = float(np.polyfit(np.arange(len(seg_norm)), seg_norm, 1)[0])

        # Count plateaus
        n_plateaus = 1
        in_plateau = True
        for i in range(1, len(seg_norm)):
            if abs(seg_norm[i] - seg_norm[i - 1]) <= 0.08:
                if not in_plateau:
                    n_plateaus += 1
                    in_plateau = True
            else:
                in_plateau = False

        if variance < FLOW_VARIANCE_LINEAR:
            flow_type = "L"
        elif variance < FLOW_VARIANCE_HYPNOTIC and max_jump < FLOW_JUMP_HYPNOTIC:
            flow_type = "H"
        elif max_jump > FLOW_JUMP_DROP:
            flow_type = "D"
        elif trend > FLOW_TREND_BUILDER and n_plateaus >= FLOW_PLATEAU_MIN:
            flow_type = "B"
        elif trend > FLOW_TREND_GROOVE:
            flow_type = "G"

    formatted = f"{intro_bars}{flow_type}"
    return {
        "structure": formatted, "intro_bars": intro_bars,
        "flow_type": flow_type, "structure_confidence": intro_conf,
    }


# ── Derive all ───────────────────────────────────────────────────────────────

def derive_all(dsp: dict, raw_analysis: dict) -> dict:
    """Recompute all derived analysis values from cached raw data.

    Returns a tagger-compatible result dict.
    """
    energy = derive_energy(dsp)
    vibe = derive_vibe(dsp)
    vocal = derive_vocal(raw_analysis)
    structure = derive_structure(raw_analysis)

    return {
        "energy": energy["energy"],
        "vibe": vibe["vibe"],
        "vocal": vocal["vocal"],
        "has_vocals": vocal["has_vocals"],
        "vocal_ratio": vocal["vocal_ratio"],
        "vibe_scores": vibe["vibe_scores"],
        "structure": structure["structure"],
        "intro_bars": structure["intro_bars"],
        "flow_type": structure["flow_type"],
        "confidences": {
            "energy": energy["energy_confidence"],
            "vibe": vibe["vibe_confidence"],
            "vocal": vocal["vocal_confidence"],
            "structure": structure["structure_confidence"],
        },
    }
