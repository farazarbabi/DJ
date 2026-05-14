"""Recompute derived analysis values from cached raw features.

All parameters are read from settings.toml. Change any value there and derived
tagger entries refresh from cached raw data on the next run.

Raw inputs:
  - dsp: dict with ~45 DSP features (from grouper extraction)
  - raw_analysis: dict with additional features not in DSP
    (bar_energies, vocal_ratio, vocal_temporal_bonus, onset_rate)

Derived outputs:
  - energy: level (E1-E5), confidence
  - vibe: taxonomy mood code, mood-score map, confidence
  - vocal: taxonomy vocal profile, has_vocals, profile scores, confidence
  - structure: intro_bars, flow_type, formatted, confidence
"""

from __future__ import annotations

import logging

import numpy as np

from .settings import get, get_section
from .vibe_scoring import vibe_metrics_from_dsp, score_vibe_metrics
from .vocals import score_vocal_profile

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


# ── Energy derivation ────────────────────────────────────────────────────────

def derive_energy(dsp: dict) -> dict:
    """Recompute energy level from cached DSP features."""
    s = get_section("energy")

    features = {
        "beat_strength": s.get("beat_strength", [1.5, 2.5, 0.35]),
        "onset_density": s.get("onset_density", [1.0, 2.0, 0.30]),
        "centroid_mean": s.get("centroid_mean", [1000, 2000, 0.20]),
        "rms_mean": s.get("rms_mean", [0.15, 0.20, 0.15]),
    }
    thresholds = s.get("thresholds", [0.25, 0.42, 0.58, 0.75])

    composite = 0.0
    for key, (lo, rng, weight) in features.items():
        raw = dsp.get(key, 0.0)
        normalized = _clip01((raw - lo) / rng)
        composite += weight * normalized

    level = 1
    for t in thresholds:
        if composite >= t:
            level += 1
    level = min(level, 5)

    distances = [abs(composite - t) for t in thresholds]
    confidence = min(1.0, min(distances) / 0.10) if distances else 1.0

    return {"energy": level, "energy_confidence": confidence}


# ── Vibe derivation ──────────────────────────────────────────────────────────

def derive_vibe(dsp: dict, audio_features: dict[str, float] | None = None) -> dict:
    """Recompute mood/vibe from cached DSP features + optional API audio features.

    audio_features: optional dict with keys like 'valence', 'instrumentalness',
    'liveness', 'energy', 'acousticness' (0-1 scale). When provided, these
    contribute additional scoring signals per mood via [vibe.songstats] settings.
    """
    metrics = vibe_metrics_from_dsp(dsp)
    label, scores, confidence = score_vibe_metrics(metrics, audio_features=audio_features)

    return {
        "vibe": label,
        "mood": label,
        "vibe_scores": scores,
        "mood_scores": scores,
        "vibe_confidence": confidence,
        "mood_confidence": confidence,
    }


# ── Vocal derivation ─────────────────────────────────────────────────────────

def derive_vocal(
    raw_analysis: dict,
    dsp: dict | None = None,
    audio_features: dict[str, float] | None = None,
    mood_scores: dict[str, float] | None = None,
) -> dict:
    """Recompute taxonomy vocal profile from cached raw analysis features."""
    s = get_section("vocal")

    vocal_ratio = raw_analysis.get("vocal_ratio", 0.0)
    temporal_bonus = raw_analysis.get("vocal_temporal_bonus", 0.0)
    base_w = s.get("score_base_weight", 0.6)
    temp_w = s.get("score_temporal_weight", 0.4)
    threshold = s.get("frame_threshold", 0.20)
    conf_scale = s.get("confidence_scale", 0.15)

    profile, profile_scores, has_vocals, profile_confidence = score_vocal_profile(
        vocal_ratio=vocal_ratio,
        temporal_bonus=temporal_bonus,
        threshold=threshold,
        base_weight=base_w,
        temporal_weight=temp_w,
        dsp=dsp,
        audio_features=audio_features,
        mood_scores=mood_scores,
    )
    vocal_score = vocal_ratio * (base_w + temp_w * temporal_bonus)
    detector_confidence = min(1.0, abs(vocal_score - threshold) / conf_scale)
    confidence = max(detector_confidence, profile_confidence)

    return {
        "vocal": profile,
        "vocal_profile": profile,
        "vocal_scores": profile_scores,
        "has_vocals": has_vocals,
        "vocal_ratio": vocal_ratio,
        "vocal_confidence": confidence,
    }


# ── Structure derivation ─────────────────────────────────────────────────────

def derive_structure(raw_analysis: dict) -> dict:
    """Recompute structure from cached bar energies."""
    s = get_section("structure")

    bar_energies = raw_analysis.get("bar_energies", [])
    if not bar_energies:
        return {"structure": "16H", "intro_bars": 16, "flow_type": "H", "structure_confidence": 0.2}

    bar_e = np.array(bar_energies)
    std_bars = s.get("standard_intro_bars", [16, 32, 64])
    sustain = s.get("intro_sustain_bars", 8)
    energy_ratio = s.get("intro_energy_ratio", 0.80)
    n_seg = s.get("n_segments", 8)

    # Detect intro
    intro_bars = std_bars[0]
    intro_conf = 0.3
    if len(bar_e) >= sustain:
        median_energy = float(np.median(bar_e))
        threshold = energy_ratio * median_energy
        intro_end = 0
        for i in range(len(bar_e) - sustain + 1):
            if np.all(bar_e[i: i + sustain] > threshold):
                intro_end = i
                break
        intro_bars = min(std_bars, key=lambda x: abs(x - intro_end))
        intro_conf = max(0.2, 1.0 - abs(intro_end - intro_bars) / 16.0)

    # Detect flow type
    flow_type = "H"
    if len(bar_e) >= n_seg:
        seg_len = len(bar_e) // n_seg
        seg_e = np.array([float(np.mean(bar_e[i * seg_len: (i + 1) * seg_len])) for i in range(n_seg)])
        seg_max = float(np.max(seg_e))
        seg_norm = seg_e / seg_max if seg_max > 0 else seg_e

        variance = float(np.var(seg_norm))
        diffs = np.diff(seg_norm)
        max_jump = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0
        trend = float(np.polyfit(np.arange(len(seg_norm)), seg_norm, 1)[0])

        tol = s.get("plateau_tolerance", 0.08)
        n_plateaus = 1
        in_plateau = True
        for i in range(1, len(seg_norm)):
            if abs(seg_norm[i] - seg_norm[i - 1]) <= tol:
                if not in_plateau:
                    n_plateaus += 1
                    in_plateau = True
            else:
                in_plateau = False

        if variance < s.get("flow_variance_linear", 0.005):
            flow_type = "L"
        elif variance < s.get("flow_variance_hypnotic", 0.02) and max_jump < s.get("flow_jump_hypnotic", 0.15):
            flow_type = "H"
        elif max_jump > s.get("flow_jump_drop", 0.40):
            flow_type = "D"
        elif trend > s.get("flow_trend_builder", 0.05) and n_plateaus >= s.get("flow_plateau_min", 2):
            flow_type = "B"
        elif trend > s.get("flow_trend_groove", 0.03):
            flow_type = "G"

    return {
        "structure": f"{intro_bars}{flow_type}",
        "intro_bars": intro_bars,
        "flow_type": flow_type,
        "structure_confidence": intro_conf,
    }


# ── Derive all ───────────────────────────────────────────────────────────────

def derive_all(dsp: dict, raw_analysis: dict, audio_features: dict[str, float] | None = None) -> dict:
    """Recompute all derived values from cached raw data."""
    energy = derive_energy(dsp)
    vibe = derive_vibe(dsp, audio_features=audio_features)
    vocal = derive_vocal(
        raw_analysis,
        dsp=dsp,
        audio_features=audio_features,
        mood_scores=vibe["mood_scores"],
    )
    structure = derive_structure(raw_analysis)

    return {
        "energy": energy["energy"],
        "bpm": round(float(raw_analysis.get("tempo", 0.0)), 1) if raw_analysis.get("tempo") is not None else None,
        "vibe": vibe["vibe"],
        "mood": vibe["mood"],
        "vocal": vocal["vocal"],
        "vocal_profile": vocal["vocal_profile"],
        "has_vocals": vocal["has_vocals"],
        "vocal_ratio": vocal["vocal_ratio"],
        "vocal_scores": vocal["vocal_scores"],
        "vibe_scores": vibe["vibe_scores"],
        "mood_scores": vibe["mood_scores"],
        "structure": structure["structure"],
        "intro_bars": structure["intro_bars"],
        "flow_type": structure["flow_type"],
        "confidences": {
            "energy": energy["energy_confidence"],
            "vibe": vibe["vibe_confidence"],
            "mood": vibe["mood_confidence"],
            "vocal": vocal["vocal_confidence"],
            "structure": structure["structure_confidence"],
        },
    }
