"""Recompute derived analysis values from cached raw features.

All parameters are read from settings.toml. Change any value there and
the derived cache auto-invalidates on next run — no manual version bumping.

Raw inputs:
  - dsp: dict with ~45 DSP features (from grouper extraction)
  - raw_analysis: dict with additional features not in DSP
    (bar_energies, vocal_ratio, vocal_temporal_bonus, onset_rate)

Derived outputs:
  - energy: level (E1-E5), confidence
  - vibe: label, scores (8 floats), confidence
  - vocal: has_vocals (V/NV), confidence
  - structure: intro_bars, flow_type, formatted, confidence
"""

from __future__ import annotations

import logging

import numpy as np

from .settings import get, get_section

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
    """Recompute vibe from cached DSP features + optional API audio features.

    audio_features: optional dict with keys like 'valence', 'instrumentalness',
    'liveness', 'energy', 'acousticness' (0-1 scale). When provided, these
    contribute additional scoring signals per vibe via [vibe.songstats] settings.
    """
    s = get_section("vibe")

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

    # Normalize to coefficient of variation
    onset_var = onset_variance / (onset_density ** 2 + 1e-8)
    centroid_cv = centroid_var / (centroid_mean ** 2 + 1e-8)
    spectral_stability = 1.0 - min(1.0, centroid_cv * 10)

    scores: dict[str, float] = {}

    # MEL
    mel_base = s.get("mel_base", 0.38)
    mel_chroma = min(1.0, chroma_var / s.get("mel_chroma_scale", 0.065))
    mel_lowflat = _clip01((s.get("mel_lowflat_center", 0.025) - flatness) / s.get("mel_lowflat_scale", 0.015))
    scores["MEL"] = mel_base + s.get("mel_chroma_weight", 0.25) * mel_chroma + s.get("mel_lowflat_weight", 0.12) * mel_lowflat

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
    scores["ACID"] = _scored(centroid_cv, s.get("acid_centroid_var", [1.0, 0.75, 0.40]))

    # ATM
    scores["ATM"] = (
        _scored_inv(onset_density, s.get("atm_sparse", [0.50, 1.5, 1.0]))
        + _scored_inv(rms, s.get("atm_quiet", [0.50, 0.20, 0.08]))
    )

    # Add Songstats audio feature contributions (when available)
    if audio_features:
        ss = s.get("songstats", {})

        af_valence = audio_features.get("valence")
        af_energy = audio_features.get("energy")
        af_instrumentalness = audio_features.get("instrumentalness")
        af_liveness = audio_features.get("liveness")
        af_acousticness = audio_features.get("acousticness")

        # DRK: low valence + high energy
        if af_valence is not None:
            scores["DRK"] += _scored_inv(af_valence, ss.get("drk_valence", [0.15, 0.35, 0.25]))
        if af_energy is not None:
            scores["DRK"] += _scored(af_energy, ss.get("drk_energy", [0.10, 0.60, 0.30]))

        # HYPN: high instrumentalness + low valence
        if af_instrumentalness is not None:
            scores["HYPN"] += _scored(af_instrumentalness, ss.get("hypn_instrumentalness", [0.15, 0.70, 0.25]))
        if af_valence is not None:
            scores["HYPN"] += _scored_inv(af_valence, ss.get("hypn_valence", [0.10, 0.40, 0.30]))

        # TRIB: high instrumentalness + high energy
        if af_instrumentalness is not None:
            scores["TRIB"] += _scored(af_instrumentalness, ss.get("trib_instrumentalness", [0.10, 0.70, 0.25]))
        if af_energy is not None:
            scores["TRIB"] += _scored(af_energy, ss.get("trib_energy", [0.10, 0.60, 0.30]))

        # DEEP: low energy + low valence
        if af_energy is not None:
            scores["DEEP"] += _scored_inv(af_energy, ss.get("deep_energy", [0.15, 0.40, 0.25]))
        if af_valence is not None:
            scores["DEEP"] += _scored_inv(af_valence, ss.get("deep_valence", [0.10, 0.35, 0.25]))

        # ATM: low energy + high acousticness
        if af_energy is not None:
            scores["ATM"] += _scored_inv(af_energy, ss.get("atm_energy", [0.15, 0.35, 0.25]))
        if af_acousticness is not None:
            scores["ATM"] += _scored(af_acousticness, ss.get("atm_acousticness", [0.10, 0.40, 0.30]))

        # RAW: high liveness
        if af_liveness is not None:
            scores["RAW"] += _scored(af_liveness, ss.get("raw_liveness", [0.15, 0.40, 0.30]))

        # MEL: high valence (earned evidence)
        if af_valence is not None:
            scores["MEL"] += _scored(af_valence, ss.get("mel_valence", [0.15, 0.50, 0.30]))

    scores = {k: _clip01(v) for k, v in scores.items()}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0

    return {"vibe": label, "vibe_scores": scores, "vibe_confidence": confidence}


# ── Vocal derivation ─────────────────────────────────────────────────────────

def derive_vocal(raw_analysis: dict) -> dict:
    """Recompute vocal V/NV from cached raw analysis features."""
    s = get_section("vocal")

    vocal_ratio = raw_analysis.get("vocal_ratio", 0.0)
    temporal_bonus = raw_analysis.get("vocal_temporal_bonus", 0.0)
    base_w = s.get("score_base_weight", 0.6)
    temp_w = s.get("score_temporal_weight", 0.4)
    threshold = s.get("frame_threshold", 0.20)
    conf_scale = s.get("confidence_scale", 0.15)

    vocal_score = vocal_ratio * (base_w + temp_w * temporal_bonus)
    has_vocals = vocal_score > threshold
    confidence = min(1.0, abs(vocal_score - threshold) / conf_scale)

    return {
        "vocal": "V" if has_vocals else "NV",
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
