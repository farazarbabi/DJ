"""Canonical mood/vibe scoring shared by analysis and cache derivation."""

from __future__ import annotations

from collections.abc import Mapping

from .moods import MOOD_LABELS
from .settings import get_section

VIBE_LABELS: tuple[str, ...] = MOOD_LABELS


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def scored(raw: float, params: list) -> float:
    """Compute weight * clamp01((raw - offset) / scale) from [weight, offset, scale]."""
    weight, offset, scale = params
    return weight * clip01((raw - offset) / scale)


def scored_inv(raw: float, params: list) -> float:
    """Inverted: weight * clamp01((offset - raw) / scale)."""
    weight, center, scale = params
    return weight * clip01((center - raw) / scale)


def vibe_metrics_from_dsp(dsp: Mapping[str, float]) -> dict[str, float]:
    """Normalize raw DSP features into the canonical mood metrics space."""
    rms = float(dsp.get("rms_mean", 0.0))
    centroid_mean = float(dsp.get("centroid_mean", 0.0))
    centroid_var = float(dsp.get("centroid_var", 0.0))
    flatness = float(dsp.get("flatness_mean", 0.0))
    onset_density = float(dsp.get("onset_density", 0.0))
    onset_variance = float(dsp.get("onset_variance", 0.0))
    perc_ratio = float(dsp.get("perc_harmonic_ratio", 0.0))
    low_ratio = float(dsp.get("low_freq_ratio", 0.0))
    flux = float(dsp.get("flux_mean", 0.0))
    chroma_var = float(dsp.get("chroma_var", 0.0))

    onset_var = onset_variance / (onset_density ** 2 + 1e-8)
    centroid_cv = centroid_var / (centroid_mean ** 2 + 1e-8)
    spectral_stability = 1.0 - min(1.0, centroid_cv * 10)

    return {
        "rms": rms,
        "centroid_mean": centroid_mean,
        "centroid_cv": centroid_cv,
        "flatness": flatness,
        "onset_density": onset_density,
        "onset_var": onset_var,
        "perc_ratio": perc_ratio,
        "low_ratio": low_ratio,
        "flux": flux,
        "chroma_var": chroma_var,
        "spectral_stability": spectral_stability,
    }


def score_vibe_metrics(
    metrics: Mapping[str, float],
    audio_features: Mapping[str, float] | None = None,
) -> tuple[str, dict[str, float], float]:
    """Score taxonomy mood labels from normalized metrics + optional Songstats audio features."""
    s = get_section("vibe")

    rms = float(metrics.get("rms", 0.0))
    centroid_mean = float(metrics.get("centroid_mean", 0.0))
    centroid_cv = float(metrics.get("centroid_cv", 0.0))
    flatness = float(metrics.get("flatness", 0.0))
    onset_density = float(metrics.get("onset_density", 0.0))
    onset_var = float(metrics.get("onset_var", 0.0))
    perc_ratio = float(metrics.get("perc_ratio", 0.0))
    low_ratio = float(metrics.get("low_ratio", 0.0))
    flux = float(metrics.get("flux", 0.0))
    chroma_var = float(metrics.get("chroma_var", 0.0))
    spectral_stability = float(metrics.get("spectral_stability", 0.0))

    base_scores: dict[str, float] = {}

    mel_base = s.get("mel_base", 0.38)
    mel_chroma = min(1.0, chroma_var / s.get("mel_chroma_scale", 0.065))
    mel_lowflat = clip01((s.get("mel_lowflat_center", 0.025) - flatness) / s.get("mel_lowflat_scale", 0.015))
    base_scores["MEL"] = mel_base + s.get("mel_chroma_weight", 0.25) * mel_chroma + s.get("mel_lowflat_weight", 0.12) * mel_lowflat

    base_scores["DRK"] = (
        scored(onset_var, s.get("drk_onset_var", [0.25, 1.5, 2.0]))
        + scored(flatness, s.get("drk_flatness", [0.30, 0.010, 0.030]))
        + scored(low_ratio, s.get("drk_low_ratio", [0.25, 45, 25]))
        + scored(flux, s.get("drk_flux", [0.20, 2.0, 2.5]))
    )

    base_scores["TRIB"] = (
        scored(perc_ratio, s.get("trib_perc_ratio", [0.70, 0.40, 0.15]))
        + scored_inv(flatness, s.get("trib_clean", [0.30, 0.03, 0.02]))
    )

    base_scores["RAW"] = (
        scored_inv(low_ratio, s.get("raw_nobass", [0.45, 42, 12]))
        + scored_inv(onset_density, s.get("raw_sparse", [0.30, 1.5, 0.8]))
        + scored_inv(onset_var, s.get("raw_lowvar", [0.25, 1.5, 1.0]))
    )

    hypn_stab_w = s.get("hypn_stability_weight", 0.50)
    hypn_onset = s.get("hypn_onset_var", [0.50, 2.0, 1.5])
    base_scores["HYPN"] = hypn_stab_w * spectral_stability + scored_inv(onset_var, hypn_onset)

    base_scores["DEEP"] = (
        scored_inv(centroid_mean, s.get("deep_centroid", [0.35, 2000, 1000]))
        + scored_inv(rms, s.get("deep_rms", [0.35, 0.22, 0.10]))
        + scored(low_ratio, s.get("deep_low", [0.30, 50, 20]))
    )

    base_scores["ACID"] = scored(centroid_cv, s.get("acid_centroid_var", [1.0, 0.75, 0.40]))

    base_scores["ATM"] = (
        scored_inv(onset_density, s.get("atm_sparse", [0.50, 1.5, 1.0]))
        + scored_inv(rms, s.get("atm_quiet", [0.50, 0.20, 0.08]))
    )

    if audio_features:
        ss = s.get("songstats", {})

        af_valence = audio_features.get("valence")
        af_energy = audio_features.get("energy")
        af_instrumentalness = audio_features.get("instrumentalness")
        af_liveness = audio_features.get("liveness")
        af_acousticness = audio_features.get("acousticness")

        if af_valence is not None:
            base_scores["DRK"] += scored_inv(float(af_valence), ss.get("drk_valence", [0.15, 0.35, 0.25]))
        if af_energy is not None:
            base_scores["DRK"] += scored(float(af_energy), ss.get("drk_energy", [0.10, 0.60, 0.30]))

        if af_instrumentalness is not None:
            base_scores["HYPN"] += scored(float(af_instrumentalness), ss.get("hypn_instrumentalness", [0.15, 0.70, 0.25]))
        if af_valence is not None:
            base_scores["HYPN"] += scored_inv(float(af_valence), ss.get("hypn_valence", [0.10, 0.40, 0.30]))

        if af_instrumentalness is not None:
            base_scores["TRIB"] += scored(float(af_instrumentalness), ss.get("trib_instrumentalness", [0.10, 0.70, 0.25]))
        if af_energy is not None:
            base_scores["TRIB"] += scored(float(af_energy), ss.get("trib_energy", [0.10, 0.60, 0.30]))

        if af_energy is not None:
            base_scores["DEEP"] += scored_inv(float(af_energy), ss.get("deep_energy", [0.15, 0.40, 0.25]))
        if af_valence is not None:
            base_scores["DEEP"] += scored_inv(float(af_valence), ss.get("deep_valence", [0.10, 0.35, 0.25]))

        if af_energy is not None:
            base_scores["ATM"] += scored_inv(float(af_energy), ss.get("atm_energy", [0.15, 0.35, 0.25]))
        if af_acousticness is not None:
            base_scores["ATM"] += scored(float(af_acousticness), ss.get("atm_acousticness", [0.10, 0.40, 0.30]))

        if af_liveness is not None:
            base_scores["RAW"] += scored(float(af_liveness), ss.get("raw_liveness", [0.15, 0.40, 0.30]))

        if af_valence is not None:
            base_scores["MEL"] += scored(float(af_valence), ss.get("mel_valence", [0.15, 0.50, 0.30]))

    scores = _expand_taxonomy_mood_scores(
        {k: clip01(v) for k, v in base_scores.items()},
        metrics,
        audio_features or {},
    )
    label = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0
    return label, scores, confidence


def _expand_taxonomy_mood_scores(
    base: Mapping[str, float],
    metrics: Mapping[str, float],
    audio_features: Mapping[str, float],
) -> dict[str, float]:
    """Project existing DSP evidence onto every taxonomy mood code."""
    valence = _optional_float(audio_features.get("valence"))
    af_energy = _optional_float(audio_features.get("energy"))
    instrumentalness = _optional_float(audio_features.get("instrumentalness"))
    liveness = _optional_float(audio_features.get("liveness"))
    acousticness = _optional_float(audio_features.get("acousticness"))

    flatness = float(metrics.get("flatness", 0.0))
    onset_density = float(metrics.get("onset_density", 0.0))
    onset_var = float(metrics.get("onset_var", 0.0))
    perc_ratio = float(metrics.get("perc_ratio", 0.0))
    low_ratio = float(metrics.get("low_ratio", 0.0))
    flux = float(metrics.get("flux", 0.0))
    chroma_var = float(metrics.get("chroma_var", 0.0))
    centroid_cv = float(metrics.get("centroid_cv", 0.0))

    valence_high = valence if valence is not None else clip01(chroma_var / 0.12)
    valence_low = (1.0 - valence) if valence is not None else base.get("DRK", 0.0)
    api_energy_high = af_energy if af_energy is not None else max(base.get("DRK", 0.0), base.get("TRIB", 0.0))
    api_energy_low = (1.0 - af_energy) if af_energy is not None else max(base.get("DEEP", 0.0), base.get("ATM", 0.0))
    acoustic = acousticness if acousticness is not None else clip01((0.03 - flatness) / 0.03)
    live = liveness if liveness is not None else clip01(perc_ratio)
    instrumental = instrumentalness if instrumentalness is not None else 0.5

    expanded = {
        "ACID": base.get("ACID", 0.0),
        "ATM": base.get("ATM", 0.0),
        "CIN": 0.58 * base.get("ATM", 0.0) + 0.34 * base.get("MEL", 0.0) + 0.08 * api_energy_low,
        "DEEP": base.get("DEEP", 0.0),
        "DRK": base.get("DRK", 0.0),
        "EMO": 0.58 * base.get("MEL", 0.0) + 0.32 * valence_low + 0.10 * base.get("DEEP", 0.0),
        "EUP": 0.48 * base.get("MEL", 0.0) + 0.32 * valence_high + 0.20 * api_energy_high,
        "GRIT": 0.54 * base.get("RAW", 0.0) + 0.30 * base.get("DRK", 0.0) + 0.16 * clip01(flux / 5.0),
        "HYPN": base.get("HYPN", 0.0),
        "MEL": base.get("MEL", 0.0),
        "MIN": 0.52 * base.get("HYPN", 0.0) + 0.28 * base.get("DEEP", 0.0) + 0.20 * clip01((2.0 - onset_density) / 2.0),
        "ORG": 0.50 * base.get("TRIB", 0.0) + 0.30 * acoustic + 0.20 * live,
        "PLAY": 0.42 * valence_high + 0.30 * clip01(onset_density / 3.0) + 0.28 * base.get("MEL", 0.0),
        "PSY": 0.52 * base.get("ACID", 0.0) + 0.30 * base.get("HYPN", 0.0) + 0.18 * clip01(centroid_cv),
        "RAW": base.get("RAW", 0.0),
        "SOUL": 0.46 * valence_high + 0.30 * base.get("MEL", 0.0) + 0.24 * acoustic,
        "SUB": 0.64 * base.get("DEEP", 0.0) + 0.36 * clip01((low_ratio - 45.0) / 35.0),
        "SUN": 0.46 * valence_high + 0.30 * api_energy_low + 0.24 * base.get("MEL", 0.0),
        "TENS": 0.46 * base.get("DRK", 0.0) + 0.28 * valence_low + 0.26 * clip01(onset_var / 4.0),
        "TRIB": base.get("TRIB", 0.0),
        "WARM": 0.42 * valence_high + 0.34 * base.get("DEEP", 0.0) + 0.24 * acoustic,
        "WHSE": 0.46 * base.get("RAW", 0.0) + 0.34 * base.get("DRK", 0.0) + 0.20 * api_energy_high,
    }
    return {code: clip01(expanded.get(code, 0.0)) for code in MOOD_LABELS}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
