"""Combine all feature layers into per-track feature bundles."""

from __future__ import annotations

import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import GrouperConfig
from ..scanner import TrackInfo, VIBE_LABELS
from .dsp import DSP_CURATED_NAMES
from .role import infer_role

logger = logging.getLogger(__name__)

# Camelot wheel positions (1A-12A, 1B-12B mapped to 0-23 for circular encoding)
_CAMELOT_POSITIONS: dict[str, int] = {}
for i in range(1, 13):
    _CAMELOT_POSITIONS[f"{i}A"] = i - 1
    _CAMELOT_POSITIONS[f"{i}B"] = i - 1 + 12

FLOW_TYPES = ["G", "H", "D", "B", "L"]
ROLE_LABELS = ["TOOL", "DRIVER", "PEAK", "RESET", "BREAKDOWN", "BRIDGE"]


# ─── Per-track raw cache entry ──────────────────────────────────────────────

@dataclass
class RawCacheEntry:
    """Per-file cached extraction result (survives across runs)."""
    mtime: float
    info: TrackInfo
    dsp: dict[str, float]
    section_dsp: dict[str, dict[str, float]] = field(default_factory=dict)


RawCache = dict[str, RawCacheEntry]


def load_raw_cache(path: str) -> RawCache:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            # Check if it's the new format (RawCacheEntry) or old format
            first_val = next(iter(data.values()), None) if data else None
            if first_val is not None and hasattr(first_val, "mtime"):
                logger.info("Loaded raw cache from %s (%d entries)", path, len(data))
                return data
        logger.warning("Cache at %s is old format, will re-extract", path)
        return {}
    except Exception:
        logger.warning("Could not load cache at %s, will re-extract", path)
        return {}


def save_raw_cache(raw_cache: RawCache, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(raw_cache, f)
    logger.info("Raw cache saved to %s (%d entries)", path, len(raw_cache))


# ─── Final features (derived from raw cache each run) ───────────────────────

@dataclass
class TrackFeatures:
    """All features for one track, ready for distance computation."""
    path: str
    info: TrackInfo
    tag_vector: NDArray[np.floating]       # encoded tags with continuous vibe
    dsp_vector: NDArray[np.floating]       # 10 curated DSP, z-score normalized
    raw_dsp: dict[str, float] = field(default_factory=dict)  # named features for scoring
    section_dsp: dict[str, dict[str, float]] = field(default_factory=dict)  # per-section
    embed_vector: NDArray[np.floating] | None = None
    role: str = "DRIVER"


@dataclass
class FeatureCache:
    """Derived features for the whole library."""
    tracks: list[TrackFeatures] = field(default_factory=list)
    dsp_mean: NDArray | None = None
    dsp_std: NDArray | None = None
    version: str = "5.0"


# ─── Scaling ─────────────────────────────────────────────────────────────────

def _percentile_rank(matrix: NDArray) -> NDArray:
    """Convert each column to percentile ranks in [0, 1].

    For each feature, the track with the lowest value gets 0.0, highest gets 1.0,
    and everything in between is linearly ranked. This guarantees the full [0, 1]
    range is used regardless of how narrow the raw distribution is.
    """
    from scipy.stats import rankdata
    n = matrix.shape[0]
    if n <= 1:
        return np.zeros_like(matrix)
    ranked = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        ranks = rankdata(matrix[:, col], method="average")
        ranked[:, col] = (ranks - 1) / (n - 1)  # scale to [0, 1]
    return ranked.astype(np.float32)


# ─── Tag encoding ───────────────────────────────────────────────────────────

def encode_tags(info: TrackInfo) -> NDArray[np.floating]:
    """Encode tag metadata into a numeric vector.

    Uses continuous vibe scores (8 floats) instead of one-hot.
    Confidence-weights uncertain features toward neutral.
    """
    v: list[float] = []
    confs = info.confidences

    # Energy: ordinal 0-1, confidence-weighted
    energy_val = (info.energy - 1) / 4.0 if info.energy else 0.5
    energy_conf = confs.get("energy", 1.0)
    v.append(energy_val * energy_conf + 0.5 * (1.0 - energy_conf))

    # BPM: normalized to tighter electronic music range (100-140)
    bpm = info.bpm or 128
    v.append(max(0.0, min(1.0, (bpm - 100) / 40.0)))

    # Key: circular, confidence-weighted toward zero (neutral)
    if info.key and info.key in _CAMELOT_POSITIONS:
        pos = _CAMELOT_POSITIONS[info.key]
        angle = 2 * math.pi * pos / 24.0
        key_conf = confs.get("key", 1.0)
        v.append(math.sin(angle) * key_conf)
        v.append(math.cos(angle) * key_conf)
    else:
        v.extend([0.0, 0.0])

    # Intro bars: ordinal, confidence-weighted
    bars = info.intro_bars or 32
    bars_val = {16: 0.0, 32: 0.5, 64: 1.0}.get(bars, 0.5)
    struct_conf = confs.get("structure", 1.0)
    v.append(bars_val * struct_conf + 0.5 * (1.0 - struct_conf))

    # Flow type: one-hot, confidence-weighted
    flow = info.flow_type or "H"
    for ft in FLOW_TYPES:
        val = 1.0 if flow == ft else 0.0
        v.append(val * struct_conf + (1.0 / len(FLOW_TYPES)) * (1.0 - struct_conf))

    # Vibe: continuous scores instead of one-hot
    # If vibe_scores available, use them directly; else fall back to one-hot
    vibe_conf = confs.get("vibe", 1.0)
    if info.vibe_scores:
        # Normalize scores to sum to 1
        total = sum(info.vibe_scores.values()) + 1e-8
        for vl in VIBE_LABELS:
            v.append(info.vibe_scores.get(vl, 0.0) / total)
    else:
        vibe = info.vibe or "HYPN"
        for vl in VIBE_LABELS:
            val = 1.0 if vibe == vl else 0.0
            v.append(val * vibe_conf + (1.0 / len(VIBE_LABELS)) * (1.0 - vibe_conf))

    # Vocal: binary, confidence-weighted toward 0.5 (uncertain)
    vocal_val = 1.0 if info.vocal == "V" else 0.0
    vocal_conf = confs.get("vocal", 1.0)
    v.append(vocal_val * vocal_conf + 0.5 * (1.0 - vocal_conf))

    return np.array(v, dtype=np.float32)


# ─── Energy classification (per-track, absolute thresholds) ─────────────────

# Absolute normalization ranges for electronic music (techno/house/downtempo).
# Derived from real electronic music analysis. Stable across library sizes.
_ENERGY_FEATURES = {
    #                 (min,  range,  weight)
    "beat_strength":  (1.5,  2.5,    0.35),   # 1.5-4.0: kick strength
    "onset_density":  (1.0,  2.0,    0.30),   # 1.0-3.0: rhythmic density
    "centroid_mean":  (1000, 2000,   0.20),   # 1000-3000: brightness/aggression
    "rms_mean":       (0.15, 0.20,   0.15),   # 0.15-0.35: loudness (de-emphasized)
}

# Composite thresholds — asymmetric, calibrated for electronic music
# E1 is wide (ambient/sparse is rare), E5 is narrow (truly aggressive is rare)
_ENERGY_THRESHOLDS = [0.25, 0.42, 0.58, 0.75]  # E1<0.25, E2<0.42, E3<0.58, E4<0.75, E5>=0.75


def _compute_track_energy(dsp: dict[str, float]) -> int:
    """Compute energy level for a single track from its DSP features.

    Returns 1-5. Deterministic per-track — does not depend on library context.
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
    return min(level, 5)


def _calibrate_energy(
    infos: list[TrackInfo],
    dsp_dicts: list[dict[str, float]],
) -> None:
    """Assign energy levels using absolute per-track scoring.

    Each track is scored independently against fixed thresholds calibrated
    for electronic music. Stable regardless of library size — adding or
    removing tracks does not change other tracks' energy levels.
    """
    for i, dsp in enumerate(dsp_dicts):
        infos[i].energy = _compute_track_energy(dsp)

    levels = [info.energy for info in infos]
    logger.info(
        "Energy assigned: E1=%d E2=%d E3=%d E4=%d E5=%d",
        levels.count(1), levels.count(2), levels.count(3), levels.count(4), levels.count(5),
    )


# ─── Build features from raw cache ──────────────────────────────────────────

def build_features_from_raw(
    raw_cache: RawCache,
    track_order: list[str],
    clap_embeddings: NDArray | None = None,
    config: GrouperConfig | None = None,
) -> FeatureCache:
    """Build feature bundles from the raw cache."""
    config = config or GrouperConfig()

    infos = [raw_cache[p].info for p in track_order]
    dsp_dicts = [raw_cache[p].dsp for p in track_order]
    section_dsps = [raw_cache[p].section_dsp for p in track_order]

    # Encode tags (with continuous vibe + confidence weighting)
    tag_vectors = np.array([encode_tags(info) for info in infos], dtype=np.float32)

    # Build DSP matrix from curated features, percentile rank scaling
    # Percentile rank guarantees full [0, 1] spread regardless of how narrow
    # the raw distribution is — critical for mastered electronic music where
    # features cluster in a tight band.
    dsp_matrix = np.array(
        [[d.get(name, 0.0) for name in DSP_CURATED_NAMES] for d in dsp_dicts],
        dtype=np.float32,
    )
    dsp_mean = np.mean(dsp_matrix, axis=0)  # kept for diagnostics
    dsp_std = np.std(dsp_matrix, axis=0) + 1e-8
    dsp_normalized = _percentile_rank(dsp_matrix)

    # Recalibrate energy relative to library distribution
    _calibrate_energy(infos, dsp_dicts)

    # Infer roles (after energy recalibration)
    roles = [infer_role(info, dsp) for info, dsp in zip(infos, dsp_dicts)]

    # Re-encode tags after energy recalibration
    tag_vectors = np.array([encode_tags(info) for info in infos], dtype=np.float32)

    # Assemble TrackFeatures
    feature_list: list[TrackFeatures] = []
    for i, path in enumerate(track_order):
        infos[i].role = roles[i]
        tf = TrackFeatures(
            path=path,
            info=infos[i],
            tag_vector=tag_vectors[i],
            dsp_vector=dsp_normalized[i],
            raw_dsp=dsp_dicts[i],
            section_dsp=section_dsps[i],
            embed_vector=clap_embeddings[i] if clap_embeddings is not None else None,
            role=roles[i],
        )
        feature_list.append(tf)

    return FeatureCache(
        tracks=feature_list,
        dsp_mean=dsp_mean,
        dsp_std=dsp_std,
    )


# ─── Legacy build_features (used by tests) ──────────────────────────────────

def build_features(
    tracks: list[TrackInfo],
    dsp_features: list[dict[str, float]],
    clap_embeddings: NDArray | None = None,
    config: GrouperConfig | None = None,
) -> FeatureCache:
    """Build features directly from track lists (no raw cache)."""
    raw_cache: RawCache = {}
    order: list[str] = []
    for i, info in enumerate(tracks):
        raw_cache[info.path] = RawCacheEntry(mtime=0.0, info=info, dsp=dsp_features[i])
        order.append(info.path)
    return build_features_from_raw(raw_cache, order, clap_embeddings, config)


# ─── Save/load ──────────────────────────────────────────────────────────────

def save_cache(cache: FeatureCache, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Feature cache saved to %s", path)


def load_cache(path: str) -> FeatureCache:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        logger.info("Cache is raw format, rebuilding features...")
        raw_cache: RawCache = data
        order = list(raw_cache.keys())
        return build_features_from_raw(raw_cache, order)
    logger.info("Feature cache loaded from %s (%d tracks)", path, len(data.tracks))
    return data
