"""Blended distance computation across feature layers."""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from ..config import GrouperConfig
from ..features.builder import TrackFeatures

logger = logging.getLogger(__name__)

_MAX_KEY_DIST = 2.0


def tag_distance(a: TrackFeatures, b: TrackFeatures, config: GrouperConfig) -> float:
    """Custom distance for tag vectors with confidence weighting.

    Tag vector layout (encode_tags):
      [0]     energy (ordinal, confidence-weighted)
      [1]     BPM (normalized)
      [2:4]   key (sin/cos, confidence-weighted)
      [4]     intro bars (ordinal, confidence-weighted)
      [5:10]  flow type (confidence-weighted soft one-hot)
      [10:18] vibe (continuous scores or confidence-weighted one-hot)
      [18]    vocal (confidence-weighted)
    """
    va, vb = a.tag_vector, b.tag_vector

    # Energy distance: ordinal [0, 1]
    d_energy = abs(float(va[0] - vb[0]))

    # BPM distance: non-linear to amplify genre boundaries
    d_bpm_raw = abs(float(va[1] - vb[1]))
    d_bpm = d_bpm_raw ** 1.5

    # Key distance: Euclidean on sin/cos, normalized to [0, 1]
    d_key_raw = math.sqrt(float((va[2] - vb[2]) ** 2 + (va[3] - vb[3]) ** 2))
    d_key = min(1.0, d_key_raw / _MAX_KEY_DIST)

    # Conditional key weight
    vibe_a = a.info.vibe or "HYPN"
    vibe_b = b.info.vibe or "HYPN"
    kw_a = config.key_weight_by_vibe.get(vibe_a, 0.2)
    kw_b = config.key_weight_by_vibe.get(vibe_b, 0.2)
    if a.info.vocal == "V":
        kw_a = min(1.0, kw_a + config.key_weight_vocal_boost)
    if b.info.vocal == "V":
        kw_b = min(1.0, kw_b + config.key_weight_vocal_boost)
    key_weight = (kw_a + kw_b) / 2.0

    # Intro bars: ordinal [0, 1]
    d_bars = abs(float(va[4] - vb[4]))

    # Flow type: Euclidean on soft one-hot (captures confidence smoothing)
    d_flow = float(np.sqrt(np.sum((va[5:10] - vb[5:10]) ** 2)))
    d_flow = min(1.0, d_flow)

    # Vibe: Euclidean on continuous scores — captures partial similarity
    d_vibe = float(np.sqrt(np.sum((va[10:18] - vb[10:18]) ** 2)))
    d_vibe = min(1.0, d_vibe)

    # Vocal: [0, 1]
    d_vocal = abs(float(va[18] - vb[18]))

    # Weighted combination
    d = (
        0.30 * d_energy
        + 0.20 * d_bpm
        + key_weight * 0.10 * d_key
        + 0.05 * d_bars
        + 0.05 * d_flow
        + 0.25 * d_vibe
        + 0.05 * d_vocal
    )
    return d


def dsp_distance(a: TrackFeatures, b: TrackFeatures) -> float:
    """Cosine distance on curated DSP features. [0, 1]."""
    if np.array_equal(a.dsp_vector, b.dsp_vector):
        return 0.0
    norm_a = float(np.linalg.norm(a.dsp_vector))
    norm_b = float(np.linalg.norm(b.dsp_vector))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    dot = float(np.dot(a.dsp_vector, b.dsp_vector))
    return max(0.0, 1.0 - dot / (norm_a * norm_b))


def embed_distance(a: TrackFeatures, b: TrackFeatures) -> float:
    """Cosine distance on CLAP embeddings. [0, 1]."""
    if a.embed_vector is None or b.embed_vector is None:
        return 0.0
    if np.array_equal(a.embed_vector, b.embed_vector):
        return 0.0
    dot = float(np.dot(a.embed_vector, b.embed_vector))
    norm_a = float(np.linalg.norm(a.embed_vector))
    norm_b = float(np.linalg.norm(b.embed_vector))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return max(0.0, 1.0 - dot / (norm_a * norm_b))


def blended_distance(a: TrackFeatures, b: TrackFeatures, config: GrouperConfig) -> float:
    """Blended distance across all layers."""
    has_embed = a.embed_vector is not None and b.embed_vector is not None

    d_tag = tag_distance(a, b, config)
    d_dsp = dsp_distance(a, b)

    if has_embed:
        d_emb = embed_distance(a, b)
        return config.w_tags * d_tag + config.w_dsp * d_dsp + config.w_embed * d_emb
    else:
        return config.w_tags_no_embed * d_tag + config.w_dsp_no_embed * d_dsp


def compute_distance_matrix(
    tracks: list[TrackFeatures],
    config: GrouperConfig,
) -> NDArray[np.floating]:
    """Compute the full pairwise distance matrix."""
    n = len(tracks)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = blended_distance(tracks[i], tracks[j], config)
            matrix[i, j] = d
            matrix[j, i] = d
    logger.info("Distance matrix computed: %dx%d", n, n)
    return matrix
