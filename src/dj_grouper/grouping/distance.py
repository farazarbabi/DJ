"""Blended distance computation across feature layers."""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from dj_tagger.moods import normalize_mood_code
from dj_tagger.vocals import has_vocal_content

from ..config import GrouperConfig
from ..features.builder import (
    TAG_BPM_IDX,
    TAG_ENERGY_IDX,
    TAG_FLOW_SLICE,
    TAG_INTRO_BARS_IDX,
    TAG_KEY_SLICE,
    TAG_MOOD_SLICE,
    TAG_VOCAL_SLICE,
    TrackFeatures,
)

logger = logging.getLogger(__name__)

_MAX_KEY_DIST = 2.0


def tag_distance(a: TrackFeatures, b: TrackFeatures, config: GrouperConfig) -> float:
    """Custom distance for tag vectors with confidence weighting.

    Tag vector layout is defined by dynamic constants in features.builder.
    Current length is TAG_VECTOR_DIM because the mood slice is taxonomy-driven.
    """
    va, vb = a.tag_vector, b.tag_vector

    # Energy distance: power 1.5 to amplify level differences
    d_energy = abs(float(va[TAG_ENERGY_IDX] - vb[TAG_ENERGY_IDX])) ** 1.5

    # BPM distance: quadratic to amplify genre boundaries
    d_bpm_raw = abs(float(va[TAG_BPM_IDX] - vb[TAG_BPM_IDX]))
    d_bpm = d_bpm_raw ** 2.0

    # Key distance: Euclidean on sin/cos, normalized to [0, 1]
    d_key_raw = math.sqrt(float(np.sum((va[TAG_KEY_SLICE] - vb[TAG_KEY_SLICE]) ** 2)))
    d_key = min(1.0, d_key_raw / _MAX_KEY_DIST)

    # Conditional key weight
    vibe_a = normalize_mood_code(a.info.vibe) or "HYPN"
    vibe_b = normalize_mood_code(b.info.vibe) or "HYPN"
    kw_a = config.key_weight_by_vibe.get(vibe_a, 0.2)
    kw_b = config.key_weight_by_vibe.get(vibe_b, 0.2)
    if has_vocal_content(a.info.vocal):
        kw_a = min(1.0, kw_a + config.key_weight_vocal_boost)
    if has_vocal_content(b.info.vocal):
        kw_b = min(1.0, kw_b + config.key_weight_vocal_boost)
    key_weight = (kw_a + kw_b) / 2.0

    # Intro bars: ordinal [0, 1]
    d_bars = abs(float(va[TAG_INTRO_BARS_IDX] - vb[TAG_INTRO_BARS_IDX]))

    # Flow type: Euclidean on soft one-hot (captures confidence smoothing)
    d_flow = float(np.sqrt(np.sum((va[TAG_FLOW_SLICE] - vb[TAG_FLOW_SLICE]) ** 2)))
    d_flow = min(1.0, d_flow)

    # Mood: Euclidean on continuous scores; captures partial similarity.
    d_mood = float(np.sqrt(np.sum((va[TAG_MOOD_SLICE] - vb[TAG_MOOD_SLICE]) ** 2)))
    d_mood = min(1.0, d_mood)

    # Vocal profile: Euclidean on continuous taxonomy profile scores.
    d_vocal = float(np.sqrt(np.sum((va[TAG_VOCAL_SLICE] - vb[TAG_VOCAL_SLICE]) ** 2)))
    d_vocal = min(1.0, d_vocal)

    # Weighted combination — BPM elevated to prevent unmixable groupings
    d = (
        0.25 * d_energy
        + 0.30 * d_bpm
        + key_weight * 0.10 * d_key
        + 0.05 * d_bars
        + 0.05 * d_flow
        + 0.20 * d_mood
        + 0.05 * d_vocal
    )
    return d


def dsp_distance(a: TrackFeatures, b: TrackFeatures) -> float:
    """L2-normalized Euclidean distance on DSP features. [0, 1].

    Unlike cosine, this preserves magnitude differences — a loud aggressive
    track and a quiet atmospheric track are far apart even if their spectral
    shapes are similar.
    """
    if np.array_equal(a.dsp_vector, b.dsp_vector):
        return 0.0
    # L2-normalize each vector, then Euclidean
    norm_a = float(np.linalg.norm(a.dsp_vector))
    norm_b = float(np.linalg.norm(b.dsp_vector))
    if norm_a < 1e-8 and norm_b < 1e-8:
        return 0.0
    va = a.dsp_vector / (norm_a + 1e-8)
    vb = b.dsp_vector / (norm_b + 1e-8)
    # Euclidean on unit vectors: max possible = 2.0, normalize to [0, 1]
    d = float(np.sqrt(np.sum((va - vb) ** 2)))
    return min(1.0, d / 2.0)


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
    """Compute the full pairwise distance matrix with contrast stretching.

    After computing raw blended distances, applies contrast stretching so
    the observed distance range maps to [0, 1]. This is critical for
    electronic music where CLAP cosine similarities cluster in a narrow band.
    """
    n = len(tracks)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = blended_distance(tracks[i], tracks[j], config)
            matrix[i, j] = d
            matrix[j, i] = d

    return _contrast_stretch(matrix)


def unified_distance_matrix(
    reduced_vectors: NDArray,
) -> NDArray[np.floating]:
    """Compute distance matrix from PCA-reduced L2-normalized unified vectors.

    Euclidean distance on L2-normalized vectors, then contrast stretched.
    This replaces blended_distance() for the constrained clustering path.

    Args:
        reduced_vectors: (n_tracks, n_components) from fit_unified_pca()

    Returns:
        (n_tracks, n_tracks) float32 distance matrix in [0, 1]
    """
    from scipy.spatial.distance import pdist, squareform

    n = reduced_vectors.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)

    # Euclidean distance on L2-normalized vectors: max possible = 2.0
    condensed = pdist(reduced_vectors, metric="euclidean")
    matrix = squareform(condensed).astype(np.float32)

    # Normalize by max possible distance for L2-normalized vectors
    matrix /= 2.0

    return _contrast_stretch(matrix)


def _contrast_stretch(matrix: NDArray) -> NDArray:
    """Contrast stretch a distance matrix to [0, 1] using 2nd-98th percentiles."""
    n = matrix.shape[0]
    upper = matrix[np.triu_indices(n, k=1)]
    if len(upper) > 0 and np.max(upper) > np.min(upper):
        d_min = float(np.percentile(upper, 2))
        d_max = float(np.percentile(upper, 98))
        if d_max > d_min:
            matrix = np.clip((matrix - d_min) / (d_max - d_min), 0.0, 1.0)
            np.fill_diagonal(matrix, 0.0)
            logger.info(
                "Distance matrix contrast stretched: raw [%.3f, %.3f] -> [0, 1]",
                d_min, d_max,
            )

    n_nan = int(np.sum(np.isnan(matrix)))
    if n_nan > 0:
        logger.warning("Distance matrix has %d NaN entries! Replacing with 1.0", n_nan)
        matrix = np.nan_to_num(matrix, nan=1.0)

    logger.info("Distance matrix computed: %dx%d", n, n)
    return matrix
