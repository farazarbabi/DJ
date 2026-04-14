"""PCA fusion for unified feature vectors."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def fit_unified_pca(
    tracks,
    target_variance: float = 0.90,
) -> tuple[NDArray, dict]:
    """Fit PCA on unified vectors and return reduced + PCA params.

    Steps:
      1. Stack unified_vectors from all tracks
      2. Standardize (z-score per dimension)
      3. PCA to target_variance explained
      4. L2-normalize rows

    Args:
        tracks: list of TrackFeatures with unified_vector set
        target_variance: cumulative explained variance ratio target (default 0.90)

    Returns:
        (reduced_matrix, pca_params) where reduced_matrix is (n_tracks, n_components)
        and pca_params contains mean, std, components, etc. for apply_unified_pca.
    """
    from sklearn.decomposition import PCA

    # Stack unified vectors
    vectors = np.array([t.unified_vector for t in tracks], dtype=np.float32)
    n_samples, n_features = vectors.shape

    # Standardize (z-score per dimension)
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    standardized = (vectors - mean) / std

    # PCA: determine n_components to reach target variance
    max_components = min(n_samples, n_features)
    pca = PCA(n_components=max_components)
    pca.fit(standardized)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, target_variance) + 1)
    n_components = max(2, min(n_components, max_components))

    # Re-fit with the selected number of components
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(standardized)

    # L2-normalize rows
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    reduced = reduced / norms

    pca_params = {
        "mean": mean,
        "std": std,
        "pca_mean": pca.mean_,
        "components": pca.components_,
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }

    total_var = float(np.sum(pca.explained_variance_ratio_) * 100)
    logger.info(
        "Unified PCA: %d -> %d dims (%.1f%% variance explained)",
        n_features, n_components, total_var,
    )
    return reduced.astype(np.float32), pca_params


def apply_unified_pca(
    tracks,
    pca_params: dict,
) -> NDArray:
    """Apply saved PCA transform to track unified vectors.

    Args:
        tracks: list of TrackFeatures with unified_vector set
        pca_params: dict from fit_unified_pca

    Returns:
        L2-normalized PCA-reduced matrix (n_tracks, n_components)
    """
    vectors = np.array([t.unified_vector for t in tracks], dtype=np.float32)

    # Standardize with saved params
    standardized = (vectors - pca_params["mean"]) / pca_params["std"]

    # Apply PCA
    centered = standardized - pca_params["pca_mean"]
    reduced = (centered @ pca_params["components"].T).astype(np.float32)

    # L2-normalize
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    reduced = reduced / norms

    return reduced
