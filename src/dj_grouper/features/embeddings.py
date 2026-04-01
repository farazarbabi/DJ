"""CLAP audio embedding extraction with PCA reduction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_clap_model = None


def is_clap_available() -> bool:
    """Check if CLAP dependencies are installed."""
    try:
        import torch  # noqa: F401
        import laion_clap  # noqa: F401
        return True
    except ImportError:
        return False


def _get_clap_model():
    """Lazy-load the CLAP model."""
    global _clap_model
    if _clap_model is None:
        import laion_clap
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        _clap_model.load_ckpt()
        logger.info("CLAP model loaded")
    return _clap_model


def extract_clap_embeddings(file_paths: list[str]) -> NDArray[np.floating]:
    """Extract CLAP embeddings for a list of audio files.

    Returns array of shape (n_files, 512).
    """
    model = _get_clap_model()
    embeddings = model.get_audio_embedding_from_filelist(
        x=file_paths, use_tensor=False,
    )
    return np.array(embeddings, dtype=np.float32)


def fit_pca(embeddings: NDArray, n_components: int = 64) -> tuple[NDArray, dict]:
    """Fit PCA on embeddings and return reduced embeddings + PCA params.

    Returns (reduced_embeddings, pca_params) where pca_params can be saved/loaded.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, embeddings.shape[0], embeddings.shape[1]))
    reduced = pca.fit_transform(embeddings)
    pca_params = {
        "mean": pca.mean_,
        "components": pca.components_,
        "n_components": pca.n_components_,
    }
    logger.info(
        "PCA: %d -> %d dims (%.1f%% variance explained)",
        embeddings.shape[1], pca.n_components_,
        sum(pca.explained_variance_ratio_) * 100,
    )
    return reduced.astype(np.float32), pca_params


def apply_pca(embeddings: NDArray, pca_params: dict) -> NDArray:
    """Apply saved PCA transform to new embeddings."""
    centered = embeddings - pca_params["mean"]
    return (centered @ pca_params["components"].T).astype(np.float32)


def save_pca_params(pca_params: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(pca_params, f)


def load_pca_params(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
