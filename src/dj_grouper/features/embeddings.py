"""CLAP audio embedding extraction with per-file caching and PCA reduction."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_clap_model = None

CLAP_BATCH_SIZE = 20  # save cache after every batch


# ─── CLAP availability ──────────────────────────────────────────────────────

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


# ─── Per-file CLAP cache ────────────────────────────────────────────────────

ClapCache = dict[str, NDArray]  # path -> 512-dim embedding


def load_clap_cache(path: str) -> ClapCache:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            logger.info("Loaded CLAP cache from %s (%d entries)", path, len(cache))
            return cache
        return {}
    except Exception:
        logger.warning("Could not load CLAP cache at %s", path)
        return {}


def save_clap_cache(cache: ClapCache, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("CLAP cache saved to %s (%d entries)", path, len(cache))


def extract_clap_incremental(
    file_paths: list[str],
    cache_path: str,
    force: bool = False,
) -> NDArray[np.floating]:
    """Extract CLAP embeddings incrementally with per-file caching.

    Only extracts embeddings for files not already in cache.
    Saves cache after every batch of CLAP_BATCH_SIZE tracks.

    Returns array of shape (n_files, 512) aligned with file_paths order.
    """
    cache = {} if force else load_clap_cache(cache_path)

    # Find which files need extraction
    to_extract: list[tuple[int, str]] = []
    for i, path in enumerate(file_paths):
        if path not in cache:
            to_extract.append((i, path))

    if to_extract:
        model = _get_clap_model()
        n_total = len(to_extract)

        for batch_start in range(0, n_total, CLAP_BATCH_SIZE):
            batch = to_extract[batch_start:batch_start + CLAP_BATCH_SIZE]
            batch_paths = [p for _, p in batch]
            batch_end = min(batch_start + CLAP_BATCH_SIZE, n_total)

            print(f"    CLAP batch {batch_start + 1}-{batch_end}/{n_total}", flush=True)

            try:
                embeddings = model.get_audio_embedding_from_filelist(
                    x=batch_paths, use_tensor=False,
                )
                embeddings = np.array(embeddings, dtype=np.float32)

                for j, (idx, path) in enumerate(batch):
                    cache[path] = embeddings[j]

                # Save after each batch so crashes don't lose all progress
                save_clap_cache(cache, cache_path)
            except Exception as e:
                logger.warning("CLAP batch failed: %s", e)
                print(f"    CLAP batch failed: {e}")
    else:
        print(f"    CLAP: all {len(file_paths)} tracks cached")

    # Remove deleted files from cache
    current_set = set(file_paths)
    removed = [p for p in cache if p not in current_set]
    for p in removed:
        del cache[p]
    if removed:
        save_clap_cache(cache, cache_path)

    # Build output array aligned with file_paths
    result = np.zeros((len(file_paths), 512), dtype=np.float32)
    for i, path in enumerate(file_paths):
        if path in cache:
            result[i] = cache[path]

    return result


# ─── PCA ─────────────────────────────────────────────────────────────────────

def fit_pca(embeddings: NDArray, n_components: int = 64) -> tuple[NDArray, dict]:
    """Fit PCA on embeddings and return reduced embeddings + PCA params."""
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
