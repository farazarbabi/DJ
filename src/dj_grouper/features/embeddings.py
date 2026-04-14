"""CLAP audio embedding extraction with per-file caching and PCA reduction."""

from __future__ import annotations

import logging
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
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        _clap_model.load_ckpt()
        logger.info("CLAP model loaded")
    return _clap_model


# ─── CLAP cache (via universal cache) ───────────────────────────────────────


def extract_clap_incremental(
    file_paths: list[str],
    cache_path: str,
    force: bool = False,
) -> NDArray[np.floating]:
    """Extract CLAP embeddings incrementally using the universal cache.

    Only extracts embeddings for files not already cached.
    Saves to universal cache after every batch.

    Returns array of shape (n_files, 512) aligned with file_paths order.
    """
    from dj_tagger.universal_cache import get_cache, quick_duration
    ucache_path = str(Path(cache_path).parent / "raw_cache.pkl")
    ucache = get_cache(ucache_path)

    # Build in-memory cache from universal cache + legacy migration
    cache: dict[str, NDArray] = {}

    if not force:
        # Load from universal cache
        for fpath in file_paths:
            filename = Path(fpath).name
            dur = quick_duration(fpath)
            embedding = ucache.get_track(filename, dur, "clap")
            if embedding is not None:
                cache[fpath] = embedding

        # Migrate from legacy clap_cache.pkl if it exists
        legacy_path = Path(cache_path)
        if legacy_path.exists():
            try:
                with open(legacy_path, "rb") as f:
                    legacy = pickle.load(f)
                if isinstance(legacy, dict):
                    migrated = 0
                    for fpath, emb in legacy.items():
                        if fpath not in cache:
                            cache[fpath] = emb
                            ucache.put_track(Path(fpath).name, quick_duration(fpath), "clap", emb)
                            migrated += 1
                    if migrated:
                        logger.info("Migrated %d entries from legacy CLAP cache", migrated)
                        ucache.save()
            except Exception:
                pass

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

            pct = batch_end * 100 // n_total
            print(f"    [{batch_end}/{n_total}] {pct:>3}%  extracting CLAP embeddings...", flush=True)

            try:
                embeddings = model.get_audio_embedding_from_filelist(
                    x=batch_paths, use_tensor=False,
                )
                embeddings = np.array(embeddings, dtype=np.float32)

                for j, (idx, path) in enumerate(batch):
                    cache[path] = embeddings[j]
                    ucache.put_track(Path(path).name, quick_duration(path), "clap", embeddings[j])

                ucache.save()
            except Exception as e:
                logger.warning("CLAP batch failed: %s", e)
                print(f"    CLAP batch {batch_start + 1}-{batch_end} failed: {e}")
    else:
        print(f"    All {len(file_paths)} CLAP embeddings cached")

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
