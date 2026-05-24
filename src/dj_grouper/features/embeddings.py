"""CLAP audio embedding extraction with per-file caching and PCA reduction."""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_clap_model = None

CLAP_BATCH_SIZE = 20  # save cache after every batch

# Project-local model cache. laion_clap by default downloads to its package
# directory (inside site-packages) which is invisible in the project tree;
# we mirror the weight file here so it's discoverable and survives venv
# rebuilds.
_CLAP_WEIGHT_FILENAME = "630k-audioset-best.pt"
_CLAP_LOCAL_DIR = Path("cache") / "models"
_CLAP_LOCAL_PATH = _CLAP_LOCAL_DIR / _CLAP_WEIGHT_FILENAME


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
    """Lazy-load the CLAP HTSAT-tiny model, caching weights under cache/models/.

    First call: loads from cache/models/630k-audioset-best.pt if present,
    otherwise lets laion_clap download to its package dir and mirrors the
    weight file into cache/models/ for next time.

    laion_clap's loader is verbose (one line per parameter tensor and a few
    unconditional banner prints). We suppress all of that and emit a single
    info line summarizing where the weight came from.
    """
    global _clap_model
    if _clap_model is not None:
        return _clap_model

    # Set env vars BEFORE importing laion_clap; huggingface_hub checks these
    # on its own import and prints an HF-auth warning when missing.
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    explicit_ckpt = str(_CLAP_LOCAL_PATH) if _CLAP_LOCAL_PATH.exists() else None
    t0 = time.time()

    # laion_clap.load_ckpt prints unconditionally even with verbose=False; the
    # verbose flag only gates per-parameter status lines. transformers prints
    # a missing/unexpected-keys table directly to stderr. huggingface_hub
    # prints an auth-warning at import time. Wrap the entire import-and-load
    # block in stdout+stderr capture so the populate script's output stays
    # readable. Errors during load propagate via exception, not stderr.
    silent_out = io.StringIO()
    silent_err = io.StringIO()
    with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(silent_err):
        import laion_clap
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            import huggingface_hub
            huggingface_hub.logging.set_verbosity_error()
        except Exception:
            pass
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        _clap_model.load_ckpt(ckpt=explicit_ckpt, verbose=False)

    elapsed = time.time() - t0

    # If we just downloaded, copy the weight from laion_clap's package dir into
    # cache/models/ so subsequent runs read from the project-local path.
    if explicit_ckpt is None:
        try:
            pkg_dir = Path(laion_clap.__file__).resolve().parent
            src_path = pkg_dir / _CLAP_WEIGHT_FILENAME
            if src_path.exists():
                _CLAP_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                if not _CLAP_LOCAL_PATH.exists():
                    shutil.copy2(src_path, _CLAP_LOCAL_PATH)
        except Exception as exc:
            logger.debug("Could not mirror CLAP weight to %s: %s", _CLAP_LOCAL_PATH, exc)

    location = str(_CLAP_LOCAL_PATH) if _CLAP_LOCAL_PATH.exists() else "package default"
    logger.info("CLAP HTSAT-tiny ready (%.1fs, %s)", elapsed, location)
    return _clap_model


# ─── CLAP cache (via raw cache) ─────────────────────────────────────────────


def extract_clap_incremental(
    file_paths: list[str],
    cache_path: str,
    force: bool = False,
) -> NDArray[np.floating]:
    """Extract CLAP embeddings incrementally using the raw cache.

    Only extracts embeddings for files not already cached.
    Saves to raw cache after every batch.

    Returns array of shape (n_files, 512) aligned with file_paths order.
    """
    from dj_tagger.universal_cache import get_cache, quick_duration
    ucache_path = str(Path(cache_path).parent / "raw_cache.pkl")
    ucache = get_cache(ucache_path)

    # Build in-memory cache from raw cache + legacy migration
    cache: dict[str, NDArray] = {}

    if not force:
        # Load from raw cache
        for fpath in file_paths:
            filename = Path(fpath).name
            dur = quick_duration(fpath)
            embedding = ucache.get_track(filename, dur, "clap")
            if embedding is not None:
                cache[fpath] = embedding

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
