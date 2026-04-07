"""Analysis result cache for dj-tagger."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump this when any analyzer logic changes to invalidate all cached results.
ANALYZER_VERSION = "3"


@dataclass
class TaggerCacheEntry:
    mtime: float
    version: str
    result: dict


TaggerCache = dict[str, TaggerCacheEntry]


def cache_key(filepath: str) -> str:
    """Return the cache key for a file (filename with extension)."""
    return Path(filepath).name


def load_cache(path: str) -> TaggerCache:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            first_val = next(iter(data.values()), None) if data else None
            if first_val is None or isinstance(first_val, TaggerCacheEntry):
                logger.info("Loaded tagger cache from %s (%d entries)", path, len(data))
                return data
        logger.warning("Cache at %s has unrecognized format, starting fresh", path)
        return {}
    except Exception:
        logger.warning("Could not load cache at %s, starting fresh", path)
        return {}


def save_cache(cache: TaggerCache, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Tagger cache saved to %s (%d entries)", path, len(cache))


def get_cached(cache: TaggerCache, filepath: str, mtime: float) -> dict | None:
    key = cache_key(filepath)
    entry = cache.get(key)
    if entry is None:
        return None
    if entry.mtime != mtime:
        return None
    if entry.version != ANALYZER_VERSION:
        return None
    return entry.result


def put_cached(cache: TaggerCache, filepath: str, mtime: float, result: dict) -> None:
    key = cache_key(filepath)
    cache[key] = TaggerCacheEntry(mtime=mtime, version=ANALYZER_VERSION, result=result)
