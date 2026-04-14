"""Analysis result cache for dj-tagger.

Cache key: filename + rounded duration (stable across tag edits, avoids
collisions when the same filename exists in different folders).
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump this when any analyzer logic changes to invalidate all cached results.
ANALYZER_VERSION = "5"


@dataclass
class TaggerCacheEntry:
    mtime: float
    version: str
    result: dict


TaggerCache = dict[str, TaggerCacheEntry]


def cache_key(filepath: str, duration: float | None = None) -> str:
    """Return the cache key for a file.

    Uses filename + rounded duration for robust matching.
    Falls back to filename-only if duration is not provided (backward compat).
    """
    name = Path(filepath).name
    if duration is not None and duration > 0:
        return f"{name}|{duration:.1f}"
    return name


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


def get_cached(
    cache: TaggerCache,
    filepath: str,
    mtime: float,
    duration: float | None = None,
) -> dict | None:
    """Look up a cached result. Tries duration-keyed first, falls back to name-only."""
    # Try new key format (filename|duration)
    if duration is not None and duration > 0:
        key = cache_key(filepath, duration)
        entry = cache.get(key)
        if entry and entry.version == ANALYZER_VERSION:
            return entry.result

    # Fall back to old key format (filename only) for backward compat
    key = cache_key(filepath)
    entry = cache.get(key)
    if entry is None:
        return None
    if entry.mtime != mtime:
        return None
    if entry.version != ANALYZER_VERSION:
        return None
    return entry.result


def put_cached(
    cache: TaggerCache,
    filepath: str,
    mtime: float,
    result: dict,
    duration: float | None = None,
) -> None:
    """Store a result in cache. Writes both key formats during migration."""
    entry = TaggerCacheEntry(mtime=mtime, version=ANALYZER_VERSION, result=result)
    # Always write duration-keyed entry if duration available
    if duration is not None and duration > 0:
        cache[cache_key(filepath, duration)] = entry
    # Also write name-only key for backward compat with dj-tagger
    cache[cache_key(filepath)] = entry
