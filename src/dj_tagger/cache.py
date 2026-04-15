"""Analysis result cache for dj-tagger.

Delegates to raw cache (cache/raw_cache.pkl) and derived cache
(cache/derived_cache.pkl). All analysis results are stored once and
shared across all modules. Once analyzed, never re-analyzed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Auto-computed from settings.toml — no manual bumping needed.
def _get_analyzer_version() -> str:
    try:
        from .settings import derived_version
        return derived_version()
    except Exception:
        return "5.1"


ANALYZER_VERSION = _get_analyzer_version()


@dataclass
class TaggerCacheEntry:
    mtime: float
    version: str
    result: dict


TaggerCache = dict[str, TaggerCacheEntry]


def cache_key(filepath: str, duration: float | None = None) -> str:
    """Return the cache key for a file: filename + rounded duration."""
    name = Path(filepath).name
    if duration is not None and duration > 0:
        return f"{name}|{duration:.1f}"
    return name


def load_cache(path: str) -> TaggerCache:
    """Load tagger cache from raw + derived cache files."""
    from .universal_cache import get_cache

    ucache = get_cache(_universal_path(path))
    # Build a TaggerCache dict view from cache entries
    result: TaggerCache = {}
    for key, entry in ucache._entries.items():
        if not key.endswith("|tagger"):
            continue
        base_key = key[: -len("|tagger")]
        result[base_key] = TaggerCacheEntry(
            mtime=entry.mtime,
            version=entry.version,
            result=entry.data,
        )
    logger.info("Loaded tagger layer from derived cache (%d entries)", len(result))
    return result


def save_cache(cache: TaggerCache, path: str) -> None:
    """Save tagger cache to raw + derived cache files."""
    from .universal_cache import get_cache, DERIVED_VERSIONS

    ucache = get_cache(_universal_path(path))
    derived_ver = DERIVED_VERSIONS.get("tagger", "5.1")
    for base_key, entry in cache.items():
        ukey = f"{base_key}|tagger"
        ucache.put(ukey, entry.result, version=derived_ver, mtime=entry.mtime)
    ucache.save()


def get_cached(
    cache: TaggerCache,
    filepath: str,
    mtime: float,
    duration: float | None = None,
) -> dict | None:
    """Look up a cached result by filename + duration."""
    key = cache_key(filepath, duration)
    entry = cache.get(key)
    if entry is None:
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
    """Store a result in cache by filename + duration."""
    entry = TaggerCacheEntry(mtime=mtime, version=ANALYZER_VERSION, result=result)
    cache[cache_key(filepath, duration)] = entry


def _universal_path(tagger_path: str) -> str:
    """Derive cache path from tagger cache path (used to find the cache directory)."""
    parent = str(Path(tagger_path).parent)
    return str(Path(parent) / "raw_cache.pkl")
