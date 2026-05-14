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
_CURRENT_UCACHE_PATH: str | None = None

# Auto-computed from settings.toml — no manual bumping needed.
def _get_analyzer_version() -> str:
    try:
        from .settings import tagger_version
        return tagger_version()
    except Exception:
        return "5.1"


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

    global _CURRENT_UCACHE_PATH
    _CURRENT_UCACHE_PATH = _universal_path(path)
    ucache = get_cache(_CURRENT_UCACHE_PATH)
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
    from .tagger_cache import hydrate_tagger_result
    from .universal_cache import get_cache

    global _CURRENT_UCACHE_PATH
    _CURRENT_UCACHE_PATH = _universal_path(path)
    ucache = get_cache(_CURRENT_UCACHE_PATH)
    derived_ver = _get_analyzer_version()
    for base_key, entry in cache.items():
        ukey = f"{base_key}|tagger"
        ucache.put(ukey, hydrate_tagger_result(entry.result), version=derived_ver, mtime=entry.mtime)
    ucache.save()


def get_cached(
    cache: TaggerCache,
    filepath: str,
    mtime: float,
    duration: float | None = None,
) -> dict | None:
    """Look up a cached result by filename + duration.

    Collected analysis facts are identity-keyed. A downstream version/signature
    change may trigger re-derivation from cached raw layers, but it must not
    force audio re-analysis when a filename + duration cache identity exists.
    """
    key = cache_key(filepath, duration)
    entry = cache.get(key)
    if entry is not None:
        if entry.version == _get_analyzer_version():
            return entry.result
        rederived = _rederive_from_raw(filepath, duration, entry.result)
        if rederived is not None:
            entry.result = rederived
            entry.version = _get_analyzer_version()
            return rederived
        return entry.result

    return _rederive_from_raw(filepath, duration, None)


def put_cached(
    cache: TaggerCache,
    filepath: str,
    mtime: float,
    result: dict,
    duration: float | None = None,
) -> None:
    """Store a result in cache by filename + duration."""
    from .tagger_cache import hydrate_tagger_result

    entry = TaggerCacheEntry(mtime=mtime, version=_get_analyzer_version(), result=hydrate_tagger_result(result))
    cache[cache_key(filepath, duration)] = entry


def _universal_path(tagger_path: str) -> str:
    """Derive cache path from tagger cache path (used to find the cache directory)."""
    parent = str(Path(tagger_path).parent)
    return str(Path(parent) / "raw_cache.pkl")


def _rederive_from_raw(filepath: str, duration: float | None, existing: dict | None) -> dict | None:
    """Rebuild derived tagger fields from cached raw layers without audio I/O."""
    try:
        from .derive import derive_all
        from .tagger_cache import merge_rederived_tagger
        from .universal_cache import get_cache
    except Exception:
        return None

    ucache = get_cache(_CURRENT_UCACHE_PATH) if _CURRENT_UCACHE_PATH else get_cache()
    filename = Path(filepath).name
    dsp = ucache.get_track(filename, duration, "dsp")
    raw_analysis = ucache.get_track(filename, duration, "raw_analysis")
    if not isinstance(dsp, dict) or not isinstance(raw_analysis, dict):
        return None
    try:
        return merge_rederived_tagger(existing, derive_all(dsp, raw_analysis))
    except Exception:
        logger.warning("Could not rederive cached tagger result for %s", filepath, exc_info=True)
        return None
