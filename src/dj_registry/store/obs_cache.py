"""Persistent observation cache — delegates to universal cache.

All data is stored in cache/raw_cache.pkl alongside tagger and
grouper data. This ensures one single cache file for everything.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path

from ..models import SourceObservation

logger = logging.getLogger(__name__)

# Legacy path — kept for backward-compatible loading. New data goes to universal cache.
CACHE_PATH = os.path.join("cache", "registry_cache.pkl")
_CACHE_VERSION = "2"


def _make_key(filename: str, duration: float | None, source: str) -> str:
    """Cache key: filename|duration|source — path-independent."""
    name = Path(filename).name
    dur = f"{duration:.1f}" if duration and duration > 0 else "0"
    return f"{name}|{dur}|{source}"


def _make_key_by_isrc(isrc: str, source: str) -> str:
    """Cache key for ISRC-based lookups (Songstats, Spotify)."""
    return f"isrc:{isrc}|{source}"


class ObsCache:
    """Persistent pickle cache for source observations.

    Delegates to the universal cache when available, falls back to
    its own pickle file for backward compatibility.
    """

    def __init__(self, path: str = CACHE_PATH) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._ucache = None
        self._load()

    def _get_universal_cache(self):
        """Lazy-load the universal cache."""
        if self._ucache is None:
            try:
                from dj_tagger.universal_cache import get_cache
                cache_dir = os.path.dirname(self.path) or "cache"
                ucache_path = os.path.join(cache_dir, "raw_cache.pkl")
                self._ucache = get_cache(ucache_path)
            except ImportError:
                pass
        return self._ucache

    def _load(self) -> None:
        # Try universal cache first
        ucache = self._get_universal_cache()
        if ucache is not None:
            # Load registry entries from universal cache
            for key, entry in ucache._entries.items():
                # Registry keys contain source separators like "songstats", "rekordbox", etc.
                # or start with "isrc:"
                if ("|" in key and not key.endswith("|tagger") and
                    not key.endswith("|dsp") and not key.endswith("|section_dsp") and
                    not key.endswith("|clap")):
                    # Check if it looks like a registry key (has 3 parts or is ISRC)
                    parts = key.split("|")
                    if len(parts) >= 3 or key.startswith("isrc:"):
                        if isinstance(entry.data, dict):
                            self._entries[key] = entry.data
            if self._entries:
                logger.debug("ObsCache loaded %d entries from universal cache", len(self._entries))
                return

        # Fall back to legacy registry_cache.pkl
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            if data.get("_version") != _CACHE_VERSION:
                logger.debug("Obs cache version changed, starting fresh")
                return
            self._entries = data.get("entries", {})
            logger.debug("Obs cache loaded: %d entries", len(self._entries))
            # Migrate to universal cache
            self._migrate_to_universal()
        except Exception:
            logger.debug("Could not load obs cache, starting fresh")

    def _migrate_to_universal(self) -> None:
        """Push all entries to universal cache."""
        ucache = self._get_universal_cache()
        if ucache is None or not self._entries:
            return
        for key, data in self._entries.items():
            if not key.startswith("_"):
                ucache.put(key, data)
        ucache.save()
        logger.debug("Migrated %d registry entries to universal cache", len(self._entries))

    def save(self) -> None:
        if not self._dirty:
            return
        # Write to universal cache
        ucache = self._get_universal_cache()
        if ucache is not None:
            for key, data in self._entries.items():
                if not key.startswith("_"):
                    ucache.put(key, data)
            ucache.save()

        # Also write legacy file for backward compat
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"_version": _CACHE_VERSION, "entries": self._entries}, f)
        self._dirty = False

    def get_by_file(
        self, filename: str, duration: float | None, source: str
    ) -> SourceObservation | None:
        key = _make_key(filename, duration, source)
        data = self._entries.get(key)
        if data is None:
            # Try universal cache
            ucache = self._get_universal_cache()
            if ucache is not None:
                data = ucache.get(key)
                if data is not None and isinstance(data, dict):
                    self._entries[key] = data
        if data is None:
            return None
        return SourceObservation(**data)

    def put_by_file(
        self, filename: str, duration: float | None, source: str, obs: SourceObservation
    ) -> None:
        key = _make_key(filename, duration, source)
        self._entries[key] = asdict(obs)
        self._dirty = True

    def get_by_isrc(self, isrc: str, source: str) -> SourceObservation | None:
        key = _make_key_by_isrc(isrc, source)
        data = self._entries.get(key)
        if data is None:
            ucache = self._get_universal_cache()
            if ucache is not None:
                data = ucache.get(key)
                if data is not None and isinstance(data, dict):
                    self._entries[key] = data
        if data is None:
            return None
        return SourceObservation(**data)

    def put_by_isrc(self, isrc: str, source: str, obs: SourceObservation) -> None:
        key = _make_key_by_isrc(isrc, source)
        self._entries[key] = asdict(obs)
        self._dirty = True

    def has_isrc(self, isrc: str, source: str) -> bool:
        key = _make_key_by_isrc(isrc, source)
        if key in self._entries:
            return True
        ucache = self._get_universal_cache()
        if ucache is not None:
            return ucache.has(key)
        return False

    def __len__(self) -> int:
        return len(self._entries)


# Need pickle for legacy load
import pickle
