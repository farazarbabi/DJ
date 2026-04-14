"""Persistent observation cache — survives registry resets.

Stores all collected data per (filename, source) so APIs and file parsing
are never repeated for unchanged files. Keyed by (filename|duration, source_system).
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import asdict
from pathlib import Path

from ..models import SourceObservation

logger = logging.getLogger(__name__)

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
    """Persistent pickle cache for source observations."""

    def __init__(self, path: str = CACHE_PATH) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
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
        except Exception:
            logger.debug("Could not load obs cache, starting fresh")

    def save(self) -> None:
        if not self._dirty:
            return
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
            return None
        return SourceObservation(**data)

    def put_by_isrc(self, isrc: str, source: str, obs: SourceObservation) -> None:
        key = _make_key_by_isrc(isrc, source)
        self._entries[key] = asdict(obs)
        self._dirty = True

    def has_isrc(self, isrc: str, source: str) -> bool:
        return _make_key_by_isrc(isrc, source) in self._entries

    def __len__(self) -> int:
        return len(self._entries)
