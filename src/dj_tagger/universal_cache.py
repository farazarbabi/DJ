"""Universal cache: single file for all analysis, features, and API data.

Every module (dj-tagger, dj-grouper, dj-registry) reads and writes here.
Once a track is analyzed, it is never re-analyzed — any module can reuse
results from any other module.

Cache file: cache/universal_cache.pkl

Key format:
  - Track data:  "{filename}|{duration:.1f}|{layer}"
  - ISRC-based API data: "isrc:{isrc}|{layer}"

Layers:
  - "tagger"      — analysis results (energy, key, vibe, vocal, structure, bpm, tag)
  - "dsp"         — ~45 DSP features dict
  - "section_dsp" — per-section DSP features dict
  - "clap"        — 512-dim CLAP embedding (NDArray)
  - "songstats"   — Songstats API observation
  - "rekordbox"   — Rekordbox XML observation
  - "tag"         — file tag observation
  - "spotify"     — Spotify lookup observation
  - "analysis_librosa"  — registry librosa key analysis
  - "analysis_essentia" — registry essentia key analysis
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_VERSION = "1"
DEFAULT_CACHE_PATH = os.path.join("cache", "universal_cache.pkl")

# Per-layer version constants. Bump these when extraction logic changes
# to automatically invalidate stale cached data for that layer only.
LAYER_VERSIONS: dict[str, str] = {
    "tagger": "5",       # matches ANALYZER_VERSION in cache.py
    "dsp": "1",          # bump when DSP extraction changes
    "section_dsp": "1",  # bump when section DSP changes
    "clap": "1",         # bump when CLAP model changes
}


@dataclass
class CacheEntry:
    """Generic cache entry with version gating."""
    version: str
    mtime: float  # file modification time (0.0 for API data)
    data: object  # the actual payload (dict, ndarray, etc.)


class UniversalCache:
    """Single-file cache for all DJ toolkit data."""

    def __init__(self, path: str = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        self._dirty = False
        self._load()

    # ── Key construction ─────────────────────────────────────────────────

    @staticmethod
    def track_key(filename: str, duration: float | None, layer: str) -> str:
        """Build a track-scoped cache key."""
        name = Path(filename).name
        if duration is not None and duration > 0:
            return f"{name}|{duration:.1f}|{layer}"
        return f"{name}|{layer}"

    @staticmethod
    def isrc_key(isrc: str, layer: str) -> str:
        """Build an ISRC-scoped cache key (for API lookups)."""
        return f"isrc:{isrc}|{layer}"

    # ── Generic get/put ──────────────────────────────────────────────────

    def get(self, key: str, version: str | None = None) -> object | None:
        """Get cached data by key. Returns None on miss or version mismatch."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if version is not None and entry.version != version:
            return None
        return entry.data

    def put(self, key: str, data: object, version: str = "1", mtime: float = 0.0) -> None:
        """Store data in cache."""
        self._entries[key] = CacheEntry(version=version, mtime=mtime, data=data)
        self._dirty = True

    def has(self, key: str) -> bool:
        return key in self._entries

    # ── Track-scoped convenience methods ─────────────────────────────────

    def get_track(
        self,
        filename: str,
        duration: float | None,
        layer: str,
        version: str | None = None,
    ) -> object | None:
        """Look up track data.

        Tries duration-keyed first. Falls back to name-only for backward
        compat with entries written before duration was required.

        Auto-checks layer version from LAYER_VERSIONS if version is not
        explicitly provided.
        """
        if version is None:
            version = LAYER_VERSIONS.get(layer)

        # Try with duration
        if duration is not None and duration > 0:
            key = self.track_key(filename, duration, layer)
            result = self.get(key, version)
            if result is not None:
                return result

        # Fall back to name-only (legacy entries)
        key = self.track_key(filename, None, layer)
        result = self.get(key, version)
        if result is not None:
            logger.debug("Cache hit on name-only key for %s|%s (legacy entry)", Path(filename).name, layer)
        return result

    def put_track(
        self,
        filename: str,
        duration: float | None,
        layer: str,
        data: object,
        version: str | None = None,
        mtime: float = 0.0,
    ) -> None:
        """Store track data.

        Writes only the duration-keyed entry when duration is available.
        Falls back to name-only key only when duration is unknown.

        Auto-sets version from LAYER_VERSIONS if not explicitly provided.
        """
        if version is None:
            version = LAYER_VERSIONS.get(layer, "1")

        key = self.track_key(filename, duration, layer)
        self.put(key, data, version, mtime)

    # ── ISRC-scoped convenience methods ──────────────────────────────────

    def get_isrc(self, isrc: str, layer: str) -> object | None:
        key = self.isrc_key(isrc, layer)
        return self.get(key)

    def put_isrc(self, isrc: str, layer: str, data: object) -> None:
        key = self.isrc_key(isrc, layer)
        self.put(key, data)

    def has_isrc(self, isrc: str, layer: str) -> bool:
        return self.has(self.isrc_key(isrc, layer))

    # ── Load / Save ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._try_migrate()
            return
        try:
            with open(self.path, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict) or raw.get("_version") != CACHE_VERSION:
                logger.info("Universal cache version mismatch, migrating...")
                self._try_migrate()
                return
            entries = raw.get("entries", {})
            for k, v in entries.items():
                if isinstance(v, CacheEntry):
                    self._entries[k] = v
                elif isinstance(v, dict) and "data" in v:
                    self._entries[k] = CacheEntry(
                        version=v.get("version", "1"),
                        mtime=v.get("mtime", 0.0),
                        data=v["data"],
                    )
            logger.info("Universal cache loaded: %d entries from %s", len(self._entries), self.path)
        except Exception as e:
            logger.warning("Could not load universal cache: %s — migrating from legacy", e)
            self._try_migrate()

    def save(self) -> None:
        """Save cache to disk atomically (temp file + os.replace)."""
        if not self._dirty:
            return
        cache_dir = os.path.dirname(self.path) or "."
        os.makedirs(cache_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".pkl.tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump({"_version": CACHE_VERSION, "entries": self._entries}, f)
            os.replace(tmp_path, self.path)
            self._dirty = False
            logger.info("Universal cache saved: %d entries to %s", len(self._entries), self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def force_save(self) -> None:
        """Save even if not dirty."""
        self._dirty = True
        self.save()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __len__(self) -> int:
        return len(self._entries)

    # ── Migration from legacy cache files ────────────────────────────────

    def _try_migrate(self) -> None:
        """Auto-import data from legacy cache files."""
        cache_dir = os.path.dirname(self.path) or "cache"
        migrated = 0

        tagger_path = os.path.join(cache_dir, "tagger_cache.pkl")
        if os.path.exists(tagger_path):
            migrated += self._migrate_tagger(tagger_path)

        features_path = os.path.join(cache_dir, "features_cache.pkl")
        if os.path.exists(features_path):
            migrated += self._migrate_grouper(features_path)

        clap_path = os.path.join(cache_dir, "clap_cache.pkl")
        if os.path.exists(clap_path):
            migrated += self._migrate_clap(clap_path)

        registry_path = os.path.join(cache_dir, "registry_cache.pkl")
        if os.path.exists(registry_path):
            migrated += self._migrate_registry(registry_path)

        if migrated > 0:
            logger.info("Migrated %d entries from legacy caches", migrated)
            self._dirty = True
            self.save()

    def _migrate_tagger(self, path: str) -> int:
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                return 0
            count = 0
            for key, entry in cache.items():
                if not hasattr(entry, "result"):
                    continue
                tagger_key = f"{key}|tagger"
                self._entries[tagger_key] = CacheEntry(
                    version=getattr(entry, "version", "5"),
                    mtime=getattr(entry, "mtime", 0.0),
                    data=entry.result,
                )
                count += 1
            logger.info("Migrated %d entries from tagger cache", count)
            return count
        except Exception as e:
            logger.warning("Failed to migrate tagger cache: %s", e)
            return 0

    def _migrate_grouper(self, path: str) -> int:
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                return 0
            count = 0
            for filepath, entry in cache.items():
                if not hasattr(entry, "dsp"):
                    continue
                filename = Path(filepath).name
                mtime = getattr(entry, "mtime", 0.0)
                if entry.dsp:
                    dsp_key = f"{filename}|dsp"
                    self._entries[dsp_key] = CacheEntry(version="1", mtime=mtime, data=entry.dsp)
                    count += 1
                section_dsp = getattr(entry, "section_dsp", None)
                if section_dsp:
                    sdsp_key = f"{filename}|section_dsp"
                    self._entries[sdsp_key] = CacheEntry(version="1", mtime=mtime, data=section_dsp)
                    count += 1
            logger.info("Migrated %d entries from grouper cache", count)
            return count
        except Exception as e:
            logger.warning("Failed to migrate grouper cache: %s", e)
            return 0

    def _migrate_clap(self, path: str) -> int:
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                return 0
            count = 0
            for filepath, embedding in cache.items():
                filename = Path(filepath).name
                clap_key = f"{filename}|clap"
                self._entries[clap_key] = CacheEntry(version="1", mtime=0.0, data=embedding)
                count += 1
            logger.info("Migrated %d entries from CLAP cache", count)
            return count
        except Exception as e:
            logger.warning("Failed to migrate CLAP cache: %s", e)
            return 0

    def _migrate_registry(self, path: str) -> int:
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                return 0
            entries = raw.get("entries", raw)
            count = 0
            for key, data in entries.items():
                if key.startswith("_"):
                    continue
                self._entries[key] = CacheEntry(version="1", mtime=0.0, data=data)
                count += 1
            logger.info("Migrated %d entries from registry cache", count)
            return count
        except Exception as e:
            logger.warning("Failed to migrate registry cache: %s", e)
            return 0


# ── Utility: quick duration ──────────────────────────────────────────────────

def quick_duration(path: str) -> float | None:
    """Get audio duration cheaply via mutagen (no audio decoding)."""
    try:
        import mutagen
        m = mutagen.File(path)
        if m and m.info:
            return getattr(m.info, "length", None)
    except Exception:
        pass
    return None


# ── Module-level singleton ───────────────────────────────────────────────────

_instance: UniversalCache | None = None


def get_cache(path: str = DEFAULT_CACHE_PATH) -> UniversalCache:
    """Get the singleton UniversalCache instance."""
    global _instance
    if _instance is None or _instance.path != path:
        _instance = UniversalCache(path)
    return _instance


def reset_cache() -> None:
    """Reset the singleton (for tests)."""
    global _instance
    _instance = None
