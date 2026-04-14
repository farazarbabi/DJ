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
DEFAULT_RAW_PATH = os.path.join("cache", "raw_cache.pkl")
DEFAULT_DERIVED_PATH = os.path.join("cache", "derived_cache.pkl")
# Legacy — kept for migration only
DEFAULT_CACHE_PATH = os.path.join("cache", "universal_cache.pkl")

# Raw layers: permanent, never invalidated. No version checks.
# These contain fixed algorithm outputs or external data.
RAW_LAYERS = frozenset({
    "dsp",            # ~45 DSP features (librosa fixed algorithms)
    "section_dsp",    # per-section DSP features
    "clap",           # 512-dim CLAP embedding (fixed model)
    "raw_analysis",   # intermediate features for re-derivation (bar_energies, vocal_ratio, etc.)
    "tag",            # file embedded tags
    "rekordbox",      # Rekordbox XML data
    "songstats",      # Songstats API response
    "spotify",        # Spotify lookup
    "analysis_librosa",   # registry librosa key analysis
    "analysis_essentia",  # registry essentia key analysis
})

def _get_derived_versions() -> dict[str, str]:
    """Get derived layer versions — auto-computed from settings.toml hash."""
    try:
        from .settings import derived_version
        ver = derived_version()
    except Exception:
        ver = "5.1"  # fallback if settings.toml not found
    return {"tagger": ver}


# Derived layer versions — auto-recomputed from settings.toml hash.
# No manual bumping needed: change any parameter in settings.toml and
# the hash changes, invalidating the derived cache.
DERIVED_VERSIONS: dict[str, str] = _get_derived_versions()

# Combined for backward compat with code that checks LAYER_VERSIONS
LAYER_VERSIONS: dict[str, str] = {
    **{layer: "1" for layer in RAW_LAYERS},
    **DERIVED_VERSIONS,
}


@dataclass
class CacheEntry:
    """Generic cache entry with version gating."""
    version: str
    mtime: float  # file modification time (0.0 for API data)
    data: object  # the actual payload (dict, ndarray, etc.)


class UniversalCache:
    """Two-file cache: raw_cache.pkl (permanent) + derived_cache.pkl (versioned).

    Externally behaves as one cache. Internally splits reads/writes by layer type.
    """

    def __init__(self, path: str = DEFAULT_CACHE_PATH) -> None:
        # path is kept for API compat — used to derive the cache directory
        cache_dir = os.path.dirname(path) or "cache"
        self.raw_path = os.path.join(cache_dir, "raw_cache.pkl")
        self.derived_path = os.path.join(cache_dir, "derived_cache.pkl")
        self.path = path  # legacy, for callers that reference it
        self._entries: dict[str, CacheEntry] = {}
        self._dirty_raw = False
        self._dirty_derived = False
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
        layer = self._layer_from_key(key)
        if layer in RAW_LAYERS:
            self._dirty_raw = True
        else:
            self._dirty_derived = True

    @staticmethod
    def _layer_from_key(key: str) -> str | None:
        """Extract layer name from a cache key."""
        if key.startswith("isrc:"):
            parts = key.split("|")
            return parts[-1] if len(parts) >= 2 else None
        parts = key.split("|")
        return parts[-1] if len(parts) >= 2 else None

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

        Raw layers (dsp, clap, raw_analysis, etc.) are never version-checked
        — they are permanent. Derived layers auto-check against DERIVED_VERSIONS.
        """
        if version is None and layer not in RAW_LAYERS:
            version = DERIVED_VERSIONS.get(layer)

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

        Auto-sets version: raw layers get "1" (permanent), derived layers
        get their version from DERIVED_VERSIONS.
        """
        if version is None:
            if layer in RAW_LAYERS:
                version = "1"
            else:
                version = DERIVED_VERSIONS.get(layer, "1")

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
        loaded = False

        # Load raw_cache.pkl
        if os.path.exists(self.raw_path):
            loaded |= self._load_file(self.raw_path)

        # Load derived_cache.pkl
        if os.path.exists(self.derived_path):
            loaded |= self._load_file(self.derived_path)

        # Fall back to legacy universal_cache.pkl
        if not loaded and os.path.exists(self.path) and self.path != self.raw_path:
            loaded = self._load_file(self.path)

        if not loaded:
            self._try_migrate()

    def _load_file(self, path: str) -> bool:
        """Load entries from a single pickle file. Returns True on success."""
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict) or raw.get("_version") != CACHE_VERSION:
                return False
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
            logger.info("Loaded %d entries from %s", len(entries), path)
            return bool(entries)
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)
            return False

    def save(self) -> None:
        """Save cache to disk atomically. Writes two files: raw + derived."""
        if self._dirty_raw:
            raw_entries = {k: v for k, v in self._entries.items()
                          if self._layer_from_key(k) in RAW_LAYERS}
            self._save_file(self.raw_path, raw_entries)
            self._dirty_raw = False

        if self._dirty_derived:
            derived_entries = {k: v for k, v in self._entries.items()
                              if self._layer_from_key(k) not in RAW_LAYERS}
            self._save_file(self.derived_path, derived_entries)
            self._dirty_derived = False

    def _save_file(self, path: str, entries: dict) -> None:
        """Atomically write entries to a single pickle file."""
        cache_dir = os.path.dirname(path) or "."
        os.makedirs(cache_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".pkl.tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump({"_version": CACHE_VERSION, "entries": entries}, f)
            os.replace(tmp_path, path)
            logger.info("Cache saved: %d entries to %s", len(entries), path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def force_save(self) -> None:
        """Save even if not dirty."""
        self._dirty_raw = True
        self._dirty_derived = True
        self.save()

    @property
    def dirty(self) -> bool:
        return self._dirty_raw or self._dirty_derived

    def __len__(self) -> int:
        return len(self._entries)

    # ── Migration from legacy cache files ────────────────────────────────

    def _try_migrate(self) -> None:
        """Auto-import data from legacy cache files."""
        cache_dir = os.path.dirname(self.raw_path) or "cache"
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

        # Also try legacy universal_cache.pkl
        legacy_universal = os.path.join(cache_dir, "universal_cache.pkl")
        if os.path.exists(legacy_universal):
            self._load_file(legacy_universal)
            migrated += 1  # count as migrated to trigger save

        if migrated > 0:
            logger.info("Migrated %d entries from legacy caches", migrated)
            self._dirty_raw = True
            self._dirty_derived = True
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
                # Use current derived version so migrated entries aren't immediately stale
                self._entries[tagger_key] = CacheEntry(
                    version=DERIVED_VERSIONS.get("tagger", "5.1"),
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
