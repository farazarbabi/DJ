"""Cache manager: two files (raw + derived) for all analysis, features, and API data.

Every module (dj-tagger, dj-grouper, dj-registry) reads and writes here.
Once a track is analyzed, it is never re-analyzed — any module can reuse
results from any other module.

Two cache files:
  - cache/raw_cache.pkl     — raw data (DSP, CLAP, API results; some layers versioned)
  - cache/derived_cache.pkl — versioned derived data (energy, vibe, vocal, structure)

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

# Raw layers are stored in raw_cache.pkl. Some raw layers are versioned
# automatically so feature-extraction changes invalidate stale entries.
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
VERSIONED_RAW_LAYERS = frozenset({
    "dsp",
    "section_dsp",
    "raw_analysis",
})


def _get_raw_layer_versions() -> dict[str, str]:
    """Get raw layer versions for feature extraction outputs."""
    try:
        from .settings import dsp_version, raw_analysis_version, section_dsp_version
    except Exception:
        return {layer: "1" for layer in VERSIONED_RAW_LAYERS}
    return {
        "dsp": dsp_version(),
        "section_dsp": section_dsp_version(),
        "raw_analysis": raw_analysis_version(),
    }

def _get_derived_versions() -> dict[str, str]:
    """Get derived layer versions — auto-computed from settings.toml hash."""
    try:
        from .settings import tagger_version
        ver = tagger_version()
    except Exception:
        ver = "5.1"  # fallback if settings.toml not found
    return {"tagger": ver}


# Derived layer versions — auto-recomputed from settings.toml hash.
# No manual bumping needed: change any parameter in settings.toml and
# the hash changes, invalidating the derived cache.
RAW_LAYER_VERSIONS: dict[str, str] = _get_raw_layer_versions()
DERIVED_VERSIONS: dict[str, str] = _get_derived_versions()

# Combined for backward compat with code that checks LAYER_VERSIONS
LAYER_VERSIONS: dict[str, str] = {}


def refresh_layer_versions() -> dict[str, str]:
    """Refresh layer versions after settings or code changes."""
    global RAW_LAYER_VERSIONS, DERIVED_VERSIONS, LAYER_VERSIONS
    RAW_LAYER_VERSIONS = _get_raw_layer_versions()
    DERIVED_VERSIONS = _get_derived_versions()
    LAYER_VERSIONS = {
        **{layer: "1" for layer in RAW_LAYERS},
        **RAW_LAYER_VERSIONS,
        **DERIVED_VERSIONS,
    }
    return LAYER_VERSIONS


refresh_layer_versions()


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
        """Store data in cache. Overwrites if new data is richer or version differs."""
        existing = self._entries.get(key)
        if existing is not None and existing.version == version:
            # Same version — skip only if new data is not richer.
            # A dict with more keys (e.g. full tagger result) should overwrite
            # a partial entry (e.g. key-only result).
            if isinstance(existing.data, dict) and isinstance(data, dict):
                if len(data) <= len(existing.data) and all(existing.data.get(k) == v for k, v in data.items()):
                    return
            elif existing.data == data:
                return
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
        """Look up track data by filename + duration + layer.

        Raw feature layers auto-check against their extractor signature when
        applicable. Derived layers auto-check against the current tagger version.
        """
        if version is None:
            refresh_layer_versions()
            version = LAYER_VERSIONS.get(layer)

        key = self.track_key(filename, duration, layer)
        return self.get(key, version)

    def get_track_any_version(
        self,
        filename: str,
        duration: float | None,
        layer: str,
    ) -> object | None:
        """Look up track data without applying layer-version checks."""
        key = self.track_key(filename, duration, layer)
        entry = self._entries.get(key)
        return entry.data if entry is not None else None

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

        Auto-sets version from the current layer signature map so both raw and
        derived layers are invalidated safely when their logic changes.
        """
        if version is None:
            refresh_layer_versions()
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
        # Load raw_cache.pkl
        if os.path.exists(self.raw_path):
            self._load_file(self.raw_path)

        # Load derived_cache.pkl
        if os.path.exists(self.derived_path):
            self._load_file(self.derived_path)

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
    refresh_layer_versions()
    if _instance is None or _instance.path != path:
        _instance = UniversalCache(path)
    return _instance


def reset_cache() -> None:
    """Reset the singleton (for tests)."""
    global _instance
    _instance = None
