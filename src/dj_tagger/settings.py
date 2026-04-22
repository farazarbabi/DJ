"""Central settings loader.

Reads settings.toml from the project root. All tunable parameters live there.
Cache signatures are auto-computed from the relevant settings sections and
source files so changes invalidate stale cache entries without a manual bump.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_settings: dict | None = None
_settings_path: str | None = None
_settings_mtime_ns: int | None = None
_file_digest_cache: dict[str, tuple[int, int, str]] = {}

# Sections that affect derived analysis values.
# A change in any of these triggers derived cache invalidation.
_DERIVED_SECTIONS = ("energy", "vibe", "vocal", "structure", "key")

_RAW_VERSION_FILES = (
    "src/dj_tagger/audio.py",
    "src/dj_tagger/raw_features.py",
    "src/dj_grouper/features/dsp.py",
    "src/dj_tagger/analyzers/sections.py",
)
_DERIVED_VERSION_FILES = (
    "src/dj_tagger/derive.py",
    "src/dj_tagger/vibe_scoring.py",
)
_KEY_VERSION_FILES = (
    "src/dj_tagger/audio.py",
    "src/dj_tagger/analyzers/key.py",
    "src/dj_tagger/constants.py",
)


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _find_settings_path() -> str:
    """Find settings.toml by walking up from cwd or this file's location."""
    for start in [Path.cwd(), Path(__file__).parent.parent.parent]:
        p = start / "settings.toml"
        if p.exists():
            return str(p)
    return str(Path.cwd() / "settings.toml")


def load_settings(path: str | None = None) -> dict:
    """Load settings.toml. Returns cached result on subsequent calls."""
    global _settings, _settings_path, _settings_mtime_ns

    if path is None:
        path = _find_settings_path()

    mtime_ns: int | None = None
    if os.path.exists(path):
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            mtime_ns = None

    if _settings is not None and _settings_path == path and _settings_mtime_ns == mtime_ns:
        return _settings

    if not os.path.exists(path):
        logger.warning("Settings file not found at %s, using defaults", path)
        _settings = {}
        _settings_path = path
        _settings_mtime_ns = None
        return _settings

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        _settings = tomllib.load(f)

    _settings_path = path
    _settings_mtime_ns = mtime_ns
    logger.info("Settings loaded from %s", path)
    return _settings


def get(section: str, key: str, default=None):
    """Get a setting value. Example: get("energy", "thresholds")"""
    s = load_settings()
    return s.get(section, {}).get(key, default)


def get_section(section: str) -> dict:
    """Get an entire section as a dict."""
    return load_settings().get(section, {})


def derived_version() -> str:
    """Compute a version hash from derived-affecting settings and scorer code."""
    combined = _settings_signature(_DERIVED_SECTIONS) + "|" + _code_signature(_DERIVED_VERSION_FILES)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def raw_version() -> str:
    """Compute a version hash for raw feature extraction code paths."""
    combined = _code_signature(_RAW_VERSION_FILES)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def key_version() -> str:
    """Compute a version hash for key-analysis logic and settings."""
    combined = _settings_signature(("key",)) + "|" + _code_signature(_KEY_VERSION_FILES)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def tagger_version() -> str:
    """Combined tagger cache version across key and derived logic."""
    combined = f"derived={derived_version()}|key={key_version()}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def _stable_repr(obj) -> str:
    """Produce a deterministic string representation for hashing."""
    if isinstance(obj, dict):
        items = sorted(obj.items())
        return "{" + ",".join(f"{k}:{_stable_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    return str(obj)


def _settings_signature(sections: tuple[str, ...]) -> str:
    s = load_settings()
    parts = []
    for section in sorted(sections):
        section_data = s.get(section, {})
        parts.append(f"{section}={_stable_repr(section_data)}")
    return "|".join(parts)


def _code_signature(relative_paths: tuple[str, ...]) -> str:
    digests = [_file_digest(_project_root() / rel) for rel in relative_paths]
    return "|".join(f"{rel}={digest}" for rel, digest in zip(relative_paths, digests))


def _file_digest(path: Path) -> str:
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return "missing"

    cached = _file_digest_cache.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    _file_digest_cache[key] = (stat.st_mtime_ns, stat.st_size, digest)
    return digest


def reset():
    """Reset cached settings (for tests)."""
    global _settings, _settings_path, _settings_mtime_ns
    _settings = None
    _settings_path = None
    _settings_mtime_ns = None
