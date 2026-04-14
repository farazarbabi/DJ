"""Central settings loader.

Reads settings.toml from the project root. All tunable parameters live there.
The derived cache version is auto-computed as a hash of the derived sections
(energy, vibe, vocal, structure, key). Change any parameter → hash changes →
derived cache auto-invalidates on next run. No manual version bumping needed.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_settings: dict | None = None
_settings_path: str | None = None

# Sections that affect derived analysis values.
# A change in any of these triggers derived cache invalidation.
_DERIVED_SECTIONS = ("energy", "vibe", "vocal", "structure", "key")


def _find_settings_path() -> str:
    """Find settings.toml by walking up from cwd or this file's location."""
    for start in [Path.cwd(), Path(__file__).parent.parent.parent]:
        p = start / "settings.toml"
        if p.exists():
            return str(p)
    return str(Path.cwd() / "settings.toml")


def load_settings(path: str | None = None) -> dict:
    """Load settings.toml. Returns cached result on subsequent calls."""
    global _settings, _settings_path

    if path is None:
        path = _find_settings_path()

    if _settings is not None and _settings_path == path:
        return _settings

    if not os.path.exists(path):
        logger.warning("Settings file not found at %s, using defaults", path)
        _settings = {}
        _settings_path = path
        return _settings

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        _settings = tomllib.load(f)

    _settings_path = path
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
    """Compute a version hash from all derived-affecting settings.

    This replaces manual version bumping. Any change to energy, vibe,
    vocal, structure, or key settings produces a different hash, which
    invalidates the derived cache automatically.
    """
    s = load_settings()
    # Build a stable string representation of derived sections
    parts = []
    for section in sorted(_DERIVED_SECTIONS):
        section_data = s.get(section, {})
        parts.append(f"{section}={_stable_repr(section_data)}")
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def _stable_repr(obj) -> str:
    """Produce a deterministic string representation for hashing."""
    if isinstance(obj, dict):
        items = sorted(obj.items())
        return "{" + ",".join(f"{k}:{_stable_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    return str(obj)


def reset():
    """Reset cached settings (for tests)."""
    global _settings, _settings_path
    _settings = None
    _settings_path = None
