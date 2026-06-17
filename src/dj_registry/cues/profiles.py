"""Cue profile loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HOT_ROLES = ("mix_in", "drop_1", "mix_out")
MEMORY_ROLES = ("intro_start", "breakdown", "peak", "outro_start")
LOOP_ROLES = ("intro_loop", "outro_loop")
SUPPORTED_ROLES = set(HOT_ROLES + MEMORY_ROLES + LOOP_ROLES)


@dataclass(frozen=True)
class CueProfile:
    """Validated cue profile used by the deterministic selector."""

    name: str
    version: str = "1"
    enabled_roles: tuple[str, ...] = HOT_ROLES
    loop_bars: int = 16
    min_confidence: float = 0.65
    source_system: str = "auto_v1"
    phrase_align_sections: bool = False


def load_cue_profile(
    profile: str = "v1",
    *,
    profile_file: str | None = None,
    include_memory: bool = False,
    include_loops: bool = False,
    loop_bars: int | None = None,
) -> CueProfile:
    """Load a built-in or file-backed cue profile."""
    if profile_file:
        loaded = _load_profile_file(profile_file)
        if loop_bars is not None:
            loaded = CueProfile(
                name=loaded.name,
                version=loaded.version,
                enabled_roles=loaded.enabled_roles,
                loop_bars=loop_bars,
                min_confidence=loaded.min_confidence,
                source_system=loaded.source_system,
                phrase_align_sections=loaded.phrase_align_sections,
            )
        return loaded

    name = (profile or "v1").strip().lower()
    if name == "v1":
        roles = list(HOT_ROLES)
        source_system = "auto_v1"
        if include_memory:
            roles.extend(MEMORY_ROLES)
            source_system = "auto_v2"
        if include_loops:
            roles.extend(LOOP_ROLES)
            source_system = "auto_v2"
        return CueProfile(
            name="v1" if source_system == "auto_v1" else "v1-custom",
            enabled_roles=_dedupe_roles(roles),
            loop_bars=loop_bars or 16,
            source_system=source_system,
        )
    if name == "v2":
        return CueProfile(
            name="v2",
            version="2",
            enabled_roles=HOT_ROLES + MEMORY_ROLES + LOOP_ROLES,
            loop_bars=loop_bars or 16,
            source_system="auto_v2",
        )
    if name == "v3-default":
        return CueProfile(
            name="v3-default",
            version="3",
            enabled_roles=HOT_ROLES + MEMORY_ROLES + LOOP_ROLES,
            loop_bars=loop_bars or 16,
            min_confidence=0.7,
            source_system="auto_v3",
            phrase_align_sections=True,
        )
    raise ValueError(f"Unsupported cue profile: {profile!r}")


def _load_profile_file(path: str) -> CueProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Cue profile file not found: {path}")

    text = profile_path.read_text(encoding="utf-8")
    suffix = profile_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("YAML cue profiles require PyYAML; use JSON or install PyYAML") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("Cue profile must be a JSON/YAML object")
    return _profile_from_dict(data, default_name=profile_path.stem)


def _profile_from_dict(data: dict[str, Any], *, default_name: str) -> CueProfile:
    name = str(data.get("name") or data.get("profile_name") or default_name).strip()
    if not name:
        raise ValueError("Cue profile name cannot be empty")

    raw_roles = data.get("enabled_roles")
    if raw_roles is None:
        roles = _roles_from_legacy_fields(data)
    elif isinstance(raw_roles, list):
        roles = tuple(str(role).strip().lower() for role in raw_roles)
    else:
        raise ValueError("Cue profile enabled_roles must be a list")
    roles = _dedupe_roles(roles)
    if not roles:
        raise ValueError("Cue profile must enable at least one cue role")

    unsupported = [role for role in roles if role not in SUPPORTED_ROLES]
    if unsupported:
        raise ValueError(f"Unsupported cue role(s): {', '.join(unsupported)}")

    try:
        loop_bars = int(data.get("loop_bars", 16))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cue profile loop_bars must be an integer") from exc
    if loop_bars <= 0:
        raise ValueError("Cue profile loop_bars must be greater than zero")

    try:
        min_confidence = float(data.get("min_confidence", 0.65))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cue profile min_confidence must be numeric") from exc
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("Cue profile min_confidence must be between 0 and 1")

    source_system = str(data.get("source_system") or "auto_v3").strip()
    if not source_system:
        raise ValueError("Cue profile source_system cannot be empty")

    return CueProfile(
        name=name,
        version=str(data.get("version") or "custom"),
        enabled_roles=roles,
        loop_bars=loop_bars,
        min_confidence=min_confidence,
        source_system=source_system,
        phrase_align_sections=bool(data.get("phrase_align_sections", True)),
    )


def _roles_from_legacy_fields(data: dict[str, Any]) -> tuple[str, ...]:
    roles: list[str] = list(HOT_ROLES)
    if data.get("include_memory", True):
        roles.extend(MEMORY_ROLES)
    if data.get("include_loops", True):
        roles.extend(LOOP_ROLES)
    for key in ("memory_roles", "loop_roles"):
        if key in data:
            values = data[key]
            if not isinstance(values, list):
                raise ValueError(f"Cue profile {key} must be a list")
            roles.extend(str(role).strip().lower() for role in values)
    return tuple(roles)


def _dedupe_roles(roles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for role in roles:
        normalized = str(role).strip().lower()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)
