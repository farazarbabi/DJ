"""Shared taxonomy-derived mood vocabulary.

The public registry/tagger field is still named ``vibe`` for backward
compatibility, but its values are now mood codes derived from the DJ taxonomy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoodDefinition:
    code: str
    mood: str
    cues: tuple[str, ...]


_FALLBACK_MOODS: tuple[str, ...] = (
    "acidic",
    "atmospheric",
    "cinematic",
    "dark",
    "deep",
    "emotional",
    "euphoric",
    "gritty",
    "hypnotic",
    "melodic",
    "minimal",
    "organic",
    "playful",
    "psychedelic",
    "raw",
    "soulful",
    "subby",
    "sunlit",
    "tense",
    "tribal",
    "warehouse",
    "warm",
)

_MOOD_CODE_OVERRIDES: dict[str, str] = {
    "acidic": "ACID",
    "atmospheric": "ATM",
    "cinematic": "CIN",
    "dark": "DRK",
    "deep": "DEEP",
    "emotional": "EMO",
    "euphoric": "EUP",
    "gritty": "GRIT",
    "hypnotic": "HYPN",
    "melodic": "MEL",
    "minimal": "MIN",
    "organic": "ORG",
    "playful": "PLAY",
    "psychedelic": "PSY",
    "raw": "RAW",
    "soulful": "SOUL",
    "subby": "SUB",
    "sunlit": "SUN",
    "tense": "TENS",
    "tribal": "TRIB",
    "warehouse": "WHSE",
    "warm": "WARM",
}

_EXTRA_CUES: dict[str, tuple[str, ...]] = {
    "ACID": ("acid", "303", "psychedelic"),
    "ATM": ("ambient", "atmosphere", "pad"),
    "CIN": ("cinema", "cinematic", "soundtrack"),
    "DRK": ("dark", "nocturnal", "gothic", "industrial"),
    "DEEP": ("deep", "dub", "late night"),
    "EMO": ("emotional", "romantic", "melancholic"),
    "EUP": ("euphoric", "uplifting", "trance"),
    "GRIT": ("gritty", "rough", "distorted"),
    "HYPN": ("hypnotic", "rolling", "loop", "mental"),
    "MEL": ("melodic", "harmony", "cinematic"),
    "MIN": ("minimal", "micro", "sparse"),
    "ORG": ("organic", "earthy", "desert", "ethnic", "middle eastern"),
    "PLAY": ("playful", "funky", "bouncy"),
    "PSY": ("psychedelic", "psy", "acid", "mental"),
    "RAW": ("raw", "warehouse", "industrial"),
    "SOUL": ("soulful", "soul", "gospel"),
    "SUB": ("subby", "sub", "bass"),
    "SUN": ("sunlit", "sunset", "balearic"),
    "TENS": ("tense", "tension", "suspense"),
    "TRIB": ("tribal", "percussive", "chant", "shamanic"),
    "WHSE": ("warehouse", "rave", "industrial"),
    "WARM": ("warm", "rounded", "soulful"),
}

_ALIASES: dict[str, str] = {
    "ACIDIC": "ACID",
    "AMBIENT": "ATM",
    "ATMOSPHERE": "ATM",
    "ATMOSPHERIC": "ATM",
    "DARK": "DRK",
    "EMOTIONAL": "EMO",
    "EUPHORIC": "EUP",
    "FUNKY": "PLAY",
    "HYP": "HYPN",
    "HYPNOTIC": "HYPN",
    "MELODIC": "MEL",
    "NOCTURNAL": "DRK",
    "ORGANIC": "ORG",
    "PERC": "TRIB",
    "PERCUSSIVE": "TRIB",
    "PSYCHEDELIC": "PSY",
    "SUNSET": "SUN",
    "TRIBAL": "TRIB",
}


def load_taxonomy_moods(path: str | Path | None = None) -> tuple[str, ...]:
    """Load unique mood names from ``dj_taxonomy.json``.

    The loader intentionally has a fallback so tagger/grouper commands still
    work in editable or partially installed environments.
    """
    taxonomy_path = Path(path) if path else _default_taxonomy_path()
    try:
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        moods = {
            str(mood).strip().lower()
            for category in data.get("categories", [])
            for mood in category.get("moods", [])
            if str(mood).strip()
        }
        if moods:
            return tuple(sorted(moods))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return _FALLBACK_MOODS


def build_mood_definitions(path: str | Path | None = None) -> tuple[MoodDefinition, ...]:
    definitions = []
    for mood in load_taxonomy_moods(path):
        code = _code_for_mood(mood)
        cues = tuple(dict.fromkeys((mood, *_EXTRA_CUES.get(code, ()))))
        definitions.append(MoodDefinition(code=code, mood=mood, cues=cues))
    return tuple(sorted(definitions, key=lambda item: item.code))


MOOD_DEFINITIONS: tuple[MoodDefinition, ...]
MOOD_LABELS: tuple[str, ...]
MOOD_NAME_BY_CODE: dict[str, str]
MOOD_CODE_BY_NAME: dict[str, str]
MOOD_CUES_BY_CODE: dict[str, tuple[str, ...]]
VIBE_LABELS: tuple[str, ...]


def normalize_mood_code(value: Any) -> str:
    """Normalize a stored mood/vibe token to the canonical taxonomy mood code."""
    token = str(value or "").strip()
    if not token:
        return ""
    normalized = re.sub(r"[\s-]+", "_", token).upper()
    normalized = _ALIASES.get(normalized, normalized)
    lower = token.strip().lower().replace("_", " ")
    if lower in MOOD_CODE_BY_NAME:
        return MOOD_CODE_BY_NAME[lower]
    return normalized if normalized in MOOD_LABELS else normalized


def normalize_mood_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize score dictionaries that may contain legacy vibe tokens."""
    normalized = {code: 0.0 for code in MOOD_LABELS}
    for raw_code, raw_score in (scores or {}).items():
        code = normalize_mood_code(raw_code)
        if code in normalized:
            try:
                normalized[code] += float(raw_score)
            except (TypeError, ValueError):
                continue
    return normalized


def mood_tag_pattern() -> str:
    codes = sorted({*MOOD_LABELS, "HYP"}, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(code) for code in codes) + r"|\?\?)"


def _default_taxonomy_path() -> Path:
    package_sibling = Path(__file__).resolve().parents[1] / "dj_registry" / "taxonomy" / "dj_taxonomy.json"
    repo_root = Path(__file__).resolve().parents[2] / "src" / "dj_registry" / "taxonomy" / "dj_taxonomy.json"
    for candidate in (package_sibling, repo_root, Path.cwd() / "src" / "dj_registry" / "taxonomy" / "dj_taxonomy.json"):
        if candidate.exists():
            return candidate
    return package_sibling


def _code_for_mood(mood: str) -> str:
    normalized = mood.strip().lower()
    if normalized in _MOOD_CODE_OVERRIDES:
        return _MOOD_CODE_OVERRIDES[normalized]
    return re.sub(r"[^A-Z0-9]+", "", normalized.upper())[:4]


MOOD_DEFINITIONS = build_mood_definitions()
MOOD_LABELS = tuple(defn.code for defn in MOOD_DEFINITIONS)
MOOD_NAME_BY_CODE = {defn.code: defn.mood for defn in MOOD_DEFINITIONS}
MOOD_CODE_BY_NAME = {defn.mood: defn.code for defn in MOOD_DEFINITIONS}
MOOD_CUES_BY_CODE = {defn.code: defn.cues for defn in MOOD_DEFINITIONS}

# Backward-compatible public name used by older grouper/tagger code.
VIBE_LABELS = MOOD_LABELS
