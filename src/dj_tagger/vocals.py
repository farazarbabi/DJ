"""Shared taxonomy-derived vocal-profile vocabulary and scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .moods import normalize_mood_scores


@dataclass(frozen=True)
class VocalProfileDefinition:
    code: str
    profile: str
    cues: tuple[str, ...]
    has_vocals: bool


_FALLBACK_PROFILES: tuple[str, ...] = (
    "chant",
    "dub",
    "featured_vocal",
    "instrumental",
    "spoken",
    "tool",
    "vocal",
)

_PROFILE_CODE_OVERRIDES: dict[str, str] = {
    "chant": "CHANT",
    "dub": "DUB",
    "featured_vocal": "FVOC",
    "instrumental": "INST",
    "spoken": "SPK",
    "tool": "TOOL",
    "vocal": "VOC",
}

_PROFILE_HAS_VOCALS: dict[str, bool] = {
    "CHANT": True,
    "DUB": False,
    "FVOC": True,
    "INST": False,
    "SPK": True,
    "TOOL": False,
    "VOC": True,
}

_EXTRA_CUES: dict[str, tuple[str, ...]] = {
    "CHANT": ("chant", "ritual", "shamanic", "tribal", "ceremony"),
    "DUB": ("dub", "dubby", "echo", "reduced vocal", "version"),
    "FVOC": ("featured vocal", "vocal hook", "song", "singer", "lead vocal"),
    "INST": ("instrumental", "no vocal", "no vocals"),
    "SPK": ("spoken", "speech", "spoken word", "voiceover"),
    "TOOL": ("tool", "percussion", "drums", "loop", "tracky"),
    "VOC": ("vocal", "voice", "lyrics"),
}

_ALIASES: dict[str, str] = {
    "CHNT": "CHANT",
    "DUBBY": "DUB",
    "FEATURED": "FVOC",
    "FEATURED_VOCAL": "FVOC",
    "FEATURING": "FVOC",
    "FV": "FVOC",
    "HOOK": "FVOC",
    "INSTR": "INST",
    "INSTRUMENTAL": "INST",
    "LOW_VOCAL": "VOC",
    "NO_VOCAL": "INST",
    "NO_VOCALS": "INST",
    "NONVOCAL": "INST",
    "SPEECH": "SPK",
    "SPOKEN": "SPK",
    "SPOKEN_WORD": "SPK",
    "NV": "INST",
    "V": "VOC",
    "VOICE": "VOC",
    "VOCAL": "VOC",
}


def load_taxonomy_vocal_profiles(path: str | Path | None = None) -> tuple[str, ...]:
    """Load unique vocal profile names from ``dj_taxonomy.json``."""
    taxonomy_path = Path(path) if path else _default_taxonomy_path()
    try:
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        profiles = {
            str(profile).strip().lower()
            for category in data.get("categories", [])
            for profile in category.get("vocal_profiles", [])
            if str(profile).strip()
        }
        if profiles:
            return tuple(sorted(profiles))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return _FALLBACK_PROFILES


def build_vocal_profile_definitions(path: str | Path | None = None) -> tuple[VocalProfileDefinition, ...]:
    definitions = []
    for profile in load_taxonomy_vocal_profiles(path):
        code = _code_for_profile(profile)
        cues = tuple(dict.fromkeys((profile.replace("_", " "), *_EXTRA_CUES.get(code, ()))))
        definitions.append(
            VocalProfileDefinition(
                code=code,
                profile=profile,
                cues=cues,
                has_vocals=_PROFILE_HAS_VOCALS.get(code, code not in {"INST", "TOOL", "DUB"}),
            )
        )
    return tuple(sorted(definitions, key=lambda item: item.code))


VOCAL_PROFILE_DEFINITIONS: tuple[VocalProfileDefinition, ...]
VOCAL_PROFILE_LABELS: tuple[str, ...]
VOCAL_PROFILE_NAME_BY_CODE: dict[str, str]
VOCAL_PROFILE_CODE_BY_NAME: dict[str, str]
VOCAL_PROFILE_CUES_BY_CODE: dict[str, tuple[str, ...]]


def normalize_vocal_profile(value: Any) -> str:
    """Normalize a stored vocal token to the canonical taxonomy profile code."""
    token = str(value or "").strip()
    if not token:
        return ""
    normalized = re.sub(r"[\s-]+", "_", token).upper()
    normalized = _ALIASES.get(normalized, normalized)
    lower = token.strip().lower().replace("-", "_").replace(" ", "_")
    if lower in VOCAL_PROFILE_CODE_BY_NAME:
        return VOCAL_PROFILE_CODE_BY_NAME[lower]
    return normalized if normalized in VOCAL_PROFILE_LABELS else normalized


def normalize_vocal_profile_scores(scores: dict[str, float] | None) -> dict[str, float]:
    """Normalize score dictionaries to the current vocal-profile code set."""
    normalized = {code: 0.0 for code in VOCAL_PROFILE_LABELS}
    for raw_code, raw_score in (scores or {}).items():
        code = normalize_vocal_profile(raw_code)
        if code in normalized:
            try:
                normalized[code] += float(raw_score)
            except (TypeError, ValueError):
                continue
    return normalized


def vocal_profile_tag_pattern() -> str:
    codes = sorted(VOCAL_PROFILE_LABELS, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(code) for code in codes) + r"|\?\?)"


def has_vocal_content(value: Any) -> bool:
    """Return true for vocal-led/spoken/chant profiles."""
    code = normalize_vocal_profile(value)
    return bool(code and _PROFILE_HAS_VOCALS.get(code, code not in {"INST", "TOOL", "DUB"}))


def vocal_profile_from_has_vocals(has_vocals: bool | None) -> str:
    if has_vocals is True:
        return "VOC"
    if has_vocals is False:
        return "INST"
    return ""


def _stem_vocal_presence(stem_ratio_db: float, stem_activity_frac: float) -> tuple[float, float]:
    """Map stem metrics to (vocal_presence, sustained) in [0, 1].

    Calibrated on a spike of melodic-techno / prog-house tracks (see
    scripts/vocal_spike.py). The two anchors are:
      - Pure instrumental:  ratio < -40 dB, activity ~= 0
      - Continuous vocals:  ratio > -15 dB, activity > 0.7
    """
    # Stem-to-mix ratio dominates for presence detection. Map [-40, -10] -> [0, 1].
    ratio_norm = _clip01((float(stem_ratio_db) + 40.0) / 30.0)
    # Activity fraction tells us how continuous the vocal is (FVOC vs hooky).
    activity_norm = _clip01(float(stem_activity_frac) / 0.7)
    # Vocal presence is a weighted blend; activity is required to avoid
    # false positives from stem bleed-through during loud instrumental sections.
    vocal_presence = _clip01(0.55 * ratio_norm + 0.45 * activity_norm)
    sustained = activity_norm
    return vocal_presence, sustained


def _stem_has_vocals(stem_ratio_db: float, stem_activity_frac: float) -> bool:
    """Binary vocal decision from stem signals.

    Calibrated thresholds. The rules favor preserving uncertainty in the
    middle band because melodic-techno tracks regularly have vocal-like
    instruments (lyrical synth leads, processed pads) that can produce
    moderate stem-ratio readings without real vocal content.

      - ratio < -35 dB                      -> definitely instrumental
                                               (stem inaudible regardless
                                                of activity)
      - ratio < -30 dB AND activity < 0.20  -> definitely instrumental
      - ratio > -25 dB OR activity > 0.40   -> has vocals
    """
    if stem_ratio_db < -35.0:
        return False
    if stem_ratio_db < -30.0 and stem_activity_frac < 0.20:
        return False
    if stem_ratio_db > -25.0 or stem_activity_frac > 0.40:
        return True
    # Ambiguous middle: combined score.
    presence, _ = _stem_vocal_presence(stem_ratio_db, stem_activity_frac)
    return presence > 0.45


def score_vocal_profile(
    *,
    vocal_ratio: float,
    temporal_bonus: float,
    threshold: float,
    base_weight: float = 0.6,
    temporal_weight: float = 0.4,
    dsp: dict[str, float] | None = None,
    audio_features: dict[str, float] | None = None,
    mood_scores: dict[str, float] | None = None,
    stem_ratio_db: float | None = None,
    stem_activity_frac: float | None = None,
) -> tuple[str, dict[str, float], bool, float]:
    """Score taxonomy vocal profiles.

    When Demucs stem metrics are provided they are the primary signal: the
    spectral vocal_ratio/temporal_bonus heuristic is biased by synth leads in
    the 300-3400 Hz band and routinely mistags melodic instrumentals as FVOC.
    Stem-based ratio + activity-fraction directly measure separated vocals
    and avoid that failure mode.

    Falls back to the spectral heuristic when stem metrics are unavailable.
    """
    dsp = dsp or {}
    audio_features = audio_features or {}
    mood_scores = normalize_mood_scores(mood_scores or {})

    stem_available = stem_ratio_db is not None and stem_activity_frac is not None
    if stem_available:
        vocal_presence, sustained = _stem_vocal_presence(stem_ratio_db, stem_activity_frac)
        has_vocals = _stem_has_vocals(stem_ratio_db, stem_activity_frac)
    else:
        vocal_score = float(vocal_ratio) * (float(base_weight) + float(temporal_weight) * float(temporal_bonus))
        vocal_presence = _clip01(vocal_score / max(threshold, 1e-6))
        sustained = _clip01(float(temporal_bonus))
        has_vocals = vocal_score > threshold

    no_vocal = 1.0 - vocal_presence
    onset_density = float(dsp.get("onset_density", 0.0) or 0.0)
    chroma_var = float(dsp.get("chroma_var", 0.0) or 0.0)
    perc_harmonic_ratio = float(dsp.get("perc_harmonic_ratio", 0.0) or 0.0)

    instrumentalness = _feature(audio_features, "instrumentalness")
    speechiness = _feature(audio_features, "speechiness")
    acousticness = _feature(audio_features, "acousticness")

    # CHANT requires a TRIBAL signal specifically. ORG (organic) is much
    # broader — every deep melodic-techno track scores high on organic, so
    # including it caused brief-hook vocal tracks with deep mood to be
    # mislabeled CHANT.
    tribal_mood = mood_scores.get("TRIB", 0.0)
    deep_dub_mood = (
        mood_scores.get("DEEP", 0.0)
        + mood_scores.get("SUB", 0.0)
        + mood_scores.get("MIN", 0.0)
        + mood_scores.get("ATM", 0.0)
    )
    tool_motion = _clip01((onset_density - 1.8) / 1.6)
    sparse_harmony = _clip01((0.06 - chroma_var) / 0.06)
    percussion = _clip01((perc_harmonic_ratio - 0.35) / 0.40)

    # DUB means "reduced vocals" (echo/version/sample), not "no vocals". Gate
    # the DUB score on residual vocal activity so pure instrumentals with a
    # deep mood don't beat INST on the mood term alone.
    dub_gate = _clip01(sustained / 0.15) * _clip01((0.60 - sustained) / 0.40)

    scores = {code: 0.0 for code in VOCAL_PROFILE_LABELS}
    _add(scores, "INST", 0.65 * no_vocal + 0.35 * instrumentalness)
    _add(scores, "TOOL", 0.45 * no_vocal + 0.30 * tool_motion + 0.15 * sparse_harmony + 0.10 * percussion)
    _add(scores, "DUB", dub_gate * (0.50 * no_vocal + 0.30 * deep_dub_mood + 0.10 * acousticness + 0.10 * instrumentalness))
    _add(scores, "VOC", 0.70 * vocal_presence + 0.20 * (1.0 - instrumentalness) + 0.10 * sustained)
    _add(scores, "FVOC", 0.55 * vocal_presence + 0.30 * sustained + 0.15 * (1.0 - instrumentalness))
    _add(scores, "SPK", 0.65 * speechiness + 0.20 * vocal_presence + 0.15 * (1.0 - sustained))
    # CHANT is for ritual / tribal / shamanic group vocals — typically
    # SUSTAINED, not brief. The previous formula rewarded `(1 - sustained)`
    # which gave free score to any low-activity track with a deep/organic
    # mood, causing brief-vocal-hook tracks to be mislabeled CHANT.
    _add(scores, "CHANT", 0.35 * vocal_presence + 0.45 * tribal_mood + 0.20 * sustained)

    if has_vocals:
        for code in ("INST", "TOOL", "DUB"):
            scores[code] *= 0.65
    else:
        for code in ("VOC", "FVOC", "SPK", "CHANT"):
            scores[code] *= 0.45

    if not any(scores.values()):
        scores["INST"] = 1.0

    label = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    top = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    confidence = _clip01((top - second) / (top + 1e-8))
    return label, scores, has_vocals, confidence


def _default_taxonomy_path() -> Path:
    package_sibling = Path(__file__).resolve().parents[1] / "dj_registry" / "taxonomy" / "dj_taxonomy.json"
    repo_root = Path(__file__).resolve().parents[2] / "src" / "dj_registry" / "taxonomy" / "dj_taxonomy.json"
    for candidate in (package_sibling, repo_root, Path.cwd() / "src" / "dj_registry" / "taxonomy" / "dj_taxonomy.json"):
        if candidate.exists():
            return candidate
    return package_sibling


def _code_for_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized in _PROFILE_CODE_OVERRIDES:
        return _PROFILE_CODE_OVERRIDES[normalized]
    return re.sub(r"[^A-Z0-9]+", "", normalized.upper())[:6]


def _feature(audio_features: dict[str, float], key: str) -> float:
    try:
        return _clip01(float(audio_features.get(key, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _add(scores: dict[str, float], code: str, value: float) -> None:
    if code in scores:
        scores[code] += _clip01(value)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


VOCAL_PROFILE_DEFINITIONS = build_vocal_profile_definitions()
VOCAL_PROFILE_LABELS = tuple(defn.code for defn in VOCAL_PROFILE_DEFINITIONS)
VOCAL_PROFILE_NAME_BY_CODE = {defn.code: defn.profile for defn in VOCAL_PROFILE_DEFINITIONS}
VOCAL_PROFILE_CODE_BY_NAME = {defn.profile: defn.code for defn in VOCAL_PROFILE_DEFINITIONS}
VOCAL_PROFILE_CUES_BY_CODE = {defn.code: defn.cues for defn in VOCAL_PROFILE_DEFINITIONS}
