"""Feature extraction for learned genre taxonomy classification."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from dj_tagger.moods import MOOD_CUES_BY_CODE, normalize_mood_code
from dj_tagger.vocals import VOCAL_PROFILE_CUES_BY_CODE, normalize_vocal_profile

from ..models import FileRecord, LogicalTrack, SourceObservation


TEXT_CUES = {
    "acid",
    "afro",
    "ambient",
    "bass",
    "bassline",
    "breakbeat",
    "breaks",
    "broken",
    "chant",
    "cinematic",
    "club",
    "dark",
    "deep",
    "desert",
    "disco",
    "driving",
    "dub",
    "dubby",
    "electro",
    "extended",
    "funk",
    "funky",
    "garage",
    "groove",
    "hard",
    "house",
    "hypnotic",
    "indie",
    "instrumental",
    "latin",
    "melodic",
    "minimal",
    "organic",
    "peak",
    "percussion",
    "percussive",
    "progressive",
    "psy",
    "psychedelic",
    "punk",
    "raw",
    "remix",
    "rock",
    "rolling",
    "tech",
    "techno",
    "trance",
    "tribal",
    "ukg",
    "vocal",
    "warehouse",
}

TAGGER_VIBE_CUES = MOOD_CUES_BY_CODE

TAGGER_VOCAL_CUES = VOCAL_PROFILE_CUES_BY_CODE


def build_track_features(
    track: LogicalTrack | None = None,
    observations: list[SourceObservation] | None = None,
    file_record: FileRecord | None = None,
    *,
    label_row: dict[str, str] | None = None,
    feature_mode: str = "external",
) -> dict[str, float]:
    """Build a DictVectorizer-ready feature map from all available evidence.

    Provider genre fields are intentionally ordinary text features. They are
    never treated as labels or hard overrides by this extractor.
    """
    track = track or LogicalTrack()
    observations = observations or []
    label_row = label_row or {}
    features: dict[str, float] = {}

    _add_identity_features(features, track, file_record, label_row)
    _add_tagger_features(features, track, observations, label_row)
    if feature_mode not in {"internal", "external"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    if feature_mode == "external":
        _add_provider_features(features, observations, file_record, label_row)
        _add_audio_features(features, observations)

    return features


def summarize_feature_groups(features: dict[str, float]) -> list[str]:
    groups = sorted({key.split(":", 1)[0] for key, value in features.items() if value})
    return groups


def _add_identity_features(
    features: dict[str, float],
    track: LogicalTrack,
    file_record: FileRecord | None,
    label_row: dict[str, str],
) -> None:
    artist = track.artist_canonical or label_row.get("artist") or label_row.get("Artist")
    title = track.title_canonical or label_row.get("title") or label_row.get("Track Title")
    mix = track.mix_canonical or label_row.get("mix")
    label = track.label_canonical or label_row.get("label")
    album = track.album_canonical or label_row.get("album") or label_row.get("Album")

    _add_text(features, "artist", artist)
    _add_text(features, "title", title)
    _add_text(features, "mix", mix)
    _add_text(features, "label", label)
    _add_text(features, "album", album)

    if file_record:
        _add_text(features, "filename", file_record.file_name or Path(file_record.path_abs).name)
        _add_text(features, "path", file_record.path_rel or file_record.path_abs)
        _add_text(features, "embedded_title", file_record.embedded_title)
        _add_text(features, "embedded_artist", file_record.embedded_artist)
        _add_text(features, "embedded_album", file_record.embedded_album)
        _add_text(features, "embedded_comment", file_record.embedded_comment)
    else:
        _add_text(features, "filename", label_row.get("file_name") or label_row.get("filename"))


def _add_tagger_features(
    features: dict[str, float],
    track: LogicalTrack,
    observations: list[SourceObservation],
    label_row: dict[str, str],
) -> None:
    tagger_obs = next(
        (
            obs
            for obs in observations
            if obs.tagger_energy or obs.tagger_vibe or obs.tagger_vocal or obs.tagger_structure
        ),
        None,
    )

    energy = track.tagger_energy or (tagger_obs.tagger_energy if tagger_obs else "") or label_row.get("pred_Energy_v1", "")
    vibe = track.tagger_vibe or (tagger_obs.tagger_vibe if tagger_obs else "") or label_row.get("pred_Vibe_v1", "")
    vocal = track.tagger_vocal or (tagger_obs.tagger_vocal if tagger_obs else "") or label_row.get("pred_Vocal_v1", "")
    structure = track.tagger_structure or (tagger_obs.tagger_structure if tagger_obs else "") or label_row.get("structure", "")
    vibe_scores = track.tagger_vibe_scores or (tagger_obs.tagger_vibe_scores if tagger_obs else "")
    vocal_scores = track.tagger_vocal_scores or (tagger_obs.tagger_vocal_scores if tagger_obs else "")
    confidences = track.tagger_confidences or (tagger_obs.tagger_confidences if tagger_obs else "")
    bpm = (
        _num(track.tagger_bpm)
        or _num(track.canonical_bpm)
        or (_num(tagger_obs.bpm) if tagger_obs else None)
        or _num(label_row.get("BPM") or label_row.get("bpm_hint"))
    )

    energy_token = _energy_token(energy)
    if energy_token:
        features[f"tagger:energy={energy_token}"] = 1.0
        energy_int = _energy_int(energy)
        if energy_int is not None:
            features["num:tagger_energy"] = float(energy_int)
            if energy_int <= 2:
                _add_cue(features, "warmup")
                _add_cue(features, "deep")
                _add_cue(features, "downtempo")
            if energy_int == 4:
                _add_cue(features, "driver")
                _add_cue(features, "rolling")
            if energy_int >= 5:
                _add_cue(features, "peak")
                _add_cue(features, "hard")

    for token in _split_mood_codes(vibe):
        features[f"tagger:vibe={token}"] = 1.0
        for cue in TAGGER_VIBE_CUES.get(token, []):
            _add_cue(features, cue)
    _add_score_map_features(features, "tagger:mood_score", vibe_scores, normalizer=normalize_mood_code)

    for token in _split_vocal_codes(vocal):
        features[f"tagger:vocal={token}"] = 1.0
        for cue in TAGGER_VOCAL_CUES.get(token, []):
            _add_cue(features, cue)
    _add_score_map_features(features, "tagger:vocal_score", vocal_scores, normalizer=normalize_vocal_profile)
    _add_score_map_features(features, "tagger:confidence", confidences)

    structure_token = _normalize_code(structure)
    if structure_token:
        features[f"tagger:structure={structure_token}"] = 1.0
        if structure_token.endswith("H") or structure_token == "ROLLING":
            _add_cue(features, "rolling")
            _add_cue(features, "hypnotic")
        if structure_token.endswith("D") or structure_token == "LINEAR":
            _add_cue(features, "driving")
        if structure_token.endswith("B") or structure_token == "BREAKS":
            _add_cue(features, "breaks")
            _add_cue(features, "garage")
        if structure_token.endswith("G"):
            _add_cue(features, "groove")
        if structure_token.endswith("L"):
            _add_cue(features, "downtempo")

    if bpm is not None:
        features["num:bpm"] = bpm
        features[f"bpm_band:{_bpm_band(bpm)}"] = 1.0


def _add_provider_features(
    features: dict[str, float],
    observations: list[SourceObservation],
    file_record: FileRecord | None,
    label_row: dict[str, str],
) -> None:
    if file_record:
        _add_text(features, "provider:embedded_genre", file_record.embedded_genre)
        _add_numeric(features, "embedded_bpm", _num(file_record.embedded_bpm))

    _add_text(features, "provider:label_row_genre", label_row.get("Genre"))

    for obs in observations:
        source = _normalize_text(obs.source_system) or "source"
        _add_text(features, f"provider:{source}:genre", obs.genre)
        _add_text(features, f"provider:{source}:genres_all", obs.genres_all)
        _add_text(features, f"source:{source}:comments", obs.comments)
        _add_text(features, f"source:{source}:label", obs.label)
        _add_text(features, f"source:{source}:artist", obs.artist)
        _add_text(features, f"source:{source}:title", obs.title)
        _add_numeric(features, f"{source}:bpm", _num(obs.bpm))


def _add_audio_features(features: dict[str, float], observations: list[SourceObservation]) -> None:
    for obs in observations:
        source = _normalize_text(obs.source_system) or "source"
        for name in (
            "acousticness",
            "danceability",
            "energy",
            "instrumentalness",
            "liveness",
            "speechiness",
            "valence",
        ):
            value = _num(getattr(obs, name, ""))
            _add_numeric(features, f"{source}:{name}", value)
            if value is not None:
                features[f"audio_band:{name}:{_value_band(value)}"] = 1.0
        if _num(obs.acousticness) is not None and _num(obs.acousticness) >= 0.60:
            _add_cue(features, "acoustic")
            _add_cue(features, "rock")
        if _num(obs.instrumentalness) is not None and _num(obs.instrumentalness) >= 0.75:
            _add_cue(features, "instrumental")
        if _num(obs.valence) is not None and _num(obs.valence) <= 0.30:
            _add_cue(features, "dark")
        if _num(obs.energy) is not None and _num(obs.energy) >= 0.78:
            _add_cue(features, "driving")
            _add_cue(features, "peak")


def _add_text(features: dict[str, float], prefix: str, value: Any) -> None:
    norm = _normalize_text(value)
    if not norm:
        return
    features[f"text:{prefix}:present"] = 1.0
    if len(norm) <= 80:
        features[f"text:{prefix}:value={norm}"] = 1.0
    tokens = _tokens(norm)
    for token in tokens:
        features[f"text:{prefix}:token={token}"] = 1.0
        if token in TEXT_CUES:
            _add_cue(features, token)
    for phrase in _ngrams(tokens, 2):
        features[f"text:{prefix}:phrase={phrase}"] = 1.0
        if phrase in {"tech house", "indie dance", "deep house", "melodic techno", "organic house"}:
            _add_cue(features, phrase.replace(" ", "_"))
    for phrase in _ngrams(tokens, 3):
        features[f"text:{prefix}:phrase={phrase}"] = 1.0


def _add_numeric(features: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        features[f"num:{name}"] = float(value)


def _add_score_map_features(
    features: dict[str, float],
    prefix: str,
    value: Any,
    *,
    normalizer: Any | None = None,
) -> None:
    if not value:
        return
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return
    if not isinstance(parsed, dict):
        return
    ranked: list[tuple[str, float]] = []
    for key, raw_score in parsed.items():
        norm_key = normalizer(key) if normalizer else _normalize_text(key).replace(" ", "_")
        score = _num(raw_score)
        if norm_key and score is not None:
            ranked.append((norm_key, score))
            features[f"num:{prefix}:{norm_key}"] = score
            features[f"{prefix}_band:{norm_key}:{_value_band(score)}"] = 1.0
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (key, _score) in enumerate(ranked[:3], start=1):
        features[f"{prefix}:rank{rank}={key}"] = 1.0


def _add_cue(features: dict[str, float], cue: str) -> None:
    norm = _normalize_text(cue).replace(" ", "_")
    if norm:
        features[f"cue:{norm}"] = 1.0


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\.(mp3|aiff?|wav|flac|m4a)$", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[_/|,;()\[\]{}]+", " ", text)
    text = re.sub(r"(?<=\D)-(?=\D)", " ", text)
    text = re.sub(r"[^a-z0-9.+#' -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(value: str) -> list[str]:
    return [token for token in value.split() if token and token not in {"the", "and", "feat", "ft"}]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _split_codes(value: Any) -> list[str]:
    norm = _normalize_code(value)
    if not norm:
        return []
    return [part for part in re.split(r"[\s,;/|]+", norm) if part]


def _split_mood_codes(value: Any) -> list[str]:
    return [normalize_mood_code(part) for part in _split_codes(value) if normalize_mood_code(part)]


def _split_vocal_codes(value: Any) -> list[str]:
    return [normalize_vocal_profile(part) for part in _split_codes(value) if normalize_vocal_profile(part)]


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_")


def _energy_token(value: Any) -> str:
    energy = _normalize_code(value)
    if not energy:
        return ""
    number = _energy_int(energy)
    return f"E{number}" if number is not None else energy


def _energy_int(value: Any) -> int | None:
    match = re.search(r"([1-5])", str(value or ""))
    if not match:
        return None
    return int(match.group(1))


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _value_band(value: float) -> str:
    if value < 0.25:
        return "very_low"
    if value < 0.45:
        return "low"
    if value < 0.65:
        return "mid"
    if value < 0.82:
        return "high"
    return "very_high"


def _bpm_band(bpm: float) -> str:
    if bpm < 95:
        return "60_95"
    if bpm < 115:
        return "95_115"
    if bpm < 124:
        return "115_124"
    if bpm < 130:
        return "124_130"
    if bpm < 146:
        return "130_146"
    if bpm < 160:
        return "146_160"
    if bpm < 181:
        return "160_180"
    return "180_plus"
