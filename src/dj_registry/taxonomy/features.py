"""Feature extraction for learned genre taxonomy classification."""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from dj_tagger.moods import MOOD_CUES_BY_CODE, normalize_mood_code
from dj_tagger.vocals import VOCAL_PROFILE_CUES_BY_CODE, normalize_vocal_profile

from ..models import FileRecord, LogicalTrack, SourceObservation


# ── Spec §6: feature engineering data ───────────────────────────────────────

# Mix-name phrase → canonical token. Substring (case-insensitive) match against
# the mix_canonical / title / embedded_title / embedded_comment text pool.
_MIX_NAME_TOKENS: dict[str, str] = {
    "dub mix": "dub",
    "dub remix": "dub",
    "extended mix": "extended",
    "extended version": "extended",
    "tribal mix": "tribal",
    "tribal remix": "tribal",
    "afro mix": "afro",
    "afro remix": "afro",
    "club mix": "club",
    "club edit": "club",
    "rework": "rework",
    "vip mix": "rework",
    "vip edit": "rework",
    "instrumental mix": "instrumental",
    "radio edit": "radio",
    "radio mix": "radio",
    "shamanic mix": "shamanic",
    "ritual mix": "ritual",
}

# Spec §6.6 — words that disambiguate to different families. The LR learns the
# joint distribution with other signals.
_DISAMBIGUATION_TERMS: tuple[str, ...] = (
    "deep", "dark", "melodic", "progressive", "garage", "tribal", "dub", "organic",
)

# Spec §6.2 — the anti-collapse rule: any candidate category whose family
# matches one of these patterns requires an explicit afro/tribal provider
# signal (or artist/label prior, or mix-name cue) to be eligible for selection.
AFRO_TRIBAL_FAMILY_PATTERNS: tuple[str, ...] = (
    "afro",            # matches "Afro House", "Afro-Tech", "Afro House / *", "Afro / Deep House"
    "tribal house",    # matches "Tribal House" but NOT "Techno / Tribal" (tribal techno stays)
    "organic / tribal",
)

# Tokens that count as explicit afro/tribal evidence in any text field.
_AFRO_TRIBAL_KEYWORDS_RE = re.compile(
    r"\b(afro|tribal|spiritual\s+house|3[- ]step|shamanic|ritual)\b",
    re.IGNORECASE,
)

# Substrings that count as afro/tribal evidence in an artist/label prior tag.
_AFRO_TRIBAL_PRIOR_TAGS: tuple[str, ...] = ("afro", "tribal", "spiritual")


def is_afro_tribal_family(family: str) -> bool:
    """Return True if the family is gated by the §6.2 eligibility rule."""
    if not family:
        return False
    fam = family.lower()
    return any(pattern in fam for pattern in AFRO_TRIBAL_FAMILY_PATTERNS)


def is_eligible_for_afro_tribal(
    track: LogicalTrack | None,
    observations: list[SourceObservation] | None,
    file_record: FileRecord | None,
    *,
    artist_priors: dict[str, Any] | None = None,
    label_priors: dict[str, Any] | None = None,
) -> bool:
    """Spec §6.2 hard rule for afro/tribal family eligibility.

    A track may score for afro/tribal categories only if at least one holds:
      (a) provider genre fields contain afro/tribal/spiritual/3-step token
      (b) artist appears in artist_priors with an afro/tribal tag
      (c) label appears in label_priors with an afro/tribal tag
      (d) mix_canonical or title contains an afro/tribal/shamanic/ritual mix token
    """
    track = track or LogicalTrack()
    observations = observations or []

    # (a) Explicit provider genre token
    provider_pool: list[str] = []
    if file_record and file_record.embedded_genre:
        provider_pool.append(file_record.embedded_genre)
    for obs in observations:
        if obs.genre:
            provider_pool.append(obs.genre)
        if obs.genres_all:
            provider_pool.append(obs.genres_all)
    if _AFRO_TRIBAL_KEYWORDS_RE.search(" | ".join(provider_pool)):
        return True

    # (b) Artist prior
    artist = (track.artist_canonical or "").strip()
    if artist and artist_priors and artist in artist_priors:
        for prior in artist_priors[artist]:
            tag = str(prior.get("tag", "") if isinstance(prior, dict) else prior).lower()
            if any(t in tag for t in _AFRO_TRIBAL_PRIOR_TAGS):
                return True

    # (c) Label prior
    label = (track.label_canonical or "").strip()
    if label and label_priors and label in label_priors:
        for prior in label_priors[label]:
            tag = str(prior.get("tag", "") if isinstance(prior, dict) else prior).lower()
            if any(t in tag for t in _AFRO_TRIBAL_PRIOR_TAGS):
                return True

    # (d) Mix-name / title token
    mix_pool_parts = [track.mix_canonical or "", track.title_canonical or ""]
    if file_record:
        mix_pool_parts.append(file_record.embedded_title or "")
        mix_pool_parts.append(file_record.embedded_comment or "")
    if _AFRO_TRIBAL_KEYWORDS_RE.search(" | ".join(mix_pool_parts)):
        return True

    return False


# ── Priors loading (spec §6.7) ──────────────────────────────────────────────

_ARTIST_PRIORS_PATH = Path(__file__).parent / "data" / "artist_priors.json"
_LABEL_PRIORS_PATH = Path(__file__).parent / "data" / "label_priors.json"


def _load_priors_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@functools.lru_cache(maxsize=1)
def load_artist_priors() -> dict[str, Any]:
    return _load_priors_json(_ARTIST_PRIORS_PATH)


@functools.lru_cache(maxsize=1)
def load_label_priors() -> dict[str, Any]:
    return _load_priors_json(_LABEL_PRIORS_PATH)


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
    ucache: Any | None = None,
) -> dict[str, float]:
    """Build a DictVectorizer-ready feature map from all available evidence.

    Provider genre fields are intentionally ordinary text features. They are
    never treated as labels or hard overrides by this extractor.

    When ``ucache`` is provided and ``file_record`` has a usable file name,
    dense audio features (DSP scalars, CLAP embedding, vocal-stem scalars)
    are looked up from the universal cache and added to the feature map.
    Cache misses are silent — the helpers no-op so models trained without
    these features stay usable.
    """
    track = track or LogicalTrack()
    observations = observations or []
    label_row = label_row or {}
    features: dict[str, float] = {}

    _add_identity_features(features, track, file_record, label_row)
    _add_tagger_features(features, track, observations, label_row)
    _add_mix_name_features(features, track, file_record)
    _add_disambiguation_flags(features, track, file_record)
    if feature_mode not in {"internal", "external"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    if feature_mode == "external":
        _add_provider_features(features, observations, file_record, label_row)
        _add_audio_features(features, observations)
        _add_priors_features(features, track)
        cache_key = _cache_key_from_file_record(file_record) if ucache is not None else None
        if ucache is not None and cache_key is not None:
            _add_dsp_features(features, ucache, cache_key)
            _add_clap_features(features, ucache, cache_key)
            _add_vocal_stem_features(features, ucache, cache_key)
    _add_afro_tribal_eligibility_feature(features, track, observations, file_record)

    return features


def _cache_key_from_file_record(
    file_record: FileRecord | None,
) -> tuple[str, float | None] | None:
    """Build a ``(filename, duration)`` cache key from a FileRecord.

    Uses ``quick_duration(path_abs)`` so the key matches the convention used by
    the grouper and registry (rounded librosa duration, not the persisted
    ``audio_duration_sec`` which can drift by a fraction of a second).
    """
    if file_record is None:
        return None
    filename = file_record.file_name or ""
    if not filename and file_record.path_abs:
        filename = Path(file_record.path_abs).name
    if not filename:
        return None
    duration: float | None = None
    if file_record.path_abs:
        try:
            from dj_tagger.universal_cache import quick_duration

            duration = quick_duration(file_record.path_abs)
        except Exception:
            duration = None
    if duration is None and file_record.audio_duration_sec:
        try:
            duration = float(file_record.audio_duration_sec)
        except (TypeError, ValueError):
            duration = None
    return (filename, duration)


def _add_dsp_features(
    features: dict[str, float],
    ucache: Any,
    cache_key: tuple[str, float | None],
) -> None:
    """Emit ``num:dsp:<name>`` scalars for the curated DSP feature set.

    No-op on cache miss (track wasn't analyzed yet, or analyzed before the
    DSP layer existed).
    """
    filename, duration = cache_key
    try:
        data = ucache.get_track(filename, duration, "dsp")
    except Exception:
        return
    if not isinstance(data, dict):
        return
    try:
        from dj_grouper.features.dsp import DSP_CURATED_NAMES
    except ImportError:
        return
    for name in DSP_CURATED_NAMES:
        value = _num(data.get(name))
        if value is not None:
            features[f"num:dsp:{name}"] = value


def _add_clap_features(
    features: dict[str, float],
    ucache: Any,
    cache_key: tuple[str, float | None],
) -> None:
    """Emit 512 L2-normalized CLAP audio-embedding dims as ``clap:d<i>``.

    The CLAP embedding ships as a 512-dim float vector from
    laion_clap HTSAT-tiny. We L2-normalize at emission so the vector lives on
    the unit sphere — this keeps the values in a tight range so LR's L2
    penalty doesn't trivially zero them out next to small text features, and
    leaves XGB (tree-based, scale-invariant) unaffected.
    """
    filename, duration = cache_key
    try:
        data = ucache.get_track(filename, duration, "clap")
    except Exception:
        return
    if data is None:
        return
    try:
        import numpy as np
    except ImportError:
        return
    try:
        arr = np.asarray(data, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return
    if arr.size == 0:
        return
    norm = float(np.linalg.norm(arr))
    if not (norm > 0) or not np.isfinite(norm):
        return
    unit = arr / norm
    for i, value in enumerate(unit):
        v = float(value)
        if np.isfinite(v):
            features[f"clap:d{i}"] = v


def _add_vocal_stem_features(
    features: dict[str, float],
    ucache: Any,
    cache_key: tuple[str, float | None],
) -> None:
    """Emit Demucs per-stem scalars as model features.

    Three families of keys (all read from the ``vocal_stem`` cache layer):
      - ``num:vocal_stem:<field>``  — legacy vocals-only metrics (RMS, mix
        ratio, activity, envelope_var). Kept so older models keep working.
      - ``num:stem:<name>:<metric>`` — per-stem metrics for drums / bass /
        other / vocals (rms_db, mix_ratio_db, activity_frac, centroid_hz,
        flatness, onset_rate, zcr, envelope_var).
      - ``num:dominance:<name>``    — fraction of total stem energy by stem.

    The non-vocals stems are typically the strongest single signal for
    subgenre: drum dominance separates tribal/afro from melodic; bass
    centroid + flatness separates amapiano log drums from straight 4/4;
    other-stem onset density separates minimal tools from melodic builders.
    """
    filename, duration = cache_key
    try:
        data = ucache.get_track(filename, duration, "vocal_stem")
    except Exception:
        return
    if not isinstance(data, dict):
        return

    # Legacy vocals metrics (cache schema v1).
    for name in (
        "vocal_stem_rms_db",
        "vocal_stem_mix_ratio_db",
        "vocal_stem_activity_frac",
        "vocal_stem_envelope_var",
    ):
        value = _num(data.get(name))
        if value is not None:
            features[f"num:vocal_stem:{name.removeprefix('vocal_stem_')}"] = value

    # Per-stem metrics (cache schema v2 — additive, optional).
    for key, value in data.items():
        if key.startswith("stem_"):
            num = _num(value)
            if num is None:
                continue
            # key is "stem_<name>_<metric...>"; split on first two underscores.
            remainder = key[len("stem_"):]
            try:
                stem_name, metric = remainder.split("_", 1)
            except ValueError:
                continue
            features[f"num:stem:{stem_name}:{metric}"] = num
        elif key.startswith("dominance_"):
            num = _num(value)
            if num is not None:
                features[f"num:dominance:{key.removeprefix('dominance_')}"] = num


def _add_mix_name_features(
    features: dict[str, float],
    track: LogicalTrack,
    file_record: FileRecord | None,
) -> None:
    """Spec §6.3 — emit `mix:<token>` for recognized phrases in title / mix / filename / comment."""
    pool_parts = [track.mix_canonical or "", track.title_canonical or ""]
    if file_record:
        pool_parts.append(file_record.file_name or "")
        pool_parts.append(file_record.embedded_title or "")
        pool_parts.append(file_record.embedded_comment or "")
    pool = " ".join(pool_parts).lower()
    if not pool.strip():
        return
    for phrase, token in _MIX_NAME_TOKENS.items():
        if phrase in pool:
            features[f"mix:{token}"] = 1.0


def _add_disambiguation_flags(
    features: dict[str, float],
    track: LogicalTrack,
    file_record: FileRecord | None,
) -> None:
    """Spec §6.6 — `disamb:<term>` for ambiguous words in title / mix / filename / embedded_genre."""
    pool_parts = [track.title_canonical or "", track.mix_canonical or ""]
    if file_record:
        pool_parts.append(file_record.file_name or "")
        pool_parts.append(file_record.embedded_genre or "")
        pool_parts.append(file_record.embedded_comment or "")
    pool = " ".join(pool_parts).lower()
    for term in _DISAMBIGUATION_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", pool):
            features[f"disamb:{term}"] = 1.0


def _add_priors_features(features: dict[str, float], track: LogicalTrack) -> None:
    """Spec §6.7 — `prior:artist:<cat>=<weight>` / `prior:label:<cat>=<weight>` from JSON priors."""
    artist = (track.artist_canonical or "").strip()
    label = (track.label_canonical or "").strip()
    if artist:
        for prior in load_artist_priors().get(artist, []):
            if isinstance(prior, dict):
                cat = prior.get("category_id", "")
                weight = float(prior.get("weight", 0.0))
                if cat and weight > 0:
                    features[f"prior:artist:{cat}"] = weight
    if label:
        for prior in load_label_priors().get(label, []):
            if isinstance(prior, dict):
                cat = prior.get("category_id", "")
                weight = float(prior.get("weight", 0.0))
                if cat and weight > 0:
                    features[f"prior:label:{cat}"] = weight


def _add_afro_tribal_eligibility_feature(
    features: dict[str, float],
    track: LogicalTrack,
    observations: list[SourceObservation],
    file_record: FileRecord | None,
) -> None:
    """Spec §6.2 — surface eligibility flag so LR can learn it as a feature.

    The hard candidate-filter rule is enforced post-prediction in dj_model.py
    using is_eligible_for_afro_tribal() directly. The training-time feature here
    gives the LR a chance to learn correlations.
    """
    eligible = is_eligible_for_afro_tribal(
        track,
        observations,
        file_record,
        artist_priors=load_artist_priors(),
        label_priors=load_label_priors(),
    )
    if eligible:
        features["eligibility:afro_tribal_allowed"] = 1.0
    else:
        features["eligibility:afro_tribal_blocked"] = 1.0


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
        features[f"bpm_role:{_bpm_semantic_band(bpm)}"] = 1.0


def _add_provider_features(
    features: dict[str, float],
    observations: list[SourceObservation],
    file_record: FileRecord | None,
    label_row: dict[str, str],
) -> None:
    if file_record:
        _add_text(features, "provider:embedded_genre", file_record.embedded_genre)
        _add_numeric(features, "embedded_bpm", _num(file_record.embedded_bpm))
        # Spec §6.1 — emit normalized provider-genre tokens with elevated weight
        # so the LR can lean on them ahead of tagger fields.
        _add_provider_genre_tokens(features, file_record.embedded_genre, source="embedded")

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
        # Spec §6.1 — elevated provider-genre tokens
        _add_provider_genre_tokens(features, obs.genre, source=source)
        _add_provider_genre_tokens(features, obs.genres_all, source=source)


# Spec §6.1 source-weight uplift values. The DictVectorizer treats these as
# scalar feature weights; the LR multiplies them by learned coefficients, so
# values > 1 effectively up-weight the signal vs unit-valued tagger features.
_PROVIDER_SOURCE_WEIGHTS: dict[str, float] = {
    "rekordbox": 1.5,   # manually curated, highest trust
    "embedded": 1.2,    # embedded tag genre, second-highest
    "songstats": 1.3,   # provider with full genres_all
    "spotify": 1.1,
}


def _add_provider_genre_tokens(
    features: dict[str, float],
    value: str | None,
    *,
    source: str,
) -> None:
    """Emit `provider_genre:<token>` with source-weighted value for each genre token."""
    if not value:
        return
    weight = _PROVIDER_SOURCE_WEIGHTS.get(source, 1.0)
    pool = _normalize_text(value)
    if not pool:
        return
    # genres_all comes as semicolon-separated; split on common separators
    for piece in re.split(r"[;,/|]+", pool):
        token = piece.strip()
        if not token:
            continue
        features[f"provider_genre:{token.replace(' ', '_')}"] = weight


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
        # Spotify popularity (0-100). Normalize to 0-1 for the numeric feature
        # and emit a 3-band categorical for the LR to lean on without scaling.
        popularity_raw = _num(getattr(obs, "popularity", ""))
        if popularity_raw is not None and popularity_raw >= 0:
            popularity = max(0.0, min(1.0, popularity_raw / 100.0))
            features[f"num:{source}:popularity"] = popularity
            features[f"popularity_band:{source}:{_popularity_band(popularity)}"] = 1.0
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


def _popularity_band(popularity: float) -> str:
    """3-band popularity bucket; mid covers the most common Spotify range."""
    if popularity < 0.25:
        return "lo"
    if popularity < 0.60:
        return "mid"
    return "hi"


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


def _bpm_semantic_band(bpm: float) -> str:
    """Spec §6.5 — DJ-set-role-oriented BPM bands."""
    if bpm < 100:
        return "slow"
    if bpm < 115:
        return "warmup"
    if bpm < 122:
        return "builder"
    if bpm < 128:
        return "driver"
    if bpm < 134:
        return "peak"
    if bpm < 145:
        return "peak_plus"
    if 150 <= bpm <= 180:
        return "dnb"
    return "other"
