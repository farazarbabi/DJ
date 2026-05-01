"""Deterministic 3-level genre classifier constrained to a reference taxonomy."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dj_tagger.moods import MOOD_CUES_BY_CODE, normalize_mood_code
from dj_tagger.vocals import (
    VOCAL_PROFILE_CUES_BY_CODE,
    has_vocal_content,
    normalize_vocal_profile,
)

from ..models import FileRecord, LogicalTrack, SourceObservation
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

TAXONOMY_VERSION = "genre-taxonomy-3level-v1"

SOURCE_WEIGHTS = {
    "trusted_dj_source_genre": 1.00,
    "rekordbox_genre_manual": 0.90,
    "rekordbox_genre_unknown_origin": 0.75,
    "embedded_genre": 0.60,
    "songstats_track_genre": 0.70,
    "spotify_track_genre": 0.65,
    "spotify_artist_genre": 0.45,
    "genres_all": 0.70,
    "label_prior": 0.65,
    "artist_prior": 0.55,
    "remix_artist_prior": 0.65,
    "title_or_mix_text_cue": 0.50,
    "filename_cue": 0.35,
    "audio_feature": 0.35,
    "local_tagger_field": 0.50,
    "derived_dj_signal": 0.45,
}

TRIBAL_MOOD_CODES = {"TRIB", "ORG"}
DEEP_MOOD_CODES = {"DEEP", "SUB", "MIN"}
MELODIC_MOOD_CODES = {"MEL", "EMO", "EUP", "SOUL", "WARM", "CIN", "SUN"}
DARK_MOOD_CODES = {"DRK", "TENS", "WHSE", "GRIT"}
HYPNOTIC_MOOD_CODES = {"HYPN", "MIN"}

BROAD_SUBGENRE_LABELS = {
    "Ambient",
    "Classic Rock",
    "Dark Disco",
    "Downtempo",
    "Drum & Bass",
    "Dubstep",
    "Electro",
    "Hypnotic Techno",
    "Indie Dance",
    "Jungle",
    "Nu-Disco",
    "Organic House",
    "Psytrance",
    "Rave",
    "UK Garage",
}


@dataclass(frozen=True)
class TaxonomyPath:
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None

    def label(self) -> str:
        return self.subgenre or self.genre or self.family or ""

    def full_label(self) -> str:
        return " > ".join(part for part in (self.family, self.genre, self.subgenre) if part)

    def id(self) -> str:
        return _slug(self.full_label())


@dataclass
class EvidenceItem:
    source_field: str
    raw_value: str | float | int | None = None
    normalized_value: str | None = None
    matched_label: str | None = None
    matched_path: TaxonomyPath | None = None
    evidence_type: str = ""
    weight: float = 0.0
    score_delta: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_field": self.source_field,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "matched_label": self.matched_label,
            "matched_path": _path_to_dict(self.matched_path),
            "evidence_type": self.evidence_type,
            "weight": round(self.weight, 3),
            "score_delta": round(self.score_delta, 3),
            "explanation": self.explanation,
        }


@dataclass
class GenreCandidate:
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def path(self) -> TaxonomyPath:
        return TaxonomyPath(self.family, self.genre, self.subgenre)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "genre": self.genre,
            "subgenre": self.subgenre,
            "score": round(self.score, 3),
            "evidence": self.evidence,
        }


@dataclass
class GenreClassificationResult:
    track_id: str | None = None
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None
    confidence: float = 0.0
    confidence_level: str = "unknown"
    evidence: dict[str, list[str]] = field(default_factory=dict)
    alternatives: list[GenreCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def taxonomy_id(self) -> str:
        return self.path().id()

    @property
    def label(self) -> str:
        return self.path().full_label()

    def path(self) -> TaxonomyPath:
        return TaxonomyPath(self.family, self.genre, self.subgenre)


# Backwards-compatible public name from the original flat taxonomy classifier.
TaxonomyResult = GenreClassificationResult


class GenreTaxonomy:
    """Reference taxonomy with validated family -> genre -> subgenre paths."""

    def __init__(self, data: dict[str, dict[str, list[str]]]) -> None:
        if not isinstance(data, dict) or "categories" in data:
            raise ValueError("Expected a 3-level taxonomy shaped as family -> genre -> subgenres")

        self._data: dict[str, dict[str, list[str]]] = {}
        self._paths_by_label: dict[str, list[TaxonomyPath]] = defaultdict(list)
        self._labels: list[tuple[str, str, TaxonomyPath]] = []

        for family, genres in data.items():
            if not isinstance(genres, dict):
                raise ValueError(f"Family {family!r} must contain a genre map")
            self._data[family] = {}
            self._index_label(family, TaxonomyPath(family=family))
            for genre, subgenres in genres.items():
                if not isinstance(subgenres, list):
                    raise ValueError(f"Genre {family!r}/{genre!r} must contain a subgenre list")
                self._data[family][genre] = list(subgenres)
                self._index_label(genre, TaxonomyPath(family=family, genre=genre))
                for subgenre in subgenres:
                    self._index_label(
                        subgenre,
                        TaxonomyPath(family=family, genre=genre, subgenre=subgenre),
                    )

    def families(self) -> list[str]:
        return list(self._data)

    def genres(self, family: str | None = None) -> list[str]:
        if family:
            return list(self._data.get(family, {}))
        return [genre for genres in self._data.values() for genre in genres]

    def subgenres(self, family: str | None = None, genre: str | None = None) -> list[str]:
        if family and genre:
            return list(self._data.get(family, {}).get(genre, []))
        if family:
            return [sub for subs in self._data.get(family, {}).values() for sub in subs]
        return [sub for genres in self._data.values() for subs in genres.values() for sub in subs]

    def validate_path(
        self,
        family: str | None,
        genre: str | None,
        subgenre: str | None,
    ) -> bool:
        if not family:
            return genre is None and subgenre is None
        if family not in self._data:
            return False
        if not genre:
            return subgenre is None
        if genre not in self._data[family]:
            return False
        if not subgenre:
            return True
        return subgenre in self._data[family][genre]

    def find_paths_for_label(self, label: str) -> list[TaxonomyPath]:
        return list(self._paths_by_label.get(_normalize_text(label), []))

    def labels(self) -> list[tuple[str, str, TaxonomyPath]]:
        """Return (normalized label, display label, path) for all taxonomy labels."""
        return list(self._labels)

    def _index_label(self, label: str, path: TaxonomyPath) -> None:
        norm = _normalize_text(label)
        if path not in self._paths_by_label[norm]:
            self._paths_by_label[norm].append(path)
            self._labels.append((norm, label, path))


@dataclass
class _Signal:
    source_field: str
    raw_value: Any
    normalized_values: list[str]
    weight: float
    evidence_type: str


@dataclass
class _Cue:
    source_field: str
    raw_value: Any
    terms: list[str]
    weight: float
    evidence_type: str


@dataclass
class _EvidenceBundle:
    genre_signals: list[_Signal] = field(default_factory=list)
    text_signals: list[_Signal] = field(default_factory=list)
    cue_signals: list[_Cue] = field(default_factory=list)
    source_genres_used: set[str] = field(default_factory=set)
    metadata_signals: list[str] = field(default_factory=list)
    audio_feature_signals: list[str] = field(default_factory=list)
    tagger_signals: list[str] = field(default_factory=list)
    bpm: float | None = None
    tagger_energy_int: int | None = None
    tagger_vibes: set[str] = field(default_factory=set)
    tagger_vocals: set[str] = field(default_factory=set)
    tagger_structure: str = ""
    acousticness: float | None = None
    danceability: float | None = None
    energy: float | None = None
    instrumentalness: float | None = None
    valence: float | None = None


class _Scorer:
    def __init__(self, taxonomy: GenreTaxonomy) -> None:
        self.taxonomy = taxonomy
        self.family_scores: dict[str, float] = defaultdict(float)
        self.genre_scores: dict[tuple[str, str], float] = defaultdict(float)
        self.subgenre_scores: dict[tuple[str, str, str], float] = defaultdict(float)
        self.evidence_by_path: dict[TaxonomyPath, list[EvidenceItem]] = defaultdict(list)
        self.matched_terms: set[str] = set()
        self.warnings: list[str] = []
        self._has_source_genres = False
        self._has_audio_features = False
        self._has_tagger_signals = False

    def score(self, evidence: _EvidenceBundle) -> None:
        self._has_source_genres = bool(evidence.source_genres_used)
        self._has_audio_features = any(
            value is not None
            for value in (
                evidence.acousticness,
                evidence.danceability,
                evidence.energy,
                evidence.instrumentalness,
                evidence.valence,
            )
        )
        self._has_tagger_signals = bool(evidence.tagger_signals)

        for signal in evidence.genre_signals:
            for value in signal.normalized_values:
                self._score_matches(signal, value, allow_phrase=True)

        for signal in evidence.text_signals:
            for value in signal.normalized_values:
                self._score_matches(signal, value, allow_phrase=True, text_signal=True)

        self._apply_cue_modifiers(evidence)
        self._apply_bpm_priors(evidence.bpm)
        self._apply_audio_priors(evidence)
        self._apply_disambiguation(evidence)
        self._apply_generic_source_priors(evidence)
        self._apply_tagger_combination_priors(evidence)
        self._detect_source_disagreement()

    def selected_path(self) -> tuple[TaxonomyPath, str | None]:
        family, family_score, family_margin = self._best_family()
        global_family, global_genre, global_genre_score = self._best_global_genre()
        if (
            global_family
            and global_genre
            and (not family or family_score < 0.25)
            and global_genre_score >= 0.55
        ):
            family = global_family
            family_score = self._family_selection_score(global_family)
        if not family or family_score < 0.25:
            return TaxonomyPath(), "Insufficient genre evidence."

        genre, genre_score, genre_margin = self._best_genre(family)
        if (
            global_family
            and global_genre
            and global_family != family
            and global_genre_score >= 0.55
            and (not genre or genre_score < 0.25 or global_genre_score >= genre_score + 0.25)
        ):
            family = global_family
            genre = global_genre
            genre_score = global_genre_score
            genre_margin = self._best_genre(family)[2]
        if not genre or genre_score < 0.25:
            return TaxonomyPath(family=family), "Family inferred, but genre/subgenre evidence is weak."

        subgenre, sub_score, sub_margin = self._best_subgenre(family, genre)
        if not subgenre:
            return TaxonomyPath(family=family, genre=genre), None

        if sub_score < 0.26:
            return (
                TaxonomyPath(family=family, genre=genre),
                "Subgenre left blank because subgenre evidence is weak.",
            )
        if sub_margin < 0.06 and sub_score < 0.90:
            return (
                TaxonomyPath(family=family, genre=genre),
                "Subgenre left blank because candidate margin was too small.",
            )

        # Family/genre margins are useful for confidence, but should not force a null
        # when a direct taxonomy match has already made the path valid.
        _ = (family_margin, genre_margin)
        return TaxonomyPath(family=family, genre=genre, subgenre=subgenre), None

    def alternatives(self, selected: TaxonomyPath, limit: int = 3) -> list[GenreCandidate]:
        candidates: list[GenreCandidate] = []
        for family in self.taxonomy.families():
            family_score = self.family_scores.get(family, 0.0)
            if family_score <= 0:
                continue
            for genre in self.taxonomy.genres(family):
                genre_score = self.genre_scores.get((family, genre), 0.0)
                if genre_score <= 0:
                    continue
                best_sub = None
                best_sub_score = 0.0
                for subgenre in self.taxonomy.subgenres(family, genre):
                    score = self.subgenre_scores.get((family, genre, subgenre), 0.0)
                    if score > best_sub_score:
                        best_sub = subgenre
                        best_sub_score = score
                subgenre = best_sub if best_sub_score >= 0.30 else None
                path = TaxonomyPath(family, genre, subgenre)
                if path == selected:
                    continue
                raw_score = family_score * 0.25 + genre_score + best_sub_score * 0.75
                candidates.append(
                    GenreCandidate(
                        family=family,
                        genre=genre,
                        subgenre=subgenre,
                        score=_score_to_display(raw_score),
                        evidence=self._evidence_strings(path),
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]

    def confidence(self, selected: TaxonomyPath) -> float:
        if not selected.family:
            return 0.22

        family_score = self.family_scores.get(selected.family, 0.0)
        genre_score = self.genre_scores.get((selected.family, selected.genre), 0.0) if selected.genre else 0.0
        sub_score = (
            self.subgenre_scores.get((selected.family, selected.genre, selected.subgenre), 0.0)
            if selected.genre and selected.subgenre
            else 0.0
        )
        total = family_score * 0.5 + genre_score + sub_score * 0.8

        level_bonus = 0.18
        if selected.genre:
            level_bonus += 0.22
        if selected.subgenre:
            level_bonus += 0.14

        family_margin = self._best_family()[2]
        genre_margin = self._best_genre(selected.family)[2] if selected.family else 0.0
        sub_margin = (
            self._best_subgenre(selected.family, selected.genre)[2]
            if selected.family and selected.genre
            else 0.0
        )
        margin_bonus = min(0.18, 0.08 * family_margin + 0.06 * genre_margin + 0.04 * sub_margin)
        strength_bonus = 0.34 * (1.0 - math.exp(-max(total, 0.0) / 3.6))
        missing_penalty = 0.0 if selected.subgenre else 0.08 if selected.genre else 0.16

        data_penalty = 0.0
        if not self._has_source_genres:
            data_penalty += 0.08
        if not self._has_audio_features:
            data_penalty += 0.04
        if not self._has_tagger_signals:
            data_penalty += 0.04

        raw = 0.20 + level_bonus + strength_bonus + margin_bonus - missing_penalty - data_penalty
        return round(max(0.2, min(0.96, raw)), 3)

    def evidence_summary(self, evidence: _EvidenceBundle, selected: TaxonomyPath) -> dict[str, list[str]]:
        selected_evidence = self._evidence_strings(selected)
        summary = {
            "matched_terms": sorted(self.matched_terms),
            "source_genres_used": sorted(evidence.source_genres_used),
            "metadata_signals": _dedupe(evidence.metadata_signals),
            "audio_feature_signals": _dedupe(evidence.audio_feature_signals),
            "tagger_signals": _dedupe(evidence.tagger_signals),
            "selected_path_evidence": selected_evidence,
        }
        return {key: values for key, values in summary.items() if values}

    def _score_matches(
        self,
        signal: _Signal,
        value: str,
        *,
        allow_phrase: bool,
        text_signal: bool = False,
    ) -> None:
        matches = _matches_for_value(value, self.taxonomy, allow_phrase=allow_phrase)
        for label, path, match_type, strength in matches:
            multiplier = 0.58 if text_signal else 1.0
            delta = signal.weight * strength * multiplier
            self._add_path_score(
                path,
                delta,
                EvidenceItem(
                    source_field=signal.source_field,
                    raw_value=signal.raw_value,
                    normalized_value=value,
                    matched_label=label,
                    matched_path=path,
                    evidence_type=f"{signal.evidence_type}:{match_type}",
                    weight=signal.weight,
                    score_delta=delta,
                    explanation=f"{signal.source_field} matched {label}",
                ),
            )

    def _add_path_score(self, path: TaxonomyPath, delta: float, evidence: EvidenceItem) -> None:
        if delta <= 0 or not path.family:
            return
        self.matched_terms.add(_normalize_text(evidence.matched_label or path.label()))

        if path.genre is None:
            self.family_scores[path.family] += delta
            self.evidence_by_path[TaxonomyPath(path.family)].append(evidence)
            return

        if path.subgenre is None:
            self.family_scores[path.family] += delta * 0.30
            self.genre_scores[(path.family, path.genre)] += delta
            self.evidence_by_path[TaxonomyPath(path.family, path.genre)].append(evidence)
            return

        family_delta = delta * 0.22
        genre_delta = delta * 0.52
        subgenre_delta = delta
        if (
            path.subgenre in BROAD_SUBGENRE_LABELS
            and (":exact" in evidence.evidence_type or ":alias" in evidence.evidence_type)
        ):
            # Labels such as "Organic House" or "Hypnotic Techno" often identify
            # the style cluster, while tagger/text cues select the final subgenre.
            family_delta = delta * 0.28
            genre_delta = delta * 0.78
            subgenre_delta = delta * 0.42

        self.family_scores[path.family] += family_delta
        self.genre_scores[(path.family, path.genre)] += genre_delta
        self.subgenre_scores[(path.family, path.genre, path.subgenre)] += subgenre_delta
        self.evidence_by_path[path].append(evidence)

    def _add_family_prior(self, family: str, delta: float, evidence: EvidenceItem) -> None:
        if family not in self.taxonomy.families() or delta <= 0:
            return
        self.family_scores[family] += delta
        self.evidence_by_path[TaxonomyPath(family)].append(evidence)

    def _add_genre_prior(
        self,
        family: str,
        genre: str,
        delta: float,
        evidence: EvidenceItem,
    ) -> None:
        if not self.taxonomy.validate_path(family, genre, None) or delta <= 0:
            return
        self.family_scores[family] += delta * 0.18
        self.genre_scores[(family, genre)] += delta
        self.evidence_by_path[TaxonomyPath(family, genre)].append(evidence)

    def _add_subgenre_modifier(
        self,
        family: str,
        genre: str,
        subgenre: str,
        delta: float,
        evidence: EvidenceItem,
    ) -> None:
        if not self.taxonomy.validate_path(family, genre, subgenre) or delta <= 0:
            return
        self.subgenre_scores[(family, genre, subgenre)] += delta
        self.evidence_by_path[TaxonomyPath(family, genre, subgenre)].append(evidence)

    def _apply_cue_modifiers(self, evidence: _EvidenceBundle) -> None:
        for cue in evidence.cue_signals:
            for term in cue.terms:
                norm_term = _normalize_text(term)
                if not norm_term:
                    continue
                for family in self.taxonomy.families():
                    for genre in self.taxonomy.genres(family):
                        genre_norm = _normalize_text(genre)
                        if _contains_phrase(genre_norm, norm_term):
                            self._add_genre_prior(
                                family,
                                genre,
                                cue.weight * 0.22,
                                _cue_evidence(cue, term, genre, TaxonomyPath(family, genre)),
                            )
                        for subgenre in self.taxonomy.subgenres(family, genre):
                            sub_norm = _normalize_text(subgenre)
                            if _contains_phrase(sub_norm, norm_term):
                                self._add_subgenre_modifier(
                                    family,
                                    genre,
                                    subgenre,
                                    cue.weight * _cue_strength(norm_term, sub_norm),
                                    _cue_evidence(
                                        cue,
                                        term,
                                        subgenre,
                                        TaxonomyPath(family, genre, subgenre),
                                    ),
                                )

    def _apply_bpm_priors(self, bpm: float | None) -> None:
        if bpm is None:
            return
        priors: list[tuple[str, str | None, float, str]] = []
        if 60 <= bpm <= 95:
            priors.extend([
                ("Downtempo / Slow Electronic", None, 0.18, "60-95 BPM slow range"),
                ("Ambient / Experimental", None, 0.15, "60-95 BPM ambient range"),
            ])
        if 95 <= bpm <= 115:
            priors.extend([
                ("Indie Dance / Dark Disco / Nu-Disco", None, 0.15, "95-115 BPM slow disco range"),
                ("Downtempo / Slow Electronic", "Slow Club Music", 0.16, "95-115 BPM slow club range"),
            ])
        if 115 <= bpm <= 124:
            priors.extend([
                ("House", "Deep / Minimal / Groove House", 0.18, "115-124 BPM house range"),
                ("House", "Organic / Afro / Tribal House", 0.16, "115-124 BPM organic house range"),
                ("Indie Dance / Dark Disco / Nu-Disco", None, 0.14, "115-124 BPM indie dance range"),
            ])
        if 122 <= bpm <= 128:
            priors.extend([
                ("House", "Tech House", 0.18, "122-128 BPM tech-house range"),
                ("House", "Melodic / Progressive House", 0.17, "122-128 BPM progressive house range"),
            ])
        if 126 <= bpm <= 134:
            priors.extend([
                ("Techno", "Driving / Peak-Time Techno", 0.17, "126-134 BPM techno range"),
                ("Techno", "Minimal / Hypnotic Techno", 0.15, "126-134 BPM hypnotic techno range"),
                ("Trance / Progressive / Psy", None, 0.12, "126-134 BPM trance range"),
            ])
        if 130 <= bpm <= 145:
            priors.extend([
                ("Garage / UK Bass / Breaks", None, 0.14, "130-145 BPM garage/breaks range"),
                ("Donk / Hard Dance / Rave", None, 0.14, "130-145 BPM hard dance range"),
                ("Trance / Progressive / Psy", None, 0.14, "130-145 BPM trance range"),
            ])
        if 160 <= bpm <= 180:
            priors.append(("Drum & Bass / Jungle", None, 0.26, "160-180 BPM DnB range"))

        for family, genre, delta, reason in priors:
            item = EvidenceItem(
                source_field="tagger_bpm",
                raw_value=bpm,
                evidence_type="bpm_prior",
                weight=SOURCE_WEIGHTS["derived_dj_signal"],
                score_delta=delta,
                explanation=reason,
            )
            if genre:
                self._add_genre_prior(family, genre, delta, item)
            else:
                self._add_family_prior(family, delta, item)

    def _apply_audio_priors(self, evidence: _EvidenceBundle) -> None:
        if evidence.danceability is not None and evidence.danceability >= 0.72:
            for family, delta in (
                ("House", 0.08),
                ("Indie Dance / Dark Disco / Nu-Disco", 0.06),
                ("Downtempo / Slow Electronic", 0.05),
            ):
                self._add_family_prior(
                    family,
                    delta,
                    EvidenceItem(
                        source_field="danceability",
                        raw_value=evidence.danceability,
                        evidence_type="audio_feature",
                        weight=SOURCE_WEIGHTS["audio_feature"],
                        score_delta=delta,
                        explanation="high danceability supports DJ-oriented electronic styles",
                    ),
                )
        if evidence.acousticness is not None and evidence.acousticness >= 0.65:
            self._add_family_prior(
                "Rock / Metal / Guitar-Based",
                0.32,
                EvidenceItem(
                    source_field="acousticness",
                    raw_value=evidence.acousticness,
                    evidence_type="audio_feature",
                    weight=SOURCE_WEIGHTS["audio_feature"],
                    score_delta=0.32,
                    explanation="high acousticness supports guitar-based music",
                ),
            )
        if evidence.instrumentalness is not None and evidence.instrumentalness >= 0.80:
            for family in ("Techno", "House", "Ambient / Experimental"):
                self._add_family_prior(
                    family,
                    0.08,
                    EvidenceItem(
                        source_field="instrumentalness",
                        raw_value=evidence.instrumentalness,
                        evidence_type="audio_feature",
                        weight=SOURCE_WEIGHTS["audio_feature"],
                        score_delta=0.08,
                        explanation="high instrumentalness supports instrumental DJ material",
                    ),
                )
            if evidence.valence is not None and evidence.valence <= 0.30:
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    "Deep Downtempo",
                    0.32,
                    EvidenceItem(
                        source_field="instrumentalness_valence",
                        raw_value=f"{evidence.instrumentalness},{evidence.valence}",
                        evidence_type="audio_feature",
                        weight=SOURCE_WEIGHTS["audio_feature"],
                        score_delta=0.32,
                        explanation="high instrumentalness plus low valence supports Deep Downtempo",
                    ),
                )
        if evidence.valence is not None and evidence.valence <= 0.30:
            cue = _Cue(
                source_field="valence",
                raw_value=evidence.valence,
                terms=["dark", "nocturnal", "industrial"],
                weight=0.18,
                evidence_type="audio_feature",
            )
            self._apply_cue_modifiers(
                _EvidenceBundle(cue_signals=[cue])
            )

    def _apply_disambiguation(self, evidence: _EvidenceBundle) -> None:
        text = " ".join(
            signal.normalized_values[0]
            for signal in [*evidence.genre_signals, *evidence.text_signals]
            if signal.normalized_values
        )
        terms = set(text.split())

        if "garage" in terms:
            uk_terms = {"ukg", "2", "step", "bassline", "breaks", "breakbeat", "speed"}
            rock_terms = {"rock", "punk", "psych", "psychedelic", "guitar"}
            house_terms = {"house", "soulful"}
            if terms & uk_terms:
                self._add_genre_prior(
                    "Garage / UK Bass / Breaks",
                    "UK Garage",
                    0.42,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="garage_disambiguation",
                        weight=1.0,
                        score_delta=0.42,
                        explanation="garage with UKG/2-step/bassline cues",
                    ),
                )
            if terms & rock_terms or (evidence.acousticness is not None and evidence.acousticness >= 0.60):
                self._add_subgenre_modifier(
                    "Rock / Metal / Guitar-Based",
                    "Garage / Punk / Post-Punk",
                    "Garage Rock",
                    0.50,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="garage_disambiguation",
                        weight=1.0,
                        score_delta=0.50,
                        explanation="garage with rock/punk/acoustic cues",
                    ),
                )
            if terms & house_terms:
                self._add_family_prior(
                    "House",
                    0.22,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="garage_disambiguation",
                        weight=1.0,
                        score_delta=0.22,
                        explanation="garage with house cues",
                    ),
                )

        if "progressive" in terms:
            if "rock" in terms or "metal" in terms or (evidence.acousticness or 0.0) >= 0.60:
                self._add_subgenre_modifier(
                    "Rock / Metal / Guitar-Based",
                    "Classic Rock",
                    "Progressive Rock",
                    0.40,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="progressive_disambiguation",
                        weight=1.0,
                        score_delta=0.40,
                        explanation="progressive with rock/acoustic cues",
                    ),
                )
            elif "trance" in terms or (evidence.bpm is not None and 128 <= evidence.bpm <= 140):
                self._add_genre_prior(
                    "Trance / Progressive / Psy",
                    "Progressive",
                    0.26,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="progressive_disambiguation",
                        weight=1.0,
                        score_delta=0.26,
                        explanation="progressive with trance/up-tempo cues",
                    ),
                )
            elif "house" in terms or (evidence.bpm is not None and 120 <= evidence.bpm <= 128):
                self._add_genre_prior(
                    "House",
                    "Melodic / Progressive House",
                    0.30,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="progressive_disambiguation",
                        weight=1.0,
                        score_delta=0.30,
                        explanation="progressive with house/BPM cues",
                    ),
                )

        if _contains_phrase(text, "organic house"):
            self._add_genre_prior(
                "House",
                "Organic / Afro / Tribal House",
                0.38,
                EvidenceItem(
                    source_field="disambiguation",
                    raw_value=text,
                    evidence_type="organic_house_disambiguation",
                    weight=1.0,
                    score_delta=0.38,
                    explanation="organic house is a house style even when downtempo is also present",
                ),
            )
            if "desert" in terms:
                self._add_subgenre_modifier(
                    "House",
                    "Organic / Afro / Tribal House",
                    "Desert House",
                    0.36,
                    EvidenceItem(
                        source_field="disambiguation",
                        raw_value=text,
                        evidence_type="organic_house_disambiguation",
                        weight=1.0,
                        score_delta=0.36,
                        explanation="organic house with desert cues",
                    ),
                )

    def _apply_generic_source_priors(self, evidence: _EvidenceBundle) -> None:
        source_text = _normalize_text(" ".join(evidence.source_genres_used))
        if not source_text:
            return

        generic_electronic = any(
            _contains_phrase(source_text, term)
            for term in ("dance", "electronic", "electronica", "chill electronic")
        )
        has_house = _contains_phrase(source_text, "house")
        has_techno = _contains_phrase(source_text, "techno")
        has_downtempo = _contains_phrase(source_text, "downtempo")
        has_pop = _contains_phrase(source_text, "pop")

        if generic_electronic:
            item = EvidenceItem(
                source_field="generic_source_genre",
                raw_value=source_text,
                evidence_type="generic_electronic_prior",
                weight=SOURCE_WEIGHTS["genres_all"],
                score_delta=0.0,
                explanation="generic dance/electronic source genre needs BPM/tagger disambiguation",
            )
            if evidence.bpm is not None:
                if evidence.bpm < 112:
                    self._add_genre_prior("Downtempo / Slow Electronic", "Slow Club Music", 0.24, item)
                    self._add_genre_prior("Downtempo / Slow Electronic", "Downtempo", 0.16, item)
                elif evidence.bpm < 124:
                    self._add_family_prior("House", 0.22, item)
                    self._add_genre_prior("House", "Melodic / Progressive House", 0.18, item)
                    self._add_genre_prior("House", "Deep / Minimal / Groove House", 0.14, item)
                elif evidence.bpm < 130:
                    self._add_family_prior("House", 0.24, item)
                    self._add_genre_prior("House", "Tech House", 0.16, item)
                    self._add_genre_prior("House", "Melodic / Progressive House", 0.14, item)
                elif evidence.bpm < 146:
                    self._add_family_prior("Techno", 0.20, item)
                    self._add_family_prior("Trance / Progressive / Psy", 0.16, item)
                else:
                    self._add_family_prior("Drum & Bass / Jungle", 0.16, item)

            if evidence.instrumentalness is not None and evidence.instrumentalness >= 0.72:
                self._add_family_prior("House", 0.08, item)
                self._add_family_prior("Techno", 0.08, item)
                self._add_family_prior("Downtempo / Slow Electronic", 0.08, item)

        if has_house:
            house_item = EvidenceItem(
                source_field="source_genre",
                raw_value=source_text,
                evidence_type="broad_house_prior",
                weight=SOURCE_WEIGHTS["genres_all"],
                score_delta=0.24,
                explanation="source genre contains house",
            )
            self._add_family_prior(
                "House",
                0.24,
                house_item,
            )
            if evidence.tagger_vibes & TRIBAL_MOOD_CODES:
                self._add_genre_prior(
                    "House",
                    "Organic / Afro / Tribal House",
                    0.35,
                    house_item,
                )
                self._add_subgenre_modifier(
                    "House",
                    "Organic / Afro / Tribal House",
                    "Tribal House",
                    0.48,
                    house_item,
                )
            elif evidence.tagger_energy_int is not None and evidence.tagger_energy_int <= 2:
                self._add_genre_prior(
                    "House",
                    "Deep / Minimal / Groove House",
                    0.30,
                    house_item,
                )
                self._add_subgenre_modifier(
                    "House",
                    "Deep / Minimal / Groove House",
                    "Vocal Deep House" if any(has_vocal_content(v) for v in evidence.tagger_vocals) else "Deep House",
                    0.28,
                    house_item,
                )
        if has_techno:
            self._add_family_prior(
                "Techno",
                0.24,
                EvidenceItem(
                    source_field="source_genre",
                    raw_value=source_text,
                    evidence_type="broad_techno_prior",
                    weight=SOURCE_WEIGHTS["genres_all"],
                    score_delta=0.24,
                    explanation="source genre contains techno",
                ),
            )
        if has_downtempo:
            self._add_genre_prior(
                "Downtempo / Slow Electronic",
                "Downtempo",
                0.24,
                EvidenceItem(
                    source_field="source_genre",
                    raw_value=source_text,
                    evidence_type="broad_downtempo_prior",
                    weight=SOURCE_WEIGHTS["genres_all"],
                    score_delta=0.24,
                    explanation="source genre contains downtempo",
                ),
            )
        if has_pop:
            pop_item = EvidenceItem(
                source_field="source_genre",
                raw_value=source_text,
                evidence_type="broad_pop_prior",
                weight=SOURCE_WEIGHTS["genres_all"],
                score_delta=0.24,
                explanation="source genre contains pop",
            )
            if generic_electronic or _contains_phrase(source_text, "dance"):
                self._add_subgenre_modifier(
                    "Pop / Alternative / Hypermodern",
                    "Pop",
                    "Dance Pop",
                    0.42,
                    pop_item,
                )
            elif _contains_phrase(source_text, "alternative"):
                self._add_genre_prior(
                    "Pop / Alternative / Hypermodern",
                    "Alternative Pop",
                    0.34,
                    pop_item,
                )
                self._add_subgenre_modifier(
                    "Pop / Alternative / Hypermodern",
                    "Alternative Pop",
                    "Alt-Pop",
                    0.34,
                    pop_item,
                )

    def _apply_tagger_combination_priors(self, evidence: _EvidenceBundle) -> None:
        vibes = evidence.tagger_vibes
        structure = evidence.tagger_structure.upper()
        bpm = evidence.bpm
        energy = evidence.tagger_energy_int
        if not vibes and not structure and bpm is None:
            return

        rolling = structure.endswith("H") or structure in {"16H", "32H", "ROLLING"}
        driving = structure.endswith("D") or structure in {"16D", "32D", "LINEAR"}
        loose = structure.endswith("L")
        groovy = structure.endswith("G")
        broken = structure.endswith("B") or structure == "BREAKS"
        low_valence = evidence.valence is not None and evidence.valence <= 0.30
        high_inst = evidence.instrumentalness is not None and evidence.instrumentalness >= 0.72
        text_blob = " ".join(
            value
            for signal in evidence.text_signals
            for value in signal.normalized_values
        )
        desert_text = any(
            _contains_phrase(text_blob, term)
            for term in ("desert", "middle eastern", "sunset", "ethnic")
        )

        def item(reason: str) -> EvidenceItem:
            return EvidenceItem(
                source_field="tagger_composite",
                raw_value=",".join(sorted(vibes | ({structure} if structure else set()))),
                evidence_type="tagger_combination_prior",
                weight=SOURCE_WEIGHTS["local_tagger_field"],
                score_delta=0.0,
                explanation=reason,
            )

        if vibes & TRIBAL_MOOD_CODES:
            if bpm is not None and bpm < 112:
                self._add_genre_prior(
                    "Downtempo / Slow Electronic",
                    "Psychedelic / Ethnic Downtempo",
                    0.52,
                    item("tribal/organic tagger cues at slow BPM support ethnic downtempo"),
                )
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Psychedelic / Ethnic Downtempo",
                    "Tribal Downtempo",
                    0.36,
                    item("tribal/organic slow-BPM tagger cues support Tribal Downtempo"),
                )
                if rolling or high_inst:
                    self._add_subgenre_modifier(
                        "Downtempo / Slow Electronic",
                        "Psychedelic / Ethnic Downtempo",
                        "Tribal Downtempo",
                        0.28,
                        item("tribal/organic slow rolling instrumental cues support Tribal Downtempo"),
                    )
            else:
                self._add_genre_prior(
                    "House",
                    "Organic / Afro / Tribal House",
                    0.46,
                    item("tribal/organic tagger cues support Organic / Afro / Tribal House"),
                )
                if desert_text:
                    self._add_subgenre_modifier(
                        "House",
                        "Organic / Afro / Tribal House",
                        "Desert House",
                        0.50,
                        item("organic tagger cues with desert/middle-eastern text support Desert House"),
                    )
                elif energy is not None and energy <= 2:
                    self._add_subgenre_modifier(
                        "House",
                        "Organic / Afro / Tribal House",
                        "Organic House",
                        0.28,
                        item("low-energy tribal/organic house cues support Organic House"),
                    )
                if rolling or high_inst:
                    self._add_subgenre_modifier(
                        "House",
                        "Organic / Afro / Tribal House",
                        "Tribal House",
                        0.22,
                        item("rolling tribal tagger cues support Tribal House"),
                    )

        if vibes & DEEP_MOOD_CODES:
            if bpm is not None and bpm < 112:
                self._add_family_prior(
                    "Downtempo / Slow Electronic",
                    0.24,
                    item("deep tagger cue at slow BPM supports Downtempo / Slow Electronic"),
                )
                self._add_genre_prior(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    0.60,
                    item("deep tagger cue at slow BPM supports Downtempo"),
                )
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    "Deep Downtempo",
                    0.56,
                    item("deep slow tagger cue supports Deep Downtempo"),
                )
            elif bpm is not None and bpm <= 124:
                self._add_genre_prior(
                    "House",
                    "Deep / Minimal / Groove House",
                    0.42,
                    item("deep tagger cue at house BPM supports Deep / Minimal / Groove House"),
                )
                self._add_subgenre_modifier(
                    "House",
                    "Deep / Minimal / Groove House",
                    "Deep House",
                    0.28,
                    item("deep house-BPM tagger cue supports Deep House"),
                )
            else:
                self._add_genre_prior(
                    "Techno",
                    "Deep / Dark / Afterhours Techno",
                    0.32,
                    item("deep tagger cue at techno BPM supports Deep / Dark / Afterhours Techno"),
                )
                self._add_subgenre_modifier(
                    "Techno",
                    "Deep / Dark / Afterhours Techno",
                    "Deep Techno",
                    0.24,
                    item("deep techno-BPM tagger cue supports Deep Techno"),
                )

        if vibes & MELODIC_MOOD_CODES:
            if bpm is not None and bpm < 112:
                self._add_family_prior(
                    "Downtempo / Slow Electronic",
                    0.18,
                    item("melodic tagger cue at slow BPM supports Downtempo / Slow Electronic"),
                )
                self._add_genre_prior(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    0.30,
                    item("melodic tagger cue at slow BPM supports melodic downtempo"),
                )
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    "Cinematic Downtempo",
                    0.60,
                    item("melodic slow tagger cue supports Cinematic Downtempo"),
                )
            elif bpm is not None and bpm <= 128:
                self._add_genre_prior(
                    "House",
                    "Melodic / Progressive House",
                    0.40,
                    item("melodic tagger cue at house BPM supports Melodic / Progressive House"),
                )
                if low_valence:
                    subgenre = "Dark Progressive House"
                elif energy is not None and energy <= 2:
                    subgenre = "Deep Melodic House"
                elif driving:
                    subgenre = "Progressive House"
                else:
                    subgenre = "Melodic House"
                self._add_subgenre_modifier(
                    "House",
                    "Melodic / Progressive House",
                    subgenre,
                    0.78,
                    item(f"melodic house-BPM tagger cue supports {subgenre}"),
                )
            else:
                self._add_genre_prior(
                    "Techno",
                    "Melodic / Emotional Techno",
                    0.34,
                    item("melodic tagger cue at faster BPM supports Melodic / Emotional Techno"),
                )
                subgenre = "Dark Melodic Techno" if low_valence else "Melodic Techno"
                self._add_subgenre_modifier(
                    "Techno",
                    "Melodic / Emotional Techno",
                    subgenre,
                    0.24,
                    item(f"melodic techno-BPM tagger cue supports {subgenre}"),
                )
            if bpm is not None and bpm <= 125:
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Downtempo",
                    "Cinematic Downtempo",
                    0.34,
                    item("melodic tagger cue supports Cinematic Downtempo when downtempo is otherwise likely"),
                )

        if vibes & DARK_MOOD_CODES:
            if bpm is not None and bpm <= 124:
                self._add_genre_prior(
                    "House",
                    "Dark / Indie-Adjacent House",
                    0.30,
                    item("dark tagger cue at house BPM supports dark/indie-adjacent house"),
                )
                self._add_subgenre_modifier(
                    "House",
                    "Dark / Indie-Adjacent House",
                    "Dark House",
                    0.22,
                    item("dark house-BPM tagger cue supports Dark House"),
                )
            else:
                self._add_genre_prior(
                    "Techno",
                    "Deep / Dark / Afterhours Techno",
                    0.36,
                    item("dark tagger cue at techno BPM supports Deep / Dark / Afterhours Techno"),
                )
                subgenre = "Dark Driving Techno" if driving or energy and energy >= 4 else "Dark Techno"
                self._add_subgenre_modifier(
                    "Techno",
                    "Deep / Dark / Afterhours Techno",
                    subgenre,
                    0.26,
                    item(f"dark techno-BPM tagger cue supports {subgenre}"),
                )

        if vibes & HYPNOTIC_MOOD_CODES and rolling:
            if bpm is not None and bpm >= 126:
                self._add_genre_prior(
                    "Techno",
                    "Minimal / Hypnotic Techno",
                    0.40,
                    item("hypnotic rolling tagger cues at techno BPM support Minimal / Hypnotic Techno"),
                )
                self._add_subgenre_modifier(
                    "Techno",
                    "Minimal / Hypnotic Techno",
                    "Rolling Hypnotic Techno",
                    0.36,
                    item("hypnotic rolling tagger cues support Rolling Hypnotic Techno"),
                )
            else:
                self._add_genre_prior(
                    "House",
                    "Melodic / Progressive House",
                    0.24,
                    item("hypnotic rolling tagger cues at house BPM support Progressive House"),
                )
                self._add_subgenre_modifier(
                    "House",
                    "Melodic / Progressive House",
                    "Hypnotic Progressive House",
                    0.24,
                    item("hypnotic rolling tagger cues support Hypnotic Progressive House"),
                )

        if broken:
            self._add_family_prior(
                "Garage / UK Bass / Breaks",
                0.22,
                item("broken/breaks tagger structure supports Garage / UK Bass / Breaks"),
            )
        if groovy and bpm is not None and 112 <= bpm <= 128:
            self._add_genre_prior(
                "House",
                "Deep / Minimal / Groove House",
                0.22,
                item("groove tagger structure at house BPM supports groove house"),
            )
            if high_inst:
                self._add_subgenre_modifier(
                    "House",
                    "Deep / Minimal / Groove House",
                    "Rolling Deep Tech",
                    0.34,
                    item("groovy instrumental house-BPM cues support Rolling Deep Tech"),
                )
        if rolling and bpm is not None and 118 <= bpm <= 128:
            if vibes & TRIBAL_MOOD_CODES:
                self._add_subgenre_modifier(
                    "House",
                    "Tech House",
                    "Percussive Tech House",
                    0.35,
                    item("tribal rolling house-BPM tagger cues support Percussive Tech House"),
                )
            if energy is not None and energy >= 3:
                self._add_subgenre_modifier(
                    "House",
                    "Tech House",
                    "Rolling Tech House",
                    0.22,
                    item("rolling house-BPM tagger cues support Rolling Tech House"),
                )
        if loose and bpm is not None and bpm < 115:
            self._add_genre_prior(
                "Downtempo / Slow Electronic",
                "Slow Club Music",
                0.18,
                item("loose tagger structure at slow BPM supports slow club music"),
            )
            if vibes & TRIBAL_MOOD_CODES:
                self._add_subgenre_modifier(
                    "Downtempo / Slow Electronic",
                    "Slow Club Music",
                    "Organic Slow House",
                    0.30,
                    item("loose tribal/organic slow-BPM cues support Organic Slow House"),
                )

    def _detect_source_disagreement(self) -> None:
        ranked = sorted(self.family_scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            return
        (family_a, score_a), (family_b, score_b) = ranked[:2]
        if score_a >= 0.75 and score_b >= 0.60 and score_a - score_b < 0.22:
            self.warnings.append(
                f"Source genres disagree or overlap: {family_a} and {family_b} are close."
            )

    def _best_family(self) -> tuple[str | None, float, float]:
        rows = [(family, self._family_selection_score(family)) for family in self.family_scores]
        return _top_with_margin(rows)

    def _best_genre(self, family: str) -> tuple[str | None, float, float]:
        rows = [
            (genre, score)
            for (fam, genre), score in self.genre_scores.items()
            if fam == family
        ]
        return _top_with_margin(rows)

    def _best_global_genre(self) -> tuple[str | None, str | None, float]:
        if not self.genre_scores:
            return None, None, 0.0
        (family, genre), score = max(self.genre_scores.items(), key=lambda item: item[1])
        return family, genre, score

    def _family_selection_score(self, family: str) -> float:
        score = self.family_scores.get(family, 0.0)
        best_genre = max(
            (
                genre_score
                for (fam, _genre), genre_score in self.genre_scores.items()
                if fam == family
            ),
            default=0.0,
        )
        best_subgenre = max(
            (
                subgenre_score
                for (fam, _genre, _subgenre), subgenre_score in self.subgenre_scores.items()
                if fam == family
            ),
            default=0.0,
        )
        return score + best_genre * 0.22 + best_subgenre * 0.06

    def _best_subgenre(self, family: str, genre: str) -> tuple[str | None, float, float]:
        rows = [
            (subgenre, score)
            for (fam, gen, subgenre), score in self.subgenre_scores.items()
            if fam == family and gen == genre
        ]
        return _top_with_margin(rows)

    def _evidence_strings(self, path: TaxonomyPath) -> list[str]:
        evidence: list[str] = []
        for item in self.evidence_by_path.get(path, []):
            evidence.append(item.explanation)
        if path.subgenre:
            genre_path = TaxonomyPath(path.family, path.genre)
            for item in self.evidence_by_path.get(genre_path, []):
                evidence.append(item.explanation)
        if path.family:
            family_path = TaxonomyPath(path.family)
            for item in self.evidence_by_path.get(family_path, []):
                evidence.append(item.explanation)
        return _dedupe(evidence)[:8]


def load_taxonomy(path: str | None = None) -> GenreTaxonomy:
    taxonomy_path = Path(path) if path else _default_taxonomy_path()
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GenreTaxonomy(data)


def classify_all_taxonomies(
    store: CsvStore,
    taxonomy_path: str | None = None,
    *,
    model_dir: str | None = None,
    use_model: bool = True,
    show_progress: bool = False,
) -> int:
    """Classify all tracks and persist family/genre/subgenre fields."""
    taxonomy = load_taxonomy(taxonomy_path)
    taxonomy_model = None
    if use_model:
        from .model import load_taxonomy_model_if_available

        default_model_dir = Path(store.output_dir) / "taxonomy_model"
        taxonomy_model = load_taxonomy_model_if_available(
            model_dir or str(default_model_dir),
            taxonomy_path=taxonomy_path,
        )

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()

    primary_file_by_track = {
        f.track_id: f for f in files if f.track_id and f.is_primary_file
    }
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    updated = 0
    progress = ProgressBar(len(tracks), label="Classify taxonomy", enabled=show_progress)
    for index, track in enumerate(tracks, start=1):
        result = classify_track(
            track,
            obs_by_track.get(track.track_id, []),
            primary_file_by_track.get(track.track_id),
            taxonomy=taxonomy,
            taxonomy_model=taxonomy_model,
        )
        _persist_result(track, result)
        updated += 1
        progress.update(index, track.title_canonical, classified=updated)

    progress.finish()
    store.save_tracks(tracks)
    return updated


def classify_track(
    track: LogicalTrack,
    observations: list[SourceObservation],
    file_record: FileRecord | None = None,
    *,
    taxonomy: GenreTaxonomy | None = None,
    taxonomy_model: Any | None = None,
) -> GenreClassificationResult:
    """Classify a track into family, genre, and optional subgenre."""
    taxonomy = taxonomy or load_taxonomy()
    evidence = _collect_evidence(track, observations, file_record)
    scorer = _Scorer(taxonomy)
    scorer.score(evidence)

    selected, selection_warning = scorer.selected_path()
    warnings = list(scorer.warnings)
    if selection_warning:
        warnings.append(selection_warning)

    if not taxonomy.validate_path(selected.family, selected.genre, selected.subgenre):
        warnings.append("Classifier produced an invalid taxonomy path; result was reset.")
        selected = TaxonomyPath()

    if taxonomy_model is not None:
        model_prediction = taxonomy_model.predict(track, observations, file_record, taxonomy)
        model_path = model_prediction.path
        if taxonomy.validate_path(model_path.family, model_path.genre, model_path.subgenre):
            model_warnings = list(model_prediction.warnings)
            if selected.family and model_path != selected:
                model_warnings.append("Trained model selected a taxonomy path that differs from deterministic rule scores.")
            evidence_summary = scorer.evidence_summary(evidence, model_path)
            for key, values in model_prediction.evidence.items():
                evidence_summary[key] = _dedupe([*evidence_summary.get(key, []), *values])
            return GenreClassificationResult(
                track_id=track.track_id or None,
                family=model_path.family,
                genre=model_path.genre,
                subgenre=model_path.subgenre,
                confidence=model_prediction.confidence,
                confidence_level=_confidence_level(model_prediction.confidence),
                evidence=evidence_summary,
                alternatives=model_prediction.alternatives or scorer.alternatives(model_path),
                warnings=_dedupe([*warnings, *model_warnings]),
            )
        warnings.append("Trained taxonomy model produced an invalid path; deterministic classifier was used.")

    confidence = scorer.confidence(selected)
    return GenreClassificationResult(
        track_id=track.track_id or None,
        family=selected.family,
        genre=selected.genre,
        subgenre=selected.subgenre,
        confidence=confidence,
        confidence_level=_confidence_level(confidence),
        evidence=scorer.evidence_summary(evidence, selected),
        alternatives=scorer.alternatives(selected),
        warnings=_dedupe(warnings),
    )


def _persist_result(track: LogicalTrack, result: GenreClassificationResult) -> None:
    alternatives = [alt.to_dict() for alt in result.alternatives]

    track.genre_family = result.family or ""
    track.genre = result.genre or ""
    track.subgenre = result.subgenre or ""
    track.genre_confidence = result.confidence
    track.genre_confidence_level = result.confidence_level
    track.genre_alternatives = json.dumps(alternatives, sort_keys=True)
    track.genre_evidence = json.dumps(result.evidence, sort_keys=True)
    track.genre_warnings = json.dumps(result.warnings, sort_keys=True)
    track.genre_taxonomy_version = TAXONOMY_VERSION

    # Keep legacy taxonomy columns populated for existing downstream consumers.
    track.taxonomy_id = result.taxonomy_id
    track.taxonomy_label = result.label
    track.taxonomy_confidence = result.confidence
    track.taxonomy_alternatives = track.genre_alternatives
    track.taxonomy_evidence = track.genre_evidence
    track.taxonomy_version = TAXONOMY_VERSION
    track.canonical_genre = result.subgenre or result.genre or result.family or ""


def _collect_evidence(
    track: LogicalTrack,
    observations: list[SourceObservation],
    file_record: FileRecord | None,
) -> _EvidenceBundle:
    evidence = _EvidenceBundle()

    if file_record:
        _add_genre_signal(
            evidence,
            "embedded_genre",
            file_record.embedded_genre,
            SOURCE_WEIGHTS["embedded_genre"],
            "embedded_genre",
        )
        _add_text_signal(
            evidence,
            "filename",
            file_record.file_name or file_record.path_rel or file_record.path_abs,
            SOURCE_WEIGHTS["filename_cue"],
            "filename_cue",
        )
        _add_text_signal(
            evidence,
            "embedded_comment",
            file_record.embedded_comment,
            SOURCE_WEIGHTS["title_or_mix_text_cue"],
            "text_cue",
        )

    _add_text_signal(
        evidence,
        "title_or_mix_text",
        " ".join([track.title_canonical, track.mix_canonical]),
        SOURCE_WEIGHTS["title_or_mix_text_cue"],
        "title_or_mix_text_cue",
    )
    _add_text_signal(
        evidence,
        "label",
        track.label_canonical,
        SOURCE_WEIGHTS["label_prior"],
        "label_prior",
    )

    tag_obs = None
    tagger_obs = None
    songstats_obs = None
    for obs in observations:
        source = (obs.source_system or "").lower()
        if source == "tag":
            tag_obs = obs
        if tagger_obs is None and any(
            [obs.tagger_energy, obs.tagger_vibe, obs.tagger_vocal, obs.tagger_structure]
        ):
            tagger_obs = obs
        if source == "songstats":
            songstats_obs = obs

        if source == "rekordbox":
            weight = SOURCE_WEIGHTS["rekordbox_genre_unknown_origin"]
            field_name = "rekordbox_genre"
            evidence_type = "rekordbox_genre"
        elif source == "songstats":
            weight = SOURCE_WEIGHTS["songstats_track_genre"]
            field_name = "songstats_genre"
            evidence_type = "songstats_track_genre"
        elif source == "spotify":
            weight = SOURCE_WEIGHTS["spotify_track_genre"]
            field_name = "spotify_genre"
            evidence_type = "spotify_track_genre"
        elif source == "manual":
            weight = SOURCE_WEIGHTS["trusted_dj_source_genre"]
            field_name = "manual_genre"
            evidence_type = "trusted_dj_source_genre"
        else:
            weight = SOURCE_WEIGHTS["embedded_genre"] if source == "tag" else 0.55
            field_name = f"{source or 'source'}_genre"
            evidence_type = "source_genre"

        _add_genre_signal(evidence, field_name, obs.genre, weight, evidence_type)
        _add_genre_signal(
            evidence,
            f"{source or 'source'}_genres_all",
            obs.genres_all,
            SOURCE_WEIGHTS["genres_all"],
            "genres_all",
        )
        _add_text_signal(
            evidence,
            f"{source or 'source'}_comments",
            obs.comments,
            SOURCE_WEIGHTS["title_or_mix_text_cue"],
            "source_text_cue",
        )
        _add_text_signal(
            evidence,
            f"{source or 'source'}_label",
            obs.label,
            SOURCE_WEIGHTS["label_prior"],
            "label_prior",
        )

    tagger_source = tagger_obs or tag_obs
    bpm = _num(track.tagger_bpm) or _num(track.canonical_bpm)
    if bpm is None and tagger_source:
        bpm = _num(tagger_source.bpm)
    evidence.bpm = bpm

    tagger_energy = track.tagger_energy or (tagger_source.tagger_energy if tagger_source else "")
    tagger_vibe = track.tagger_vibe or (tagger_source.tagger_vibe if tagger_source else "")
    tagger_vocal = track.tagger_vocal or (tagger_source.tagger_vocal if tagger_source else "")
    tagger_structure = track.tagger_structure or (tagger_source.tagger_structure if tagger_source else "")

    _add_tagger_cues(evidence, tagger_energy, tagger_vibe, tagger_vocal, tagger_structure, bpm)

    ss = songstats_obs
    evidence.acousticness = _num(ss.acousticness) if ss else None
    evidence.danceability = _num(ss.danceability) if ss else None
    evidence.energy = _num(ss.energy) if ss else None
    evidence.instrumentalness = _num(ss.instrumentalness) if ss else None
    evidence.valence = _num(ss.valence) if ss else None
    _add_audio_cues(evidence)

    return evidence


def _add_genre_signal(
    evidence: _EvidenceBundle,
    source_field: str,
    raw_value: Any,
    weight: float,
    evidence_type: str,
) -> None:
    values = _split_genre_values(raw_value)
    if not values:
        return
    evidence.genre_signals.append(_Signal(source_field, raw_value, values, weight, evidence_type))
    evidence.source_genres_used.update(values)
    evidence.metadata_signals.append(f"{source_field}={raw_value}")


def _add_text_signal(
    evidence: _EvidenceBundle,
    source_field: str,
    raw_value: Any,
    weight: float,
    evidence_type: str,
) -> None:
    norm = _normalize_text(raw_value)
    if not norm:
        return
    evidence.text_signals.append(_Signal(source_field, raw_value, [norm], weight, evidence_type))
    for cue in _text_cues(norm):
        evidence.cue_signals.append(
            _Cue(source_field, raw_value, [cue], weight * 0.65, "text_cue")
        )


def _add_tagger_cues(
    evidence: _EvidenceBundle,
    energy: str,
    vibe: str,
    vocal: str,
    structure: str,
    bpm: float | None,
) -> None:
    if energy:
        energy_int = _int(energy)
        evidence.tagger_energy_int = energy_int
        evidence.tagger_signals.append(f"tagger_energy={energy}")
        terms: list[str] = []
        if energy_int is not None:
            if energy_int <= 2:
                terms.extend(["deep", "organic", "ambient", "downtempo"])
            if energy_int == 4:
                terms.extend(["driving", "rolling"])
            if energy_int >= 5:
                terms.extend(["peak", "hard", "rave", "big room"])
        if terms:
            evidence.cue_signals.append(
                _Cue("tagger_energy", energy, terms, SOURCE_WEIGHTS["local_tagger_field"], "local_tagger_field")
            )

    if vibe:
        evidence.tagger_signals.append(f"tagger_vibe={vibe}")
        mood_tokens = _split_mood_tokens(vibe)
        evidence.tagger_vibes.update(mood_tokens)
        terms = []
        for token in mood_tokens:
            terms.extend(MOOD_CUES_BY_CODE.get(token, ()))
        if terms:
            evidence.cue_signals.append(
                _Cue("tagger_vibe", vibe, terms, SOURCE_WEIGHTS["local_tagger_field"], "local_tagger_field")
            )

    if vocal:
        evidence.tagger_signals.append(f"tagger_vocal={vocal}")
        vocal_tokens = _split_vocal_tokens(vocal)
        evidence.tagger_vocals.update(vocal_tokens)
        terms = []
        for token in vocal_tokens:
            terms.extend(VOCAL_PROFILE_CUES_BY_CODE.get(token, ()))
        if terms:
            evidence.cue_signals.append(
                _Cue("tagger_vocal", vocal, terms, SOURCE_WEIGHTS["local_tagger_field"], "local_tagger_field")
            )

    if structure:
        evidence.tagger_signals.append(f"tagger_structure={structure}")
        evidence.tagger_structure = structure.upper()
        terms = []
        upper = structure.upper()
        if upper in {"16H", "32H", "ROLLING"} or upper.endswith("H"):
            terms.extend(["rolling", "rolling hypnotic", "hypnotic", "steady"])
        if upper in {"16D", "32D", "LINEAR"} or upper.endswith("D"):
            terms.extend(["driving", "linear", "functional"])
        if upper.endswith("G"):
            terms.extend(["groove", "funky", "deep"])
        if upper.endswith("L"):
            terms.extend(["loose", "downtempo", "ambient"])
        if upper.endswith("B"):
            terms.extend(["broken", "breakbeat", "breaks"])
        if upper == "BREAKS":
            terms.extend(["breaks", "breakbeat", "garage", "jungle", "bass"])
        if terms:
            evidence.cue_signals.append(
                _Cue("tagger_structure", structure, terms, SOURCE_WEIGHTS["local_tagger_field"], "local_tagger_field")
            )

    if bpm is not None:
        evidence.tagger_signals.append(f"tagger_bpm={bpm:g}")


def _add_audio_cues(evidence: _EvidenceBundle) -> None:
    if evidence.energy is not None:
        if evidence.energy >= 0.78:
            evidence.audio_feature_signals.append("energy high")
            evidence.cue_signals.append(
                _Cue("energy", evidence.energy, ["driving", "peak", "hard"], 0.16, "audio_feature")
            )
        elif evidence.energy <= 0.35:
            evidence.audio_feature_signals.append("energy low")
            evidence.cue_signals.append(
                _Cue("energy", evidence.energy, ["ambient", "downtempo", "deep"], 0.12, "audio_feature")
            )
    if evidence.valence is not None:
        if evidence.valence <= 0.30:
            evidence.audio_feature_signals.append("valence low")
        elif evidence.valence >= 0.62:
            evidence.audio_feature_signals.append("valence high")
            evidence.cue_signals.append(
                _Cue("valence", evidence.valence, ["funky", "euphoric", "uplifting"], 0.12, "audio_feature")
            )
    if evidence.instrumentalness is not None and evidence.instrumentalness >= 0.75:
        evidence.audio_feature_signals.append("instrumentalness high")
    if evidence.acousticness is not None and evidence.acousticness >= 0.60:
        evidence.audio_feature_signals.append("acousticness high")


def _matches_for_value(
    value: str,
    taxonomy: GenreTaxonomy,
    *,
    allow_phrase: bool,
) -> list[tuple[str, TaxonomyPath, str, float]]:
    norm = _normalize_text(value)
    matches: dict[tuple[str, TaxonomyPath], tuple[str, TaxonomyPath, str, float]] = {}

    for label in _aliases_for(norm):
        for path in _paths_for_match(taxonomy, label):
            _store_match(matches, label, path, "alias", 1.25 + 0.12 * _specificity(label))

    for path in _paths_for_match(taxonomy, norm):
        label = path.label()
        _store_match(matches, label, path, "exact", 1.32 + 0.16 * _specificity(label))

    if allow_phrase:
        for label_norm, label, path in taxonomy.labels():
            word_count = len(label_norm.split())
            if word_count == 1:
                if norm == label_norm:
                    _store_match(matches, label, path, "exact", 0.95)
                elif path.genre is None and _contains_phrase(norm, label_norm):
                    _store_match(matches, label, path, "family-token", 0.62)
                continue
            if _contains_phrase(norm, label_norm):
                strength = 0.82 + 0.10 * _specificity(label_norm)
                _store_match(matches, label, path, "phrase", strength)

    return sorted(matches.values(), key=lambda item: item[3], reverse=True)


def _paths_for_match(taxonomy: GenreTaxonomy, label: str) -> list[TaxonomyPath]:
    paths = taxonomy.find_paths_for_label(label)
    genre_keys = {
        (path.family, path.genre)
        for path in paths
        if path.family and path.genre and path.subgenre is None
    }
    filtered = []
    for path in paths:
        duplicate_genre_subgenre = (
            path.family,
            path.genre,
        ) in genre_keys and _normalize_text(path.subgenre) == _normalize_text(path.genre)
        if duplicate_genre_subgenre:
            continue
        filtered.append(path)
    return filtered


def _store_match(
    matches: dict[tuple[str, TaxonomyPath], tuple[str, TaxonomyPath, str, float]],
    label: str,
    path: TaxonomyPath,
    match_type: str,
    strength: float,
) -> None:
    key = (_normalize_text(label), path)
    existing = matches.get(key)
    if existing is None or strength > existing[3]:
        matches[key] = (label, path, match_type, strength)


def _aliases_for(norm: str) -> list[str]:
    aliases = {
        "techhouse": ["Tech House"],
        "tech house": ["Tech House"],
        "deep tech": ["Deep Tech"],
        "melodic house and techno": ["Melodic House", "Melodic Techno", "Progressive House"],
        "melodic house techno": ["Melodic House", "Melodic Techno", "Progressive House"],
        "organic house": ["Organic House"],
        "desert house": ["Desert House"],
        "middle eastern": ["Middle Eastern Organic House"],
        "sunset": ["Sunset Organic House"],
        "afro house": ["Afro House"],
        "tribal house": ["Tribal House"],
        "indie dance": ["Indie Dance"],
        "dark disco": ["Dark Disco"],
        "nu disco": ["Nu-Disco"],
        "ukg": ["UK Garage"],
        "uk garage": ["UK Garage"],
        "2step": ["2-Step Garage"],
        "2 step": ["2-Step Garage"],
        "speed garage": ["Speed Garage"],
        "bassline": ["Bassline"],
        "chill electronic": ["Downtempo", "Ambient Chill"],
        "dnb": ["Drum & Bass"],
        "drum n bass": ["Drum & Bass"],
        "drum and bass": ["Drum & Bass"],
        "future garage": ["Future Garage"],
        "jungle": ["Jungle"],
        "garage rock": ["Garage Rock"],
        "garage punk": ["Garage Punk"],
        "psych rock": ["Psychedelic Rock"],
        "alternative rock": ["Alternative Rock"],
        "alt rock": ["Alternative Rock"],
        "classic rock": ["Classic Rock"],
        "post rock": ["Post-Rock"],
        "postrock": ["Post-Rock"],
        "psy trance": ["Psytrance"],
        "psytrance": ["Psytrance"],
        "rnb": ["R&B"],
        "r&b": ["R&B"],
        "techno peak time": ["Peak-Time Techno"],
        "techno peak time driving": ["Peak-Time Techno", "Driving Techno"],
        "peak time driving": ["Peak-Time Techno", "Driving Techno"],
        "hip hop": ["Hip-Hop"],
        "hiphop": ["Hip-Hop"],
    }
    return aliases.get(norm, [])


def _cue_strength(term: str, label_norm: str) -> float:
    if term == label_norm:
        return 0.85
    if label_norm.startswith(term) or label_norm.endswith(term):
        return 0.72
    if len(term.split()) > 1:
        return 0.70
    return 0.56


def _cue_evidence(cue: _Cue, term: str, label: str, path: TaxonomyPath) -> EvidenceItem:
    return EvidenceItem(
        source_field=cue.source_field,
        raw_value=cue.raw_value,
        normalized_value=_normalize_text(term),
        matched_label=label,
        matched_path=path,
        evidence_type=cue.evidence_type,
        weight=cue.weight,
        score_delta=cue.weight,
        explanation=f"{cue.source_field} cue {term!r} supports {label}",
    )


def _split_genre_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = [str(v) for v in value if v not in (None, "")]
    else:
        raw_values = [str(value)]

    values: list[str] = []
    for raw in raw_values:
        parts = re.split(r"[;/|,]+", raw)
        if len(parts) == 1:
            parts = [raw]
        for part in parts:
            norm = _normalize_text(part)
            if norm:
                values.append(norm)
    return _dedupe(values)


def _text_cues(norm: str) -> list[str]:
    cue_terms = [
        "acid",
        "afro",
        "ambient",
        "bassline",
        "breakbeat",
        "breaks",
        "chant",
        "cinematic",
        "dark",
        "desert",
        "disco",
        "driving",
        "dub",
        "ethnic",
        "funky",
        "gothic",
        "hard",
        "hypnotic",
        "industrial",
        "instrumental",
        "linear",
        "melancholic",
        "melodic",
        "mental",
        "middle eastern",
        "minimal",
        "organic",
        "peak",
        "percussive",
        "psychedelic",
        "rave",
        "ritual",
        "rolling",
        "romantic",
        "shamanic",
        "spoken",
        "tribal",
        "uplifting",
        "vocal",
        "warehouse",
    ]
    found = [term for term in cue_terms if _contains_phrase(norm, term)]
    if "desert" in found:
        found.append("desert house")
    return found


def _normalize_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\.[a-zA-Z0-9]{2,5}$", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"rock\s*['’]?n\s*['’]?\s*roll", "rock and roll", text)
    text = re.sub(r"\bdnb\b", "drum and bass", text)
    text = re.sub(r"\bdrum\s*(?:n|&)\s*bass\b", "drum and bass", text)
    text = re.sub(r"\br\s*&\s*b\b", " r_and_b_token ", text)
    text = re.sub(r"\brnb\b", " r_and_b_token ", text)
    text = text.replace("&", " and ")
    text = text.replace("r_and_b_token", "r&b")
    text = re.sub(r"[-_/|,;:()\[\]{}]+", " ", text)
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    return " ".join(text.split())


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return f" {needle} " in f" {haystack} "


def _specificity(label: str) -> float:
    words = len(_normalize_text(label).split())
    if words >= 3:
        return 1.25
    if words == 2:
        return 1.0
    return 0.35


def _top_with_margin(rows: Iterable[tuple[Any, float]]) -> tuple[Any | None, float, float]:
    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    best_value, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_value, best_score, max(0.0, best_score - second)


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium-high"
    if confidence >= 0.55:
        return "medium"
    if confidence >= 0.35:
        return "low"
    return "unknown"


def _score_to_display(raw_score: float) -> float:
    return round(max(0.0, min(0.99, 1.0 - math.exp(-raw_score / 2.6))), 3)


def _path_to_dict(path: TaxonomyPath | None) -> dict[str, str | None] | None:
    if path is None:
        return None
    return {"family": path.family, "genre": path.genre, "subgenre": path.subgenre}


def _default_taxonomy_path() -> Path:
    candidates = [
        Path.cwd() / "music_genre_taxonomy_3_level.json",
        Path(__file__).resolve().parents[3] / "music_genre_taxonomy_3_level.json",
        Path(__file__).with_name("genre_taxonomy_3_level.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("music_genre_taxonomy_3_level.json was not found")


def _split_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[,;/| ]+", (value or "").upper()) if token]


def _split_mood_tokens(value: str) -> list[str]:
    return [normalize_mood_code(token) for token in _split_tokens(value) if normalize_mood_code(token)]


def _split_vocal_tokens(value: str) -> list[str]:
    return [normalize_vocal_profile(token) for token in _split_tokens(value) if normalize_vocal_profile(token)]


def _num(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).strip().removeprefix("E"))
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    text = str(value or "")
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    num = _num(value)
    return int(round(num)) if num is not None else None


def _slug(value: str) -> str:
    norm = _normalize_text(value)
    return re.sub(r"[^a-z0-9]+", "_", norm).strip("_")


def _dedupe(values: Iterable[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        marker = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        if value in (None, "") or marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
