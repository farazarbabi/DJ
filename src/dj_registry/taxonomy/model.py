"""Supervised model layer for genre taxonomy classification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .classifier import GenreCandidate, GenreTaxonomy, TaxonomyPath, load_taxonomy
from .features import build_track_features, summarize_feature_groups

MODEL_VERSION = "taxonomy-supervised-v1"
FEATURE_SCHEMA_VERSION = "taxonomy-feature-schema-v2"
MODEL_FILENAME = "model.pkl"
REPORT_FILENAME = "training_report.json"
AUDIT_FILENAME = "training_audit.csv"
LABEL_SEPARATOR = "|||"


@dataclass
class TaxonomyPrediction:
    path: TaxonomyPath
    confidence: float
    alternatives: list[GenreCandidate] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingStats:
    labels_path: str
    model_dir: str
    examples: int
    family_classes: int
    genre_classes: int
    subgenre_classes: int
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _ConstantClassifier:
    label: str

    def predict_proba(self, x: Any) -> np.ndarray:
        rows = x.shape[0] if hasattr(x, "shape") else len(x)
        return np.ones((rows, 1), dtype=float)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([self.label])


@dataclass
class TaxonomyModel:
    vectorizer: DictVectorizer
    family_model: Any
    genre_model: Any
    subgenre_model: Any
    taxonomy_version: str
    taxonomy_hash: str
    feature_schema_version: str
    trained_at: str
    labels_hash: str
    examples: int

    def predict(
        self,
        track: LogicalTrack,
        observations: list[SourceObservation],
        file_record: FileRecord | None,
        taxonomy: GenreTaxonomy,
    ) -> TaxonomyPrediction:
        features = build_track_features(track, observations, file_record)
        x = self.vectorizer.transform([features])

        family_probs = _probability_map(self.family_model, x)
        genre_probs = _probability_map(self.genre_model, x)
        subgenre_probs = _probability_map(self.subgenre_model, x)

        candidates: list[tuple[TaxonomyPath, float]] = []
        for label, sub_prob in subgenre_probs.items():
            path = _decode_subgenre_label(label)
            if not taxonomy.validate_path(path.family, path.genre, path.subgenre):
                continue
            genre_label = _encode_genre_label(path.family or "", path.genre or "")
            score = (
                0.22 * family_probs.get(path.family or "", 0.0)
                + 0.30 * genre_probs.get(genre_label, 0.0)
                + 0.48 * sub_prob
            )
            candidates.append((path, score))

        if not candidates:
            for label, genre_prob in genre_probs.items():
                path = _decode_genre_label(label)
                if taxonomy.validate_path(path.family, path.genre, None):
                    score = 0.38 * family_probs.get(path.family or "", 0.0) + 0.62 * genre_prob
                    candidates.append((path, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        if not candidates:
            return TaxonomyPrediction(
                TaxonomyPath(),
                0.22,
                evidence={"model_signals": ["trained model produced no valid taxonomy path"]},
                warnings=["Trained taxonomy model produced no valid taxonomy path."],
            )

        selected, selected_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = max(0.0, selected_score - second_score)
        confidence = _confidence_from_model_score(selected_score, margin, features)

        alternatives = [
            GenreCandidate(
                family=path.family,
                genre=path.genre,
                subgenre=path.subgenre,
                score=round(_confidence_from_model_score(score, max(0.0, score - selected_score), features), 3),
                evidence=[f"learned_path_score={score:.3f}"],
            )
            for path, score in candidates[1:4]
        ]

        warnings: list[str] = []
        if confidence < 0.55:
            warnings.append("Low-confidence taxonomy inferred by trained model.")
        if margin < 0.08:
            warnings.append("Low taxonomy model margin between top candidates.")

        evidence = {
            "model_signals": [
                f"model_version={MODEL_VERSION}",
                f"learned_path_score={selected_score:.3f}",
                f"learned_margin={margin:.3f}",
            ],
            "feature_groups": summarize_feature_groups(features),
            "top_model_paths": [
                f"{path.full_label()} ({score:.3f})" for path, score in candidates[:3]
            ],
        }
        return TaxonomyPrediction(selected, confidence, alternatives, evidence, warnings)

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        artifact_path = path / MODEL_FILENAME
        with artifact_path.open("wb") as f:
            pickle.dump(self, f)
        return artifact_path


def train_taxonomy_model(
    store: CsvStore,
    labels_path: str,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    show_progress: bool = False,
) -> TrainingStats:
    """Train and persist the unified learned taxonomy classifier."""
    taxonomy = load_taxonomy(taxonomy_path)
    labels = _load_label_rows(labels_path, taxonomy)
    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    examples, skipped = _build_training_examples(
        labels,
        tracks,
        files,
        observations,
        show_progress=show_progress,
    )
    if not examples:
        raise ValueError("No valid taxonomy training examples could be built from labels CSV")

    feature_dicts = [example["features"] for example in examples]
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(feature_dicts)

    family_labels = [example["family"] for example in examples]
    genre_labels = [_encode_genre_label(example["family"], example["genre"]) for example in examples]
    subgenre_labels = [
        _encode_subgenre_label(example["family"], example["genre"], example["subgenre"])
        for example in examples
    ]

    model = TaxonomyModel(
        vectorizer=vectorizer,
        family_model=_fit_classifier(x, family_labels),
        genre_model=_fit_classifier(x, genre_labels),
        subgenre_model=_fit_classifier(x, subgenre_labels),
        taxonomy_version=_taxonomy_version(taxonomy_path),
        taxonomy_hash=_file_hash(_taxonomy_file_path(taxonomy_path)),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        trained_at=now_iso(),
        labels_hash=_file_hash(Path(labels_path)),
        examples=len(examples),
    )

    out_dir = Path(model_dir or Path(store.output_dir) / "taxonomy_model")
    model.save(out_dir)
    _write_training_audit(out_dir / AUDIT_FILENAME, examples)

    stats = TrainingStats(
        labels_path=labels_path,
        model_dir=str(out_dir),
        examples=len(examples),
        family_classes=len(set(family_labels)),
        genre_classes=len(set(genre_labels)),
        subgenre_classes=len(set(subgenre_labels)),
        skipped_rows=skipped,
    )
    _write_training_report(out_dir / REPORT_FILENAME, stats, model)
    return stats


def load_taxonomy_model(model_dir: str | Path, *, taxonomy_path: str | None = None) -> TaxonomyModel:
    artifact_path = Path(model_dir) / MODEL_FILENAME
    with artifact_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TaxonomyModel):
        raise ValueError(f"{artifact_path} does not contain a taxonomy model artifact")
    if model.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Taxonomy model feature schema mismatch: {model.feature_schema_version} != {FEATURE_SCHEMA_VERSION}"
        )
    current_taxonomy_hash = _file_hash(_taxonomy_file_path(taxonomy_path))
    if model.taxonomy_hash and current_taxonomy_hash and model.taxonomy_hash != current_taxonomy_hash:
        raise ValueError("Taxonomy model was trained against a different taxonomy JSON")
    return model


def load_taxonomy_model_if_available(
    model_dir: str | Path | None,
    *,
    taxonomy_path: str | None = None,
) -> TaxonomyModel | None:
    if not model_dir:
        return None
    artifact_path = Path(model_dir) / MODEL_FILENAME
    if not artifact_path.exists():
        return None
    return load_taxonomy_model(model_dir, taxonomy_path=taxonomy_path)


def evaluate_taxonomy_model(
    store: CsvStore,
    labels_path: str,
    *,
    model_dir: str,
    taxonomy_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    model = load_taxonomy_model(model_dir, taxonomy_path=taxonomy_path)
    labels = _load_label_rows(labels_path, taxonomy)
    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    examples, skipped = _build_training_examples(
        labels,
        tracks,
        files,
        observations,
        show_progress=show_progress,
    )

    totals = {"examples": len(examples), "skipped_rows": skipped, "family": 0, "genre": 0, "subgenre": 0}
    files_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        obs_by_track.setdefault(obs.track_id, []).append(obs)

    progress = ProgressBar(len(examples), label="Evaluate taxonomy model", enabled=show_progress)
    for index, example in enumerate(examples, start=1):
        track = example["track"]
        prediction = model.predict(track, obs_by_track.get(track.track_id, []), files_by_track.get(track.track_id), taxonomy)
        if prediction.path.family == example["family"]:
            totals["family"] += 1
        if prediction.path.genre == example["genre"] and prediction.path.family == example["family"]:
            totals["genre"] += 1
        if (
            prediction.path.subgenre == example["subgenre"]
            and prediction.path.genre == example["genre"]
            and prediction.path.family == example["family"]
        ):
            totals["subgenre"] += 1
        progress.update(index, track.title_canonical, subgenre=totals["subgenre"])
    progress.finish()

    denominator = max(1, len(examples))
    return {
        "examples": len(examples),
        "skipped_rows": skipped,
        "family_accuracy": round(totals["family"] / denominator, 3),
        "genre_accuracy": round(totals["genre"] / denominator, 3),
        "subgenre_accuracy": round(totals["subgenre"] / denominator, 3),
    }


def _load_label_rows(labels_path: str, taxonomy: GenreTaxonomy) -> list[dict[str, str]]:
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    valid_rows: list[dict[str, str]] = []
    invalid: list[str] = []
    for index, row in enumerate(rows, start=2):
        family = (row.get("family") or "").strip()
        genre = (row.get("genre") or "").strip()
        subgenre = (row.get("subgenre") or "").strip()
        if not taxonomy.validate_path(family, genre, subgenre):
            invalid.append(f"row {index}: {family} > {genre} > {subgenre}")
            continue
        valid_rows.append(row)
    if invalid and not valid_rows:
        raise ValueError("No valid taxonomy labels found: " + "; ".join(invalid[:5]))
    return valid_rows


def _build_training_examples(
    labels: list[dict[str, str]],
    tracks: list[LogicalTrack],
    files: list[FileRecord],
    observations: list[SourceObservation],
    *,
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    track_by_id = {track.track_id: track for track in tracks if track.track_id}
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    file_by_name = {f.file_name.lower(): f for f in files if f.file_name}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    examples: list[dict[str, Any]] = []
    skipped = 0
    progress = ProgressBar(len(labels), label="Build training data", enabled=show_progress)
    for index, row in enumerate(labels, start=1):
        track = None
        file_record = None
        track_id = (row.get("track_id") or "").strip()
        if track_id:
            track = track_by_id.get(track_id)
        if not track:
            file_name = (row.get("file_name") or row.get("filename") or "").strip().lower()
            file_record = file_by_name.get(file_name)
            if file_record:
                track = track_by_id.get(file_record.track_id)
        if not track:
            track = _track_from_label_row(row)
        if not file_record and track.track_id:
            file_record = file_by_track.get(track.track_id)

        features = build_track_features(
            track,
            obs_by_track.get(track.track_id, []),
            file_record,
            label_row=row,
        )
        if not features:
            skipped += 1
            progress.update(index, row.get("file_name", ""), examples=len(examples), skipped=skipped)
            continue
        examples.append(
            {
                "track": track,
                "features": features,
                "family": row["family"].strip(),
                "genre": row["genre"].strip(),
                "subgenre": row["subgenre"].strip(),
                "file_name": row.get("file_name") or (file_record.file_name if file_record else ""),
            }
        )
        progress.update(index, row.get("file_name", ""), examples=len(examples), skipped=skipped)
    progress.finish()
    return examples, skipped


def _track_from_label_row(row: dict[str, str]) -> LogicalTrack:
    return LogicalTrack(
        track_id=(row.get("track_id") or row.get("file_name") or "").strip(),
        artist_canonical=(row.get("artist") or row.get("Artist") or "").strip(),
        title_canonical=(row.get("title") or row.get("Track Title") or "").strip(),
        mix_canonical=(row.get("mix") or "").strip(),
        album_canonical=(row.get("album") or row.get("Album") or "").strip(),
        label_canonical=(row.get("label") or "").strip(),
        canonical_bpm=(row.get("bpm_hint") or row.get("BPM") or "").strip(),
        tagger_energy=(row.get("pred_Energy_v1") or row.get("energy") or "").strip(),
        tagger_vibe=(row.get("pred_Vibe_v1") or row.get("vibe") or "").strip(),
        tagger_vocal=(row.get("pred_Vocal_v1") or row.get("vocal") or "").strip(),
        tagger_structure=(row.get("structure") or "").strip(),
    )


def _fit_classifier(x: Any, labels: list[str]) -> Any:
    unique = sorted(set(labels))
    if len(unique) == 1:
        return _ConstantClassifier(unique[0])
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x, labels)
    return classifier


def _probability_map(model: Any, x: Any) -> dict[str, float]:
    probabilities = model.predict_proba(x)[0]
    return {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}


def _confidence_from_model_score(score: float, margin: float, features: dict[str, float]) -> float:
    richness = min(0.12, len(features) / 420.0)
    raw = 0.24 + 0.58 * max(0.0, min(score, 1.0)) + 0.18 * max(0.0, min(margin, 1.0)) + richness
    return round(max(0.2, min(0.96, raw)), 3)


def _encode_genre_label(family: str, genre: str) -> str:
    return f"{family}{LABEL_SEPARATOR}{genre}"


def _encode_subgenre_label(family: str, genre: str, subgenre: str) -> str:
    return f"{family}{LABEL_SEPARATOR}{genre}{LABEL_SEPARATOR}{subgenre}"


def _decode_genre_label(label: str) -> TaxonomyPath:
    parts = label.split(LABEL_SEPARATOR)
    return TaxonomyPath(parts[0] if len(parts) > 0 else None, parts[1] if len(parts) > 1 else None)


def _decode_subgenre_label(label: str) -> TaxonomyPath:
    parts = label.split(LABEL_SEPARATOR)
    return TaxonomyPath(
        parts[0] if len(parts) > 0 else None,
        parts[1] if len(parts) > 1 else None,
        parts[2] if len(parts) > 2 else None,
    )


def _write_training_audit(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "track_id", "artist", "title", "family", "genre", "subgenre", "feature_count"],
        )
        writer.writeheader()
        for example in examples:
            track = example["track"]
            writer.writerow(
                {
                    "file_name": example.get("file_name", ""),
                    "track_id": track.track_id,
                    "artist": track.artist_canonical,
                    "title": track.title_canonical,
                    "family": example["family"],
                    "genre": example["genre"],
                    "subgenre": example["subgenre"],
                    "feature_count": len(example["features"]),
                }
            )


def _write_training_report(path: Path, stats: TrainingStats, model: TaxonomyModel) -> None:
    payload = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at": model.trained_at,
        "labels_path": stats.labels_path,
        "labels_hash": model.labels_hash,
        "taxonomy_hash": model.taxonomy_hash,
        "examples": stats.examples,
        "family_classes": stats.family_classes,
        "genre_classes": stats.genre_classes,
        "subgenre_classes": stats.subgenre_classes,
        "skipped_rows": stats.skipped_rows,
        "warnings": stats.warnings,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _taxonomy_file_path(taxonomy_path: str | None) -> Path:
    if taxonomy_path:
        return Path(taxonomy_path)
    candidates = [
        Path.cwd() / "music_genre_taxonomy_3_level.json",
        Path(__file__).resolve().parents[3] / "music_genre_taxonomy_3_level.json",
        Path(__file__).with_name("genre_taxonomy_3_level.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _taxonomy_version(taxonomy_path: str | None) -> str:
    path = _taxonomy_file_path(taxonomy_path)
    return path.name


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
