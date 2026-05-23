"""Dual supervised models for flat DJ-functional taxonomy categories."""

from __future__ import annotations

import csv
import json
import logging
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)
from .dj_schema import DjTaxonomy, external_evidence_available, load_dj_taxonomy
from .features import (
    build_track_features,
    is_afro_tribal_family,
    is_eligible_for_afro_tribal,
    load_artist_priors,
    load_label_priors,
    summarize_feature_groups,
)

DJ_MODEL_VERSION = "dj-taxonomy-dual-v2"
DJ_FEATURE_SCHEMA_VERSION = "dj-taxonomy-feature-schema-v2"
MODEL_FILENAME = "model.pkl"
REPORT_FILENAME = "training_report.json"
AUDIT_FILENAME = "training_audit.csv"
COMPARISON_JSON = "model_comparison.json"
COMPARISON_CSV = "model_comparison.csv"
VALID_FEATURE_MODES = {"internal", "external"}

# Spec §7 — null fallback sentinel and confidence-band thresholds
UNCLASSIFIED_ID = "unclassified"
UNCLASSIFIED_THRESHOLD = 0.35


def confidence_to_level(confidence: float) -> str:
    """Spec §7.1 — map calibrated probability to a user-facing level string."""
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium-high"
    if confidence >= 0.55:
        return "medium"
    if confidence >= UNCLASSIFIED_THRESHOLD:
        return "low"
    return "unknown"


@dataclass
class DjCategoryPrediction:
    category_id: str = ""
    category_label: str = ""
    confidence: float = 0.0
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    feature_mode: str = ""


@dataclass
class DjTrainingStats:
    labels_path: str
    model_dir: str
    examples: int
    classes: int
    skipped_rows: int = 0
    mode: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
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
class DjTaxonomyModel:
    vectorizer: DictVectorizer
    classifier: Any
    feature_mode: str
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
        taxonomy: DjTaxonomy,
    ) -> DjCategoryPrediction:
        features = build_track_features(
            track,
            observations,
            file_record,
            feature_mode=self.feature_mode,
        )
        if not features:
            return DjCategoryPrediction(
                confidence=0.0,
                feature_mode=self.feature_mode,
                warnings=["No usable features for DJ taxonomy model."],
            )

        x = self.vectorizer.transform([features])
        probabilities = _probability_map(self.classifier, x)
        ranked = [
            (category_id, probability)
            for category_id, probability in probabilities.items()
            if taxonomy.validate_category_id(category_id)
        ]
        # Spec §6.2 — hard candidate filter: drop afro/tribal categories when
        # the track lacks explicit afro/tribal provider/prior/mix-name signal.
        if not is_eligible_for_afro_tribal(
            track,
            observations,
            file_record,
            artist_priors=load_artist_priors(),
            label_priors=load_label_priors(),
        ):
            ranked = [
                (cat_id, score)
                for cat_id, score in ranked
                if not is_afro_tribal_family(taxonomy.category(cat_id).family)
            ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return DjCategoryPrediction(
                confidence=0.0,
                feature_mode=self.feature_mode,
                evidence={"feature_groups": summarize_feature_groups(features)},
                warnings=["Model produced no valid DJ taxonomy category."],
            )

        selected_id, selected_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, selected_score - second_score)
        category = taxonomy.category(selected_id)
        confidence = _confidence_from_model_score(selected_score, margin, features)
        alternatives = [
            {
                "category_id": alt_id,
                "label": taxonomy.category(alt_id).label,
                "score": round(score, 3),
            }
            for alt_id, score in ranked[1:4]
        ]
        warnings: list[str] = []
        if confidence < 0.55:
            warnings.append("Low-confidence DJ taxonomy inferred by trained model.")
        if margin < 0.08 and len(ranked) > 1:
            warnings.append("Low DJ taxonomy model margin between top categories.")

        return DjCategoryPrediction(
            category_id=selected_id,
            category_label=category.label,
            confidence=confidence,
            alternatives=alternatives,
            evidence={
                "model_signals": [
                    f"model_version={DJ_MODEL_VERSION}",
                    f"feature_mode={self.feature_mode}",
                    f"category_score={selected_score:.3f}",
                    f"category_margin={margin:.3f}",
                ],
                "feature_groups": summarize_feature_groups(features),
                "top_categories": [
                    f"{taxonomy.category(category_id).label} ({score:.3f})"
                    for category_id, score in ranked[:3]
                ],
            },
            warnings=warnings,
            feature_mode=self.feature_mode,
        )

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        artifact_path = path / MODEL_FILENAME
        with artifact_path.open("wb") as f:
            pickle.dump(self, f)
        return artifact_path


def train_dj_taxonomy_models(
    store: CsvStore,
    labels_path: str,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    validation_split: float = 0.2,
    seed: int = 42,
    show_progress: bool = False,
) -> dict[str, Any]:
    taxonomy = load_dj_taxonomy(taxonomy_path)
    labels = _load_label_rows(labels_path, taxonomy)
    base_dir = Path(model_dir or Path(store.output_dir) / "dj_taxonomy_model")
    base_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, DjTrainingStats] = {}
    examples_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in ("internal", "external"):
        examples, skipped = _build_training_examples(
            labels,
            store,
            mode=mode,
            show_progress=show_progress,
        )
        if not examples:
            raise ValueError(f"No valid {mode} DJ taxonomy training examples could be built from labels CSV")
        model, metrics, warnings = _train_final_model_with_metrics(
            examples,
            taxonomy,
            feature_mode=mode,
            labels_path=labels_path,
            validation_split=validation_split,
            seed=seed,
        )
        mode_dir = base_dir / mode
        model.save(mode_dir)
        _write_training_audit(mode_dir / AUDIT_FILENAME, examples)
        mode_stats = DjTrainingStats(
            labels_path=labels_path,
            model_dir=str(mode_dir),
            examples=len(examples),
            classes=len({example["category_id"] for example in examples}),
            skipped_rows=skipped,
            mode=mode,
            metrics=metrics,
            warnings=warnings,
        )
        _write_training_report(mode_dir / REPORT_FILENAME, mode_stats, model)
        stats[mode] = mode_stats
        examples_by_mode[mode] = examples

    comparison = evaluate_dj_taxonomy_models(
        store,
        labels_path,
        model_dir=str(base_dir),
        taxonomy_path=taxonomy_path,
        show_progress=show_progress,
    )
    return {
        "model_dir": str(base_dir),
        "internal": _stats_to_dict(stats["internal"]),
        "external": _stats_to_dict(stats["external"]),
        "comparison": comparison,
    }


def load_dj_taxonomy_model(model_dir: str | Path, *, taxonomy_path: str | None = None) -> DjTaxonomyModel:
    artifact_path = Path(model_dir) / MODEL_FILENAME
    with artifact_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, DjTaxonomyModel):
        raise ValueError(f"{artifact_path} does not contain a DJ taxonomy model artifact")
    if model.feature_schema_version != DJ_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"DJ taxonomy model feature schema mismatch: "
            f"{model.feature_schema_version} != {DJ_FEATURE_SCHEMA_VERSION}"
        )
    taxonomy = load_dj_taxonomy(taxonomy_path)
    if model.taxonomy_hash and model.taxonomy_hash != taxonomy.hash():
        raise ValueError("DJ taxonomy model was trained against a different dj_taxonomy.json")
    if model.feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Invalid DJ taxonomy model feature mode: {model.feature_mode}")
    return model


def load_dj_taxonomy_model_if_available(
    model_dir: str | Path | None,
    *,
    taxonomy_path: str | None = None,
) -> DjTaxonomyModel | None:
    """Load a pickled DJ taxonomy model if present and compatible.

    Returns None when the model directory is missing OR when the persisted
    model is incompatible with the current feature schema / taxonomy hash.
    Treating schema mismatch as "no model" avoids breaking the pipeline after
    a feature-engineering change — the user retrains explicitly when ready.
    """
    if not model_dir:
        return None
    artifact_path = Path(model_dir) / MODEL_FILENAME
    if not artifact_path.exists():
        return None
    try:
        return load_dj_taxonomy_model(model_dir, taxonomy_path=taxonomy_path)
    except ValueError as exc:
        # Schema mismatch, taxonomy drift, or corrupt artifact — skip.
        import logging
        logging.getLogger(__name__).info(
            "DJ taxonomy model at %s is incompatible (%s); skipping until retrain.",
            artifact_path,
            exc,
        )
        return None


def evaluate_dj_taxonomy_models(
    store: CsvStore,
    labels_path: str,
    *,
    model_dir: str,
    taxonomy_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    taxonomy = load_dj_taxonomy(taxonomy_path)
    labels = _load_label_rows(labels_path, taxonomy)
    base_dir = Path(model_dir)
    internal_model = load_dj_taxonomy_model(base_dir / "internal", taxonomy_path=taxonomy_path)
    external_model = load_dj_taxonomy_model(base_dir / "external", taxonomy_path=taxonomy_path)

    internal_examples, internal_skipped = _build_training_examples(
        labels,
        store,
        mode="internal",
        show_progress=show_progress,
    )
    external_examples, external_skipped = _build_training_examples(
        labels,
        store,
        mode="external",
        show_progress=show_progress,
    )
    examples_by_key = {
        _example_key(example): {"internal": example}
        for example in internal_examples
    }
    for example in external_examples:
        examples_by_key.setdefault(_example_key(example), {})["external"] = example

    rows: list[dict[str, Any]] = []
    progress = ProgressBar(len(examples_by_key), label="Evaluate DJ taxonomy models", enabled=show_progress)
    for index, (key, pair) in enumerate(sorted(examples_by_key.items()), start=1):
        internal_example = pair.get("internal")
        external_example = pair.get("external")
        example = external_example or internal_example
        if not example:
            continue
        expected = example["category_id"]
        internal_prediction = (
            _predict_for_example(internal_model, internal_example, taxonomy)
            if internal_example else DjCategoryPrediction(feature_mode="internal")
        )
        external_prediction = (
            _predict_for_example(external_model, external_example, taxonomy)
            if external_example else DjCategoryPrediction(feature_mode="external")
        )
        rows.append(
            {
                "track_id": example["track"].track_id,
                "file_name": example.get("file_name", ""),
                "artist": example["track"].artist_canonical,
                "title": example["track"].title_canonical,
                "expected_category_id": expected,
                "expected_category_label": taxonomy.category(expected).label,
                "internal_category_id": internal_prediction.category_id,
                "internal_label": internal_prediction.category_label,
                "internal_confidence": internal_prediction.confidence,
                "internal_correct": internal_prediction.category_id == expected,
                "internal_top3_correct": _top3_contains(internal_prediction, expected),
                "external_category_id": external_prediction.category_id,
                "external_label": external_prediction.category_label,
                "external_confidence": external_prediction.confidence,
                "external_correct": external_prediction.category_id == expected,
                "external_top3_correct": _top3_contains(external_prediction, expected),
                "models_agree": internal_prediction.category_id == external_prediction.category_id,
                "confidence_delta": round(external_prediction.confidence - internal_prediction.confidence, 3),
                "external_evidence_available": example.get("external_evidence_available", False),
            }
        )
        progress.update(index, example["track"].title_canonical)
    progress.finish()

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_csv(out_dir / COMPARISON_CSV, rows)
    metrics = {
        "examples": len(rows),
        "skipped_rows": {"internal": internal_skipped, "external": external_skipped},
        "internal": _metrics_from_rows(rows, "internal"),
        "external": _metrics_from_rows(rows, "external"),
        "agreement_rate": _safe_rate(sum(1 for row in rows if row["models_agree"]), len(rows)),
        "external_improved": sum(
            1 for row in rows if row["external_correct"] and not row["internal_correct"]
        ),
        "external_worsened": sum(
            1 for row in rows if row["internal_correct"] and not row["external_correct"]
        ),
    }
    with (out_dir / COMPARISON_JSON).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def classify_all_dj_taxonomies(
    store: CsvStore,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    primary_model: str = "external",
    show_progress: bool = False,
) -> int:
    if primary_model not in VALID_FEATURE_MODES:
        raise ValueError(f"Invalid primary DJ taxonomy model: {primary_model}")
    taxonomy = load_dj_taxonomy(taxonomy_path)
    base_dir = Path(model_dir or Path(store.output_dir) / "dj_taxonomy_model")
    internal_model = load_dj_taxonomy_model_if_available(base_dir / "internal", taxonomy_path=taxonomy_path)
    external_model = load_dj_taxonomy_model_if_available(base_dir / "external", taxonomy_path=taxonomy_path)
    if internal_model is None and external_model is None:
        raise FileNotFoundError(f"No DJ taxonomy models found under {base_dir}")

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    progress = ProgressBar(len(tracks), label="Classify DJ taxonomy", enabled=show_progress)
    for index, track in enumerate(tracks, start=1):
        track_obs = obs_by_track.get(track.track_id, [])
        file_record = file_by_track.get(track.track_id)
        internal_prediction = (
            internal_model.predict(track, track_obs, file_record, taxonomy)
            if internal_model else DjCategoryPrediction(feature_mode="internal")
        )
        external_prediction = (
            external_model.predict(track, track_obs, file_record, taxonomy)
            if external_model else DjCategoryPrediction(feature_mode="external")
        )
        _apply_predictions_to_track(
            track,
            taxonomy,
            internal_prediction,
            external_prediction,
            has_external_evidence=external_evidence_available(track_obs),
            primary_model=primary_model,
        )
        progress.update(index, track.title_canonical, category=track.dj_taxonomy_id)
    progress.finish()
    store.save_tracks(tracks)
    return len(tracks)


def _apply_predictions_to_track(
    track: LogicalTrack,
    taxonomy: DjTaxonomy,
    internal_prediction: DjCategoryPrediction,
    external_prediction: DjCategoryPrediction,
    *,
    has_external_evidence: bool,
    primary_model: str = "external",
) -> None:
    if primary_model == "internal":
        primary = internal_prediction if internal_prediction.category_id else external_prediction
        source_model = "internal" if internal_prediction.category_id else "external"
    else:
        primary = external_prediction if external_prediction.category_id else internal_prediction
        source_model = "external" if external_prediction.category_id else "internal"

    # Spec §7.2 — null fallback when calibrated confidence is below threshold
    if primary.category_id and primary.confidence < UNCLASSIFIED_THRESHOLD:
        track.dj_taxonomy_id = UNCLASSIFIED_ID
        track.dj_taxonomy_label = ""
        track.dj_taxonomy_family = ""
        track.dj_taxonomy_moods = ""
        track.dj_taxonomy_grooves = ""
        track.dj_taxonomy_set_roles = ""
        track.dj_taxonomy_bpm_range = ""
        track.dj_taxonomy_energy_range = ""
        track.dj_taxonomy_vocal_profiles = ""
        track.dj_taxonomy_source_genres = ""
        track.dj_taxonomy_keywords = ""
        track.dj_taxonomy_confidence = primary.confidence
        track.dj_taxonomy_confidence_level = "unknown"
        track.dj_taxonomy_source_model = source_model
    elif primary.category_id:
        category = taxonomy.category(primary.category_id)
        track.dj_taxonomy_id = category.id
        track.dj_taxonomy_label = category.label
        track.dj_taxonomy_family = category.family
        track.dj_taxonomy_moods = ";".join(category.moods)
        track.dj_taxonomy_grooves = ";".join(category.grooves)
        track.dj_taxonomy_set_roles = ";".join(category.set_roles)
        track.dj_taxonomy_bpm_range = _range_text(category.bpm_range)
        track.dj_taxonomy_energy_range = _range_text(category.energy_range)
        track.dj_taxonomy_vocal_profiles = ";".join(category.vocal_profiles)
        track.dj_taxonomy_source_genres = ";".join(category.source_genres)
        track.dj_taxonomy_keywords = ";".join(category.keywords)
        track.dj_taxonomy_confidence = primary.confidence
        track.dj_taxonomy_confidence_level = confidence_to_level(primary.confidence)
        track.dj_taxonomy_source_model = source_model
    track.dj_taxonomy_internal_id = internal_prediction.category_id
    track.dj_taxonomy_internal_label = internal_prediction.category_label
    track.dj_taxonomy_internal_confidence = internal_prediction.confidence
    track.dj_taxonomy_external_id = external_prediction.category_id
    track.dj_taxonomy_external_label = external_prediction.category_label
    track.dj_taxonomy_external_confidence = external_prediction.confidence
    track.dj_taxonomy_models_agree = (
        "YES"
        if internal_prediction.category_id
        and external_prediction.category_id
        and internal_prediction.category_id == external_prediction.category_id
        else "NO"
    )
    track.dj_taxonomy_external_evidence_available = "YES" if has_external_evidence else "NO"
    track.dj_taxonomy_alternatives = json.dumps(
        {
            "internal": internal_prediction.alternatives,
            "external": external_prediction.alternatives,
        },
        sort_keys=True,
    )
    track.dj_taxonomy_evidence = json.dumps(
        {
            "internal": internal_prediction.evidence,
            "external": external_prediction.evidence,
            "warnings": {
                "internal": internal_prediction.warnings,
                "external": external_prediction.warnings,
            },
        },
        sort_keys=True,
    )
    track.dj_taxonomy_version = taxonomy.version


def _train_final_model_with_metrics(
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
    *,
    feature_mode: str,
    labels_path: str,
    validation_split: float,
    seed: int,
) -> tuple[DjTaxonomyModel, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if len(examples) < 8 or validation_split <= 0:
        train_examples = examples
        validation_examples = examples
        validation_kind = "in_sample"
        warnings.append("Dataset is small; metrics are in-sample and should be treated as smoke-test metrics.")
    else:
        train_examples, validation_examples = _split_examples(examples, validation_split=validation_split, seed=seed)
        validation_kind = "holdout"
        if not validation_examples:
            validation_examples = train_examples
            validation_kind = "in_sample"
            warnings.append("Validation split produced no holdout rows; metrics are in-sample.")

    validation_model = _fit_model(
        train_examples,
        taxonomy,
        feature_mode=feature_mode,
        labels_path=labels_path,
    )
    metrics = _evaluate_examples(validation_model, validation_examples, taxonomy)
    metrics["validation_kind"] = validation_kind
    metrics["validation_examples"] = len(validation_examples)

    final_model = _fit_model(
        examples,
        taxonomy,
        feature_mode=feature_mode,
        labels_path=labels_path,
    )
    return final_model, metrics, warnings


def _fit_model(
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
    *,
    feature_mode: str,
    labels_path: str,
) -> DjTaxonomyModel:
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([example["features"] for example in examples])
    labels = [example["category_id"] for example in examples]
    return DjTaxonomyModel(
        vectorizer=vectorizer,
        classifier=_fit_classifier(x, labels),
        feature_mode=feature_mode,
        taxonomy_version=taxonomy.version,
        taxonomy_hash=taxonomy.hash(),
        feature_schema_version=DJ_FEATURE_SCHEMA_VERSION,
        trained_at=now_iso(),
        labels_hash=_file_hash(Path(labels_path)),
        examples=len(examples),
    )


def _load_label_rows(labels_path: str, taxonomy: DjTaxonomy) -> list[dict[str, str]]:
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    valid_rows: list[dict[str, str]] = []
    invalid: list[str] = []
    for index, row in enumerate(rows, start=2):
        category_id = (row.get("category_id") or row.get("dj_taxonomy_id") or "").strip()
        label_source = (row.get("label_source") or "").strip().lower()
        if label_source.endswith("_error"):
            invalid.append(f"row {index}: error row {category_id or '<blank>'}")
            continue
        if not taxonomy.validate_category_id(category_id):
            invalid.append(f"row {index}: unknown category_id {category_id or '<blank>'}")
            continue
        row["category_id"] = category_id
        valid_rows.append(row)
    if invalid and not valid_rows:
        raise ValueError("No valid DJ taxonomy labels found: " + "; ".join(invalid[:5]))
    return valid_rows


def _build_training_examples(
    labels: list[dict[str, str]],
    store: CsvStore,
    *,
    mode: str,
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    track_by_id = {track.track_id: track for track in tracks if track.track_id}
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    file_by_name = {f.file_name.lower(): f for f in files if f.file_name}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    examples: list[dict[str, Any]] = []
    skipped = 0
    progress = ProgressBar(len(labels), label=f"Build {mode} DJ taxonomy data", enabled=show_progress)
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
        track_obs = obs_by_track.get(track.track_id, [])
        features = build_track_features(
            track,
            track_obs,
            file_record,
            label_row=row,
            feature_mode=mode,
        )
        if not features:
            skipped += 1
            progress.update(index, row.get("file_name", ""), examples=len(examples), skipped=skipped)
            continue
        examples.append(
            {
                "track": track,
                "features": features,
                "category_id": row["category_id"],
                "file_name": row.get("file_name") or (file_record.file_name if file_record else ""),
                "observations": track_obs,
                "file_record": file_record,
                "external_evidence_available": external_evidence_available(track_obs),
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
        tagger_energy=(row.get("tagger_energy") or row.get("energy") or "").strip(),
        tagger_vibe=(row.get("tagger_mood") or row.get("tagger_vibe") or row.get("vibe") or "").strip(),
        tagger_vocal=(row.get("tagger_vocal") or row.get("vocal") or "").strip(),
        tagger_structure=(row.get("tagger_structure") or row.get("structure") or "").strip(),
    )


def _fit_classifier(x: Any, labels: list[str], *, tune: bool = False, seed: int = 42) -> Any:
    """Fit a LogisticRegression classifier. Optionally hyperparameter-tune via
    RandomizedSearchCV across C / solver / max_iter when `tune=True`.

    max_iter raised from 1000 -> 2000 (the v1 model warned convergence issues
    on the high-dim sparse feature space).
    """
    unique = sorted(set(labels))
    if len(unique) == 1:
        return _ConstantClassifier(unique[0])
    if tune:
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        import numpy as np
        # CV requires at least 2 examples per class; fold count clamped accordingly.
        from collections import Counter
        min_per_class = min(Counter(labels).values())
        cv_folds = min(3, min_per_class) if min_per_class >= 2 else 0
        if cv_folds >= 2:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            base = LogisticRegression(class_weight="balanced", random_state=seed)
            search = RandomizedSearchCV(
                base,
                param_distributions={
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["lbfgs", "saga"],
                    "max_iter": [2000, 4000],
                    "penalty": ["l2"],
                },
                n_iter=10,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                random_state=seed,
                refit=True,
            )
            search.fit(x, labels)
            logger.info("LR best params: %s (CV f1_macro=%.3f)", search.best_params_, search.best_score_)
            return search.best_estimator_
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(x, labels)
    return classifier


def _probability_map(model: Any, x: Any) -> dict[str, float]:
    probabilities = model.predict_proba(x)[0]
    return {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}


def _confidence_from_model_score(score: float, margin: float, features: dict[str, float]) -> float:
    richness = min(0.12, len(features) / 420.0)
    raw = 0.18 + 0.64 * max(0.0, min(score, 1.0)) + 0.18 * max(0.0, min(margin, 1.0)) + richness
    return round(max(0.05, min(0.98, raw)), 3)


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_split)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


# ── Spec §9.2 / §9.3 — Library-distribution check + anti-collapse guard ─────


class CollapseGuardError(RuntimeError):
    """Raised by train_with_collapse_guard when the largest predicted bucket
    persistently exceeds the 20% cap after the maximum retrain attempts."""


def run_library_distribution_check(
    store: CsvStore,
    *,
    model_dir: str | Path,
    taxonomy_path: str | None = None,
    primary_model: str = "external",
    max_bucket_share: float = 0.20,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Predict on every track and report bucket-share distribution.

    Returns a dict matching the spec §9.2 library_distribution_check schema:
        {
            "total_tracks": <int>,
            "predictions": {"<id>": <count>},
            "largest_bucket_id": "<id>",
            "largest_bucket_share": <float>,
            "passes_cap": <bool>,
            "max_bucket_share": <float>,  # the threshold tested
        }
    """
    taxonomy = load_dj_taxonomy(taxonomy_path)
    base_dir = Path(model_dir)
    model = load_dj_taxonomy_model_if_available(base_dir / primary_model, taxonomy_path=taxonomy_path)
    if model is None:
        # Fall back to the other mode if requested one is missing
        other = "internal" if primary_model == "external" else "external"
        model = load_dj_taxonomy_model_if_available(base_dir / other, taxonomy_path=taxonomy_path)
    if model is None:
        raise FileNotFoundError(f"No DJ taxonomy model found under {base_dir}")

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    counts: dict[str, int] = {}
    progress = ProgressBar(len(tracks), label="Library distribution", enabled=show_progress)
    for index, track in enumerate(tracks, start=1):
        track_obs = obs_by_track.get(track.track_id, [])
        file_record = file_by_track.get(track.track_id)
        prediction = model.predict(track, track_obs, file_record, taxonomy)
        # Apply the same null-fallback rule the writer uses, so the audit
        # reflects what would actually end up on tracks_master.csv.
        if prediction.category_id and prediction.confidence < UNCLASSIFIED_THRESHOLD:
            cat_id = UNCLASSIFIED_ID
        else:
            cat_id = prediction.category_id or UNCLASSIFIED_ID
        counts[cat_id] = counts.get(cat_id, 0) + 1
        progress.update(index, track.title_canonical, category=cat_id)
    progress.finish()

    total = max(1, len(tracks))
    largest_id = max(counts, key=counts.get) if counts else ""
    largest_count = counts.get(largest_id, 0)
    largest_share = largest_count / total
    return {
        "total_tracks": len(tracks),
        "predictions": counts,
        "largest_bucket_id": largest_id,
        "largest_bucket_share": round(largest_share, 4),
        "passes_cap": largest_share <= max_bucket_share,
        "max_bucket_share": max_bucket_share,
    }


def train_with_collapse_guard(
    store: CsvStore,
    labels_path: str,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    validation_split: float = 0.2,
    seed: int = 42,
    max_bucket_share: float = 0.20,
    max_retrains: int = 3,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Spec §9.3 — train with anti-collapse guard.

    Trains normally, then runs run_library_distribution_check. If the largest
    bucket exceeds the cap, retrains up to max_retrains more times. The retrain
    penalty (per spec) is a class-weight reduction on the offending category;
    this implementation logs the offending category and re-shuffles the seed,
    leaving deeper class-weight surgery as a follow-up if simple re-seeding
    doesn't break the collapse on its own.
    """
    last_check: dict[str, Any] = {}
    offending_history: list[str] = []
    for attempt in range(max_retrains + 1):
        attempt_seed = seed + attempt * 17
        result = train_dj_taxonomy_models(
            store,
            labels_path,
            taxonomy_path=taxonomy_path,
            model_dir=model_dir,
            validation_split=validation_split,
            seed=attempt_seed,
            show_progress=show_progress,
        )
        last_check = run_library_distribution_check(
            store,
            model_dir=result["model_dir"],
            taxonomy_path=taxonomy_path,
            show_progress=show_progress,
            max_bucket_share=max_bucket_share,
        )
        if last_check["passes_cap"]:
            result["library_distribution_check"] = last_check
            result["collapse_guard"] = {
                "attempts": attempt + 1,
                "offending_history": offending_history,
                "passed": True,
            }
            return result
        offending_history.append(last_check["largest_bucket_id"])
        if attempt < max_retrains:
            # log to make iteration diagnostics surface to the operator
            print(
                f"[collapse-guard] attempt {attempt + 1}: largest bucket "
                f"{last_check['largest_bucket_id']!r} at "
                f"{last_check['largest_bucket_share']:.1%} — retraining (seed={attempt_seed})"
            )
    raise CollapseGuardError(
        f"Library distribution still exceeds {max_bucket_share:.0%} after "
        f"{max_retrains + 1} attempts. Offending history: {offending_history}. "
        f"Last check: {last_check}"
    )


# ── Unified single-model training (LR vs XGB head-to-head) ──────────────────


def train_dj_taxonomy_unified(
    store: CsvStore,
    labels_path: str,
    *,
    model_type: str = "lr",
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    validation_split: float = 0.2,
    seed: int = 42,
    tune_lr: bool = False,
    xgb_n_iter: int = 30,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train ONE model per algorithm on the unified (external) feature set.

    model_type:
      - "lr"   : LogisticRegression (default)
      - "xgb"  : XGBoost with RandomizedSearchCV tuning
      - "both" : train both and write a comparison report

    Output directory layout:
      <model_dir>/lr/model.pkl + training_report.json
      <model_dir>/xgb/model.pkl + training_report.json + best_params.json
      <model_dir>/comparison.json  (only when model_type='both')
    """
    if model_type not in {"lr", "xgb", "both"}:
        raise ValueError(f"model_type must be one of lr|xgb|both, got {model_type!r}")

    taxonomy = load_dj_taxonomy(taxonomy_path)
    labels = _load_label_rows(labels_path, taxonomy)
    base_dir = Path(model_dir or Path(store.output_dir) / "dj_taxonomy_model")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build feature examples once (unified = external mode = superset of internal)
    examples, skipped = _build_training_examples(
        labels, store, mode="external", show_progress=show_progress
    )
    if not examples:
        raise ValueError("No valid training examples could be built from labels CSV")

    results: dict[str, Any] = {"model_dir": str(base_dir), "examples_total": len(examples), "skipped_rows": skipped}

    if model_type in {"lr", "both"}:
        results["lr"] = _train_lr_unified(
            examples, taxonomy, labels_path=labels_path,
            model_dir=base_dir / "lr",
            validation_split=validation_split, seed=seed, tune=tune_lr,
        )

    if model_type in {"xgb", "both"}:
        from .dj_model_xgb import train_xgb_model

        fit_start = time.perf_counter()
        xgb_model, xgb_metrics, xgb_warnings = train_xgb_model(
            examples, taxonomy,
            labels_path=labels_path,
            validation_split=validation_split,
            seed=seed,
            n_iter=xgb_n_iter,
            show_progress=show_progress,
        )
        xgb_dir = base_dir / "xgb"
        xgb_model.save(xgb_dir)
        xgb_stats = DjTrainingStats(
            labels_path=labels_path,
            model_dir=str(xgb_dir),
            examples=len(examples),
            classes=len({example["category_id"] for example in examples}),
            skipped_rows=skipped,
            mode="external",
            metrics=xgb_metrics,
            warnings=xgb_warnings,
        )
        _write_xgb_training_report(xgb_dir / REPORT_FILENAME, xgb_stats, xgb_model)
        results["xgb"] = {
            "model_dir": str(xgb_dir),
            "examples": len(examples),
            "classes": xgb_stats.classes,
            "metrics": xgb_metrics,
            "warnings": xgb_warnings,
            "best_params": xgb_model.best_params,
            "fit_time_seconds": xgb_metrics.get("fit_time_seconds", round(time.perf_counter() - fit_start, 2)),
        }

    if model_type == "both":
        from .dj_model_comparison import write_comparison_json

        write_comparison_json(base_dir / "comparison.json", results.get("lr"), results.get("xgb"))
        results["comparison_path"] = str(base_dir / "comparison.json")

    return results


def _train_lr_unified(
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
    *,
    labels_path: str,
    model_dir: Path,
    validation_split: float,
    seed: int,
    tune: bool,
) -> dict[str, Any]:
    """Train a single LR model on the unified feature set + write artifacts."""
    fit_start = time.perf_counter()

    # Validation split, mirroring _train_final_model_with_metrics
    if len(examples) < 8 or validation_split <= 0:
        train_examples = examples
        validation_examples = examples
        validation_kind = "in_sample"
        warnings_list = ["Dataset is small; LR metrics are in-sample."]
    else:
        train_examples, validation_examples = _split_examples(
            examples, validation_split=validation_split, seed=seed
        )
        validation_kind = "holdout"
        if not validation_examples:
            validation_examples = train_examples
            validation_kind = "in_sample"
            warnings_list = ["Validation split produced no holdout rows; metrics are in-sample."]
        else:
            warnings_list = []

    # Fit on train split for validation metrics
    vec_val = DictVectorizer(sparse=True)
    x_train = vec_val.fit_transform([e["features"] for e in train_examples])
    y_train = [e["category_id"] for e in train_examples]
    val_clf = _fit_classifier(x_train, y_train, tune=tune, seed=seed)
    val_model = DjTaxonomyModel(
        vectorizer=vec_val,
        classifier=val_clf,
        feature_mode="external",
        taxonomy_version=taxonomy.version,
        taxonomy_hash=taxonomy.hash(),
        feature_schema_version=DJ_FEATURE_SCHEMA_VERSION,
        trained_at=now_iso(),
        labels_hash=_file_hash(Path(labels_path)),
        examples=len(train_examples),
    )
    metrics = _evaluate_examples(val_model, validation_examples, taxonomy)
    metrics["validation_kind"] = validation_kind
    metrics["validation_examples"] = len(validation_examples)
    metrics["fit_time_seconds"] = round(time.perf_counter() - fit_start, 2)
    metrics["model_version"] = DJ_MODEL_VERSION
    metrics["model_type"] = "lr"

    # Refit on all examples for the final shipped model
    final_model = _fit_model(examples, taxonomy, feature_mode="external", labels_path=labels_path)
    final_model.save(model_dir)
    stats = DjTrainingStats(
        labels_path=labels_path,
        model_dir=str(model_dir),
        examples=len(examples),
        classes=len({e["category_id"] for e in examples}),
        skipped_rows=0,
        mode="external",
        metrics=metrics,
        warnings=warnings_list,
    )
    _write_training_audit(model_dir / AUDIT_FILENAME, examples)
    _write_training_report(model_dir / REPORT_FILENAME, stats, final_model)
    return {
        "model_dir": str(model_dir),
        "examples": len(examples),
        "classes": stats.classes,
        "metrics": metrics,
        "warnings": warnings_list,
        "fit_time_seconds": metrics["fit_time_seconds"],
    }


def _write_xgb_training_report(path: Path, stats: DjTrainingStats, model: "Any") -> None:
    """Write a training report for an XGB model (mirrors LR's _write_training_report)."""
    payload = _stats_to_dict(stats)
    payload.update(
        {
            "model_version": getattr(model, "best_params", None)
            and stats.metrics.get("model_version", "dj-taxonomy-xgb-v1"),
            "feature_schema_version": DJ_FEATURE_SCHEMA_VERSION,
            "trained_at": model.trained_at,
            "labels_hash": model.labels_hash,
            "taxonomy_hash": model.taxonomy_hash,
            "best_params": getattr(model, "best_params", {}),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _evaluate_examples(
    model: DjTaxonomyModel,
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
) -> dict[str, Any]:
    rows = []
    for example in examples:
        prediction = _predict_for_example(model, example, taxonomy)
        top3 = [alt["category_id"] for alt in prediction.alternatives]
        top3.insert(0, prediction.category_id)
        rows.append(
            {
                "expected_category_id": example["category_id"],
                f"{model.feature_mode}_category_id": prediction.category_id,
                f"{model.feature_mode}_confidence": prediction.confidence,
                f"{model.feature_mode}_correct": prediction.category_id == example["category_id"],
                f"{model.feature_mode}_top3_correct": example["category_id"] in top3,
            }
        )
    metrics = _metrics_from_rows(rows, model.feature_mode)
    metrics["examples"] = len(rows)
    return metrics


def _predict_for_example(
    model: DjTaxonomyModel,
    example: dict[str, Any] | None,
    taxonomy: DjTaxonomy,
) -> DjCategoryPrediction:
    if not example:
        return DjCategoryPrediction(feature_mode=model.feature_mode)
    return model.predict(
        example["track"],
        example.get("observations", []),
        example.get("file_record"),
        taxonomy,
    )


def _metrics_from_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "average_confidence": 0.0,
        }
    y_true = [row["expected_category_id"] for row in rows]
    y_pred = [row.get(f"{prefix}_category_id", "") for row in rows]
    top1_correct = sum(1 for row in rows if row.get(f"{prefix}_correct"))
    top3_key = f"{prefix}_top3_correct"
    top3_values = [row.get(top3_key) for row in rows if top3_key in row]
    confidences = [float(row.get(f"{prefix}_confidence") or 0.0) for row in rows]
    return {
        "top1_accuracy": _safe_rate(top1_correct, len(rows)),
        "top3_accuracy": _safe_rate(sum(1 for value in top3_values if value), len(top3_values))
        if top3_values else 0.0,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 3),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 3),
        "average_confidence": round(sum(confidences) / len(confidences), 3),
        "confidence_buckets": _confidence_buckets(rows, prefix),
        "per_category": _per_category_metrics(rows, prefix),
    }


def _confidence_buckets(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    buckets = {
        "0.00-0.50": [0.0, 0],
        "0.50-0.70": [0.0, 0],
        "0.70-0.85": [0.0, 0],
        "0.85-1.00": [0.0, 0],
    }
    correct_key = f"{prefix}_correct"
    confidence_key = f"{prefix}_confidence"
    for row in rows:
        confidence = float(row.get(confidence_key) or 0.0)
        if confidence < 0.50:
            bucket = "0.00-0.50"
        elif confidence < 0.70:
            bucket = "0.50-0.70"
        elif confidence < 0.85:
            bucket = "0.70-0.85"
        else:
            bucket = "0.85-1.00"
        buckets[bucket][0] += 1.0 if row.get(correct_key) else 0.0
        buckets[bucket][1] += 1
    return {
        bucket: {"accuracy": _safe_rate(int(values[0]), values[1]), "support": values[1]}
        for bucket, values in buckets.items()
    }


def _per_category_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row["expected_category_id"]
        stats = totals.setdefault(expected, {"support": 0, "correct": 0})
        stats["support"] += 1
        if row.get(f"{prefix}_correct"):
            stats["correct"] += 1
    return {
        category_id: {
            "support": stats["support"],
            "accuracy": _safe_rate(stats["correct"], stats["support"]),
        }
        for category_id, stats in sorted(totals.items())
    }


def _write_training_audit(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "track_id",
                "artist",
                "title",
                "category_id",
                "feature_count",
                "external_evidence_available",
            ],
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
                    "category_id": example["category_id"],
                    "feature_count": len(example["features"]),
                    "external_evidence_available": example.get("external_evidence_available", False),
                }
            )


def _write_training_report(path: Path, stats: DjTrainingStats, model: DjTaxonomyModel) -> None:
    payload = _stats_to_dict(stats)
    payload.update(
        {
            "model_version": DJ_MODEL_VERSION,
            "feature_schema_version": DJ_FEATURE_SCHEMA_VERSION,
            "trained_at": model.trained_at,
            "labels_hash": model.labels_hash,
            "taxonomy_hash": model.taxonomy_hash,
        }
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "file_name",
        "artist",
        "title",
        "expected_category_id",
        "expected_category_label",
        "internal_category_id",
        "internal_label",
        "internal_confidence",
        "internal_correct",
        "internal_top3_correct",
        "external_category_id",
        "external_label",
        "external_confidence",
        "external_correct",
        "external_top3_correct",
        "models_agree",
        "confidence_delta",
        "external_evidence_available",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stats_to_dict(stats: DjTrainingStats) -> dict[str, Any]:
    return {
        "labels_path": stats.labels_path,
        "model_dir": stats.model_dir,
        "examples": stats.examples,
        "classes": stats.classes,
        "skipped_rows": stats.skipped_rows,
        "mode": stats.mode,
        "metrics": stats.metrics,
        "warnings": stats.warnings,
    }


def _example_key(example: dict[str, Any]) -> str:
    track = example["track"]
    return track.track_id or example.get("file_name", "")


def _top3_contains(prediction: DjCategoryPrediction, expected: str) -> bool:
    top_ids = [prediction.category_id]
    top_ids.extend(str(alt.get("category_id") or "") for alt in prediction.alternatives)
    return expected in top_ids[:3]


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _range_text(value: tuple[int, int] | tuple[()]) -> str:
    if len(value) != 2:
        return ""
    return f"{value[0]}-{value[1]}"
