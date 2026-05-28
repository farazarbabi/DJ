"""XGBoost-only supervised classifier for the flat DJ-functional taxonomy.

This module holds the orchestration + shared helpers (feature building,
splits, metrics, prediction-to-track wiring). The XGBoost training code
itself lives in ``dj_model_xgb.py``. The previous LogisticRegression
pipeline has been removed; XGB is the only model.

The dual ``dj_taxonomy_internal_*`` / ``dj_taxonomy_external_*`` columns
on ``LogicalTrack`` are preserved (exports + tag-writer depend on them)
and now both receive the SAME XGB prediction.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score

from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .dj_schema import DjTaxonomy, external_evidence_available, load_dj_taxonomy
from .features import build_track_features

logger = logging.getLogger(__name__)

DJ_MODEL_VERSION = "dj-taxonomy-xgb-v1"
# Bumped v2 -> v3 when dense audio features (DSP, popularity, CLAP, vocal_stem)
# were wired through build_track_features. Old pickles are auto-skipped by
# load_dj_taxonomy_model_if_available / load_xgb_model.
DJ_FEATURE_SCHEMA_VERSION = "dj-taxonomy-feature-schema-v4"
MIN_EXAMPLES_PER_CATEGORY = 3
MODEL_FILENAME = "model.pkl"
REPORT_FILENAME = "training_report.json"
AUDIT_FILENAME = "training_audit.csv"
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
class LabelLoadDiagnostics:
    total_rows: int = 0
    valid_rows: int = 0
    error_rows: int = 0
    invalid_rows: int = 0
    alias_counts: dict[str, int] = field(default_factory=dict)
    alias_targets: dict[str, str] = field(default_factory=dict)
    invalid_category_counts: dict[str, int] = field(default_factory=dict)

    def metric_fields(self) -> dict[str, Any]:
        return {
            "label_rows_total": self.total_rows,
            "label_rows_loaded": self.valid_rows,
            "label_rows_error_skipped": self.error_rows,
            "label_rows_invalid_skipped": self.invalid_rows,
            "label_alias_counts": dict(sorted(self.alias_counts.items())),
            "label_alias_targets": dict(sorted(self.alias_targets.items())),
            "invalid_category_counts": dict(sorted(self.invalid_category_counts.items())),
        }

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.error_rows:
            out.append(f"Skipped {self.error_rows} ground-truth error rows with no category_id.")
        if self.invalid_rows:
            details = ", ".join(
                f"{category_id}={count}"
                for category_id, count in sorted(self.invalid_category_counts.items())
            )
            out.append(
                f"Skipped {self.invalid_rows} rows with category IDs missing from the current "
                f"dj_taxonomy.json: {details}"
            )
        return out


def train_dj_taxonomy_models(
    store: CsvStore,
    labels_path: str,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    validation_split: float = 0.2,
    seed: int = 42,
    xgb_n_iter: int = 30,
    feature_mode: str = "external",
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train the XGB DJ taxonomy classifier.

    The unified feature mode is ``external`` (superset of internal). The
    trained model lives at ``<model_dir>/xgb/model.pkl``. Reports go to
    ``training_report.json`` + ``training_audit.csv`` alongside it.
    """
    from .dj_model_xgb import train_xgb_model

    if feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {sorted(VALID_FEATURE_MODES)}, got {feature_mode!r}")

    taxonomy = load_dj_taxonomy(taxonomy_path)
    labels, label_diagnostics = _load_label_rows_with_diagnostics(labels_path, taxonomy)
    base_dir = Path(model_dir or Path(store.output_dir) / "dj_taxonomy_model")
    base_dir.mkdir(parents=True, exist_ok=True)

    ucache = _resolve_cache()
    examples, skipped, dropped_categories = _build_training_examples(
        labels, store, mode=feature_mode, show_progress=show_progress, ucache=ucache,
    )
    if not examples:
        raise ValueError("No valid DJ taxonomy training examples could be built from labels CSV")

    fit_start = time.perf_counter()
    model, metrics, warnings_list = train_xgb_model(
        examples, taxonomy,
        labels_path=labels_path,
        validation_split=validation_split,
        seed=seed,
        n_iter=xgb_n_iter,
        feature_mode=feature_mode,
        show_progress=show_progress,
    )
    metrics.update(label_diagnostics.metric_fields())
    warnings_list = label_diagnostics.warnings() + list(warnings_list)
    if dropped_categories:
        warnings_list = list(warnings_list) + [
            f"Dropped {sum(dropped_categories.values())} rows in "
            f"{len(dropped_categories)} under-supported categories "
            f"(<{MIN_EXAMPLES_PER_CATEGORY} examples): {sorted(dropped_categories.keys())}"
        ]

    xgb_dir = base_dir / "xgb"
    model.save(xgb_dir)
    stats = DjTrainingStats(
        labels_path=labels_path,
        model_dir=str(xgb_dir),
        examples=len(examples),
        classes=len({example["category_id"] for example in examples}),
        skipped_rows=skipped,
        mode=feature_mode,
        metrics=metrics,
        warnings=warnings_list,
    )
    _write_training_audit(xgb_dir / AUDIT_FILENAME, examples)
    _write_xgb_training_report(xgb_dir / REPORT_FILENAME, stats, model)
    return {
        "model_dir": str(base_dir),
        "xgb": {
            "model_dir": str(xgb_dir),
            "examples": len(examples),
            "classes": stats.classes,
            "metrics": metrics,
            "warnings": warnings_list,
            "best_params": model.best_params,
            "fit_time_seconds": metrics.get("fit_time_seconds", round(time.perf_counter() - fit_start, 2)),
        },
        "examples_total": len(examples),
        "skipped_rows": skipped,
        "dropped_categories": dropped_categories,
    }


def evaluate_dj_taxonomy_models(
    store: CsvStore,
    labels_path: str,
    *,
    model_dir: str,
    taxonomy_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Re-evaluate the trained XGB model against the labels CSV."""
    taxonomy = load_dj_taxonomy(taxonomy_path)
    labels, label_diagnostics = _load_label_rows_with_diagnostics(labels_path, taxonomy)
    model = load_dj_taxonomy_model_if_available(model_dir, taxonomy_path=taxonomy_path)
    if model is None:
        raise FileNotFoundError(f"No XGB DJ taxonomy model found under {model_dir}")

    ucache = _resolve_cache()
    examples, skipped, _dropped = _build_training_examples(
        labels, store, mode=model.feature_mode, show_progress=show_progress, ucache=ucache,
    )
    metrics = _evaluate_examples(model, examples, taxonomy, ucache=ucache)
    metrics["examples"] = len(examples)
    metrics["skipped_rows"] = skipped
    metrics["evaluation_kind"] = "in_sample_refit"
    metrics["evaluation_warning"] = (
        "This evaluates the saved model after it was refit on all retained examples; "
        "use training_report.json for held-out validation metrics."
    )
    metrics.update(label_diagnostics.metric_fields())
    return metrics


def classify_all_dj_taxonomies(
    store: CsvStore,
    *,
    taxonomy_path: str | None = None,
    model_dir: str | None = None,
    show_progress: bool = False,
    primary_model: str | None = None,  # accepted + ignored for back-compat
) -> int:
    """Classify every track via the XGB model. Writes results to track columns.

    ``primary_model`` is accepted but ignored — there's only one model now.
    Both ``dj_taxonomy_internal_*`` and ``dj_taxonomy_external_*`` columns
    are populated with the same XGB prediction so downstream consumers
    (exports, tag-writer) keep working.
    """
    del primary_model  # only one model now
    taxonomy = load_dj_taxonomy(taxonomy_path)
    base_dir = Path(model_dir or Path(store.output_dir) / "dj_taxonomy_model")
    model = load_dj_taxonomy_model_if_available(base_dir, taxonomy_path=taxonomy_path)
    if model is None:
        raise FileNotFoundError(f"No XGB DJ taxonomy model found under {base_dir}")

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    ucache = _resolve_cache()
    progress = ProgressBar(len(tracks), label="Classify DJ taxonomy", enabled=show_progress)
    for index, track in enumerate(tracks, start=1):
        track_obs = obs_by_track.get(track.track_id, [])
        file_record = file_by_track.get(track.track_id)
        prediction = model.predict(track, track_obs, file_record, taxonomy, ucache=ucache)
        _apply_prediction_to_track(
            track,
            taxonomy,
            prediction,
            has_external_evidence=external_evidence_available(track_obs),
        )
        progress.update(index, track.title_canonical, category=track.dj_taxonomy_id)
    progress.finish()
    store.save_tracks(tracks)
    return len(tracks)


def load_dj_taxonomy_model_if_available(
    model_dir: str | Path | None,
    *,
    taxonomy_path: str | None = None,
):
    """Load a pickled XGB DJ taxonomy model if present and compatible.

    Accepts either the base model dir (containing ``xgb/``) or the
    ``xgb/`` dir directly. Returns ``None`` when nothing usable is found.
    """
    if not model_dir:
        return None
    from .dj_model_xgb import load_xgb_model

    candidates: list[Path] = []
    base = Path(model_dir)
    candidates.append(base / "xgb")
    candidates.append(base)
    for cand in candidates:
        if (cand / MODEL_FILENAME).exists():
            model = load_xgb_model(cand, taxonomy_path=taxonomy_path)
            if model is not None:
                return model
    return None


def _apply_prediction_to_track(
    track: LogicalTrack,
    taxonomy: DjTaxonomy,
    prediction: DjCategoryPrediction,
    *,
    has_external_evidence: bool,
) -> None:
    """Write a single XGB prediction to BOTH internal and external track columns.

    With only one model, the two column sets are populated identically. We
    keep the dual columns so the existing CSV schema, tag-writer, and
    downstream consumers don't break.
    """
    source_model = "xgb"
    if prediction.category_id and prediction.confidence < UNCLASSIFIED_THRESHOLD:
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
        track.dj_taxonomy_confidence = prediction.confidence
        track.dj_taxonomy_confidence_level = "unknown"
        track.dj_taxonomy_source_model = source_model
    elif prediction.category_id:
        category = taxonomy.category(prediction.category_id)
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
        track.dj_taxonomy_confidence = prediction.confidence
        track.dj_taxonomy_confidence_level = confidence_to_level(prediction.confidence)
        track.dj_taxonomy_source_model = source_model
    track.dj_taxonomy_internal_id = prediction.category_id
    track.dj_taxonomy_internal_label = prediction.category_label
    track.dj_taxonomy_internal_confidence = prediction.confidence
    track.dj_taxonomy_external_id = prediction.category_id
    track.dj_taxonomy_external_label = prediction.category_label
    track.dj_taxonomy_external_confidence = prediction.confidence
    track.dj_taxonomy_models_agree = "YES" if prediction.category_id else "NO"
    track.dj_taxonomy_external_evidence_available = "YES" if has_external_evidence else "NO"
    track.dj_taxonomy_alternatives = json.dumps(
        {"internal": prediction.alternatives, "external": prediction.alternatives},
        sort_keys=True,
    )
    track.dj_taxonomy_evidence = json.dumps(
        {
            "internal": prediction.evidence,
            "external": prediction.evidence,
            "warnings": {"internal": prediction.warnings, "external": prediction.warnings},
        },
        sort_keys=True,
    )
    track.dj_taxonomy_version = taxonomy.version


def _resolve_cache() -> Any | None:
    """Resolve the UniversalCache singleton; return None if unavailable.

    Used by training and inference to feed dense audio features (DSP scalars,
    CLAP, vocal_stem) when they're cached. Returns None silently so callers
    without a cache fall back to today's behavior — no error path is needed.
    """
    try:
        from dj_tagger.universal_cache import get_cache
        return get_cache()
    except Exception:
        logger.debug("Universal cache unavailable; dense audio features will be skipped.")
        return None


def _load_label_rows(labels_path: str, taxonomy: DjTaxonomy) -> list[dict[str, str]]:
    rows, _diagnostics = _load_label_rows_with_diagnostics(labels_path, taxonomy)
    return rows


def _load_label_rows_with_diagnostics(
    labels_path: str,
    taxonomy: DjTaxonomy,
) -> tuple[list[dict[str, str]], LabelLoadDiagnostics]:
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    diagnostics = LabelLoadDiagnostics(total_rows=len(rows))
    valid_rows: list[dict[str, str]] = []
    invalid: list[str] = []
    for index, row in enumerate(rows, start=2):
        category_id = (row.get("category_id") or row.get("dj_taxonomy_id") or "").strip()
        label_source = (row.get("label_source") or "").strip().lower()
        if label_source.endswith("_error"):
            invalid.append(f"row {index}: error row {category_id or '<blank>'}")
            diagnostics.error_rows += 1
            continue
        resolved_category_id = taxonomy.resolve_category_id(category_id)
        if resolved_category_id != category_id and taxonomy.validate_category_id(resolved_category_id):
            row = dict(row)
            diagnostics.alias_counts[category_id] = diagnostics.alias_counts.get(category_id, 0) + 1
            diagnostics.alias_targets[category_id] = resolved_category_id
            category_id = resolved_category_id
        if not taxonomy.validate_category_id(category_id):
            invalid.append(f"row {index}: unknown category_id {category_id or '<blank>'}")
            diagnostics.invalid_rows += 1
            key = category_id or "<blank>"
            diagnostics.invalid_category_counts[key] = diagnostics.invalid_category_counts.get(key, 0) + 1
            continue
        row["category_id"] = category_id
        valid_rows.append(row)
    diagnostics.valid_rows = len(valid_rows)
    if invalid and not valid_rows:
        raise ValueError("No valid DJ taxonomy labels found: " + "; ".join(invalid[:5]))
    return valid_rows, diagnostics


def _build_training_examples(
    labels: list[dict[str, str]],
    store: CsvStore,
    *,
    mode: str,
    show_progress: bool = False,
    ucache: Any | None = None,
    min_examples_per_category: int = MIN_EXAMPLES_PER_CATEGORY,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Build training examples from the registry + labels CSV.

    Returns ``(examples, skipped, dropped_categories)``:
      - ``examples``: feature-bearing rows ready for vectorization.
      - ``skipped``: count of label rows that yielded no usable features.
      - ``dropped_categories``: ``{category_id: count}`` for classes with
        fewer than ``min_examples_per_category`` rows — surfaced to warnings
        so the operator can target them for data growth.
    """
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
            ucache=ucache,
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

    # Spec §9 — drop categories with <N examples. They overfit instead of learn,
    # and they consume train/val budget that the long-tail can't afford.
    # Skipped when the dataset is below 8 rows total — smoke-test fixtures
    # must still train successfully even though every category has 1 example.
    dropped: dict[str, int] = {}
    if min_examples_per_category > 1 and len(examples) >= 8:
        from collections import Counter

        counts = Counter(example["category_id"] for example in examples)
        keep = {cid for cid, n in counts.items() if n >= min_examples_per_category}
        if len(keep) < len(counts) and len(keep) >= 2:
            dropped = {cid: n for cid, n in counts.items() if cid not in keep}
            examples = [e for e in examples if e["category_id"] in keep]
            logger.info(
                "Dropped %d categories with <%d examples: %s",
                len(dropped), min_examples_per_category, sorted(dropped),
            )

    return examples, skipped, dropped


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


def _confidence_from_model_score(score: float, margin: float, features: dict[str, float]) -> float:
    # Richness scales with the *number of distinct feature groups* present (not
    # the raw key count). Counting groups makes the bonus invariant when wide
    # dense blocks (CLAP, DSP) are added — adding 512 CLAP dims shouldn't
    # trivially saturate confidence.
    group_count = len({key.split(":", 1)[0] for key in features})
    richness = min(0.12, group_count / 30.0)
    raw = 0.18 + 0.64 * max(0.0, min(score, 1.0)) + 0.18 * max(0.0, min(margin, 1.0)) + richness
    return round(max(0.05, min(0.98, raw)), 3)


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train/validation with stratification + same-artist leakage guard.

    Priority order:
      1. ``GroupShuffleSplit`` by ``track.artist_canonical`` — prevents the
         classifier from being graded on tracks by the same artist it trained
         on (the identity-token features make that a real leak).
      2. Plain stratified ``train_test_split`` — keeps every class in both
         splits when group-aware splitting can't satisfy the class coverage
         (e.g., a class has only one artist).
      3. Random shuffle — fall-through when even stratification is infeasible
         (any class has <2 examples).
    """
    from collections import Counter

    ys = [example["category_id"] for example in examples]
    class_counts = Counter(ys)

    # Fall-through: any class with <2 examples means we can't stratify cleanly.
    if not class_counts or min(class_counts.values()) < 2:
        shuffled = list(examples)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * validation_split)))
        n_val = min(n_val, len(shuffled) - 1)
        return shuffled[n_val:], shuffled[:n_val]

    # Try group-aware split by artist when most rows have an artist.
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    artists = [(example["track"].artist_canonical or "").strip() for example in examples]
    if sum(1 for a in artists if a) >= int(len(artists) * 0.6):
        groups = [
            artist or (example["track"].track_id or example.get("file_name") or f"_row{i}")
            for i, (artist, example) in enumerate(zip(artists, examples))
        ]
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=seed)
            train_idx, val_idx = next(splitter.split(examples, ys, groups=groups))
            # GroupShuffleSplit is not stratified; verify class coverage and
            # fall through if any class is entirely missing or represented by
            # only one row in training. Singleton train classes disable the
            # stratified CV used by XGB tuning, which is worse than the small
            # artist-leakage risk on this long-tail dataset.
            train_classes = {ys[i] for i in train_idx}
            train_counts = Counter(ys[i] for i in train_idx)
            if (
                set(ys).issubset(train_classes)
                and len(val_idx) > 0
                and min(train_counts.values()) >= 2
            ):
                return [examples[i] for i in train_idx], [examples[i] for i in val_idx]
        except ValueError:
            pass

    # Stratified split — every class represented proportionally in both.
    try:
        train_examples, val_examples = train_test_split(
            examples, test_size=validation_split, stratify=ys, random_state=seed
        )
        return train_examples, val_examples
    except ValueError:
        shuffled = list(examples)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * validation_split)))
        n_val = min(n_val, len(shuffled) - 1)
        return shuffled[n_val:], shuffled[:n_val]


# ── Spec §9.2 / §9.3 — Library-distribution check + anti-collapse guard ─────


class CollapseGuardError(RuntimeError):
    """Raised by train_with_collapse_guard when the largest predicted bucket
    persistently exceeds the 20% cap after the maximum retrain attempts."""


def run_library_distribution_check(
    store: CsvStore,
    *,
    model_dir: str | Path,
    taxonomy_path: str | None = None,
    max_bucket_share: float = 0.20,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Predict on every track and report bucket-share distribution."""
    taxonomy = load_dj_taxonomy(taxonomy_path)
    model = load_dj_taxonomy_model_if_available(model_dir, taxonomy_path=taxonomy_path)
    if model is None:
        raise FileNotFoundError(f"No DJ taxonomy model found under {model_dir}")

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    ucache = _resolve_cache()
    counts: dict[str, int] = {}
    progress = ProgressBar(len(tracks), label="Library distribution", enabled=show_progress)
    for index, track in enumerate(tracks, start=1):
        track_obs = obs_by_track.get(track.track_id, [])
        file_record = file_by_track.get(track.track_id)
        prediction = model.predict(track, track_obs, file_record, taxonomy, ucache=ucache)
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
    bucket exceeds the cap, retrains up to max_retrains more times with a
    re-seeded RandomizedSearchCV.
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


def _write_xgb_training_report(path: Path, stats: DjTrainingStats, model: Any) -> None:
    """Write a training report for the XGB model."""
    payload = _stats_to_dict(stats)
    payload.update(
        {
            "model_version": stats.metrics.get("model_version", DJ_MODEL_VERSION),
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
    model: Any,
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
    *,
    ucache: Any | None = None,
) -> dict[str, Any]:
    if ucache is None:
        ucache = _resolve_cache()
    rows = []
    for example in examples:
        prediction = _predict_for_example(model, example, taxonomy, ucache=ucache)
        top3 = [prediction.category_id] + [
            alt["category_id"] for alt in prediction.alternatives[:2]
        ]
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
    model: Any,
    example: dict[str, Any] | None,
    taxonomy: DjTaxonomy,
    *,
    ucache: Any | None = None,
) -> DjCategoryPrediction:
    if not example:
        return DjCategoryPrediction(feature_mode=getattr(model, "feature_mode", "external"))
    return model.predict(
        example["track"],
        example.get("observations", []),
        example.get("file_record"),
        taxonomy,
        ucache=ucache,
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
        "per_class": _per_class_classification_report(y_true, y_pred),
    }


def _per_class_classification_report(
    y_true: list[str], y_pred: list[str]
) -> dict[str, dict[str, float]]:
    """Full precision/recall/F1 per category from sklearn.classification_report."""
    if not y_true:
        return {}
    from sklearn.metrics import classification_report

    labels = sorted({label for label in y_true + y_pred if label})
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        label: {
            "precision": round(float(stats.get("precision", 0.0)), 3),
            "recall": round(float(stats.get("recall", 0.0)), 3),
            "f1": round(float(stats.get("f1-score", 0.0)), 3),
            "support": int(stats.get("support", 0)),
        }
        for label, stats in report.items()
        if isinstance(stats, dict) and label not in {"accuracy", "macro avg", "weighted avg"}
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
