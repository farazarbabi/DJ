"""XGBoost classifier with hyperparameter tuning for DJ taxonomy.

Parallel to dj_model.py's LogisticRegression pipeline. Shares the same
feature extraction (features.py:build_track_features in external mode),
the same anti-collapse rule (features.is_eligible_for_afro_tribal), and
the same DjCategoryPrediction output shape — so downstream consumers
(tag writing, playlists, reports) work unchanged.

Tuning: scikit-learn RandomizedSearchCV with stratified K-fold CV.
n_iter and random_state are configurable; default n_iter=30, seed=42.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from .dj_model import (
    DJ_FEATURE_SCHEMA_VERSION,
    DJ_MODEL_VERSION,
    DjCategoryPrediction,
    UNCLASSIFIED_ID,
    UNCLASSIFIED_THRESHOLD,
    _confidence_from_model_score,
    confidence_to_level,
)
from .dj_schema import DjTaxonomy
from .features import (
    build_track_features,
    is_afro_tribal_family,
    is_eligible_for_afro_tribal,
    load_artist_priors,
    load_label_priors,
    summarize_feature_groups,
)

logger = logging.getLogger(__name__)

XGB_MODEL_VERSION = "dj-taxonomy-xgb-v1"
MODEL_FILENAME = "model.pkl"


@dataclass
class XGBTaxonomyModel:
    """XGBoost classifier wrapped to match the DjTaxonomyModel.predict() contract."""

    vectorizer: DictVectorizer
    label_encoder: LabelEncoder
    classifier: XGBClassifier
    feature_mode: str
    taxonomy_version: str
    taxonomy_hash: str
    feature_schema_version: str
    trained_at: str
    labels_hash: str
    examples: int
    best_params: dict[str, Any] = field(default_factory=dict)

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
                warnings=["No usable features for XGB taxonomy model."],
            )

        x = self.vectorizer.transform([features])
        probabilities = self._probability_map(x)
        ranked = [
            (cat_id, probability)
            for cat_id, probability in probabilities.items()
            if taxonomy.validate_category_id(cat_id)
        ]
        # Spec §6.2 — anti-collapse rule (same as LR predict)
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
                warnings=["XGB model produced no valid DJ taxonomy category."],
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
            warnings.append("Low-confidence DJ taxonomy inferred by trained XGB model.")
        if margin < 0.08 and len(ranked) > 1:
            warnings.append("Low DJ taxonomy XGB model margin between top categories.")

        return DjCategoryPrediction(
            category_id=selected_id,
            category_label=category.label,
            confidence=confidence,
            alternatives=alternatives,
            evidence={
                "model_signals": [
                    f"model_version={XGB_MODEL_VERSION}",
                    f"feature_mode={self.feature_mode}",
                    f"category_score={selected_score:.3f}",
                    f"category_margin={margin:.3f}",
                ],
                "feature_groups": summarize_feature_groups(features),
                "top_categories": [
                    f"{taxonomy.category(cat_id).label} ({score:.3f})"
                    for cat_id, score in ranked[:3]
                ],
            },
            warnings=warnings,
            feature_mode=self.feature_mode,
        )

    def _probability_map(self, x: Any) -> dict[str, float]:
        """Map XGB integer-class probabilities back to taxonomy IDs."""
        probabilities = self.classifier.predict_proba(x)[0]
        # XGBClassifier.classes_ holds the integer-encoded labels in the
        # order matching predict_proba's columns. label_encoder.inverse_transform
        # converts those ints back to the original category_id strings.
        int_classes = self.classifier.classes_
        str_classes = self.label_encoder.inverse_transform(int_classes)
        return {str(label): float(prob) for label, prob in zip(str_classes, probabilities)}

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        artifact_path = path / MODEL_FILENAME
        with artifact_path.open("wb") as f:
            pickle.dump(self, f)
        return artifact_path


def _xgb_search_space() -> dict[str, Any]:
    """RandomizedSearchCV parameter distributions per spec §3 (XGBoost search)."""
    return {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),  # uniform(loc, scale) → [0.6, 1.0]
        "colsample_bytree": uniform(0.5, 0.5),  # [0.5, 1.0]
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 10.0],
    }


def train_xgb_model(
    examples: list[dict[str, Any]],
    taxonomy: DjTaxonomy,
    *,
    labels_path: str,
    validation_split: float = 0.2,
    seed: int = 42,
    n_iter: int = 30,
    feature_mode: str = "external",
    show_progress: bool = False,
) -> tuple[XGBTaxonomyModel, dict[str, Any], list[str]]:
    """Train an XGBoost classifier with RandomizedSearchCV tuning.

    Returns (model, metrics, warnings). metrics matches the LR pipeline's shape
    so dj_model_comparison can consume both side-by-side.
    """
    from .dj_model import _evaluate_examples, _file_hash, _split_examples

    warnings: list[str] = []

    if len(examples) < 8 or validation_split <= 0:
        train_examples = examples
        validation_examples = examples
        validation_kind = "in_sample"
        warnings.append(
            "Dataset is small; XGB metrics are in-sample and should be treated as smoke-test metrics."
        )
    else:
        train_examples, validation_examples = _split_examples(
            examples, validation_split=validation_split, seed=seed
        )
        validation_kind = "holdout"
        if not validation_examples:
            validation_examples = train_examples
            validation_kind = "in_sample"
            warnings.append("Validation split produced no holdout rows; metrics are in-sample.")

    fit_start = time.perf_counter()
    # Build features matrix
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([example["features"] for example in train_examples])
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform([example["category_id"] for example in train_examples])

    # Compute sample weights (class_weight='balanced' equivalent for XGB)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    n_classes = len(label_encoder.classes_)
    base_clf = XGBClassifier(
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method="hist",
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )

    best_params: dict[str, Any] = {}
    # Avoid CV when there are too few examples per class for stratified splitting
    min_per_class = int(min(np.bincount(y_train))) if n_classes > 1 else len(y_train)
    cv_folds = min(3, min_per_class) if min_per_class >= 2 else 0

    if cv_folds >= 2 and len(train_examples) >= 8 and n_iter > 0:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            base_clf,
            param_distributions=_xgb_search_space(),
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        # Pass sample_weight via fit_params
        search.fit(x_train, y_train, sample_weight=sample_weight)
        classifier = search.best_estimator_
        best_params = {
            k: (float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in search.best_params_.items()
        }
        logger.info("XGB best params: %s (best CV f1_macro=%.3f)", best_params, search.best_score_)
    else:
        # Tuning skipped — train a single sensible default
        warnings.append(
            f"XGB hyperparameter tuning skipped (n_iter={n_iter}, cv_folds={cv_folds}, "
            f"min_examples_per_class={min_per_class}); using default params."
        )
        classifier = base_clf
        classifier.fit(x_train, y_train, sample_weight=sample_weight)
        best_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3}

    fit_time = time.perf_counter() - fit_start

    model = XGBTaxonomyModel(
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        classifier=classifier,
        feature_mode=feature_mode,
        taxonomy_version=taxonomy.version,
        taxonomy_hash=taxonomy.hash(),
        feature_schema_version=DJ_FEATURE_SCHEMA_VERSION,
        trained_at=now_iso(),
        labels_hash=_file_hash(Path(labels_path)),
        examples=len(examples),
        best_params=best_params,
    )

    # Evaluate on the validation set (or in-sample if dataset too small)
    metrics = _evaluate_examples(model, validation_examples, taxonomy)
    metrics["validation_kind"] = validation_kind
    metrics["validation_examples"] = len(validation_examples)
    metrics["fit_time_seconds"] = round(fit_time, 2)
    metrics["best_params"] = best_params
    metrics["model_version"] = XGB_MODEL_VERSION
    metrics["model_type"] = "xgb"

    # Refit on ALL examples for the final shipped model
    x_all = vectorizer.fit_transform([example["features"] for example in examples])
    y_all = label_encoder.fit_transform([example["category_id"] for example in examples])
    final_sample_weight = compute_sample_weight(class_weight="balanced", y=y_all)
    classifier.fit(x_all, y_all, sample_weight=final_sample_weight)
    model.vectorizer = vectorizer
    model.label_encoder = label_encoder
    model.classifier = classifier

    return model, metrics, warnings


def load_xgb_model(
    model_dir: str | Path,
    *,
    taxonomy_path: str | None = None,
) -> XGBTaxonomyModel | None:
    """Load a pickled XGB model; return None if missing or incompatible."""
    from .dj_schema import load_dj_taxonomy

    artifact_path = Path(model_dir) / MODEL_FILENAME
    if not artifact_path.exists():
        return None
    try:
        with artifact_path.open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, XGBTaxonomyModel):
            return None
        if model.feature_schema_version != DJ_FEATURE_SCHEMA_VERSION:
            logger.info(
                "XGB model at %s has stale feature schema (%s vs %s); skipping until retrain.",
                artifact_path,
                model.feature_schema_version,
                DJ_FEATURE_SCHEMA_VERSION,
            )
            return None
        taxonomy = load_dj_taxonomy(taxonomy_path)
        if model.taxonomy_hash and model.taxonomy_hash != taxonomy.hash():
            logger.info(
                "XGB model at %s was trained against a different dj_taxonomy.json; skipping.",
                artifact_path,
            )
            return None
        return model
    except (pickle.UnpicklingError, OSError, ValueError) as exc:
        logger.info("XGB model at %s incompatible (%s); skipping.", artifact_path, exc)
        return None
