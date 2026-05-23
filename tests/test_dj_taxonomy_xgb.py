"""Spec §11 — tests for the XGBoost classifier path."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.taxonomy.dj_model import train_dj_taxonomy_unified
from dj_registry.taxonomy.dj_model_xgb import (
    XGB_MODEL_VERSION,
    XGBTaxonomyModel,
    load_xgb_model,
    train_xgb_model,
)
from dj_registry.taxonomy.dj_schema import load_dj_taxonomy
from dj_registry.taxonomy.features import build_track_features


_CATEGORIES = [
    "dark_tech_house_driver",
    "vocal_hook_tech_house",
    "raw_warehouse_techno",
    "organic_chant_house",
]


def _store(tmp_path: Path) -> CsvStore:
    store = CsvStore(str(tmp_path / "registry"))
    # Two tracks per category for stratified-CV to work and to give XGB enough
    # signal to actually fit.
    tracks: list[LogicalTrack] = []
    files: list[FileRecord] = []
    obs: list[SourceObservation] = []
    counter = 0
    for cat_idx, label in enumerate(_CATEGORIES):
        for replica in range(2):
            counter += 1
            tid = f"T{counter}"
            tracks.append(LogicalTrack(
                track_id=tid,
                artist_canonical=f"A{counter}",
                title_canonical=f"Track {counter}",
                tagger_energy=f"E{3 + cat_idx % 3}",
                tagger_vibe=["dark", "warm", "warehouse", "tribal"][cat_idx],
                tagger_structure="16H",
                tagger_bpm="124",
            ))
            files.append(FileRecord(
                file_id=f"F{counter}", track_id=tid, is_primary_file=True,
                file_name=f"track_{counter}.mp3",
                embedded_genre=["Tech House", "Tech House", "Techno", "Afro House"][cat_idx],
            ))
            obs.append(SourceObservation(
                track_id=tid, source_system="songstats",
                genre=["Tech House", "Tech House", "Techno", "Afro House"][cat_idx],
                genres_all=["Tech House;House", "Tech House;House", "Techno;Peak Time Techno", "Afro House;Organic House"][cat_idx],
                danceability="0.80", energy="0.75", valence="0.30",
            ))
    store.save_tracks(tracks)
    store.save_files(files)
    store.save_observations(obs)
    return store


def _labels_csv(tmp_path: Path) -> Path:
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "file_name", "category_id"])
        writer.writeheader()
        counter = 0
        for cat_idx, label in enumerate(_CATEGORIES):
            for _replica in range(2):
                counter += 1
                writer.writerow({"track_id": f"T{counter}", "file_name": f"track_{counter}.mp3", "category_id": label})
    return labels


def _build_examples(store: CsvStore, taxonomy):
    """Mirror of _build_training_examples but for tests — uses external mode."""
    from dj_registry.taxonomy.dj_model import _build_training_examples, _load_label_rows
    # Helpers expect a labels file; use a tiny in-memory equivalent
    labels_rows = [
        {"track_id": f"T{i+1}", "file_name": f"track_{i+1}.mp3", "category_id": _CATEGORIES[i // 2]}
        for i in range(2 * len(_CATEGORIES))
    ]
    # Insert directly into _build_training_examples by faking the loader
    examples, _ = _build_training_examples(labels_rows, store, mode="external", show_progress=False)
    return examples


# ── Core training ──────────────────────────────────────────────────────────


def test_xgb_model_trains_on_synthetic_examples(tmp_path):
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    taxonomy = load_dj_taxonomy()

    examples = _build_examples(store, taxonomy)
    assert len(examples) == 8

    # n_iter=2 to keep the test fast
    model, metrics, warnings = train_xgb_model(
        examples, taxonomy,
        labels_path=str(labels),
        validation_split=0.25,
        seed=42,
        n_iter=2,
    )
    assert isinstance(model, XGBTaxonomyModel)
    assert model.feature_mode == "external"
    assert metrics["model_type"] == "xgb"
    assert "top1_accuracy" in metrics
    assert "fit_time_seconds" in metrics


def test_xgb_predict_returns_valid_category(tmp_path):
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    taxonomy = load_dj_taxonomy()

    examples = _build_examples(store, taxonomy)
    model, _, _ = train_xgb_model(
        examples, taxonomy,
        labels_path=str(labels),
        validation_split=0.25,
        seed=42,
        n_iter=2,
    )

    # Pick the first track and predict
    tracks = store.load_tracks()
    files = store.load_files()
    file_by_track = {f.track_id: f for f in files}
    track = tracks[0]
    prediction = model.predict(track, [], file_by_track.get(track.track_id), taxonomy)
    assert prediction.category_id  # something was assigned
    assert taxonomy.validate_category_id(prediction.category_id)
    assert prediction.feature_mode == "external"
    assert XGB_MODEL_VERSION in str(prediction.evidence)


# ── Anti-collapse rule (shared with LR) ────────────────────────────────────


def test_xgb_predict_filters_afro_tribal_when_ineligible(tmp_path):
    """A track without explicit afro/tribal provider signal must NOT get an afro/tribal prediction."""
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    taxonomy = load_dj_taxonomy()
    examples = _build_examples(store, taxonomy)
    model, _, _ = train_xgb_model(
        examples, taxonomy, labels_path=str(labels),
        validation_split=0.25, seed=42, n_iter=2,
    )

    # Hand-build a percussion-heavy track with no afro/tribal genre signal
    track = LogicalTrack(
        track_id="X1",
        artist_canonical="Unknown",
        title_canonical="Percussive Roller",
        mix_canonical="Original Mix",
        canonical_bpm="124",
        tagger_vibe="TRIB,HYPN",
        tagger_vocal="CHANT",
        tagger_energy="E4",
        tagger_structure="ROLLING",
    )
    obs = [SourceObservation(
        track_id="X1", source_system="songstats",
        genre="Tech House",  # NOT afro/tribal
        genres_all="Tech House;Indie Dance",
    )]
    fr = FileRecord(file_name="x1.mp3", embedded_genre="Tech House")
    prediction = model.predict(track, obs, fr, taxonomy)

    # The §6.2 anti-collapse rule must zero out afro/tribal candidates
    afro_tribal_ids = {
        "tribal_afro_driver", "afro_tribal_warmup", "afro_tribal_builder",
        "afro_house_peak", "spiritual_afro_chant", "afro_cinematic_builder",
        "afro_3_step", "afro_tech_driver", "deep_afro_house",
        "organic_chant_house", "tribal_house",
    }
    assert prediction.category_id not in afro_tribal_ids, (
        f"Predicted {prediction.category_id!r} despite no afro/tribal signal"
    )


# ── Label-encoder roundtrip ────────────────────────────────────────────────


def test_xgb_label_encoder_roundtrip(tmp_path):
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    taxonomy = load_dj_taxonomy()
    examples = _build_examples(store, taxonomy)
    model, _, _ = train_xgb_model(
        examples, taxonomy, labels_path=str(labels),
        validation_split=0.25, seed=42, n_iter=2,
    )

    save_dir = tmp_path / "xgb_model"
    model.save(save_dir)
    loaded = load_xgb_model(save_dir)
    assert loaded is not None
    assert loaded.feature_mode == model.feature_mode
    assert set(loaded.label_encoder.classes_) == set(model.label_encoder.classes_)

    # Predict via loaded model — same shape, valid category
    tracks = store.load_tracks()
    files = store.load_files()
    file_by_track = {f.track_id: f for f in files}
    track = tracks[0]
    prediction = loaded.predict(track, [], file_by_track.get(track.track_id), taxonomy)
    assert taxonomy.validate_category_id(prediction.category_id)


# ── train_dj_taxonomy_unified dispatch ────────────────────────────────────


def test_train_dj_taxonomy_unified_lr_only(tmp_path):
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    result = train_dj_taxonomy_unified(
        store, str(labels),
        model_type="lr",
        model_dir=str(tmp_path / "dj_model"),
        validation_split=0.25,
        seed=42,
    )
    assert "lr" in result
    assert "xgb" not in result
    assert (tmp_path / "dj_model" / "lr" / "model.pkl").exists()
    assert (tmp_path / "dj_model" / "lr" / "training_report.json").exists()


def test_train_dj_taxonomy_unified_both(tmp_path):
    store = _store(tmp_path)
    labels = _labels_csv(tmp_path)
    result = train_dj_taxonomy_unified(
        store, str(labels),
        model_type="both",
        model_dir=str(tmp_path / "dj_model"),
        validation_split=0.25,
        seed=42,
        xgb_n_iter=2,
    )
    assert "lr" in result
    assert "xgb" in result
    assert (tmp_path / "dj_model" / "lr" / "model.pkl").exists()
    assert (tmp_path / "dj_model" / "xgb" / "model.pkl").exists()
    assert (tmp_path / "dj_model" / "comparison.json").exists()

    with (tmp_path / "dj_model" / "comparison.json").open("r", encoding="utf-8") as f:
        comparison = json.load(f)
    assert "lr" in comparison
    assert "xgb" in comparison
    assert "deltas" in comparison
    assert "gates_pass_lr" in comparison
    assert "gates_pass_xgb" in comparison
