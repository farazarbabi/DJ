import csv
import json
from argparse import Namespace

from dj_registry.cli import cmd_dj_taxonomy
from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.sync.export import generate_overview
from dj_registry.taxonomy.dj_ground_truth import (
    generate_dj_ground_truth_csv,
    test_dj_api_connection as run_dj_api_connection_test,
)
from dj_registry.taxonomy.dj_model import (
    classify_all_dj_taxonomies,
    evaluate_dj_taxonomy_models,
    load_dj_taxonomy_model_if_available,
    train_dj_taxonomy_models,
)
from dj_registry.taxonomy.dj_schema import compact_category_label, load_dj_taxonomy
from dj_registry.taxonomy.features import build_track_features


def test_dj_taxonomy_schema_loads_flat_categories():
    taxonomy = load_dj_taxonomy()

    assert taxonomy.validate_category_id("dark_tech_house_driver")
    assert taxonomy.category("dark_tech_house_driver").label == "Dark Tech-House Driver"
    assert taxonomy.category("dark_tech_house_driver").moods
    assert taxonomy.category("dark_tech_house_driver").grooves


def test_compact_category_label_uses_dot_separated_word_codes():
    assert compact_category_label("Dark Tech-House Driver") == "DRK.TECH.HOUS.DRV"
    assert compact_category_label("Organic Chant House") == "ORG.CHNT.HOUS"
    assert compact_category_label("Lo-Fi Deep House") == "LOFI.DEEP.HOUS"


def test_internal_feature_mode_excludes_provider_evidence():
    observations = [
        SourceObservation(
            track_id="T1",
            source_system="songstats",
            genre="Melodic House",
            genres_all="Melodic House;Indie Dance",
            danceability="0.82",
        )
    ]

    internal = build_track_features(
        LogicalTrack(track_id="T1", title_canonical="Dark Driver", tagger_energy="E4", tagger_vibe="dark"),
        observations,
        FileRecord(track_id="T1", file_name="Dark Driver.mp3", embedded_genre="Tech House"),
        feature_mode="internal",
    )
    external = build_track_features(
        LogicalTrack(track_id="T1", title_canonical="Dark Driver", tagger_energy="E4", tagger_vibe="dark"),
        observations,
        FileRecord(track_id="T1", file_name="Dark Driver.mp3", embedded_genre="Tech House"),
        feature_mode="external",
    )

    assert not any("provider:" in key for key in internal)
    assert not any("songstats:danceability" in key for key in internal)
    assert any("provider:" in key for key in external)
    assert any("songstats:danceability" in key for key in external)


def test_train_evaluate_and_classify_xgb_dj_taxonomy_model(tmp_path):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)

    result = train_dj_taxonomy_models(
        store,
        str(labels_path),
        model_dir=str(tmp_path / "dj_model"),
        validation_split=0,  # 4 rows × 4 classes — skip stratified holdout
        xgb_n_iter=0,
    )
    metrics = evaluate_dj_taxonomy_models(
        store,
        str(labels_path),
        model_dir=str(tmp_path / "dj_model"),
    )
    count = classify_all_dj_taxonomies(store, model_dir=str(tmp_path / "dj_model"))

    model = load_dj_taxonomy_model_if_available(tmp_path / "dj_model")
    tracks = {track.track_id: track for track in store.load_tracks()}

    assert result["xgb"]["examples"] == 4
    assert metrics["examples"] == 4
    assert model is not None
    assert model.feature_mode == "external"
    assert count == 4
    assert tracks["T1"].dj_taxonomy_internal_id
    assert tracks["T1"].dj_taxonomy_external_id
    # Both columns are now populated identically by the single XGB prediction.
    assert tracks["T1"].dj_taxonomy_internal_id == tracks["T1"].dj_taxonomy_external_id
    assert tracks["T1"].dj_taxonomy_confidence > 0
    assert tracks["T1"].dj_taxonomy_source_model == "xgb"
    assert json.loads(tracks["T1"].dj_taxonomy_evidence)["internal"]


def test_dj_taxonomy_overview_exports_dual_columns_from_xgb(tmp_path):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)
    train_dj_taxonomy_models(
        store,
        str(labels_path),
        model_dir=str(tmp_path / "dj_model"),
        validation_split=0,
        xgb_n_iter=0,
    )
    classify_all_dj_taxonomies(store, model_dir=str(tmp_path / "dj_model"))

    overview = generate_overview(store, str(tmp_path / "registry"))

    with open(overview, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["dj_taxonomy_id"]
    assert rows[0]["dj_taxonomy_internal_confidence"]
    assert rows[0]["dj_taxonomy_external_confidence"]
    # Both columns are populated from the single XGB prediction.
    assert rows[0]["dj_taxonomy_internal_id"] == rows[0]["dj_taxonomy_external_id"]


def test_generate_dj_ground_truth_csv_with_mocked_client(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "Dark Driver.mp3").write_bytes(b"")
    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([LogicalTrack(track_id="T1", title_canonical="Dark Driver", tagger_energy="E4", tagger_vibe="dark")])
    store.save_files([FileRecord(file_id="F1", track_id="T1", is_primary_file=True, file_name="Dark Driver.mp3")])
    store.save_observations([])

    class FakeClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "category_id": "dark_tech_house_driver",
                "confidence": 0.86,
                "alternate_category_ids": ["driving_dark_indie_tech"],
                "rationale": "Dark E4 rolling club evidence.",
                "warnings": "",
            }

    out = tmp_path / "dj_taxonomy_ground_truth.csv"
    stats = generate_dj_ground_truth_csv(
        store,
        files_dir=str(files_dir),
        output_path=str(out),
        client=FakeClient(),
        collect=False,
    )

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert stats.generated == 1
    assert rows[0]["category_id"] == "dark_tech_house_driver"
    assert rows[0]["category_label"] == "Dark Tech-House Driver"
    assert rows[0]["moods"]
    assert rows[0]["alternate_category_ids"] == "driving_dark_indie_tech"


def test_generate_dj_ground_truth_default_out_is_registry_output(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "Dark Driver.mp3").write_bytes(b"")
    store = CsvStore(str(tmp_path / "registry"))

    class FakeClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "category_id": "dark_tech_house_driver",
                "confidence": 0.86,
                "alternate_category_ids": [],
                "rationale": "Dark E4 rolling club evidence.",
                "warnings": "",
            }

    stats = generate_dj_ground_truth_csv(
        store,
        files_dir=str(files_dir),
        client=FakeClient(),
        collect=False,
    )

    assert stats.output_path == str(tmp_path / "dj_taxonomy_ground_truth.csv")
    assert (tmp_path / "dj_taxonomy_ground_truth.csv").exists()


def test_generate_dj_ground_truth_legacy_files_out_redirects_to_registry_output(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "Dark Driver.mp3").write_bytes(b"")
    store = CsvStore(str(tmp_path / "registry"))

    class FakeClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "category_id": "dark_tech_house_driver",
                "confidence": 0.86,
                "alternate_category_ids": [],
                "rationale": "Dark E4 rolling club evidence.",
                "warnings": "",
            }

    stats = generate_dj_ground_truth_csv(
        store,
        files_dir=str(files_dir),
        output_path="files/dj_taxonomy_ground_truth.csv",
        client=FakeClient(),
        collect=False,
    )

    assert stats.output_path == str(tmp_path / "dj_taxonomy_ground_truth.csv")
    assert (tmp_path / "dj_taxonomy_ground_truth.csv").exists()


def test_dj_taxonomy_cli_train_models_runs_xgb(tmp_path, capsys):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)

    rc = cmd_dj_taxonomy(
        Namespace(
            dj_taxonomy_command="train-models",
            output=str(tmp_path / "registry"),
            labels=str(labels_path),
            taxonomy=None,
            model_dir=str(tmp_path / "dj_model"),
            validation_split=0,
            seed=42,
            xgb_n_iter=0,
            quiet=False,
            no_progress=True,
        )
    )

    assert rc == 0
    assert (tmp_path / "dj_model" / "xgb" / "model.pkl").exists()
    assert (tmp_path / "dj_model" / "xgb" / "training_report.json").exists()
    output = capsys.readouterr().out
    assert "DJ Taxonomy Model (XGB)" in output
    assert "top-1 accuracy" in output


def test_dj_taxonomy_cli_evaluate_prints_summary(tmp_path, capsys):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)
    cmd_dj_taxonomy(
        Namespace(
            dj_taxonomy_command="train-models",
            output=str(tmp_path / "registry"),
            labels=str(labels_path),
            taxonomy=None,
            model_dir=str(tmp_path / "dj_model"),
            validation_split=0,
            seed=42,
            xgb_n_iter=0,
            quiet=False,
            no_progress=True,
        )
    )
    capsys.readouterr()

    rc = cmd_dj_taxonomy(
        Namespace(
            dj_taxonomy_command="evaluate",
            output=str(tmp_path / "registry"),
            labels=str(labels_path),
            taxonomy=None,
            model_dir=str(tmp_path / "dj_model"),
            quiet=False,
            no_progress=True,
        )
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "DJ taxonomy evaluation complete" in output
    assert "top-1" in output


def test_dj_api_connection_success_with_fake_client():
    class FakeClient:
        provider = "test"
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "category_id": "dark_tech_house_driver",
                "confidence": 0.9,
                "alternate_category_ids": [],
                "rationale": "connection test",
                "warnings": "",
            }

    result = run_dj_api_connection_test(client=FakeClient())

    assert result.ok is True
    assert result.provider == "test"


def _sample_store(tmp_path):
    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks(
        [
            LogicalTrack(track_id="T1", artist_canonical="A", title_canonical="Dark Driver", tagger_energy="E4", tagger_vibe="dark", tagger_structure="16H", tagger_bpm="126"),
            LogicalTrack(track_id="T2", artist_canonical="B", title_canonical="Vocal Hook", tagger_energy="E4", tagger_vocal="featured_vocal", tagger_structure="16G", tagger_bpm="126"),
            LogicalTrack(track_id="T3", artist_canonical="C", title_canonical="Warehouse Peak", tagger_energy="E5", tagger_vibe="warehouse", tagger_structure="16D", tagger_bpm="134"),
            LogicalTrack(track_id="T4", artist_canonical="D", title_canonical="Organic Chant", tagger_energy="E3", tagger_vibe="tribal", tagger_vocal="chant", tagger_structure="16H", tagger_bpm="120"),
        ]
    )
    store.save_files(
        [
            FileRecord(file_id="F1", track_id="T1", is_primary_file=True, file_name="dark driver.mp3", embedded_genre="Tech House"),
            FileRecord(file_id="F2", track_id="T2", is_primary_file=True, file_name="vocal hook.mp3", embedded_genre="Tech House"),
            FileRecord(file_id="F3", track_id="T3", is_primary_file=True, file_name="warehouse peak.mp3", embedded_genre="Techno"),
            FileRecord(file_id="F4", track_id="T4", is_primary_file=True, file_name="organic chant.mp3", embedded_genre="Afro House"),
        ]
    )
    store.save_observations(
        [
            SourceObservation(track_id="T1", source_system="songstats", genre="Tech House", genres_all="Tech House;House", danceability="0.82", energy="0.76", valence="0.22"),
            SourceObservation(track_id="T2", source_system="songstats", genre="Tech House", genres_all="Tech House;House", danceability="0.86", energy="0.74"),
            SourceObservation(track_id="T3", source_system="songstats", genre="Techno", genres_all="Techno;Peak Time Techno", danceability="0.78", energy="0.90", valence="0.18"),
            SourceObservation(track_id="T4", source_system="songstats", genre="Afro House", genres_all="Afro House;Organic House", danceability="0.80", energy="0.58", valence="0.62"),
        ]
    )
    return store


def _write_dj_labels(tmp_path):
    labels_path = tmp_path / "dj_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "file_name", "category_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"track_id": "T1", "file_name": "dark driver.mp3", "category_id": "dark_tech_house_driver"},
                {"track_id": "T2", "file_name": "vocal hook.mp3", "category_id": "vocal_hook_tech_house"},
                {"track_id": "T3", "file_name": "warehouse peak.mp3", "category_id": "raw_warehouse_techno"},
                {"track_id": "T4", "file_name": "organic chant.mp3", "category_id": "ritual_chant_house"},
            ]
        )
    return labels_path


# ── Phase 2: data-hygiene additions ─────────────────────────────────────────


def _make_example(category_id, *, artist=""):
    """Tiny synthetic example matching the shape of _build_training_examples."""
    return {
        "track": LogicalTrack(track_id=f"T-{category_id}-{artist}", artist_canonical=artist),
        "features": {"text:title:token=x": 1.0},
        "category_id": category_id,
        "file_name": f"{category_id}.mp3",
        "observations": [],
        "file_record": None,
        "external_evidence_available": False,
    }


def test_split_examples_stratifies_classes_when_feasible():
    """Every class should land in both train and test when each has ≥2 examples."""
    from dj_registry.taxonomy.dj_model import _split_examples

    examples = (
        [_make_example("a", artist=f"A{i}") for i in range(5)]
        + [_make_example("b", artist=f"B{i}") for i in range(5)]
        + [_make_example("c", artist=f"C{i}") for i in range(5)]
    )
    train, val = _split_examples(examples, validation_split=0.4, seed=42)
    train_classes = {e["category_id"] for e in train}
    val_classes = {e["category_id"] for e in val}
    assert train_classes == {"a", "b", "c"}, f"train classes incomplete: {train_classes}"
    assert val_classes == {"a", "b", "c"}, f"val classes incomplete: {val_classes}"


def test_split_examples_falls_back_to_random_shuffle_for_thin_classes():
    """When any class has <2 examples, stratification fails — must not crash."""
    from dj_registry.taxonomy.dj_model import _split_examples

    examples = [
        _make_example("a"),
        _make_example("a"),
        _make_example("b"),
        _make_example("c"),  # singleton — blocks stratified split
    ]
    train, val = _split_examples(examples, validation_split=0.25, seed=42)
    assert len(train) + len(val) == 4
    assert len(val) >= 1


def test_split_examples_avoids_same_artist_leakage():
    """All tracks by the same artist should land in the same split."""
    from dj_registry.taxonomy.dj_model import _split_examples

    examples = (
        [_make_example("a", artist="X") for _ in range(3)]
        + [_make_example("a", artist="Y") for _ in range(3)]
        + [_make_example("b", artist="Z") for _ in range(3)]
        + [_make_example("b", artist="W") for _ in range(3)]
    )
    train, val = _split_examples(examples, validation_split=0.25, seed=42)
    train_artists = {e["track"].artist_canonical for e in train}
    val_artists = {e["track"].artist_canonical for e in val}
    # Some artist set should be disjoint between train and val
    assert train_artists & val_artists == set(), (
        f"artist leakage: train={train_artists}, val={val_artists}"
    )


def test_build_training_examples_drops_under_supported_categories(tmp_path):
    """Categories with fewer than MIN_EXAMPLES_PER_CATEGORY rows are dropped
    when the dataset is large enough; the dropped set is returned."""
    from dj_registry.taxonomy.dj_model import _build_training_examples, MIN_EXAMPLES_PER_CATEGORY

    store = CsvStore(str(tmp_path / "registry"))
    # Make 8 rows total: one popular category × 5 + one singleton × 3, plus
    # an under-supported category × 1 that should be dropped.
    tracks = []
    files = []
    label_rows = []
    cat_popular = "dark_tech_house_driver"
    cat_under = "vocal_hook_tech_house"
    cat_tiny = "raw_warehouse_techno"
    # popular: 5
    for i in range(5):
        tid = f"T{i+1}"
        tracks.append(LogicalTrack(track_id=tid, artist_canonical=f"A{i}",
                                   tagger_energy="E4", tagger_vibe="dark", tagger_structure="16H", tagger_bpm="126"))
        files.append(FileRecord(file_id=f"F{i+1}", track_id=tid, is_primary_file=True, file_name=f"f{i+1}.mp3"))
        label_rows.append({"track_id": tid, "file_name": f"f{i+1}.mp3", "category_id": cat_popular})
    # under: 3 — at the boundary (>=3 = kept)
    for i in range(5, 8):
        tid = f"T{i+1}"
        tracks.append(LogicalTrack(track_id=tid, artist_canonical=f"B{i}",
                                   tagger_energy="E4", tagger_vibe="dark", tagger_structure="16H", tagger_bpm="126"))
        files.append(FileRecord(file_id=f"F{i+1}", track_id=tid, is_primary_file=True, file_name=f"f{i+1}.mp3"))
        label_rows.append({"track_id": tid, "file_name": f"f{i+1}.mp3", "category_id": cat_under})
    # tiny: 1 — should be dropped (< 3)
    tracks.append(LogicalTrack(track_id="T9", artist_canonical="C0",
                               tagger_energy="E4", tagger_vibe="dark", tagger_structure="16H", tagger_bpm="126"))
    files.append(FileRecord(file_id="F9", track_id="T9", is_primary_file=True, file_name="f9.mp3"))
    label_rows.append({"track_id": "T9", "file_name": "f9.mp3", "category_id": cat_tiny})

    store.save_tracks(tracks)
    store.save_files(files)
    store.save_observations([])

    examples, _, dropped = _build_training_examples(
        label_rows, store, mode="external", show_progress=False,
    )

    assert MIN_EXAMPLES_PER_CATEGORY == 3
    kept_categories = {e["category_id"] for e in examples}
    assert cat_popular in kept_categories
    assert cat_under in kept_categories
    assert cat_tiny not in kept_categories
    assert dropped == {cat_tiny: 1}


def test_metrics_include_per_class_precision_recall_f1():
    """_metrics_from_rows should now also surface a per-class report dict."""
    from dj_registry.taxonomy.dj_model import _metrics_from_rows

    rows = [
        {"expected_category_id": "a", "ext_category_id": "a", "ext_correct": True,
         "ext_top3_correct": True, "ext_confidence": 0.8},
        {"expected_category_id": "a", "ext_category_id": "b", "ext_correct": False,
         "ext_top3_correct": True, "ext_confidence": 0.5},
        {"expected_category_id": "b", "ext_category_id": "b", "ext_correct": True,
         "ext_top3_correct": True, "ext_confidence": 0.9},
    ]
    metrics = _metrics_from_rows(rows, prefix="ext")
    assert "per_class" in metrics
    assert set(metrics["per_class"].keys()) >= {"a", "b"}
    for stats in metrics["per_class"].values():
        assert {"precision", "recall", "f1", "support"} == set(stats.keys())
    # Class "b" has 1 true positive and 1 false positive → precision 0.5
    assert metrics["per_class"]["b"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["b"]["recall"] == pytest.approx(1.0)


def test_confidence_formula_uses_feature_group_count():
    """Adding more keys within a group must not inflate confidence — the bonus
    counts distinct prefixes, so wide dense blocks (CLAP, DSP) stay neutral."""
    from dj_registry.taxonomy.dj_model import _confidence_from_model_score

    sparse = {"text:a:b": 1.0, "num:bpm": 124, "cue:dark": 1.0}
    bloated = {f"num:dsp:f{i}": 0.1 for i in range(512)}
    bloated.update(sparse)

    sparse_conf = _confidence_from_model_score(0.6, 0.1, sparse)
    bloated_conf = _confidence_from_model_score(0.6, 0.1, bloated)
    # Adding hundreds of `num:dsp:*` keys creates only ONE new group ("num"
    # already exists from num:bpm), so the bonus must not jump dramatically.
    assert abs(bloated_conf - sparse_conf) < 0.05


import pytest  # noqa: E402  (used by approx assertions above)
