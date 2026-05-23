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
    load_dj_taxonomy_model,
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


def test_train_evaluate_and_classify_both_dj_taxonomy_models(tmp_path):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)

    result = train_dj_taxonomy_models(store, str(labels_path), model_dir=str(tmp_path / "dj_model"))
    metrics = evaluate_dj_taxonomy_models(
        store,
        str(labels_path),
        model_dir=str(tmp_path / "dj_model"),
    )
    count = classify_all_dj_taxonomies(store, model_dir=str(tmp_path / "dj_model"))

    internal_model = load_dj_taxonomy_model(tmp_path / "dj_model" / "internal")
    external_model = load_dj_taxonomy_model(tmp_path / "dj_model" / "external")
    tracks = {track.track_id: track for track in store.load_tracks()}

    assert result["internal"]["examples"] == 4
    assert result["external"]["examples"] == 4
    assert metrics["examples"] == 4
    assert internal_model.feature_mode == "internal"
    assert external_model.feature_mode == "external"
    assert count == 4
    assert tracks["T1"].dj_taxonomy_internal_id
    assert tracks["T1"].dj_taxonomy_external_id
    assert tracks["T1"].dj_taxonomy_confidence > 0
    assert tracks["T1"].dj_taxonomy_source_model == "external"
    assert json.loads(tracks["T1"].dj_taxonomy_evidence)["internal"]


def test_classify_can_use_internal_model_as_primary(tmp_path):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)

    train_dj_taxonomy_models(store, str(labels_path), model_dir=str(tmp_path / "dj_model"))
    classify_all_dj_taxonomies(store, model_dir=str(tmp_path / "dj_model"), primary_model="internal")

    tracks = {track.track_id: track for track in store.load_tracks()}
    assert tracks["T1"].dj_taxonomy_source_model == "internal"
    assert tracks["T1"].dj_taxonomy_id == tracks["T1"].dj_taxonomy_internal_id


def test_dj_taxonomy_overview_exports_dual_model_columns(tmp_path):
    store = _sample_store(tmp_path)
    labels_path = _write_dj_labels(tmp_path)
    train_dj_taxonomy_models(store, str(labels_path), model_dir=str(tmp_path / "dj_model"))
    classify_all_dj_taxonomies(store, model_dir=str(tmp_path / "dj_model"))

    overview = generate_overview(store, str(tmp_path / "registry"))

    with open(overview, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["dj_taxonomy_id"]
    assert rows[0]["dj_taxonomy_internal_confidence"]
    assert rows[0]["dj_taxonomy_external_confidence"]


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


def test_dj_taxonomy_cli_train_models_prints_locations_only(tmp_path, capsys):
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
            quiet=False,
            no_progress=True,
        )
    )

    assert rc == 0
    assert (tmp_path / "dj_model" / "internal" / "model.pkl").exists()
    assert (tmp_path / "dj_model" / "external" / "model.pkl").exists()
    output = capsys.readouterr().out
    assert "DJ taxonomy models trained" in output
    assert "model_comparison.json" in output
    assert '"comparison"' not in output
    assert '"metrics"' not in output


def test_dj_taxonomy_cli_evaluate_prints_locations_only(tmp_path, capsys):
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
    assert "model_comparison.json" in output
    assert '"internal"' not in output
    assert '"external"' not in output


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
                {"track_id": "T4", "file_name": "organic chant.mp3", "category_id": "organic_chant_house"},
            ]
        )
    return labels_path
