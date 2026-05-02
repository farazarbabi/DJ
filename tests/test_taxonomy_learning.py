import csv
import json
from argparse import Namespace

from dj_registry.cli import cmd_taxonomy
from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.taxonomy.features import build_track_features
from dj_registry.taxonomy.ground_truth import (
    GroundTruthStats,
    OpenAIGroundTruthClient,
    extract_chat_completion_json,
    extract_response_json,
    generate_ground_truth_csv,
    test_api_connection as run_api_connection_test,
)
from dj_registry.taxonomy.model import load_taxonomy_model, train_taxonomy_model


def test_provider_genres_are_model_features_not_labels():
    features = build_track_features(
        LogicalTrack(track_id="T1", tagger_energy="E4", tagger_structure="16H", tagger_bpm="126"),
        [SourceObservation(track_id="T1", source_system="spotify", genre="Dance Pop")],
        FileRecord(track_id="T1", file_name="Artist - Track.mp3", embedded_genre="Tech House"),
    )

    assert any("provider:spotify:genre" in key for key in features)
    assert any("provider:embedded_genre" in key for key in features)
    assert any(key == "cue:rolling" for key in features)


def test_train_model_and_classify_with_sparse_non_provider_signals(tmp_path):
    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks(
        [
            LogicalTrack(track_id="T1", artist_canonical="A", title_canonical="Rolling", tagger_energy="E4", tagger_structure="16H", tagger_bpm="126"),
            LogicalTrack(track_id="T2", artist_canonical="B", title_canonical="Peak", tagger_energy="E5", tagger_structure="16D", tagger_bpm="128"),
            LogicalTrack(track_id="T3", artist_canonical="C", title_canonical="Hypnosis", tagger_energy="E4", tagger_vibe="HYPN", tagger_structure="16H", tagger_bpm="132"),
            LogicalTrack(track_id="T4", artist_canonical="D", title_canonical="Desert", tagger_energy="E2", tagger_vibe="ORG", tagger_structure="16H", tagger_bpm="120"),
        ]
    )
    store.save_files(
        [
            FileRecord(file_id="F1", track_id="T1", is_primary_file=True, file_name="rolling.mp3"),
            FileRecord(file_id="F2", track_id="T2", is_primary_file=True, file_name="peak.mp3"),
            FileRecord(file_id="F3", track_id="T3", is_primary_file=True, file_name="hypnosis.mp3"),
            FileRecord(file_id="F4", track_id="T4", is_primary_file=True, file_name="desert.mp3"),
        ]
    )
    store.save_observations([])
    labels_path = tmp_path / "labels.csv"
    _write_labels(
        labels_path,
        [
            ("T1", "rolling.mp3", "House", "Tech House", "Rolling Tech House"),
            ("T2", "peak.mp3", "House", "Tech House", "Peak-Time Tech House"),
            ("T3", "hypnosis.mp3", "Techno", "Minimal / Hypnotic Techno", "Rolling Hypnotic Techno"),
            ("T4", "desert.mp3", "House", "Organic / Afro / Tribal House", "Desert House"),
        ],
    )

    stats = train_taxonomy_model(store, str(labels_path), model_dir=str(tmp_path / "model"))
    model = load_taxonomy_model(tmp_path / "model")
    rc = cmd_taxonomy(
        Namespace(
            taxonomy_command=None,
            output=str(tmp_path / "registry"),
            taxonomy=None,
            model_dir=str(tmp_path / "model"),
            no_model=False,
        )
    )

    loaded = {track.track_id: track for track in store.load_tracks()}
    assert stats.examples == 4
    assert model.examples == 4
    assert rc == 0
    assert loaded["T3"].genre_family == "Techno"
    assert loaded["T3"].genre == "Minimal / Hypnotic Techno"
    assert loaded["T3"].subgenre == "Rolling Hypnotic Techno"
    assert "model_signals" in json.loads(loaded["T3"].genre_evidence)


def test_generate_ground_truth_csv_with_mocked_gpt_client(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "Example Track.mp3").write_bytes(b"")
    store = CsvStore(str(tmp_path / "registry"))
    out = tmp_path / "taxonomy_ground_truth.csv"

    class FakeClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "family": "House",
                "genre": "Tech House",
                "subgenre": "Rolling Tech House",
                "energy": "E4",
                "vibe": "HYPN",
                "vocal": "INST",
                "structure": "16H",
                "set_role": "driver",
                "bpm_hint": "126",
                "confidence": 0.82,
                "alternate_family": "House",
                "alternate_genre": "Melodic / Progressive House",
                "alternate_subgenre": "Hypnotic Progressive House",
                "rationale": "Rolling 4/4 club metadata and filename cues.",
                "warnings": "",
            }

    stats = generate_ground_truth_csv(
        store,
        files_dir=str(files_dir),
        output_path=str(out),
        taxonomy_path="music_genre_taxonomy_3_level.json",
        client=FakeClient(),
    )

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert stats.generated == 1
    assert rows[0]["family"] == "House"
    assert rows[0]["genre"] == "Tech House"
    assert rows[0]["subgenre"] == "Rolling Tech House"
    assert rows[0]["label_source"] == "gpt5_seed"


def test_generate_ground_truth_retries_existing_error_rows(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "Example Track.mp3").write_bytes(b"")
    out = tmp_path / "taxonomy_ground_truth.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "family", "genre", "subgenre", "label_source", "warnings"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "Example Track.mp3",
                "family": "",
                "genre": "",
                "subgenre": "",
                "label_source": "gpt5_error",
                "warnings": "old error",
            }
        )

    class FakeClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "family": "House",
                "genre": "Tech House",
                "subgenre": "Rolling Tech House",
                "energy": "E4",
                "vibe": "HYPN",
                "vocal": "INST",
                "structure": "16H",
                "set_role": "driver",
                "bpm_hint": "126",
                "confidence": 0.82,
                "alternate_family": "",
                "alternate_genre": "",
                "alternate_subgenre": "",
                "rationale": "Retry old error row.",
                "warnings": "",
            }

    stats = generate_ground_truth_csv(
        CsvStore(str(tmp_path / "registry")),
        files_dir=str(files_dir),
        output_path=str(out),
        client=FakeClient(),
    )

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert stats.generated == 1
    assert stats.reused == 0
    assert rows[0]["label_source"] == "gpt5_seed"


def test_generate_ground_truth_fail_fast_stops_on_first_api_error(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "First.mp3").write_bytes(b"")
    (files_dir / "Second.mp3").write_bytes(b"")
    out = tmp_path / "taxonomy_ground_truth.csv"

    class FailingClient:
        model = "gpt-5"

        def __init__(self):
            self.calls = 0

        def label_track(self, context, taxonomy_json, validation_error=None):
            self.calls += 1
            raise RuntimeError("connection refused")

    client = FailingClient()
    stats = generate_ground_truth_csv(
        CsvStore(str(tmp_path / "registry")),
        files_dir=str(files_dir),
        output_path=str(out),
        client=client,
    )

    assert client.calls == 1
    assert stats.errors == 1
    assert stats.rows_written == 0
    assert "First.mp3" in stats.error_message
    assert "connection refused" in stats.error_message
    assert not out.exists()


def test_generate_ground_truth_keep_going_writes_error_rows(tmp_path):
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "First.mp3").write_bytes(b"")
    (files_dir / "Second.mp3").write_bytes(b"")
    out = tmp_path / "taxonomy_ground_truth.csv"

    class FailingClient:
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            raise RuntimeError("connection refused")

    stats = generate_ground_truth_csv(
        CsvStore(str(tmp_path / "registry")),
        files_dir=str(files_dir),
        output_path=str(out),
        client=FailingClient(),
        fail_fast=False,
    )

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert stats.errors == 2
    assert len(rows) == 2
    assert {row["label_source"] for row in rows} == {"gpt5_error"}


def test_ground_truth_cli_returns_nonzero_when_errors_remain(tmp_path, monkeypatch):
    import dj_registry.taxonomy.ground_truth as ground_truth

    def fake_generate(*args, **kwargs):
        return GroundTruthStats(
            files_seen=1,
            rows_written=1,
            generated=0,
            reused=0,
            errors=1,
            output_path=str(tmp_path / "labels.csv"),
        )

    monkeypatch.setattr(ground_truth, "generate_ground_truth_csv", fake_generate)

    rc = cmd_taxonomy(
        Namespace(
            taxonomy_command="generate-ground-truth",
            output=str(tmp_path / "registry"),
            files=str(tmp_path),
            out=str(tmp_path / "labels.csv"),
            taxonomy=None,
            model=None,
            limit=None,
            force=False,
            cache_dir=None,
        )
    )

    assert rc == 1


def test_api_connection_success_with_fake_client():
    class FakeClient:
        provider = "test"
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            return {
                "family": "House",
                "genre": "Tech House",
                "subgenre": "Rolling Tech House",
            }

    result = run_api_connection_test(client=FakeClient())

    assert result.ok is True
    assert result.provider == "test"
    assert result.error_message == ""


def test_api_connection_failure_with_fake_client():
    class FakeClient:
        provider = "test"
        model = "gpt-5"

        def label_track(self, context, taxonomy_json, validation_error=None):
            raise RuntimeError("connection refused")

    result = run_api_connection_test(client=FakeClient())

    assert result.ok is False
    assert result.provider == "test"
    assert "connection refused" in result.error_message


def test_extract_response_json_from_responses_payload():
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "{\"family\":\"House\"}",
                    }
                ],
            }
        ]
    }

    assert extract_response_json(payload) == {"family": "House"}


def test_azure_openai_env_is_supported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = OpenAIGroundTruthClient(model=None)

    assert client.use_azure is True
    assert client.api_key == "test-key"
    assert client.azure_endpoint == "https://example.openai.azure.com"
    assert client.azure_deployment == "gpt-5-deployment"


def test_azure_openai_url_normalizes_deployment_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/deployments")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = OpenAIGroundTruthClient(model=None)

    assert client._azure_chat_completions_url() == (
        "https://example.openai.azure.com/openai/deployments/gpt-5/chat/completions"
        "?api-version=2025-01-01-preview"
    )


def test_azure_openai_url_strips_deployment_prefix(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "openai/deployments/gpt-5")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = OpenAIGroundTruthClient(model=None)

    assert client._azure_chat_completions_url() == (
        "https://example.openai.azure.com/openai/deployments/gpt-5/chat/completions"
        "?api-version=2025-01-01-preview"
    )


def test_extract_chat_completion_json():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "{\"family\":\"House\"}",
                }
            }
        ]
    }

    assert extract_chat_completion_json(payload) == {"family": "House"}


def _write_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "file_name", "family", "genre", "subgenre"],
        )
        writer.writeheader()
        for track_id, file_name, family, genre, subgenre in rows:
            writer.writerow(
                {
                    "track_id": track_id,
                    "file_name": file_name,
                    "family": family,
                    "genre": genre,
                    "subgenre": subgenre,
                }
            )
