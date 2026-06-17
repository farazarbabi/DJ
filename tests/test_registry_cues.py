"""Tests for registry cue point analysis/export helpers."""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from dj_registry.cli import main as registry_main
from dj_registry.config import RegistryConfig
from dj_registry.cues.export_rekordbox import REVIEW_PLAYLIST_NAME, export_rekordbox_cues
from dj_registry.cues.selection import CueGrid, CueSection, select_default_hot_cues
from dj_registry.cues.slots import hot_cue_num, hot_cue_slot
from dj_registry.models import CuePoint, FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore


def test_hot_cue_slot_mapping_supports_a_to_h():
    for index, slot in enumerate("ABCDEFGH"):
        assert hot_cue_num(slot) == index
        assert hot_cue_slot(index) == slot


def test_default_selector_generates_three_phrase_aligned_hot_cues():
    grid = CueGrid(
        duration_sec=256.0,
        bar_times=[float(i * 2) for i in range(64)],
        bar_energies=[0.2] * 8 + [0.45] * 24 + [0.9] * 8 + [0.5] * 16 + [0.2] * 8,
        sections=[
            CueSection(label="intro", start_bar=0, end_bar=8, energy=0.2),
            CueSection(label="groove", start_bar=8, end_bar=32, energy=0.45),
            CueSection(label="peak", start_bar=32, end_bar=40, energy=0.9),
            CueSection(label="outro", start_bar=56, end_bar=64, energy=0.2),
        ],
    )

    cues = select_default_hot_cues("T1", "F1", grid)

    assert [(cue.cue_slot, cue.cue_role, cue.rekordbox_num) for cue in cues] == [
        ("A", "mix_in", 0),
        ("B", "drop_1", 1),
        ("C", "mix_out", 2),
    ]
    assert [cue.cue_bar_index for cue in cues] == [8, 32, 56]
    assert [cue.cue_time_sec for cue in cues] == [16.0, 64.0, 112.0]
    assert all(not cue.manual_review_required for cue in cues)


def test_cue_points_round_trip_through_csv_store(tmp_path):
    store = CsvStore(str(tmp_path))
    cue = CuePoint(
        cue_id="CUE-F1-A",
        track_id="T1",
        file_id="F1",
        cue_slot="A",
        cue_name="MIX IN",
        cue_time_sec=12.345,
        rekordbox_num=0,
        red=255,
        manual_review_required=True,
    )

    store.save_cue_points([cue])
    loaded = store.load_cue_points()

    assert len(loaded) == 1
    assert loaded[0].cue_time_sec == 12.345
    assert loaded[0].rekordbox_num == 0
    assert loaded[0].manual_review_required is True
    assert store.get_cue_points_for_file("F1")[0].cue_id == "CUE-F1-A"


def test_rekordbox_export_preserves_conflicts_inserts_missing_slots_and_playlist(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    output_xml = tmp_path / "exported.xml"

    stats = export_rekordbox_cues(config, store, input_xml=xml_path, output_xml=output_xml)

    assert stats.inserted == 1
    assert stats.skipped_conflict == 2
    assert stats.unmatched == 1
    assert stats.review_tracks == 1

    tree = etree.parse(str(output_xml))
    track = tree.getroot().find("COLLECTION").find("TRACK")
    marks = track.findall("POSITION_MARK")
    marks_by_num = {mark.get("Num"): mark for mark in marks}
    assert marks_by_num["0"].get("Name") == "Existing A"
    assert marks_by_num["1"].get("Name") == "DROP 1"
    assert marks_by_num["5"].get("Name") == "Existing F"

    playlist = tree.getroot().find(f".//NODE[@Name='{REVIEW_PLAYLIST_NAME}']")
    assert playlist is not None
    assert playlist.get("Entries") == "1"
    assert playlist.find("TRACK").get("Key") == "101"

    statuses = {cue.cue_id: cue.export_status for cue in store.load_cue_points()}
    assert statuses["CUE-F1-A"] == "skipped_conflict"
    assert statuses["CUE-F1-B"] == "inserted"
    assert statuses["CUE-F1-F"] == "skipped_conflict"
    assert statuses["CUE-MISSING-A"] == "unmatched"
    assert Path(stats.report_path).exists()


def test_cues_export_cli_smoke(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path)
    output_xml = tmp_path / "cli-exported.xml"

    rc = registry_main([
        "cues",
        "export-rekordbox",
        "--input-xml",
        str(xml_path),
        "--output-xml",
        str(output_xml),
        "--registry",
        store.output_dir,
    ])

    assert rc == 0
    assert output_xml.exists()


def _store_with_xml_fixture(tmp_path):
    registry_dir = tmp_path / "registry"
    store = CsvStore(str(registry_dir))
    audio_path = tmp_path / "Artist - One.mp3"
    audio_path.write_bytes(b"not audio")
    missing_path = tmp_path / "missing.mp3"

    store.save_tracks([
        LogicalTrack(track_id="T1", primary_file_id="F1"),
        LogicalTrack(track_id="TM", primary_file_id="FM"),
    ])
    store.save_files([
        FileRecord(
            file_id="F1",
            track_id="T1",
            path_abs=str(audio_path),
            file_name=audio_path.name,
            size_bytes=audio_path.stat().st_size,
            is_primary_file=True,
        ),
        FileRecord(
            file_id="FM",
            track_id="TM",
            path_abs=str(missing_path),
            file_name=missing_path.name,
            size_bytes=99,
            is_primary_file=True,
        ),
    ])
    store.save_cue_points([
        _cue("CUE-F1-A", "T1", "F1", "A", 0, "MIX IN", 8.0),
        _cue("CUE-F1-B", "T1", "F1", "B", 1, "DROP 1", 32.0),
        _cue("CUE-F1-F", "T1", "F1", "F", 5, "ALT HOT", 64.0),
        _cue("CUE-MISSING-A", "TM", "FM", "A", 0, "MIX IN", 1.0),
    ])

    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="1">
    <TRACK TrackID="101" Name="One" Artist="Artist" Location="{audio_path.resolve().as_uri()}" Size="{audio_path.stat().st_size}" TotalTime="300">
      <POSITION_MARK Name="Existing A" Type="0" Start="0.000" Num="0" Red="255" Green="55" Blue="111"/>
      <POSITION_MARK Name="Existing F" Type="0" Start="72.237" Num="5" Red="224" Green="100" Blue="27"/>
    </TRACK>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="0"/>
  </PLAYLISTS>
</DJ_PLAYLISTS>
""",
        encoding="utf-8",
    )
    return store, xml_path, audio_path


def _cue(cue_id, track_id, file_id, slot, num, name, time_sec):
    return CuePoint(
        cue_id=cue_id,
        track_id=track_id,
        file_id=file_id,
        source_system="auto_v1",
        cue_kind="hot",
        cue_role=name.lower().replace(" ", "_"),
        cue_slot=slot,
        cue_name=name,
        cue_time_sec=time_sec,
        rekordbox_num=num,
        rekordbox_type="0",
        red=1,
        green=2,
        blue=3,
        confidence=0.5 if slot == "F" else 0.8,
        manual_review_required=slot == "F",
    )
