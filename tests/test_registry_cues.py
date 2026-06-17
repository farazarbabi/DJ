"""Tests for registry cue point analysis/export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from lxml import etree

from dj_registry.cli import main as registry_main
from dj_registry.config import RegistryConfig
from dj_registry.cues.export_rekordbox import REVIEW_PLAYLIST_NAME, export_rekordbox_cues
from dj_registry.cues.profiles import load_cue_profile
from dj_registry.cues.quality import write_cue_quality_report
from dj_registry.cues.selection import CueGrid, CueSection, select_default_hot_cues, select_profile_cues
from dj_registry.cues.slots import hot_cue_num, hot_cue_slot
from dj_registry.cues.validate_rekordbox import validate_rekordbox_xml
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


def test_profile_v1_matches_default_hot_cues():
    grid = _v2_grid()

    default_cues = select_default_hot_cues("T1", "F1", grid)
    profile_cues = select_profile_cues("T1", "F1", grid, profile="v1")

    assert [(cue.cue_role, cue.cue_time_sec, cue.source_system) for cue in profile_cues] == [
        (cue.cue_role, cue.cue_time_sec, "auto_v1") for cue in default_cues
    ]
    assert [cue.cue_kind for cue in profile_cues] == ["hot", "hot", "hot"]


def test_profile_v2_generates_hot_memory_and_16_bar_loops():
    cues = select_profile_cues("T1", "F1", _v2_grid(), profile="v2")

    assert [cue.cue_kind for cue in cues].count("hot") == 3
    assert [cue.cue_kind for cue in cues].count("memory") == 4
    assert [cue.cue_kind for cue in cues].count("loop") == 2
    assert {cue.source_system for cue in cues} == {"auto_v2"}

    by_role = {cue.cue_role: cue for cue in cues}
    assert by_role["breakdown"].cue_time_sec == 64.0
    assert by_role["peak"].cue_time_sec == 80.0
    assert by_role["outro_start"].cue_time_sec == 112.0
    assert by_role["intro_loop"].cue_time_sec == 16.0
    assert by_role["intro_loop"].cue_end_sec == 48.0
    assert by_role["outro_loop"].cue_time_sec == 112.0
    assert by_role["outro_loop"].cue_end_sec == 144.0


def test_profile_v2_omits_breakdown_when_section_is_missing():
    grid = _v2_grid(sections=[
        CueSection(label="intro", start_bar=0, end_bar=8, energy=0.2),
        CueSection(label="groove", start_bar=8, end_bar=40, energy=0.45),
        CueSection(label="peak", start_bar=40, end_bar=48, energy=0.9),
        CueSection(label="outro", start_bar=56, end_bar=80, energy=0.2),
    ])

    cues = select_profile_cues("T1", "F1", grid, profile="v2")

    assert "breakdown" not in {cue.cue_role for cue in cues}
    assert [cue.cue_kind for cue in cues].count("memory") == 3


def test_profile_v2_omits_loops_without_full_default_runway():
    grid = CueGrid(
        duration_sec=30.0,
        bar_times=[float(i * 2) for i in range(15)],
        bar_energies=[0.4] * 15,
        sections=[
            CueSection(label="intro", start_bar=0, end_bar=4, energy=0.2),
            CueSection(label="peak", start_bar=8, end_bar=10, energy=0.9),
            CueSection(label="outro", start_bar=7, end_bar=15, energy=0.2),
        ],
    )

    cues = select_profile_cues("T1", "F1", grid, profile="v2")

    assert [cue.cue_kind for cue in cues].count("loop") == 0


def test_loop_bars_override_generates_short_review_loops():
    grid = CueGrid(
        duration_sec=30.0,
        bar_times=[float(i * 2) for i in range(15)],
        bar_energies=[0.4] * 15,
        sections=[
            CueSection(label="intro", start_bar=0, end_bar=4, energy=0.2),
            CueSection(label="peak", start_bar=8, end_bar=10, energy=0.9),
            CueSection(label="outro", start_bar=7, end_bar=15, energy=0.2),
        ],
    )

    cues = select_profile_cues("T1", "F1", grid, profile="v2", loop_bars=8)
    loops = [cue for cue in cues if cue.cue_kind == "loop"]

    assert {cue.cue_role for cue in loops} == {"intro_loop", "outro_loop"}
    assert all(cue.cue_end_sec - cue.cue_time_sec == 16.0 for cue in loops)
    assert all(cue.manual_review_required for cue in loops)


def test_profile_v3_default_phrase_aligns_sections_and_flags_grid_offset():
    grid = CueGrid(
        duration_sec=160.0,
        bar_times=[float(1 + i * 2) for i in range(80)],
        bar_energies=[0.2] * 8 + [0.45] * 24 + [0.35] * 8 + [0.9] * 8 + [0.5] * 8 + [0.2] * 24,
        sections=[
            CueSection(label="intro", start_bar=0, end_bar=8, energy=0.2),
            CueSection(label="groove", start_bar=8, end_bar=33, energy=0.45),
            CueSection(label="breakdown", start_bar=33, end_bar=41, energy=0.35),
            CueSection(label="peak", start_bar=41, end_bar=49, energy=0.9),
            CueSection(label="outro", start_bar=57, end_bar=80, energy=0.2),
        ],
    )

    cues = select_profile_cues("T1", "F1", grid, profile="v3-default")
    by_role = {cue.cue_role: cue for cue in cues}

    assert {cue.source_system for cue in cues} == {"auto_v3"}
    assert by_role["breakdown"].cue_bar_index == 32
    assert by_role["peak"].cue_bar_index == 40
    assert by_role["outro_start"].cue_bar_index == 56
    assert "possible_grid_offset" in by_role["breakdown"].selection_reason
    assert by_role["breakdown"].manual_review_required is True


def test_profile_file_filters_roles_and_loop_length(tmp_path):
    profile_path = tmp_path / "minimal-cues.json"
    profile_path.write_text(json.dumps({
        "name": "minimal",
        "enabled_roles": ["mix_in", "intro_loop"],
        "loop_bars": 8,
        "min_confidence": 0.5,
        "source_system": "auto_custom",
        "phrase_align_sections": False,
    }), encoding="utf-8")
    profile = load_cue_profile(profile_file=str(profile_path))

    cues = select_profile_cues("T1", "F1", _v2_grid(), profile_config=profile)

    assert [(cue.cue_role, cue.source_system) for cue in cues] == [
        ("mix_in", "auto_custom"),
        ("intro_loop", "auto_custom"),
    ]
    loop = cues[1]
    assert loop.cue_end_sec - loop.cue_time_sec == 16.0
    assert "short_loop" in loop.selection_reason


def test_profile_file_rejects_unsupported_roles(tmp_path):
    profile_path = tmp_path / "bad-cues.json"
    profile_path.write_text(json.dumps({
        "name": "bad",
        "enabled_roles": ["mix_in", "not_a_role"],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported cue role"):
        load_cue_profile(profile_file=str(profile_path))


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


def test_rekordbox_export_inserts_and_preserves_v2_memory_and_loop_cues(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path, include_v2=True)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    output_xml = tmp_path / "exported-v2.xml"

    stats = export_rekordbox_cues(config, store, input_xml=xml_path, output_xml=output_xml)

    assert stats.inserted == 3
    assert stats.skipped_conflict == 4
    assert stats.unmatched == 1
    assert stats.invalid == 1

    tree = etree.parse(str(output_xml))
    track = tree.getroot().find("COLLECTION").find("TRACK")
    marks_by_name = {mark.get("Name"): mark for mark in track.findall("POSITION_MARK")}
    assert marks_by_name["INTRO START"].get("Type") == "0"
    assert marks_by_name["INTRO START"].get("Num") == "-1"
    assert marks_by_name["INTRO LOOP"].get("Type") == "4"
    assert marks_by_name["INTRO LOOP"].get("Num") == "-1"
    assert marks_by_name["INTRO LOOP"].get("Start") == "16.000"
    assert marks_by_name["INTRO LOOP"].get("End") == "48.000"
    assert marks_by_name["Existing Memory"].get("Start") == "64.000"
    assert marks_by_name["Existing Loop"].get("End") == "144.000"

    statuses = {cue.cue_id: cue.export_status for cue in store.load_cue_points()}
    assert statuses["CUE-F1-MEM-intro_start"] == "inserted"
    assert statuses["CUE-F1-MEM-breakdown"] == "skipped_conflict"
    assert statuses["CUE-F1-LOOP-intro_loop"] == "inserted"
    assert statuses["CUE-F1-LOOP-outro_loop"] == "skipped_conflict"
    assert statuses["CUE-F1-LOOP-invalid"] == "invalid"

    with open(stats.report_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"cue_kind", "cue_end_sec"}.issubset(rows[0])
    assert any(row["cue_kind"] == "loop" and row["cue_end_sec"] == "48.000" for row in rows)


def test_rekordbox_export_dry_run_does_not_insert_or_update_store(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    output_xml = tmp_path / "dry-run.xml"

    stats = export_rekordbox_cues(
        config,
        store,
        input_xml=xml_path,
        output_xml=output_xml,
        dry_run=True,
    )

    assert stats.would_insert == 1
    assert stats.inserted == 0
    assert stats.skipped_conflict == 2
    assert stats.unmatched == 1
    tree = etree.parse(str(output_xml))
    track = tree.getroot().find("COLLECTION").find("TRACK")
    assert "1" not in {mark.get("Num") for mark in track.findall("POSITION_MARK")}
    assert {cue.cue_id: cue.export_status for cue in store.load_cue_points()}["CUE-F1-B"] == ""


def test_rekordbox_export_review_only_policy_writes_report_without_markers(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    output_xml = tmp_path / "review-only.xml"

    stats = export_rekordbox_cues(
        config,
        store,
        input_xml=xml_path,
        output_xml=output_xml,
        policy="review-only",
    )

    assert stats.review_only == 3
    assert stats.inserted == 0
    assert stats.unmatched == 1
    tree = etree.parse(str(output_xml))
    track = tree.getroot().find("COLLECTION").find("TRACK")
    assert "1" not in {mark.get("Num") for mark in track.findall("POSITION_MARK")}
    statuses = {cue.cue_id: cue.export_status for cue in store.load_cue_points()}
    assert statuses["CUE-F1-B"] == "review_only"


def test_rekordbox_export_replace_generated_policy_only_replaces_matching_markers(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path, generated_b=True)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    output_xml = tmp_path / "replace-generated.xml"

    stats = export_rekordbox_cues(
        config,
        store,
        input_xml=xml_path,
        output_xml=output_xml,
        policy="replace-generated",
    )

    assert stats.replaced == 1
    assert stats.skipped_conflict == 2
    tree = etree.parse(str(output_xml))
    track = tree.getroot().find("COLLECTION").find("TRACK")
    marks_by_num = {mark.get("Num"): mark for mark in track.findall("POSITION_MARK")}
    assert marks_by_num["1"].get("Start") == "32.000"


def test_cue_quality_report_writes_flags_without_rekordbox_xml(tmp_path):
    store, _xml_path, _audio_path = _store_with_xml_fixture(tmp_path, include_v2=True)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))

    stats = write_cue_quality_report(config, store)

    assert stats.cues_total == len(store.load_cue_points())
    assert stats.manual_review >= 1
    with open(stats.report_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"track_id", "quality_flags", "selection_reason"}.issubset(rows[0])
    assert any("low_confidence" in row["quality_flags"] for row in rows)


def test_rekordbox_xml_validation_counts_v2_marker_shapes(tmp_path):
    _store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path, include_v2=True)

    result = validate_rekordbox_xml(xml_path)

    assert result.ok
    assert result.hot_cues == 2
    assert result.memory_cues == 1
    assert result.loops == 1
    assert result.unknown_markers == 0


def test_cues_v3_cli_quality_and_validate_smoke(tmp_path):
    store, xml_path, _audio_path = _store_with_xml_fixture(tmp_path, include_v2=True)

    rc_quality = registry_main([
        "cues",
        "report-quality",
        "--registry",
        store.output_dir,
    ])
    rc_validate = registry_main([
        "cues",
        "validate-rekordbox-xml",
        "--input-xml",
        str(xml_path),
        "--registry",
        store.output_dir,
    ])

    assert rc_quality == 0
    assert rc_validate == 0
    assert (Path(store.output_dir) / "reports" / "cue_quality_report.csv").exists()


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


def _store_with_xml_fixture(tmp_path, *, include_v2: bool = False, generated_b: bool = False):
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
    cue_points = [
        _cue("CUE-F1-A", "T1", "F1", "A", 0, "MIX IN", 8.0),
        _cue("CUE-F1-B", "T1", "F1", "B", 1, "DROP 1", 32.0),
        _cue("CUE-F1-F", "T1", "F1", "F", 5, "ALT HOT", 64.0),
        _cue("CUE-MISSING-A", "TM", "FM", "A", 0, "MIX IN", 1.0),
    ]
    if include_v2:
        cue_points.extend([
            _memory("CUE-F1-MEM-intro_start", "T1", "F1", "intro_start", "INTRO START", 16.0),
            _memory("CUE-F1-MEM-breakdown", "T1", "F1", "breakdown", "BREAKDOWN", 64.0),
            _loop("CUE-F1-LOOP-intro_loop", "T1", "F1", "intro_loop", "INTRO LOOP", 16.0, 48.0),
            _loop("CUE-F1-LOOP-outro_loop", "T1", "F1", "outro_loop", "OUTRO LOOP", 112.0, 144.0),
            _loop("CUE-F1-LOOP-invalid", "T1", "F1", "intro_loop", "BROKEN LOOP", 50.0, 50.0),
        ])
    store.save_cue_points(cue_points)

    xml_path = tmp_path / "rekordbox.xml"
    v2_marks = ""
    if include_v2:
        v2_marks = """
      <POSITION_MARK Name="Existing Memory" Type="0" Start="64.000" Num="-1" Red="255" Green="202" Blue="88"/>
      <POSITION_MARK Name="Existing Loop" Type="4" Start="112.000" End="144.000" Num="-1" Red="75" Green="211" Blue="220"/>
"""
    generated_b_mark = ""
    if generated_b:
        generated_b_mark = """
      <POSITION_MARK Name="DROP 1" Type="0" Start="20.000" Num="1" Red="1" Green="2" Blue="3"/>
"""
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="1">
    <TRACK TrackID="101" Name="One" Artist="Artist" Location="{audio_path.resolve().as_uri()}" Size="{audio_path.stat().st_size}" TotalTime="300">
      <POSITION_MARK Name="Existing A" Type="0" Start="0.000" Num="0" Red="255" Green="55" Blue="111"/>
      <POSITION_MARK Name="Existing F" Type="0" Start="72.237" Num="5" Red="224" Green="100" Blue="27"/>
{generated_b_mark.rstrip()}
{v2_marks.rstrip()}
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


def _memory(cue_id, track_id, file_id, role, name, time_sec):
    return CuePoint(
        cue_id=cue_id,
        track_id=track_id,
        file_id=file_id,
        source_system="auto_v2",
        cue_kind="memory",
        cue_role=role,
        cue_name=name,
        cue_time_sec=time_sec,
        rekordbox_num=-1,
        rekordbox_type="0",
        red=11,
        green=22,
        blue=33,
        confidence=0.8,
    )


def _loop(cue_id, track_id, file_id, role, name, start_sec, end_sec):
    return CuePoint(
        cue_id=cue_id,
        track_id=track_id,
        file_id=file_id,
        source_system="auto_v2",
        cue_kind="loop",
        cue_role=role,
        cue_name=name,
        cue_time_sec=start_sec,
        cue_end_sec=end_sec,
        rekordbox_num=-1,
        rekordbox_type="4",
        red=44,
        green=55,
        blue=66,
        confidence=0.8,
    )


def _v2_grid(sections: list[CueSection] | None = None) -> CueGrid:
    return CueGrid(
        duration_sec=160.0,
        bar_times=[float(i * 2) for i in range(80)],
        bar_energies=[0.2] * 8 + [0.45] * 24 + [0.35] * 8 + [0.9] * 8 + [0.5] * 8 + [0.2] * 24,
        sections=sections or [
            CueSection(label="intro", start_bar=0, end_bar=8, energy=0.2),
            CueSection(label="groove", start_bar=8, end_bar=32, energy=0.45),
            CueSection(label="breakdown", start_bar=32, end_bar=40, energy=0.35),
            CueSection(label="peak", start_bar=40, end_bar=48, energy=0.9),
            CueSection(label="outro", start_bar=56, end_bar=80, energy=0.2),
        ],
    )
