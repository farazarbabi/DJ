"""Tests for unified dj CLI helpers."""

from types import SimpleNamespace

from dj_tools.cli import (
    _build_parser,
    _cue_work_requested,
    _load_group_ids_by_file,
    _run_cue_work,
    _run_fetch_missing,
)


def test_load_group_ids_by_file(tmp_path):
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text(
        "file,group_id,group_folder\n"
        "Track A.mp3,G001,G001_folder\n"
        "Track B.aiff,G017,G017_folder\n",
        encoding="utf-8",
    )

    assert _load_group_ids_by_file(str(groups_csv)) == {
        "Track A.mp3": "G001",
        "Track B.aiff": "G017",
    }


def test_load_group_ids_missing_file_returns_empty(tmp_path):
    assert _load_group_ids_by_file(str(tmp_path / "missing.csv")) == {}


def test_run_parser_accepts_cue_flags():
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "files",
        "--cues",
        "--cue-profile",
        "v3-default",
        "--cue-loop-bars",
        "8",
        "--cue-force",
        "--cue-force-analysis",
        "--cue-export-xml",
        "files/out.xml",
        "--cue-export-policy",
        "review-only",
        "--cue-export-dry-run",
        "--cue-quality-report",
        "--cue-validate-xml",
    ])

    assert args.command == "run"
    assert args.cues is True
    assert args.cue_profile == "v3-default"
    assert args.cue_loop_bars == 8
    assert args.cue_force is True
    assert args.cue_force_analysis is True
    assert args.cue_export_xml == "files/out.xml"
    assert args.cue_export_policy == "review-only"
    assert args.cue_export_dry_run is True
    assert args.cue_quality_report is True
    assert args.cue_validate_xml is True
    assert _cue_work_requested(args) is True


def test_run_parser_fine_playlists_defaults_off():
    parser = _build_parser()
    assert parser.parse_args(["run", "files"]).fine_playlists is False
    assert parser.parse_args(["run", "files", "--fine-playlists"]).fine_playlists is True


def test_run_fetch_missing_flag_nargs_distinguishes_absent_bare_explicit():
    parser = _build_parser()
    # Absent -> None (Phase 0 skipped).
    assert parser.parse_args(["run", "files"]).fetch_missing is None
    # Bare flag -> [] (Phase 0 uses default <library>/spotify-playlists).
    assert parser.parse_args(["run", "files", "--fetch-missing"]).fetch_missing == []
    # Explicit -> list.
    assert parser.parse_args(
        ["run", "files", "--fetch-missing", "a.csv", "b.csv"]
    ).fetch_missing == ["a.csv", "b.csv"]


def test_run_fetch_missing_defaults_to_library_playlists_dir(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    (lib / "spotify-playlists").mkdir(parents=True)
    captured = {}

    def fake_fetch_missing(playlists, library, **kwargs):
        captured["playlists"] = playlists
        captured["library"] = library

    # _run_fetch_missing does `from .spotify_fetch import fetch_missing` at call
    # time, so patching the source attribute is what takes effect.
    monkeypatch.setattr("dj_tools.spotify_fetch.fetch_missing", fake_fetch_missing)

    parser = _build_parser()
    args = parser.parse_args(["fetch-missing", "--library", str(lib)])
    rc = _run_fetch_missing(args)

    assert rc == 0
    assert captured["playlists"] == [str(lib / "spotify-playlists")]


def test_run_fetch_missing_errors_when_default_dir_absent(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()  # no spotify-playlists/ inside
    parser = _build_parser()
    args = parser.parse_args(["fetch-missing", "--library", str(lib)])
    assert _run_fetch_missing(args) == 1


def test_fetch_missing_parser_allows_prune_only_without_playlists():
    parser = _build_parser()
    args = parser.parse_args(["fetch-missing", "--prune-only", "--library", "D:/Music"])
    assert args.prune_only is True
    assert args.playlists == []


def test_run_fetch_missing_prune_only_deletes_superseded(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    orig = lib / "Artist - Song.aiff"
    marked = lib / "Artist - Song[U].aiff"
    orig.write_bytes(b"\x00")
    marked.write_bytes(b"\x00")

    parser = _build_parser()
    args = parser.parse_args(["fetch-missing", "--prune-only", "--library", str(lib)])
    rc = _run_fetch_missing(args)

    assert rc == 0
    assert orig.exists()
    assert not marked.exists()  # superseded [U] copy removed


def test_run_fetch_missing_prune_only_dry_run_keeps_files(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "Artist - Song.aiff").write_bytes(b"\x00")
    marked = lib / "Artist - Song[U].aiff"
    marked.write_bytes(b"\x00")

    parser = _build_parser()
    args = parser.parse_args(
        ["fetch-missing", "--prune-only", "--dry-run", "--library", str(lib)]
    )
    rc = _run_fetch_missing(args)

    assert rc == 0
    assert marked.exists()  # dry run reports without deleting


def test_run_cue_work_wires_analyze_export_and_report(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_validate(path):
        calls.append(("validate", {"path": path}))
        return SimpleNamespace(
            ok=True,
            markers=3,
            hot_cues=1,
            memory_cues=1,
            loops=1,
            errors=[],
        )

    def fake_analyze(config, store, **kwargs):
        calls.append(("analyze", kwargs))
        return SimpleNamespace(
            analyzed=2,
            cached=3,
            cues_written=18,
            skipped_existing=1,
            failed=0,
        )

    def fake_export(config, store, **kwargs):
        calls.append(("export", kwargs))
        return SimpleNamespace(
            inserted=4,
            replaced=1,
            would_insert=0,
            would_replace=0,
            skipped_conflict=2,
            invalid=0,
            report_path="cue-export-report.csv",
            output_xml=kwargs["output_xml"],
        )

    def fake_quality(config, store):
        calls.append(("quality", {}))
        return SimpleNamespace(
            cues_total=18,
            manual_review=3,
            report_path="cue-quality-report.csv",
        )

    monkeypatch.setattr("dj_registry.cues.validate_rekordbox.validate_rekordbox_xml", fake_validate)
    monkeypatch.setattr("dj_registry.cues.analysis.analyze_rekordbox_cues", fake_analyze)
    monkeypatch.setattr("dj_registry.cues.export_rekordbox.export_rekordbox_cues", fake_export)
    monkeypatch.setattr("dj_registry.cues.quality.write_cue_quality_report", fake_quality)

    config = SimpleNamespace(rekordbox_xml_path="library.xml")
    store = object()
    args = SimpleNamespace(
        cues=True,
        cue_profile="v3-default",
        cue_profile_file="profile.json",
        cue_loop_bars=8,
        cue_force=True,
        cue_force_analysis=True,
        cue_limit=2,
        cue_export_xml="out.xml",
        cue_export_policy="review-only",
        cue_export_dry_run=True,
        cue_quality_report=False,
        cue_validate_xml=True,
    )

    summary = _run_cue_work(config, store, args, show_progress=False)

    assert [name for name, _kwargs in calls] == ["validate", "analyze", "export", "quality"]
    analyze_kwargs = calls[1][1]
    assert analyze_kwargs["profile"] == "v3-default"
    assert analyze_kwargs["profile_file"] == "profile.json"
    assert analyze_kwargs["loop_bars"] == 8
    assert analyze_kwargs["force"] is True
    assert analyze_kwargs["force_analysis"] is True
    assert analyze_kwargs["limit"] == 2
    export_kwargs = calls[2][1]
    assert export_kwargs["input_xml"] == "library.xml"
    assert export_kwargs["output_xml"] == "out.xml"
    assert export_kwargs["policy"] == "review-only"
    assert export_kwargs["dry_run"] is True
    assert summary["cue_points_written"] == 18
    assert summary["cue_tracks_cached"] == 3
    assert summary["cue_export_inserted"] == 4
    assert summary["cue_quality_review"] == 3
