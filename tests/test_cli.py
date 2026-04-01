"""Tests for CLI argument parsing and integration."""

import json
from pathlib import Path

from dj_tagger.cli import _build_parser, find_audio_files, main


def test_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args(["/some/path"])
    assert args.paths == ["/some/path"]
    assert args.recursive is False
    assert args.dry_run is True
    assert args.write_tags is False
    assert args.overwrite is False
    assert args.workers >= 1


def test_parser_all_flags():
    parser = _build_parser()
    args = parser.parse_args([
        "-r", "-n", "--write-tags", "--overwrite",
        "-w", "4", "--limit", "10",
        "--csv", "out.csv", "--json",
        "-v", "--use-essentia",
        "/music",
    ])
    assert args.recursive is True
    assert args.write_tags is True
    assert args.overwrite is True
    assert args.workers == 4
    assert args.limit == 10
    assert args.csv_path == "out.csv"
    assert args.json_output is True
    assert args.verbose is True
    assert args.use_essentia is True


def test_find_audio_files(tmp_path):
    """find_audio_files should discover supported formats."""
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.flac").touch()
    (tmp_path / "c.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.wav").touch()

    # Non-recursive
    files = find_audio_files([str(tmp_path)], recursive=False)
    names = {f.name for f in files}
    assert "a.mp3" in names
    assert "b.flac" in names
    assert "c.txt" not in names
    assert "d.wav" not in names

    # Recursive
    files_r = find_audio_files([str(tmp_path)], recursive=True)
    names_r = {f.name for f in files_r}
    assert "d.wav" in names_r


def test_main_no_files(tmp_path):
    """main() with an empty dir should return 0."""
    rc = main([str(tmp_path)])
    assert rc == 0


def test_main_dry_run(sine_440hz, tmp_path):
    """main() dry run should process and return 0."""
    rc = main([sine_440hz, "--limit", "1", "-w", "1"])
    assert rc == 0


def test_main_json_output(sine_440hz, capsys):
    """--json should produce valid JSON on stdout."""
    main([sine_440hz, "--json", "-w", "1"])
    captured = capsys.readouterr()
    for line in captured.out.strip().split("\n"):
        if line:
            obj = json.loads(line)
            assert "tag" in obj or "error" in obj


def test_main_csv_export(sine_440hz, tmp_path):
    """--csv should create a CSV file."""
    csv_path = str(tmp_path / "results.csv")
    rc = main([sine_440hz, "--csv", csv_path, "-w", "1"])
    assert rc == 0
    assert Path(csv_path).exists()
    content = Path(csv_path).read_text()
    assert "file" in content  # header row
