"""Tests for unified dj CLI helpers."""

from dj_tools.cli import _load_group_ids_by_file


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
