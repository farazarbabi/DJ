"""Tests for half-resolution group merging and coarse-playlist output."""

from __future__ import annotations

import os

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.features.builder import TrackFeatures
from dj_grouper.grouping.assignment import GroupAssignment, GroupInfo
from dj_grouper.output.coarse_groups import (
    build_coarse_assignment,
    generate_coarse_group_playlists,
)
from dj_grouper.scanner import TrackInfo


def _track(path: str, key: str = "9A", bpm: int = 124, energy: int = 4) -> TrackFeatures:
    info = TrackInfo(
        path=path,
        key=key,
        energy=energy,
        bpm=bpm,
        structure="16H",
        vibe="DRK",
        vocal="INST",
    )
    return TrackFeatures(path, info, np.zeros(19), np.zeros(21))


def _group(gid: str, members: list[int], medoid: int, key: str) -> GroupInfo:
    return GroupInfo(
        group_id=gid,
        member_indices=members,
        medoid_index=medoid,
        key=key,
        energy=4,
        vibe="DRK",
        structure="16H",
        vocal="INST",
        bpm=124,
        folder_name=f"{key}_E4_DRK_INST_{gid}",
    )


def test_coarse_halves_group_count():
    tracks = [_track(f"/music/{i}.aiff") for i in range(8)]
    assignment = GroupAssignment(
        groups=[
            _group("G001", [0, 1], 0, "1A"),
            _group("G002", [2, 3], 2, "2A"),
            _group("G003", [4, 5], 4, "9A"),
            _group("G004", [6, 7], 6, "10A"),
        ],
    )
    coarse = build_coarse_assignment(tracks, assignment, GrouperConfig())
    assert len(coarse.groups) == 2
    # All 8 tracks accounted for.
    members = sorted(idx for g in coarse.groups for idx in g.member_indices)
    assert members == list(range(8))


def test_coarse_single_group_passes_through():
    tracks = [_track(f"/music/{i}.aiff") for i in range(3)]
    assignment = GroupAssignment(
        groups=[_group("G001", [0, 1, 2], 0, "1A")],
    )
    coarse = build_coarse_assignment(tracks, assignment, GrouperConfig())
    assert len(coarse.groups) == 1
    # ID renamed under the coarse namespace.
    assert coarse.groups[0].group_id.startswith("CG")


def test_coarse_two_groups_merged_to_one():
    tracks = [_track(f"/music/{i}.aiff") for i in range(4)]
    assignment = GroupAssignment(
        groups=[
            _group("G001", [0, 1], 0, "1A"),
            _group("G002", [2, 3], 2, "1A"),
        ],
    )
    coarse = build_coarse_assignment(tracks, assignment, GrouperConfig())
    assert len(coarse.groups) == 1
    assert sorted(coarse.groups[0].member_indices) == [0, 1, 2, 3]


def test_generate_coarse_group_playlists_writes_files(tmp_path):
    tracks = [_track(f"/music/{i}.aiff") for i in range(6)]
    assignment = GroupAssignment(
        groups=[
            _group("G001", [0, 1], 0, "1A"),
            _group("G002", [2, 3], 2, "2A"),
            _group("G003", [4, 5], 4, "9A"),
        ],
    )
    n = generate_coarse_group_playlists(tracks, assignment, GrouperConfig(), str(tmp_path))
    # 3 source groups → halved → 1 (floor division).
    assert n == 1
    coarse_dir = tmp_path / "groups_coarse"
    assert coarse_dir.exists()
    files = sorted(p.name for p in coarse_dir.glob("*.m3u8"))
    assert len(files) == 1
    contents = (coarse_dir / files[0]).read_text(encoding="utf-8")
    assert "#EXTM3U" in contents
    # All six paths appear in the merged playlist, as absolute on-disk paths
    # (Rekordbox anchors relative M3U8 entries to the playlist folder).
    for i in range(6):
        assert os.path.abspath(f"/music/{i}.aiff") in contents
