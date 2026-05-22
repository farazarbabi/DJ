"""Tests for group playlist generation."""

import numpy as np

from dj_grouper.features.builder import TrackFeatures
from dj_grouper.scanner import TrackInfo
from dj_grouper.grouping.assignment import GroupInfo, GroupAssignment
from dj_grouper.output.playlists import generate_group_playlists


def _make_assignment():
    info_a = TrackInfo(path="/music/a.aiff", energy=3, vibe="HYPN", bpm=126)
    info_b = TrackInfo(path="/music/b.aiff", energy=3, vibe="HYPN", bpm=128)
    tracks = [
        TrackFeatures("/music/a.aiff", info_a, np.zeros(19), np.zeros(21)),
        TrackFeatures("/music/b.aiff", info_b, np.zeros(19), np.zeros(21)),
    ]
    group = GroupInfo(
        group_id="G001", member_indices=[0, 1], medoid_index=0,
        key="9A", energy=3, vibe="HYPN", structure="64H",
        vocal="INST", bpm=126, folder_name="9A_E3_HYPN_INST",
    )
    assignment = GroupAssignment(
        groups=[group],
        track_to_group={"/music/a.aiff": "G001", "/music/b.aiff": "G001"},
    )
    return tracks, assignment


def test_group_playlist_created(tmp_path):
    tracks, assignment = _make_assignment()
    generate_group_playlists(tracks, assignment, str(tmp_path))
    playlist = tmp_path / "groups" / "9A_E3_HYPN_INST.m3u8"
    assert playlist.exists()
    content = playlist.read_text()
    assert "#EXTM3U" in content
    assert "/music/a.aiff" in content
