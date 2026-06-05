"""Tests for group playlist generation."""

import os

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
    # Entries must be absolute on-disk paths so Rekordbox can resolve them
    # (it anchors relative M3U8 entries to the playlist file's own folder).
    assert os.path.abspath("/music/a.aiff") in content
    assert os.path.abspath("/music/b.aiff") in content


def test_relative_paths_are_anchored_to_absolute(tmp_path):
    """Relative scan paths (the default ./files library) must be emitted as
    absolute paths, not verbatim — otherwise Rekordbox resolves them against
    the playlist folder and the playlist imports empty."""
    info = TrackInfo(path="files/track.mp3", energy=3, vibe="HYPN", bpm=126)
    tracks = [TrackFeatures("files/track.mp3", info, np.zeros(19), np.zeros(21))]
    group = GroupInfo(
        group_id="G001", member_indices=[0], medoid_index=0,
        key="9A", energy=3, vibe="HYPN", structure="64H",
        vocal="INST", bpm=126, folder_name="9A_E3_HYPN_INST",
    )
    assignment = GroupAssignment(
        groups=[group], track_to_group={"files/track.mp3": "G001"},
    )
    generate_group_playlists(tracks, assignment, str(tmp_path))
    content = (tmp_path / "groups" / "9A_E3_HYPN_INST.m3u8").read_text()
    assert os.path.abspath("files/track.mp3") in content
    assert os.path.isabs(content.strip().splitlines()[-1])
