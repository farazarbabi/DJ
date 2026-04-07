"""Tests for playlist generation."""

from pathlib import Path

import numpy as np

from dj_grouper.features.builder import TrackFeatures
from dj_grouper.scanner import TrackInfo
from dj_grouper.grouping.assignment import GroupInfo, GroupAssignment
from dj_grouper.output.playlists import generate_group_playlists
from dj_grouper.recommend.neighbors import Recommendation
from dj_grouper.output.playlists import generate_recommendation_playlists


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
        vocal="NV", bpm=126, folder_name="9A_E3_HYPN_64H_NV_126",
    )
    assignment = GroupAssignment(
        groups=[group],
        track_to_group={"/music/a.aiff": "G001", "/music/b.aiff": "G001"},
    )
    return tracks, assignment


def test_group_playlist_created(tmp_path):
    tracks, assignment = _make_assignment()
    generate_group_playlists(tracks, assignment, str(tmp_path))
    playlist = tmp_path / "groups" / "9A_E3_HYPN_64H_NV_126.m3u8"
    assert playlist.exists()
    content = playlist.read_text()
    assert "#EXTM3U" in content
    assert "/music/a.aiff" in content


def test_recommendation_playlist(tmp_path):
    recs = [
        Recommendation(
            source_path="/music/a.aiff", target_path="/music/b.aiff",
            rank=1, score=0.9,
            energy_delta=0, bpm_delta=2,
            key_compatible=True, struct_compatible=True,
        ),
    ]
    generate_recommendation_playlists(recs, str(tmp_path))
    assert (tmp_path / "recommendations").exists()
    files = list((tmp_path / "recommendations").glob("*.m3u8"))
    assert len(files) >= 1
