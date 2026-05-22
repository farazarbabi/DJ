"""Tests for group assignment descriptors and folder names."""

import pickle

import numpy as np

from dj_grouper.features.builder import TrackFeatures
from dj_grouper.config import GrouperConfig
from dj_grouper.grouping.assignment import (
    GroupAssignment,
    GroupInfo,
    assign_group_ids,
    assign_new_tracks,
    load_assignment,
    refresh_group_descriptors,
)
from dj_grouper.scanner import TrackInfo


def _track(path: str, key: str | None) -> TrackFeatures:
    info = TrackInfo(
        path=path,
        key=key,
        energy=4,
        bpm=124,
        structure="16H",
        vibe="DRK",
        vocal="INST",
    )
    return TrackFeatures(path, info, np.zeros(19), np.zeros(21))


def test_group_folder_prefers_known_key_over_unknown_placeholder():
    tracks = [
        _track("/music/a.aiff", "??"),
        _track("/music/b.aiff", "9A"),
    ]
    labels = np.array([0, 0])
    distances = np.zeros((2, 2))

    assignment = assign_group_ids(tracks, labels, distances)

    group = assignment.groups[0]
    assert group.key == "9A"
    assert group.folder_name == "9A_E4_DRK_INST_G001"
    assert "____" not in group.folder_name


def test_group_folder_uses_safe_unknown_key_placeholder():
    tracks = [_track("/music/a.aiff", "??")]
    labels = np.array([0])
    distances = np.zeros((1, 1))

    assignment = assign_group_ids(tracks, labels, distances)

    group = assignment.groups[0]
    assert group.key == "NK"
    assert group.folder_name == "NK_E4_DRK_INST_G001"


def test_load_assignment_normalizes_unknown_key_folder(tmp_path):
    assignment = GroupAssignment(
        groups=[
            GroupInfo(
                group_id="G001",
                member_indices=[0],
                medoid_index=0,
                key="??",
                energy=4,
                vibe="DRK",
                structure="16H",
                vocal="INST",
                bpm=124,
                folder_name="??|E4|DRK|INST|G001",
            )
        ],
        track_to_group={"/music/a.aiff": "G001"},
    )
    path = tmp_path / "assignment.pkl"
    path.write_bytes(pickle.dumps(assignment))

    loaded = load_assignment(str(path))

    group = loaded.groups[0]
    assert group.key == "NK"
    assert group.folder_name == "NK_E4_DRK_INST_G001"


def test_refresh_group_descriptors_updates_stale_cached_key():
    tracks = [
        _track("/music/a.aiff", "??"),
        _track("/music/b.aiff", "9A"),
    ]
    assignment = GroupAssignment(
        groups=[
            GroupInfo(
                group_id="G001",
                member_indices=[0, 1],
                medoid_index=0,
                key="??",
                energy=4,
                vibe="DRK",
                structure="16H",
                vocal="INST",
                bpm=124,
                folder_name="??|E4|DRK|INST|G001",
            )
        ],
        track_to_group={
            "/music/a.aiff": "G001",
            "/music/b.aiff": "G001",
        },
    )

    refresh_group_descriptors(tracks, assignment)

    group = assignment.groups[0]
    assert group.key == "9A"
    assert group.folder_name == "9A_E4_DRK_INST_G001"


def test_assign_new_tracks_refreshes_descriptors_without_membership_changes():
    tracks = [
        _track("/music/a.aiff", "??"),
        _track("/music/b.aiff", "9A"),
    ]
    assignment = GroupAssignment(
        groups=[
            GroupInfo(
                group_id="G001",
                member_indices=[0, 1],
                medoid_index=0,
                key="??",
                energy=4,
                vibe="DRK",
                structure="16H",
                vocal="INST",
                bpm=124,
                folder_name="??|E4|DRK|INST|G001",
            )
        ],
        track_to_group={
            "/music/a.aiff": "G001",
            "/music/b.aiff": "G001",
        },
    )

    updated = assign_new_tracks(tracks, assignment, GrouperConfig())

    group = updated.groups[0]
    assert group.key == "9A"
    assert group.folder_name == "9A_E4_DRK_INST_G001"
