"""Stable group ID assignment and new-track assignment."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from statistics import mode as stat_mode

import numpy as np
from numpy.typing import NDArray

from dj_tagger.moods import normalize_mood_code
from dj_tagger.vocals import normalize_vocal_profile

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from .distance import blended_distance
from .medoid import compute_all_medoids

logger = logging.getLogger(__name__)

_WIN_ILLEGAL = str.maketrans({c: "_" for c in r'<>:"/\|?*'})
_UNKNOWN_KEYS = {"", "??", "NK"}


def _safe_folder(name: str) -> str:
    """Strip Windows-illegal characters from a folder/file name component."""
    return name.translate(_WIN_ILLEGAL)


def _is_known_key(key: str | None) -> bool:
    """Return true only for real Camelot keys, not unknown placeholders."""
    if not key or key in _UNKNOWN_KEYS:
        return False
    if len(key) < 2 or key[-1] not in {"A", "B"}:
        return False
    if not key[:-1].isdigit():
        return False
    return 1 <= int(key[:-1]) <= 12


def _folder_key(key: str | None) -> str:
    """Use a filesystem-safe placeholder for unknown group keys."""
    return key if _is_known_key(key) else "NK"


def _build_folder_name(
    group_id: str,
    key: str | None,
    energy: int,
    vibe: str,
    structure: str,
    vocal: str,
    bpm: int,
) -> str:
    return _safe_folder(
        f"{group_id}_{_folder_key(key)}_E{energy}_{vibe}_{structure}_{vocal}_{bpm}"
    )


def _representative_values(
    tracks: list[TrackFeatures],
    members: list[int],
) -> tuple[str, int, str, int, str, str]:
    """Compute current group descriptor values from member track metadata."""
    infos = [tracks[i].info for i in members]
    keys = [i.key for i in infos if _is_known_key(i.key)]
    energies = [i.energy for i in infos if i.energy is not None]
    vibes = [normalize_mood_code(i.vibe) for i in infos if normalize_mood_code(i.vibe)]
    bpms = [i.bpm for i in infos if i.bpm is not None]
    structures = [i.structure for i in infos if i.structure]
    vocals = [normalize_vocal_profile(i.vocal) for i in infos if normalize_vocal_profile(i.vocal)]

    rep_key = _safe_mode(keys, "NK")
    rep_energy = _safe_mode(energies, 3)
    rep_vibe = _safe_mode(vibes, "HYPN")
    rep_bpm = round(np.median(bpms)) if bpms else 128
    rep_structure = _safe_mode(structures, "32H")
    rep_vocal = _safe_mode(vocals, "INST")

    return rep_key, rep_energy, rep_vibe, rep_bpm, rep_structure, rep_vocal


@dataclass
class GroupInfo:
    """Descriptor for a single group."""
    group_id: str
    member_indices: list[int]
    medoid_index: int
    key: str
    energy: int
    vibe: str
    structure: str
    vocal: str
    bpm: int
    folder_name: str


@dataclass
class GroupAssignment:
    """Full assignment state for the library."""
    groups: list[GroupInfo] = field(default_factory=list)
    track_to_group: dict[str, str] = field(default_factory=dict)  # path -> group_id


def _camelot_sort_key(key: str) -> tuple[int, int]:
    """Sort key for Camelot codes. Descending: 12B, 12A, 11B, 11A, ..., 1B, 1A."""
    if not _is_known_key(key):
        return (0, 0)
    num = int(key[:-1])
    letter = 1 if key[-1] == "B" else 0  # B before A at same number
    return (num, letter)


def assign_group_ids(
    tracks: list[TrackFeatures],
    labels: NDArray,
    distance_matrix: NDArray,
) -> GroupAssignment:
    """Create stable group IDs and descriptors from clustering labels.

    Groups are numbered by Camelot key descending (12B first, 1A last)
    so that sorting by group ID in DJ software orders tracks by key.
    """
    medoids = compute_all_medoids(distance_matrix, labels)

    # First pass: build groups with temporary IDs
    temp_groups: list[GroupInfo] = []
    for label in sorted(np.unique(labels)):
        members = list(np.where(labels == label)[0])

        rep_key, rep_energy, rep_vibe, rep_bpm, rep_structure, rep_vocal = (
            _representative_values(tracks, members)
        )

        temp_groups.append(GroupInfo(
            group_id="",  # assigned after sorting
            member_indices=members,
            medoid_index=medoids[int(label)],
            key=rep_key,
            energy=rep_energy,
            vibe=rep_vibe,
            bpm=rep_bpm,
            structure=rep_structure,
            vocal=rep_vocal,
            folder_name="",
        ))

    # Sort by Camelot key ascending (1A, 1B, 2A, 2B, ..., 12A, 12B)
    temp_groups.sort(key=lambda g: _camelot_sort_key(g.key))

    # Second pass: assign sequential IDs in sorted order
    assignment = GroupAssignment()
    for i, group in enumerate(temp_groups):
        group.group_id = f"G{i + 1:03d}"
        group.folder_name = _build_folder_name(
            group.group_id,
            group.key,
            group.energy,
            group.vibe,
            group.structure,
            group.vocal,
            group.bpm,
        )
        assignment.groups.append(group)
        for idx in group.member_indices:
            assignment.track_to_group[tracks[idx].path] = group.group_id

    logger.info("Assigned %d groups (sorted by key desc)", len(assignment.groups))
    return assignment


def assign_new_tracks(
    all_tracks: list[TrackFeatures],
    existing_assignment: GroupAssignment,
    config: GrouperConfig,
) -> GroupAssignment:
    """Assign new tracks to existing groups via nearest-medoid.

    Tracks already in the assignment keep their group.
    New tracks are assigned to the nearest medoid, or a new group is created.
    Deleted tracks are removed.

    Returns updated assignment.
    """
    known_paths = set(existing_assignment.track_to_group.keys())
    current_paths = {t.path for t in all_tracks}
    path_to_idx = {t.path: i for i, t in enumerate(all_tracks)}

    new_paths = current_paths - known_paths
    deleted_paths = known_paths - current_paths

    if not new_paths and not deleted_paths:
        logger.info("No changes — all %d tracks already assigned", len(all_tracks))
        return refresh_group_descriptors(all_tracks, existing_assignment)

    # Start from existing assignment
    assignment = GroupAssignment(
        groups=[g for g in existing_assignment.groups],
        track_to_group=dict(existing_assignment.track_to_group),
    )

    # Remove deleted tracks
    for path in deleted_paths:
        gid = assignment.track_to_group.pop(path, None)
        if gid:
            for group in assignment.groups:
                if group.group_id == gid:
                    # We can't easily remove by old index, so we'll rebuild indices later
                    break

    # Assign new tracks
    n_assigned = 0
    n_new_groups = 0
    for path in sorted(new_paths):
        idx = path_to_idx[path]
        new_track = all_tracks[idx]

        best_group = None
        best_dist = float("inf")

        for group in assignment.groups:
            if group.medoid_index < len(all_tracks):
                medoid_track = all_tracks[group.medoid_index]
                d = blended_distance(new_track, medoid_track, config)
                if d < best_dist:
                    best_dist = d
                    best_group = group

        if best_dist > config.new_group_distance_threshold or best_group is None:
            next_id = max((int(g.group_id[1:]) for g in assignment.groups), default=0) + 1
            new_gid = f"G{next_id:03d}"
            info = new_track.info
            rep_key = _folder_key(info.key)
            rep_energy = info.energy or 3
            rep_vibe = normalize_mood_code(info.vibe) or "HYPN"
            rep_bpm = info.bpm or 128
            rep_structure = info.structure or "32H"
            rep_vocal = normalize_vocal_profile(info.vocal) or "INST"
            new_group = GroupInfo(
                group_id=new_gid,
                member_indices=[idx],
                medoid_index=idx,
                key=rep_key,
                energy=rep_energy,
                vibe=rep_vibe,
                bpm=rep_bpm,
                structure=rep_structure,
                vocal=rep_vocal,
                folder_name=_build_folder_name(
                    new_gid,
                    rep_key,
                    rep_energy,
                    rep_vibe,
                    rep_structure,
                    rep_vocal,
                    rep_bpm,
                ),
            )
            assignment.groups.append(new_group)
            assignment.track_to_group[path] = new_gid
            n_new_groups += 1
            logger.info("New group %s for %s (dist=%.3f)", new_gid, path, best_dist)
        else:
            assignment.track_to_group[path] = best_group.group_id
            n_assigned += 1
            logger.info("Assigned %s to %s (dist=%.3f)", path, best_group.group_id, best_dist)

    # Rebuild member_indices for all groups based on current track list
    gid_to_members: dict[str, list[int]] = {}
    for path, gid in assignment.track_to_group.items():
        if path in path_to_idx:
            gid_to_members.setdefault(gid, []).append(path_to_idx[path])

    # Update groups, remove empty ones
    updated_groups = []
    for group in assignment.groups:
        members = gid_to_members.get(group.group_id, [])
        if not members:
            continue
        group.member_indices = members
        # Medoid: pick the first member if old medoid is invalid
        if group.medoid_index not in members:
            group.medoid_index = members[0]
        updated_groups.append(group)

    assignment.groups = updated_groups
    refresh_group_descriptors(all_tracks, assignment)

    logger.info(
        "Incremental update: %d assigned to existing, %d new groups, %d deleted",
        n_assigned, n_new_groups, len(deleted_paths),
    )
    return assignment


def refresh_group_descriptors(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
) -> GroupAssignment:
    """Refresh group key/energy/mood/BPM descriptors from current track metadata.

    Existing assignments can outlive key-resolution improvements. Recomputing
    descriptors prevents stale ``??`` group keys from leaking into output names.
    """
    for group in assignment.groups:
        if not group.member_indices:
            continue
        rep_key, rep_energy, rep_vibe, rep_bpm, rep_structure, rep_vocal = (
            _representative_values(tracks, group.member_indices)
        )
        group.key = rep_key
        group.energy = rep_energy
        group.vibe = rep_vibe
        group.bpm = rep_bpm
        group.structure = rep_structure
        group.vocal = rep_vocal
        group.folder_name = _build_folder_name(
            group.group_id,
            group.key,
            group.energy,
            group.vibe,
            group.structure,
            group.vocal,
            group.bpm,
        )
    return assignment


def save_assignment(assignment: GroupAssignment, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(assignment, f)


def load_assignment(path: str) -> GroupAssignment:
    with open(path, "rb") as f:
        assignment = pickle.load(f)
    for group in assignment.groups:
        group.key = _folder_key(group.key)
        group.vibe = normalize_mood_code(group.vibe) or "HYPN"
        group.vocal = normalize_vocal_profile(group.vocal) or "INST"
        group.folder_name = _build_folder_name(
            group.group_id,
            group.key,
            group.energy,
            group.vibe,
            group.structure,
            group.vocal,
            group.bpm,
        )
    return assignment


def _safe_mode(values: list, default):
    """Return the most common value, or default if empty."""
    if not values:
        return default
    try:
        return stat_mode(values)
    except Exception:
        return values[0]
