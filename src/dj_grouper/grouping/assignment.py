"""Stable group ID assignment and new-track assignment."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mode as stat_mode

import numpy as np
from numpy.typing import NDArray

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from .distance import blended_distance
from .medoid import compute_all_medoids

logger = logging.getLogger(__name__)


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
    if not key or key == "??":
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

        infos = [tracks[i].info for i in members]
        keys = [i.key for i in infos if i.key]
        energies = [i.energy for i in infos if i.energy is not None]
        vibes = [i.vibe for i in infos if i.vibe]
        bpms = [i.bpm for i in infos if i.bpm is not None]
        structures = [i.structure for i in infos if i.structure]
        vocals = [i.vocal for i in infos if i.vocal]

        rep_key = _safe_mode(keys, "??")
        rep_energy = _safe_mode(energies, 3)
        rep_vibe = _safe_mode(vibes, "HYPN")
        rep_bpm = round(np.median(bpms)) if bpms else 128
        rep_structure = _safe_mode(structures, "32H")
        rep_vocal = _safe_mode(vocals, "NV")

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
        group.folder_name = (
            f"{group.group_id}_{group.key}_E{group.energy}_{group.vibe}"
            f"_{group.structure}_{group.vocal}_{group.bpm}"
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
        return existing_assignment

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
            rep_key = info.key or "??"
            rep_energy = info.energy or 3
            rep_vibe = info.vibe or "HYPN"
            rep_bpm = info.bpm or 128
            rep_structure = info.structure or "32H"
            rep_vocal = info.vocal or "NV"
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
                folder_name=f"{new_gid}_{rep_key}_E{rep_energy}_{rep_vibe}_{rep_structure}_{rep_vocal}_{rep_bpm}",
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

    logger.info(
        "Incremental update: %d assigned to existing, %d new groups, %d deleted",
        n_assigned, n_new_groups, len(deleted_paths),
    )
    return assignment


def save_assignment(assignment: GroupAssignment, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(assignment, f)


def load_assignment(path: str) -> GroupAssignment:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mode(values: list, default):
    """Return the most common value, or default if empty."""
    if not values:
        return default
    try:
        return stat_mode(values)
    except Exception:
        return values[0]
