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
    energy: int
    vibe: str
    bpm: int
    structure: str
    vocal: str
    folder_name: str


@dataclass
class GroupAssignment:
    """Full assignment state for the library."""
    groups: list[GroupInfo] = field(default_factory=list)
    track_to_group: dict[str, str] = field(default_factory=dict)  # path -> group_id


def assign_group_ids(
    tracks: list[TrackFeatures],
    labels: NDArray,
    distance_matrix: NDArray,
) -> GroupAssignment:
    """Create stable group IDs and descriptors from clustering labels."""
    medoids = compute_all_medoids(distance_matrix, labels)
    assignment = GroupAssignment()

    for label in sorted(np.unique(labels)):
        members = list(np.where(labels == label)[0])
        group_id = f"G{label + 1:03d}"

        # Compute representative descriptors
        infos = [tracks[i].info for i in members]
        energies = [i.energy for i in infos if i.energy is not None]
        vibes = [i.vibe for i in infos if i.vibe]
        bpms = [i.bpm for i in infos if i.bpm is not None]
        structures = [i.structure for i in infos if i.structure]
        vocals = [i.vocal for i in infos if i.vocal]

        rep_energy = _safe_mode(energies, 3)
        rep_vibe = _safe_mode(vibes, "HYPN")
        rep_bpm = round(np.median(bpms)) if bpms else 128
        rep_structure = _safe_mode(structures, "32H")
        rep_vocal = _safe_mode(vocals, "NV")

        # Abbreviate vibe for folder name
        vibe_abbr = rep_vibe[:3] if len(rep_vibe) >= 3 else rep_vibe

        folder_name = f"{group_id}_E{rep_energy}{vibe_abbr}_{rep_bpm}_{rep_structure}"

        group = GroupInfo(
            group_id=group_id,
            member_indices=members,
            medoid_index=medoids[int(label)],
            energy=rep_energy,
            vibe=rep_vibe,
            bpm=rep_bpm,
            structure=rep_structure,
            vocal=rep_vocal,
            folder_name=folder_name,
        )
        assignment.groups.append(group)

        for idx in members:
            assignment.track_to_group[tracks[idx].path] = group_id

    logger.info("Assigned %d groups", len(assignment.groups))
    return assignment


def assign_new_track(
    new_track: TrackFeatures,
    existing_tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    config: GrouperConfig,
) -> str:
    """Assign a new track to the nearest existing group via nearest-medoid.

    Returns the group_id, or creates a new group if too distant.
    """
    best_group = None
    best_dist = float("inf")

    for group in assignment.groups:
        medoid_track = existing_tracks[group.medoid_index]
        d = blended_distance(new_track, medoid_track, config)
        if d < best_dist:
            best_dist = d
            best_group = group

    if best_dist > config.new_group_distance_threshold or best_group is None:
        # Create new group
        next_id = max(int(g.group_id[1:]) for g in assignment.groups) + 1 if assignment.groups else 1
        new_gid = f"G{next_id:03d}"
        logger.info("New track too distant (%.3f), creating group %s", best_dist, new_gid)
        return new_gid

    logger.debug("Assigned new track to %s (distance=%.3f)", best_group.group_id, best_dist)
    return best_group.group_id


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
