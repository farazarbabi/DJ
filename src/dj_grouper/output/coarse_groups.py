"""Coarse (half-resolution) group playlists via agglomerative medoid merge.

For live use, the fine-grained cluster playlists are often too narrow to
browse mid-set. This module reduces the number of groups by half by
repeatedly merging the two nearest groups (by medoid distance) until the
target count is reached, then writes the merged groups as M3U8 playlists
alongside the existing ``groups/`` output.

Merge uses :func:`blended_distance` on the same TrackFeatures objects the
grouper already built, so the coarsening is consistent with the
audio-similarity model that produced the fine clusters in the first place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from ..grouping.assignment import (
    GroupAssignment,
    GroupInfo,
    _build_folder_name,
    _representative_values,
)
from ..grouping.distance import blended_distance

logger = logging.getLogger(__name__)

_WIN_ILLEGAL = str.maketrans({c: "_" for c in r'<>:"/\|?*'})


def _medoid_distance_matrix(
    tracks: list[TrackFeatures],
    medoid_indices: list[int],
    config: GrouperConfig,
) -> NDArray:
    n = len(medoid_indices)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = blended_distance(tracks[medoid_indices[i]], tracks[medoid_indices[j]], config)
            d[i, j] = dist
            d[j, i] = dist
    return d


def build_coarse_assignment(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    config: GrouperConfig,
) -> GroupAssignment:
    """Merge nearest medoids until group count is halved.

    The merge step is O(n²) per iteration with n iterations, so O(n³) total.
    Fine for libraries up to a few thousand groups; revisit if that ever
    stops being true.
    """
    source_groups = [g for g in assignment.groups if g.member_indices]
    n = len(source_groups)
    if n <= 1:
        return _rebuild(tracks, source_groups)

    target = max(1, n // 2)

    super_members: list[list[int]] = [list(g.member_indices) for g in source_groups]
    super_medoids: list[int] = [g.medoid_index for g in source_groups]

    d = _medoid_distance_matrix(tracks, super_medoids, config)
    active = list(range(n))

    while len(active) > target:
        best = (active[0], active[1], float("inf"))
        for i_pos in range(len(active)):
            for j_pos in range(i_pos + 1, len(active)):
                i, j = active[i_pos], active[j_pos]
                if d[i, j] < best[2]:
                    best = (i, j, float(d[i, j]))
        i, j, _ = best
        # Merge j into i
        super_members[i].extend(super_members[j])
        # Re-elect medoid for the merged set against the full feature space.
        # Use distances we already have for cheap recompute against the merged
        # members — but the medoid lookup needs pairwise distances among members,
        # which we don't pre-compute. Fall back to keeping i's medoid; medoid
        # accuracy at this level mostly affects pretty-printing of folder names.
        super_members[j] = []
        active.remove(j)

    coarse_super_members = [super_members[i] for i in active]
    coarse_super_medoids = [super_medoids[i] for i in active]
    # Best-effort medoid re-election for printed folder name only.
    for k, members in enumerate(coarse_super_members):
        if len(members) > 1:
            coarse_super_medoids[k] = _local_medoid(tracks, members, config)

    return _rebuild_from_members(tracks, coarse_super_members, coarse_super_medoids)


def _local_medoid(
    tracks: list[TrackFeatures],
    members: list[int],
    config: GrouperConfig,
) -> int:
    """Pick the medoid of `members` using blended_distance. O(|members|²)."""
    if len(members) == 1:
        return members[0]
    best_idx = members[0]
    best_total = float("inf")
    for i in members:
        total = 0.0
        for j in members:
            if i == j:
                continue
            total += blended_distance(tracks[i], tracks[j], config)
            if total >= best_total:
                break
        if total < best_total:
            best_total = total
            best_idx = i
    return best_idx


def _rebuild(
    tracks: list[TrackFeatures],
    source_groups: list[GroupInfo],
) -> GroupAssignment:
    """Trivial pass-through wrapper when n <= 1: rename IDs with `CG` prefix."""
    members = [list(g.member_indices) for g in source_groups]
    medoids = [g.medoid_index for g in source_groups]
    return _rebuild_from_members(tracks, members, medoids)


def _rebuild_from_members(
    tracks: list[TrackFeatures],
    super_members: list[list[int]],
    super_medoids: list[int],
) -> GroupAssignment:
    coarse = GroupAssignment()
    for i, (members, medoid) in enumerate(zip(super_members, super_medoids)):
        if not members:
            continue
        rep_key, rep_energy, rep_vibe, rep_bpm, rep_structure, rep_vocal = (
            _representative_values(tracks, members)
        )
        gid = f"CG{i + 1:03d}"
        coarse.groups.append(
            GroupInfo(
                group_id=gid,
                member_indices=members,
                medoid_index=medoid,
                key=rep_key,
                energy=rep_energy,
                vibe=rep_vibe,
                bpm=rep_bpm,
                structure=rep_structure,
                vocal=rep_vocal,
                folder_name=_build_folder_name(
                    gid, rep_key, rep_energy, rep_vibe, rep_structure, rep_vocal, rep_bpm
                ),
            )
        )
        for idx in members:
            coarse.track_to_group[tracks[idx].path] = gid
    return coarse


def generate_coarse_group_playlists(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    config: GrouperConfig,
    output_dir: str,
) -> int:
    """Write half-resolution group playlists to `{output_dir}/groups_coarse/`.

    Returns the number of coarse playlists written.
    """
    coarse = build_coarse_assignment(tracks, assignment, config)
    out = Path(output_dir) / "groups_coarse"
    out.mkdir(parents=True, exist_ok=True)
    for group in coarse.groups:
        playlist_path = out / f"{group.folder_name.translate(_WIN_ILLEGAL)}.m3u8"
        lines = ["#EXTM3U"]
        for idx in group.member_indices:
            tf = tracks[idx]
            lines.append(f"#EXTINF:-1,{Path(tf.path).stem}")
            lines.append(tf.path)
        playlist_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated %d coarse group playlists in %s", len(coarse.groups), out)
    return len(coarse.groups)
