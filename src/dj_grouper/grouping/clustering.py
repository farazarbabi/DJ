"""Agglomerative clustering with size constraints and BPM validation."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..config import GrouperConfig

logger = logging.getLogger(__name__)


def cluster_tracks(
    distance_matrix: NDArray,
    config: GrouperConfig,
    bpms: list[int | None] | None = None,
    energies: list[int | None] | None = None,
) -> NDArray[np.intp]:
    """Run agglomerative clustering and return labels.

    Args:
        distance_matrix: pairwise distance matrix
        config: GrouperConfig
        bpms: optional list of BPM values per track (for BPM validation)
        energies: optional list of energy levels per track (for energy validation)
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([0] * n, dtype=np.intp)

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=config.linkage)

    labels = _auto_threshold(Z, n, config)
    labels = _post_process(labels, distance_matrix, config)

    # BPM validation: split groups with too-wide BPM spread
    if bpms is not None:
        labels = _split_bpm_spread(labels, bpms, distance_matrix, config)

    # Energy validation: split groups with too-wide energy spread
    if energies is not None:
        labels = _split_energy_spread(labels, energies, distance_matrix, config)

    n_groups = len(np.unique(labels))
    sizes = [int(np.sum(labels == l)) for l in np.unique(labels)]
    logger.info(
        "Clustering: %d tracks -> %d groups (sizes: %s)",
        n, n_groups, sorted(sizes, reverse=True),
    )
    return labels


def _auto_threshold(Z: NDArray, n: int, config: GrouperConfig) -> NDArray:
    """Find a distance threshold producing groups in the target size range."""
    min_target, max_target = config.target_group_size
    distances_in_Z = Z[:, 2]
    lo, hi = float(np.min(distances_in_Z)), float(np.max(distances_in_Z))

    best_labels = None
    best_score = float("inf")

    for frac in np.linspace(0.1, 0.9, 40):
        threshold = lo + frac * (hi - lo)
        labels = fcluster(Z, t=threshold, criterion="distance") - 1

        unique, counts = np.unique(labels, return_counts=True)
        median_size = float(np.median(counts))

        target_mid = (min_target + max_target) / 2
        score = abs(median_size - target_mid)
        n_outside = sum(1 for c in counts if c > config.max_group_size)
        score += n_outside * 2

        if score < best_score:
            best_score = score
            best_labels = labels.copy()

    if best_labels is None:
        threshold = lo + 0.5 * (hi - lo)
        best_labels = fcluster(Z, t=threshold, criterion="distance") - 1

    return best_labels


def _post_process(
    labels: NDArray,
    distance_matrix: NDArray,
    config: GrouperConfig,
) -> NDArray:
    """Split oversized groups and merge undersized groups."""
    labels = labels.copy()

    # Merge groups smaller than min_group_size into nearest neighbor group
    if config.min_group_size > 1:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            members = np.where(labels == label)[0]
            if len(members) < config.min_group_size:
                other_labels = [l for l in unique_labels if l != label and np.sum(labels == l) >= config.min_group_size]
                if not other_labels:
                    continue
                best_target = None
                best_dist = float("inf")
                for target_label in other_labels:
                    target_members = np.where(labels == target_label)[0]
                    avg_dist = float(np.mean(distance_matrix[np.ix_(members, target_members)]))
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_target = target_label
                if best_target is not None:
                    labels[members] = best_target

    # Split groups larger than max_group_size
    unique_labels = np.unique(labels)
    next_label = int(np.max(labels)) + 1
    for label in unique_labels:
        members = np.where(labels == label)[0]
        if len(members) > config.max_group_size:
            sub_matrix = distance_matrix[np.ix_(members, members)]
            if sub_matrix.shape[0] > 2:
                condensed = squareform(sub_matrix, checks=False)
                Z_sub = linkage(condensed, method=config.linkage)
                sub_labels = fcluster(Z_sub, t=2, criterion="maxclust") - 1
                for i, m in enumerate(members):
                    if sub_labels[i] == 1:
                        labels[m] = next_label
                next_label += 1

    # Re-number labels to be contiguous
    labels = _renumber(labels)
    return labels


def _split_bpm_spread(
    labels: NDArray,
    bpms: list[int | None],
    distance_matrix: NDArray,
    config: GrouperConfig,
) -> NDArray:
    """Split groups where BPM spread exceeds the configured maximum.

    This is the hard safety net: no matter what the distance function does,
    tracks with unmixable BPM differences cannot share a group.
    """
    labels = labels.copy()
    max_spread = config.bpm_group_max_spread_pct
    changed = True

    while changed:
        changed = False
        next_label = int(np.max(labels)) + 1

        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if len(members) <= 1:
                continue

            member_bpms = [bpms[i] or 128 for i in members]
            median_bpm = float(np.median(member_bpms))
            if median_bpm == 0:
                continue

            min_bpm = min(member_bpms)
            max_bpm = max(member_bpms)
            spread = (max_bpm - min_bpm) / median_bpm

            if spread > max_spread:
                # Split: re-cluster this subgroup into 2
                sub_matrix = distance_matrix[np.ix_(members, members)]
                if sub_matrix.shape[0] > 2:
                    condensed = squareform(sub_matrix, checks=False)
                    Z_sub = linkage(condensed, method=config.linkage)
                    sub_labels = fcluster(Z_sub, t=2, criterion="maxclust") - 1
                    for i, m in enumerate(members):
                        if sub_labels[i] == 1:
                            labels[m] = next_label
                    next_label += 1
                    changed = True
                    logger.info(
                        "Split group (BPM spread %.1f%%): %d-%d BPM",
                        spread * 100, min_bpm, max_bpm,
                    )

    labels = _renumber(labels)
    return labels


def _split_energy_spread(
    labels: NDArray,
    energies: list[int | None],
    distance_matrix: NDArray,
    config: GrouperConfig,
) -> NDArray:
    """Split groups where energy spread exceeds the configured maximum."""
    labels = labels.copy()
    max_spread = config.energy_group_max_spread
    changed = True

    while changed:
        changed = False
        next_label = int(np.max(labels)) + 1

        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if len(members) <= 1:
                continue

            member_energies = [energies[i] or 3 for i in members]
            min_e = min(member_energies)
            max_e = max(member_energies)

            if max_e - min_e > max_spread:
                sub_matrix = distance_matrix[np.ix_(members, members)]
                if sub_matrix.shape[0] > 2:
                    condensed = squareform(sub_matrix, checks=False)
                    Z_sub = linkage(condensed, method=config.linkage)
                    sub_labels = fcluster(Z_sub, t=2, criterion="maxclust") - 1
                    for i, m in enumerate(members):
                        if sub_labels[i] == 1:
                            labels[m] = next_label
                    next_label += 1
                    changed = True
                    logger.info(
                        "Split group (energy spread E%d-E%d)", min_e, max_e,
                    )

    labels = _renumber(labels)
    return labels


def _renumber(labels: NDArray) -> NDArray:
    """Re-number labels to be contiguous starting from 0."""
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels], dtype=np.intp)
