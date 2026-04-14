"""Agglomerative clustering with size constraints and BPM validation."""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..config import GrouperConfig

logger = logging.getLogger(__name__)


def camelot_distance(key_a: str, key_b: str) -> int:
    """Camelot wheel distance. 0 = same, 1 = compatible, 2+ = incompatible."""
    if key_a == key_b:
        return 0
    num_a, letter_a = int(key_a[:-1]), key_a[-1]
    num_b, letter_b = int(key_b[:-1]), key_b[-1]
    num_diff = abs(num_a - num_b)
    num_diff = min(num_diff, 12 - num_diff)  # circular
    if letter_a == letter_b:
        return num_diff
    # different letter: relative major/minor costs 1, plus any number offset
    return num_diff + 1


def cluster_tracks(
    distance_matrix: NDArray,
    config: GrouperConfig,
    bpms: list[int | None] | None = None,
    energies: list[int | None] | None = None,
    keys: list[str | None] | None = None,
) -> NDArray[np.intp]:
    """Run agglomerative clustering and return labels.

    Args:
        distance_matrix: pairwise distance matrix
        config: GrouperConfig
        bpms: optional list of BPM values per track (for BPM validation)
        energies: optional list of energy levels per track (for energy validation)
        keys: optional list of Camelot keys per track (for key validation)
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([0] * n, dtype=np.intp)

    t0 = time.perf_counter()
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=config.linkage)

    labels = _auto_threshold(Z, n, config)
    n_initial = len(np.unique(labels))
    labels = _post_process(labels, distance_matrix, config)
    print(f"    Agglomerative clustering: {n} tracks -> {n_initial} initial groups", flush=True)

    # Key validation: split groups with incompatible keys (Camelot distance > 1)
    if keys is not None:
        labels = _split_key_incompatible(labels, keys)

    # BPM validation: split groups with too-wide BPM spread
    if bpms is not None:
        n_before = len(np.unique(labels))
        labels = _split_bpm_spread(labels, bpms, distance_matrix, config)
        n_after = len(np.unique(labels))
        if n_after > n_before:
            print(f"    BPM validation: {n_after - n_before} splits", flush=True)

    # Energy validation: split groups with too-wide energy spread
    if energies is not None:
        n_before = len(np.unique(labels))
        labels = _split_energy_spread(labels, energies, distance_matrix, config)
        n_after = len(np.unique(labels))
        if n_after > n_before:
            print(f"    Energy validation: {n_after - n_before} splits", flush=True)

    # Re-merge singletons created by validation splitting
    n_before = len(np.unique(labels))
    labels = _merge_small_groups(labels, distance_matrix, config, bpms, energies, keys)
    n_after = len(np.unique(labels))
    if n_before > n_after:
        print(f"    Re-merged {n_before - n_after} small groups", flush=True)

    n_groups = len(np.unique(labels))
    elapsed = time.perf_counter() - t0
    t_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed)//60}m{int(elapsed)%60:02d}s"
    print(f"    Final: {n_groups} groups ({t_str})", flush=True)
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
        n_small = sum(1 for c in counts if c < config.min_group_size)
        score += n_small * 1.5

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
                other_labels = [lb for lb in unique_labels if lb != label and np.sum(labels == lb) >= config.min_group_size]
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


def _split_key_incompatible(
    labels: NDArray,
    keys: list[str | None],
) -> NDArray:
    """Split groups where keys are not harmonically compatible.

    Each group must have a center key such that all tracks with known keys
    are within Camelot distance <= 1 of that center. Tracks with unknown
    keys ("??" or None) float freely and are never split.
    """
    labels = labels.copy()
    changed = True
    n_splits = 0

    while changed:
        changed = False
        next_label = int(np.max(labels)) + 1

        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if len(members) <= 1:
                continue

            # Collect known keys
            member_keys = [(i, keys[idx]) for i, idx in enumerate(members)
                           if keys[idx] and keys[idx] != "??"]
            if len(member_keys) < 2:
                continue

            # Find the center key that maximizes compatible tracks
            key_values = [k for _, k in member_keys]
            best_center = None
            best_compat = 0
            for center in set(key_values):
                compat = sum(1 for k in key_values if camelot_distance(center, k) <= 1)
                if compat > best_compat:
                    best_compat = compat
                    best_center = center

            # Split off tracks incompatible with the center
            incompatible = [
                members[i] for i, k in member_keys
                if camelot_distance(best_center, k) > 1
            ]

            if incompatible:
                for idx in incompatible:
                    labels[idx] = next_label
                next_label += 1
                changed = True
                n_splits += 1

    if n_splits:
        logger.info("Key validation: %d splits (Camelot distance > 1)", n_splits)
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
    n_splits = 0
    iteration = 0
    max_iterations = 200

    while changed and iteration < max_iterations:
        iteration += 1
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
                sub_matrix = distance_matrix[np.ix_(members, members)]
                if sub_matrix.shape[0] > 2:
                    # Check for degenerate submatrix (NaN or all-zero)
                    if np.any(np.isnan(sub_matrix)):
                        logger.warning("BPM split: NaN in submatrix for group %d, skipping", label)
                        continue
                    condensed = squareform(sub_matrix, checks=False)
                    Z_sub = linkage(condensed, method=config.linkage)
                    sub_labels = fcluster(Z_sub, t=2, criterion="maxclust") - 1
                    moved = sum(1 for s in sub_labels if s == 1)
                    if moved == 0 or moved == len(members):
                        # fcluster didn't actually split — bail to avoid infinite loop
                        continue
                    for i, m in enumerate(members):
                        if sub_labels[i] == 1:
                            labels[m] = next_label
                    next_label += 1
                    changed = True
                    n_splits += 1
                    logger.debug(
                        "BPM split iter %d: group %d (%d members, spread %.1f%%) -> %d/%d",
                        iteration, label, len(members), spread * 100, len(members) - moved, moved,
                    )

    if iteration >= max_iterations:
        logger.warning("BPM validation hit iteration limit (%d)", max_iterations)
    if n_splits:
        logger.info("BPM validation: %d splits in %d iterations (max spread %.0f%%)", n_splits, iteration, max_spread * 100)
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
    n_splits = 0
    iteration = 0
    max_iterations = 200

    while changed and iteration < max_iterations:
        iteration += 1
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
                    if np.any(np.isnan(sub_matrix)):
                        logger.warning("Energy split: NaN in submatrix for group %d, skipping", label)
                        continue
                    condensed = squareform(sub_matrix, checks=False)
                    Z_sub = linkage(condensed, method=config.linkage)
                    sub_labels = fcluster(Z_sub, t=2, criterion="maxclust") - 1
                    moved = sum(1 for s in sub_labels if s == 1)
                    if moved == 0 or moved == len(members):
                        continue
                    for i, m in enumerate(members):
                        if sub_labels[i] == 1:
                            labels[m] = next_label
                    next_label += 1
                    changed = True
                    n_splits += 1

    if iteration >= max_iterations:
        logger.warning("Energy validation hit iteration limit (%d)", max_iterations)
    if n_splits:
        logger.info("Energy validation: %d splits in %d iterations (max spread %d levels)", n_splits, iteration, max_spread)
    labels = _renumber(labels)
    return labels


def _merge_small_groups(
    labels: NDArray,
    distance_matrix: NDArray,
    config: GrouperConfig,
    bpms: list[int | None] | None = None,
    energies: list[int | None] | None = None,
    keys: list[str | None] | None = None,
) -> NDArray:
    """Re-merge groups below min_group_size after validation splitting.

    Unlike the initial merge in _post_process, this checks BPM, energy,
    and key constraints before merging so we don't undo validation splits.
    """
    if config.min_group_size <= 1:
        return labels

    labels = labels.copy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        members = np.where(labels == label)[0]
        if len(members) >= config.min_group_size:
            continue

        # Find candidate groups that already meet min_group_size
        other_labels = [
            lb for lb in np.unique(labels)
            if lb != label and np.sum(labels == lb) >= config.min_group_size
        ]
        if not other_labels:
            continue

        # Rank candidates by average distance
        candidates = []
        for target_label in other_labels:
            target_members = np.where(labels == target_label)[0]
            avg_dist = float(np.mean(distance_matrix[np.ix_(members, target_members)]))
            candidates.append((avg_dist, target_label, target_members))
        candidates.sort()

        for avg_dist, target_label, target_members in candidates:
            # Check BPM constraint
            if bpms is not None:
                merged_bpms = [bpms[i] or 128 for i in list(members) + list(target_members)]
                median_bpm = float(np.median(merged_bpms))
                if median_bpm > 0:
                    spread = (max(merged_bpms) - min(merged_bpms)) / median_bpm
                    if spread > config.bpm_group_max_spread_pct:
                        continue

            # Check energy constraint
            if energies is not None:
                merged_energies = [energies[i] or 3 for i in list(members) + list(target_members)]
                if max(merged_energies) - min(merged_energies) > config.energy_group_max_spread:
                    continue

            # Check key constraint: all known keys must be within Camelot distance 1
            if keys is not None:
                all_indices = list(members) + list(target_members)
                known_keys = [keys[i] for i in all_indices if keys[i] and keys[i] != "??"]
                if len(known_keys) >= 2:
                    # Find best center for merged group
                    best_center = max(set(known_keys), key=lambda c: sum(
                        1 for k in known_keys if camelot_distance(c, k) <= 1
                    ))
                    if any(camelot_distance(best_center, k) > 1 for k in known_keys):
                        continue

            # Merge
            labels[members] = target_label
            logger.info(
                "Re-merged %d singleton(s) into group (avg dist %.3f)",
                len(members), avg_dist,
            )
            break

    return _renumber(labels)


def _renumber(labels: NDArray) -> NDArray:
    """Re-number labels to be contiguous starting from 0."""
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[lb] for lb in labels], dtype=np.intp)
