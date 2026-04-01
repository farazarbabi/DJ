"""Agglomerative clustering with size constraints."""

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
) -> NDArray[np.intp]:
    """Run agglomerative clustering and return labels.

    Uses distance threshold auto-tuning to hit the target group size range.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([0] * n, dtype=np.intp)

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix, checks=False)

    Z = linkage(condensed, method=config.linkage)

    # Auto-tune distance threshold to hit target group sizes
    labels = _auto_threshold(Z, n, config)

    # Post-process: split oversized, merge undersized
    labels = _post_process(labels, distance_matrix, config)

    n_groups = len(np.unique(labels))
    sizes = [int(np.sum(labels == l)) for l in np.unique(labels)]
    logger.info(
        "Clustering: %d tracks -> %d groups (sizes: %s)",
        n, n_groups, sorted(sizes, reverse=True),
    )
    return labels


def _auto_threshold(Z: NDArray, n: int, config: GrouperConfig) -> NDArray:
    """Find a distance threshold that produces groups in the target size range."""
    min_target, max_target = config.target_group_size

    # Try a range of thresholds
    distances_in_Z = Z[:, 2]
    lo, hi = float(np.min(distances_in_Z)), float(np.max(distances_in_Z))

    best_labels = None
    best_score = float("inf")

    for frac in np.linspace(0.2, 0.9, 30):
        threshold = lo + frac * (hi - lo)
        labels = fcluster(Z, t=threshold, criterion="distance") - 1  # 0-based

        unique, counts = np.unique(labels, return_counts=True)
        median_size = float(np.median(counts))

        # Score: penalize deviation from target range midpoint
        target_mid = (min_target + max_target) / 2
        score = abs(median_size - target_mid)

        # Also penalize having many groups outside target range
        n_outside = sum(1 for c in counts if c < config.min_group_size or c > config.max_group_size)
        score += n_outside * 2

        if score < best_score:
            best_score = score
            best_labels = labels.copy()

    if best_labels is None:
        # Fallback: just use midpoint threshold
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
    unique_labels = np.unique(labels)
    for label in unique_labels:
        members = np.where(labels == label)[0]
        if len(members) < config.min_group_size:
            # Find nearest other group by average distance
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
            # Bisect by re-clustering the subgroup
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
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.intp)

    return labels
