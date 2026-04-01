"""Medoid computation for clusters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_medoid(distance_matrix: NDArray, indices: list[int]) -> int:
    """Find the medoid (most central member) of a cluster.

    Args:
        distance_matrix: full pairwise distance matrix
        indices: indices of tracks in this cluster

    Returns:
        The index (in the full matrix) of the medoid track.
    """
    if len(indices) == 1:
        return indices[0]

    best_idx = indices[0]
    best_total = float("inf")

    for i in indices:
        total = sum(float(distance_matrix[i, j]) for j in indices if j != i)
        if total < best_total:
            best_total = total
            best_idx = i

    return best_idx


def compute_all_medoids(
    distance_matrix: NDArray,
    labels: NDArray,
) -> dict[int, int]:
    """Compute medoid for each cluster label.

    Returns:
        Dict mapping cluster_label -> track_index of medoid.
    """
    medoids: dict[int, int] = {}
    for label in np.unique(labels):
        indices = list(np.where(labels == label)[0])
        medoids[int(label)] = compute_medoid(distance_matrix, indices)
    return medoids
