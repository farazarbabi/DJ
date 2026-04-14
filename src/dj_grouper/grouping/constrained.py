"""COP-KMedoids: constrained clustering with key/BPM as hard constraints."""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray

from ..config import GrouperConfig
from .constraints import Constraints
from .medoid import compute_medoid

logger = logging.getLogger(__name__)


def cop_kmedoids(
    distance_matrix: NDArray,
    constraints: Constraints,
    config: GrouperConfig,
) -> NDArray[np.intp]:
    """Run COP-KMedoids clustering with hard constraints.

    1. Search k +/- 3 around n/target_mid for best silhouette
    2. For best k: k-medoids++ init, iterative assign+update with constraints
    3. Post-process: split oversized, merge undersized

    Args:
        distance_matrix: (n, n) pairwise distance matrix
        constraints: cannot-link and must-link pairs
        config: GrouperConfig with group size params

    Returns:
        Label array (n,) with cluster assignments
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([0] * n, dtype=np.intp)

    t0 = time.perf_counter()
    target_mid = (config.target_group_size[0] + config.target_group_size[1]) / 2
    k_center = max(2, int(np.ceil(n / target_mid)))
    k_range = range(max(2, k_center - 3), k_center + 4)

    best_labels = None
    best_score = -float("inf")
    best_k = k_center

    for k in k_range:
        if k >= n:
            continue
        labels = _run_kmedoids(distance_matrix, k, constraints, config)
        score = _evaluate_labels(labels, distance_matrix, config)
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            best_k = k

    if best_labels is None:
        best_labels = _run_kmedoids(distance_matrix, k_center, constraints, config)

    # Post-process: split oversized, merge undersized
    best_labels = _post_process(best_labels, distance_matrix, constraints, config)

    n_groups = len(np.unique(best_labels))
    elapsed = time.perf_counter() - t0
    print(
        f"    COP-KMedoids: {n} tracks -> {n_groups} groups (k={best_k}, {elapsed:.1f}s)",
        flush=True,
    )
    return best_labels


def _run_kmedoids(
    dist: NDArray,
    k: int,
    constraints: Constraints,
    config: GrouperConfig,
    max_iterations: int = 50,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Single run of COP-KMedoids for a given k."""
    n = dist.shape[0]
    rng = np.random.RandomState(seed)

    # k-medoids++ initialization
    medoids = _kmedoids_init(dist, k, rng)

    labels = np.full(n, -1, dtype=np.intp)

    for iteration in range(max_iterations):
        new_labels = _assign_step(dist, medoids, constraints)

        # Update medoids — cluster count may have grown from constraint-driven splits
        actual_k = int(np.max(new_labels)) + 1
        new_medoids = _update_medoids(dist, new_labels, actual_k)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels
        medoids = new_medoids

    return labels


def _kmedoids_init(dist: NDArray, k: int, rng: np.random.RandomState) -> NDArray:
    """K-medoids++ initialization: spread-out medoid selection."""
    n = dist.shape[0]
    medoids = [rng.randint(n)]

    for _ in range(1, k):
        # Distance from each point to nearest existing medoid
        min_dists = np.min(dist[:, medoids], axis=1)
        # Sample proportional to squared distance
        probs = min_dists ** 2
        total = probs.sum()
        if total < 1e-10:
            # All points are identical; pick randomly
            idx = rng.randint(n)
        else:
            probs /= total
            idx = rng.choice(n, p=probs)
        medoids.append(int(idx))

    return np.array(medoids, dtype=np.intp)


def _assign_step(
    dist: NDArray,
    medoids: NDArray,
    constraints: Constraints,
) -> NDArray[np.intp]:
    """Assign each track to nearest medoid respecting constraints.

    Three-tier assignment:
      1. Strict: no hard or relaxed constraint violations (same key / +-1 neighbor)
      2. Relaxed: allow relaxed violations (+-2 neighbors), still reject hard
      3. New cluster: if no cluster passes even relaxed, create a new one
    """
    n = dist.shape[0]
    labels = np.full(n, -1, dtype=np.intp)

    # Mutable medoid list — grows when new clusters are created
    med_list: list[int] = list(medoids)

    # First assign medoids to their own clusters
    for cluster_id, med_idx in enumerate(med_list):
        labels[med_idx] = cluster_id

    # Build cluster membership sets for constraint checking
    cluster_members: list[set[int]] = [set() for _ in range(len(med_list))]
    for cluster_id, med_idx in enumerate(med_list):
        cluster_members[cluster_id].add(int(med_idx))

    # Sort non-medoid points by distance to nearest medoid (closest first)
    medoid_arr = np.array(med_list, dtype=np.intp)
    non_medoid_indices = [i for i in range(n) if i not in set(med_list)]
    if not non_medoid_indices:
        return labels
    min_dists = np.min(dist[non_medoid_indices][:, medoid_arr], axis=1)
    sorted_order = [non_medoid_indices[j] for j in np.argsort(min_dists)]

    for i in sorted_order:
        # Rank clusters by distance to their medoid
        cluster_dists = [(float(dist[i, med_list[c]]), c) for c in range(len(cluster_members))]
        cluster_dists.sort()

        # Tier 1: strict — no hard or relaxed violations
        assigned = False
        for _, cluster_id in cluster_dists:
            if _violates_hard(i, cluster_members[cluster_id], constraints):
                continue
            if _violates_relaxed(i, cluster_members[cluster_id], constraints):
                continue
            labels[i] = cluster_id
            cluster_members[cluster_id].add(i)
            assigned = True
            break

        if assigned:
            continue

        # Tier 2: relaxed — allow relaxed violations (key +-2), reject hard
        for _, cluster_id in cluster_dists:
            if _violates_hard(i, cluster_members[cluster_id], constraints):
                continue
            labels[i] = cluster_id
            cluster_members[cluster_id].add(i)
            assigned = True
            break

        if assigned:
            logger.debug("Track %d: assigned via relaxed fallback (key +-2 neighbor)", i)
            continue

        # Tier 3: no cluster works — create a new one (track becomes its own medoid)
        new_cluster_id = len(cluster_members)
        labels[i] = new_cluster_id
        cluster_members.append({i})
        med_list.append(i)
        logger.debug("Track %d: created new cluster %d (no compatible cluster)", i, new_cluster_id)

    return labels


def _violates_hard(
    track_idx: int,
    cluster_members: set[int],
    constraints: Constraints,
) -> bool:
    """Check if adding track_idx would violate any hard cannot-link."""
    for member in cluster_members:
        if constraints.is_cannot_link(track_idx, member):
            return True
    return False


def _violates_relaxed(
    track_idx: int,
    cluster_members: set[int],
    constraints: Constraints,
) -> bool:
    """Check if adding track_idx would violate any relaxed cannot-link."""
    for member in cluster_members:
        if constraints.is_relaxed_cannot_link(track_idx, member):
            return True
    return False


def _update_medoids(dist: NDArray, labels: NDArray, k: int) -> NDArray:
    """Recompute medoid for each cluster."""
    medoids = np.zeros(k, dtype=np.intp)
    for c in range(k):
        members = list(np.where(labels == c)[0])
        if members:
            medoids[c] = compute_medoid(dist, members)
        else:
            # Empty cluster — pick a random unassigned point or keep old
            medoids[c] = 0
    return medoids


def _evaluate_labels(
    labels: NDArray,
    dist: NDArray,
    config: GrouperConfig,
) -> float:
    """Score a labeling by silhouette-like metric + size penalty."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= len(labels):
        return -1.0

    # Simplified silhouette: average (b - a) / max(a, b)
    try:
        from sklearn.metrics import silhouette_score
        score = float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        score = 0.0

    # Penalize groups outside target size range
    min_target, max_target = config.target_group_size
    sizes = [int(np.sum(labels == lb)) for lb in unique_labels]
    size_penalty = sum(
        1.0 for s in sizes
        if s < config.min_group_size or s > config.max_group_size
    )
    score -= 0.1 * size_penalty

    return score


def _post_process(
    labels: NDArray,
    dist: NDArray,
    constraints: Constraints,
    config: GrouperConfig,
) -> NDArray[np.intp]:
    """Split oversized and merge undersized groups respecting constraints."""
    labels = labels.copy()

    # Split groups > max_group_size
    next_label = int(np.max(labels)) + 1
    for label in np.unique(labels):
        members = list(np.where(labels == label)[0])
        if len(members) > config.max_group_size:
            # Split via 2-medoids subclustering
            sub_dist = dist[np.ix_(members, members)]
            sub_constraints = _extract_sub_constraints(members, constraints)
            sub_labels = _run_kmedoids(sub_dist, 2, sub_constraints, config)
            for i, m in enumerate(members):
                if sub_labels[i] == 1:
                    labels[m] = next_label
            next_label += 1

    # Merge singletons / undersized groups into nearest compatible neighbor
    if config.min_group_size > 1:
        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if len(members) >= config.min_group_size:
                continue

            # Find nearest compatible group
            other_labels = [
                lb for lb in np.unique(labels)
                if lb != label and np.sum(labels == lb) >= config.min_group_size
            ]
            if not other_labels:
                continue

            best_target = None
            best_dist_val = float("inf")
            for target_label in other_labels:
                target_members = np.where(labels == target_label)[0]
                # Check constraints: would merging violate any cannot-link?
                violates = False
                for m in members:
                    for tm in target_members:
                        if constraints.is_cannot_link(int(m), int(tm)):
                            violates = True
                            break
                    if violates:
                        break
                if violates:
                    continue

                avg_d = float(np.mean(dist[np.ix_(members, target_members)]))
                if avg_d < best_dist_val:
                    best_dist_val = avg_d
                    best_target = target_label

            if best_target is not None:
                labels[members] = best_target

    # Renumber contiguously
    return _renumber(labels)


def _extract_sub_constraints(
    members: list[int],
    constraints: Constraints,
) -> Constraints:
    """Extract constraints relevant to a subset, re-indexed to local indices."""
    idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(members)}
    sub = Constraints()
    for i, j in constraints.cannot_link:
        if i in idx_map and j in idx_map:
            sub.cannot_link.add((idx_map[i], idx_map[j]))
    for i, j in constraints.relaxed_cannot_link:
        if i in idx_map and j in idx_map:
            sub.relaxed_cannot_link.add((idx_map[i], idx_map[j]))
    for i, j in constraints.must_link:
        if i in idx_map and j in idx_map:
            sub.must_link.add((idx_map[i], idx_map[j]))
    return sub


def _renumber(labels: NDArray) -> NDArray[np.intp]:
    """Re-number labels to be contiguous starting from 0."""
    unique = np.unique(labels)
    remap = {int(old): new for new, old in enumerate(unique)}
    return np.array([remap[int(lb)] for lb in labels], dtype=np.intp)
