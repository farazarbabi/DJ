"""Tests for COP-KMedoids constrained clustering."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.grouping.constraints import Constraints
from dj_grouper.grouping.constrained import cop_kmedoids
from dj_grouper.grouping.distance import unified_distance_matrix


def _make_clusterable_data(n=30, n_clusters=3, seed=42):
    """Create synthetic data with clear cluster structure."""
    rng = np.random.RandomState(seed)
    centers = rng.rand(n_clusters, 10).astype(np.float32)
    data = []
    true_labels = []
    per_cluster = n // n_clusters
    for c in range(n_clusters):
        for _ in range(per_cluster):
            point = centers[c] + rng.randn(10).astype(np.float32) * 0.1
            nrm = np.linalg.norm(point)
            if nrm > 1e-8:
                point /= nrm
            data.append(point)
            true_labels.append(c)
    return np.array(data, dtype=np.float32), np.array(true_labels)


def test_cop_kmedoids_convergence():
    """COP-KMedoids should converge and produce valid labels."""
    data, _ = _make_clusterable_data(n=24, n_clusters=3)
    dist = unified_distance_matrix(data)
    constraints = Constraints()
    config = GrouperConfig(target_group_size=(4, 12), min_group_size=2, max_group_size=20)

    labels = cop_kmedoids(dist, constraints, config)
    assert len(labels) == 24
    assert len(np.unique(labels)) >= 2


def test_cannot_link_respected():
    """No cannot-linked pair should end up in the same cluster."""
    data, _ = _make_clusterable_data(n=20, n_clusters=2)
    dist = unified_distance_matrix(data)

    constraints = Constraints()
    # Cannot-link between tracks 0 and 1 (both in cluster 0)
    constraints.cannot_link.add((0, 1))
    # Cannot-link between tracks 0 and 2
    constraints.cannot_link.add((0, 2))

    config = GrouperConfig(target_group_size=(4, 12), min_group_size=2, max_group_size=20)
    labels = cop_kmedoids(dist, constraints, config)

    # Check that cannot-linked pairs are in different clusters
    for i, j in constraints.cannot_link:
        assert labels[i] != labels[j], f"Cannot-linked pair ({i}, {j}) in same cluster {labels[i]}"


def test_single_track():
    dist = np.zeros((1, 1), dtype=np.float32)
    constraints = Constraints()
    config = GrouperConfig()
    labels = cop_kmedoids(dist, constraints, config)
    assert len(labels) == 1
    assert labels[0] == 0


def test_small_group_produces_labels():
    """Small input should still produce valid labels."""
    dist = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float32)
    constraints = Constraints()
    config = GrouperConfig(target_group_size=(2, 4), min_group_size=1)
    labels = cop_kmedoids(dist, constraints, config)
    assert len(labels) == 2
    # Should produce valid labels (0 or 1)
    assert all(lb >= 0 for lb in labels)


def test_two_tracks_cannot_link():
    dist = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float32)
    constraints = Constraints()
    constraints.cannot_link.add((0, 1))
    config = GrouperConfig(target_group_size=(1, 4), min_group_size=1)
    labels = cop_kmedoids(dist, constraints, config)
    assert labels[0] != labels[1]


def test_unified_distance_matrix_shape():
    data = np.random.rand(10, 5).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    data /= norms

    dist = unified_distance_matrix(data)
    assert dist.shape == (10, 10)
    np.testing.assert_array_almost_equal(dist, dist.T)
    assert dist[0, 0] == 0.0


def test_unified_distance_matrix_values_bounded():
    data = np.random.rand(10, 5).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    data /= norms

    dist = unified_distance_matrix(data)
    assert np.all(dist >= 0.0)
    assert np.all(dist <= 1.0)


def test_many_cannot_links_still_converges():
    """Even with many constraints, clustering should produce valid output."""
    n = 20
    data, _ = _make_clusterable_data(n=n, n_clusters=4)
    dist = unified_distance_matrix(data)

    constraints = Constraints()
    # Add cannot-links between every pair in different halves
    for i in range(n // 2):
        for j in range(n // 2, n):
            constraints.cannot_link.add((i, j))

    config = GrouperConfig(target_group_size=(3, 10), min_group_size=2, max_group_size=15)
    labels = cop_kmedoids(dist, constraints, config)
    assert len(labels) == n
    # Should produce at least 2 groups
    assert len(np.unique(labels)) >= 2
