"""Tests for clustering."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.grouping.clustering import cluster_tracks


def test_cluster_two_distinct_groups():
    """Two clear clusters should be separated."""
    # Group A: indices 0-4 are close to each other
    # Group B: indices 5-9 are close to each other
    n = 10
    matrix = np.ones((n, n), dtype=np.float32) * 0.8
    for i in range(5):
        for j in range(5):
            matrix[i, j] = 0.1 if i != j else 0.0
    for i in range(5, 10):
        for j in range(5, 10):
            matrix[i, j] = 0.1 if i != j else 0.0

    config = GrouperConfig(min_group_size=2, max_group_size=8)
    labels = cluster_tracks(matrix, config)

    # Should produce exactly 2 groups
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2

    # Members 0-4 should share a label, 5-9 should share a different label
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_cluster_single_track():
    matrix = np.zeros((1, 1), dtype=np.float32)
    config = GrouperConfig()
    labels = cluster_tracks(matrix, config)
    assert len(labels) == 1
