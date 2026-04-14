"""Apply feedback to distance matrices and group assignments."""

from __future__ import annotations

import logging
from pathlib import Path

from numpy.typing import NDArray

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from .store import FeedbackEntry

logger = logging.getLogger(__name__)


def apply_feedback_to_distances(
    distance_matrix: NDArray,
    tracks: list[TrackFeatures],
    feedback: list[FeedbackEntry],
    config: GrouperConfig,
) -> NDArray:
    """Modify distance matrix based on feedback entries."""
    if not feedback:
        return distance_matrix

    modified = distance_matrix.copy()

    # Build path-to-index lookup (match by filename)
    name_to_idx: dict[str, int] = {}
    for i, tf in enumerate(tracks):
        name_to_idx[Path(tf.path).name] = i
        name_to_idx[tf.path] = i

    applied = 0
    for entry in feedback:
        if entry.type not in ("good_pair", "bad_pair"):
            continue

        idx_a = name_to_idx.get(entry.track_a)
        idx_b = name_to_idx.get(entry.track_b)
        if idx_a is None or idx_b is None:
            logger.debug(
                "Feedback entry skipped (track not found): %s <-> %s",
                entry.track_a, entry.track_b,
            )
            continue

        strength = float(entry.strength)
        if entry.type == "good_pair":
            factor = 1.0 - config.good_pair_factor * strength
        else:  # bad_pair
            factor = 1.0 + config.bad_pair_factor * strength

        modified[idx_a, idx_b] *= factor
        modified[idx_b, idx_a] *= factor
        applied += 1

    logger.info("Applied %d feedback entries to distance matrix", applied)
    return modified


def get_group_overrides(
    tracks: list[TrackFeatures],
    feedback: list[FeedbackEntry],
) -> dict[int, str]:
    """Extract group override assignments from feedback.

    Returns: dict mapping track_index -> forced_group_id
    """
    name_to_idx: dict[str, int] = {}
    for i, tf in enumerate(tracks):
        name_to_idx[Path(tf.path).name] = i
        name_to_idx[tf.path] = i

    overrides: dict[int, str] = {}
    for entry in feedback:
        if entry.type != "group_override":
            continue
        idx = name_to_idx.get(entry.track_a)
        if idx is not None:
            overrides[idx] = entry.strength  # strength field stores group_id for overrides

    return overrides
