"""Build cannot-link and must-link constraint sets from track metadata and feedback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..features.builder import TrackFeatures
from ..config import GrouperConfig
from .clustering import camelot_distance

logger = logging.getLogger(__name__)


@dataclass
class Constraints:
    """Symmetric constraint pairs for constrained clustering.

    Three tiers:
      - cannot_link: hard — never place in same cluster (key distance > 2, BPM too far)
      - relaxed_cannot_link: soft — avoid if possible, allow as fallback (key distance == 2)
      - must_link: force into same cluster (user feedback)
    """
    cannot_link: set[tuple[int, int]] = field(default_factory=set)
    relaxed_cannot_link: set[tuple[int, int]] = field(default_factory=set)
    must_link: set[tuple[int, int]] = field(default_factory=set)

    def is_cannot_link(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.cannot_link

    def is_relaxed_cannot_link(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.relaxed_cannot_link

    def is_must_link(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.must_link


def build_constraints(
    tracks: list[TrackFeatures],
    config: GrouperConfig,
    feedback: list | None = None,
) -> Constraints:
    """Build constraint sets from track key/BPM compatibility and feedback.

    Key constraints (three tiers based on Camelot distance):
      - distance <= 1: no constraint (same key or direct neighbor)
      - distance == 2: relaxed cannot-link (neighbor-of-neighbor, fallback ok)
      - distance > 2: hard cannot-link (never in same group)

    BPM constraints:
      - spread > threshold: hard cannot-link

    Unknown keys generate no key constraints (float freely).
    """
    constraints = Constraints()
    n = len(tracks)

    n_key_hard = 0
    n_key_relaxed = 0
    n_bpm_cl = 0

    for i in range(n):
        key_i = tracks[i].info.key
        bpm_i = tracks[i].info.bpm
        for j in range(i + 1, n):
            pair = (i, j)
            key_j = tracks[j].info.key
            bpm_j = tracks[j].info.bpm

            # Key constraint: tiered by Camelot distance
            if (key_i and key_i != "??" and key_j and key_j != "??"):
                try:
                    kd = camelot_distance(key_i, key_j)
                    if kd > 2:
                        # Hard: too far, never group together
                        constraints.cannot_link.add(pair)
                        n_key_hard += 1
                        continue
                    elif kd == 2:
                        # Relaxed: neighbor-of-neighbor, allow as fallback
                        constraints.relaxed_cannot_link.add(pair)
                        n_key_relaxed += 1
                except (ValueError, IndexError):
                    pass

            # BPM constraint: hard cannot-link if spread too wide
            if bpm_i and bpm_j and bpm_i > 0 and bpm_j > 0:
                avg_bpm = (bpm_i + bpm_j) / 2.0
                spread = abs(bpm_i - bpm_j) / avg_bpm
                if spread > config.bpm_group_max_spread_pct:
                    constraints.cannot_link.add(pair)
                    n_bpm_cl += 1

    # Feedback-based constraints
    n_feedback_cl = 0
    n_feedback_ml = 0
    if feedback:
        name_to_idx: dict[str, int] = {}
        for i, tf in enumerate(tracks):
            name_to_idx[Path(tf.path).name] = i
            name_to_idx[tf.path] = i

        for entry in feedback:
            idx_a = name_to_idx.get(entry.track_a)
            idx_b = name_to_idx.get(entry.track_b)
            if idx_a is None or idx_b is None:
                continue
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))

            if entry.type == "good_pair":
                constraints.must_link.add(pair)
                constraints.cannot_link.discard(pair)
                constraints.relaxed_cannot_link.discard(pair)
                n_feedback_ml += 1
            elif entry.type == "bad_pair":
                constraints.cannot_link.add(pair)
                n_feedback_cl += 1

    total_pairs = n * (n - 1) // 2
    logger.info(
        "Constraints built: %d hard cannot-link (%d key>2, %d bpm, %d feedback), "
        "%d relaxed (key==2), %d must-link out of %d pairs",
        len(constraints.cannot_link), n_key_hard, n_bpm_cl, n_feedback_cl,
        len(constraints.relaxed_cannot_link),
        len(constraints.must_link), total_pairs,
    )
    return constraints
