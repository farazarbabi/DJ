"""Top-N neighbor computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from .scoring import recommend_score

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    source_path: str
    target_path: str
    rank: int
    score: float
    energy_delta: int
    bpm_delta: int
    key_compatible: bool
    struct_compatible: bool


def compute_recommendations(
    tracks: list[TrackFeatures],
    config: GrouperConfig,
) -> list[Recommendation]:
    """Compute top-N similar track recommendations for each track."""
    all_recs: list[Recommendation] = []
    n = len(tracks)

    for i in range(n):
        source = tracks[i]
        scored: list[tuple[float, int]] = []

        for j in range(n):
            if i == j:
                continue
            s = recommend_score(source, tracks[j], config)
            if s > -float("inf"):
                scored.append((s, j))

        scored.sort(key=lambda x: -x[0])
        top = scored[: config.n_recommendations]

        for rank, (score, j) in enumerate(top, 1):
            target = tracks[j]
            e_a = source.info.energy or 3
            e_b = target.info.energy or 3
            bpm_a = source.info.bpm or 128
            bpm_b = target.info.bpm or 128

            rec = Recommendation(
                source_path=source.path,
                target_path=target.path,
                rank=rank,
                score=round(score, 4),
                energy_delta=e_b - e_a,
                bpm_delta=abs(bpm_b - bpm_a),
                key_compatible=_is_key_compatible(source.info.key, target.info.key),
                struct_compatible=_is_struct_compatible(
                    source.info.flow_type, target.info.flow_type,
                ),
            )
            all_recs.append(rec)

    logger.info(
        "Computed %d recommendations (%d tracks x %d each)",
        len(all_recs), n, config.n_recommendations,
    )
    return all_recs


def _is_key_compatible(key_a: str | None, key_b: str | None) -> bool:
    if not key_a or not key_b:
        return True
    if key_a == key_b:
        return True
    if key_a[:-1] == key_b[:-1]:
        return True
    if key_a[-1] == key_b[-1]:
        num_a = int(key_a[:-1])
        num_b = int(key_b[:-1])
        diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
        return diff <= 1
    return False


def _is_struct_compatible(flow_a: str | None, flow_b: str | None) -> bool:
    a = flow_a or "H"
    b = flow_b or "H"
    bad_pairs = {("H", "D"), ("D", "H"), ("D", "L"), ("L", "D")}
    return (a, b) not in bad_pairs
