"""Lightweight evaluation framework for subjective DJ utility."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import GrouperConfig
from .features.builder import TrackFeatures
from .recommend.scoring import recommend_score

logger = logging.getLogger(__name__)


@dataclass
class EvalPair:
    track_a: str
    track_b: str
    label: str  # "good", "bad", "same_family"


def load_eval_pairs(path: str) -> list[EvalPair]:
    """Load evaluation pairs from CSV.

    Expected columns: track_a, track_b, label
    """
    p = Path(path)
    if not p.exists():
        return []
    pairs: list[EvalPair] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(EvalPair(
                track_a=row["track_a"],
                track_b=row["track_b"],
                label=row["label"],
            ))
    return pairs


def evaluate(
    tracks: list[TrackFeatures],
    eval_pairs: list[EvalPair],
    config: GrouperConfig,
) -> dict:
    """Evaluate recommendation quality against known pairs.

    Returns metrics dict with:
        - good_avg_rank: average rank of known good pairs in recommendations
        - bad_avg_score: average score of known bad pairs (should be low)
        - good_above_bad: fraction of good pairs scoring above all bad pairs
        - n_good, n_bad: counts
    """
    # Build name-to-index lookup
    name_to_idx: dict[str, int] = {}
    for i, tf in enumerate(tracks):
        name_to_idx[Path(tf.path).name] = i
        name_to_idx[tf.path] = i

    good_scores: list[float] = []
    bad_scores: list[float] = []
    good_ranks: list[int] = []

    for pair in eval_pairs:
        idx_a = name_to_idx.get(pair.track_a)
        idx_b = name_to_idx.get(pair.track_b)
        if idx_a is None or idx_b is None:
            logger.debug("Eval pair not found: %s <-> %s", pair.track_a, pair.track_b)
            continue

        score = recommend_score(tracks[idx_a], tracks[idx_b], config)

        if pair.label == "good" or pair.label == "same_family":
            good_scores.append(score)
            # Compute rank of track_b in track_a's recommendations
            all_scores = []
            for j in range(len(tracks)):
                if j != idx_a:
                    s = recommend_score(tracks[idx_a], tracks[j], config)
                    all_scores.append(s)
            all_scores.sort(reverse=True)
            rank = 1
            for s in all_scores:
                if s > score:
                    rank += 1
                else:
                    break
            good_ranks.append(rank)

        elif pair.label == "bad":
            bad_scores.append(score)

    metrics = {
        "n_good": len(good_scores),
        "n_bad": len(bad_scores),
        "good_avg_score": sum(good_scores) / len(good_scores) if good_scores else 0.0,
        "bad_avg_score": sum(bad_scores) / len(bad_scores) if bad_scores else 0.0,
        "good_avg_rank": sum(good_ranks) / len(good_ranks) if good_ranks else 0.0,
    }

    # Fraction of good pairs scoring above all bad pairs
    if good_scores and bad_scores:
        max_bad = max(bad_scores)
        metrics["good_above_bad"] = sum(1 for s in good_scores if s > max_bad) / len(good_scores)
    else:
        metrics["good_above_bad"] = None

    return metrics
