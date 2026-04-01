"""Feedback storage: read/write feedback.csv."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    track_a: str
    track_b: str  # empty for group_override
    type: str     # "good_pair", "bad_pair", "group_override"
    strength: str  # float for pairs, group_id for overrides


def load_feedback(path: str) -> list[FeedbackEntry]:
    """Load feedback entries from CSV."""
    p = Path(path)
    if not p.exists():
        return []

    entries: list[FeedbackEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(FeedbackEntry(
                track_a=row["track_a"],
                track_b=row.get("track_b", ""),
                type=row["type"],
                strength=row["strength"],
            ))
    logger.info("Loaded %d feedback entries from %s", len(entries), path)
    return entries


def save_feedback(entries: list[FeedbackEntry], path: str) -> None:
    """Save feedback entries to CSV."""
    fieldnames = ["track_a", "track_b", "type", "strength"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "track_a": e.track_a,
                "track_b": e.track_b,
                "type": e.type,
                "strength": e.strength,
            })
    logger.info("Saved %d feedback entries to %s", len(entries), path)


def add_feedback(
    path: str,
    track_a: str,
    track_b: str,
    feedback_type: str,
    strength: str = "1.0",
) -> None:
    """Append a feedback entry to the CSV file."""
    entries = load_feedback(path)
    entries.append(FeedbackEntry(
        track_a=track_a,
        track_b=track_b,
        type=feedback_type,
        strength=strength,
    ))
    save_feedback(entries, path)
