"""Shared playlist sorting utilities for consistent track ordering."""

from __future__ import annotations

import math
from typing import Any, TypeVar

T = TypeVar("T")


def camelot_sort_key(key: str | None) -> int:
    """Map Camelot key to wheel-order sort position (0-23), or infinity if missing.

    Order: 1A (0), 1B (1), 2A (2), 2B (3), ..., 12A (22), 12B (23).
    Non-standard keys map to infinity so they sort last.
    """
    if not key or not isinstance(key, str):
        return math.inf

    key = key.strip().upper()
    if not key:
        return math.inf

    # Extract number and letter
    digits = "".join(c for c in key if c.isdigit())
    letters = "".join(c for c in key if c.isalpha()).upper()

    if not digits or letters not in ("A", "B"):
        return math.inf

    try:
        num = int(digits)
    except ValueError:
        return math.inf

    if not 1 <= num <= 12:
        return math.inf

    # 1A=0, 1B=1, 2A=2, 2B=3, ..., 12A=22, 12B=23
    return (num - 1) * 2 + (0 if letters == "A" else 1)


def bpm_sort_key(bpm: int | float | str | None) -> float:
    """Convert BPM to sort key (low to high), or infinity if missing."""
    if bpm is None:
        return math.inf

    if isinstance(bpm, str):
        try:
            bpm = float(bpm)
        except (ValueError, TypeError):
            return math.inf

    try:
        bpm = float(bpm)
    except (ValueError, TypeError):
        return math.inf

    if not math.isfinite(bpm) or bpm <= 0:
        return math.inf

    return bpm


def sort_tracks_by_bpm_and_key(
    tracks: list[T],
    *,
    bpm_getter: callable | None = None,
    key_getter: callable | None = None,
) -> list[T]:
    """Sort tracks by BPM (ascending), with key as tiebreaker (Camelot wheel order).

    Missing BPM/key values sort last (infinity).

    Args:
        tracks: List of tracks to sort
        bpm_getter: Callable(track) -> bpm or None. If None, tries track.bpm or track.info.bpm
        key_getter: Callable(track) -> key or None. If None, tries track.key or track.info.key
    """
    if bpm_getter is None:
        def bpm_getter(t):
            if hasattr(t, "bpm"):
                return t.bpm
            if hasattr(t, "info") and hasattr(t.info, "bpm"):
                return t.info.bpm
            return None

    if key_getter is None:
        def key_getter(t):
            if hasattr(t, "key"):
                return t.key
            if hasattr(t, "info") and hasattr(t.info, "key"):
                return t.info.key
            return None

    def sort_key(track: T) -> tuple[float, int]:
        bpm = bpm_getter(track)
        key = key_getter(track)
        return (bpm_sort_key(bpm), camelot_sort_key(key))

    return sorted(tracks, key=sort_key)
