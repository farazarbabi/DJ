"""Rekordbox hot-cue slot helpers."""

from __future__ import annotations

HOT_CUE_SLOTS = tuple("ABCDEFGH")
SLOT_TO_NUM = {slot: index for index, slot in enumerate(HOT_CUE_SLOTS)}
NUM_TO_SLOT = {index: slot for slot, index in SLOT_TO_NUM.items()}


def hot_cue_num(slot: str) -> int:
    """Return Rekordbox Num for a hot-cue slot A-H."""
    normalized = (slot or "").strip().upper()
    if normalized not in SLOT_TO_NUM:
        raise ValueError(f"Unsupported hot cue slot: {slot!r}")
    return SLOT_TO_NUM[normalized]


def hot_cue_slot(num: int | str) -> str:
    """Return hot-cue slot A-H for a Rekordbox Num value."""
    try:
        value = int(num)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported Rekordbox hot cue number: {num!r}") from exc
    if value not in NUM_TO_SLOT:
        raise ValueError(f"Unsupported Rekordbox hot cue number: {num!r}")
    return NUM_TO_SLOT[value]
