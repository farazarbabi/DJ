"""Read-only Rekordbox XML marker validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from defusedxml.lxml import parse as _safe_parse


@dataclass
class RekordboxXmlValidation:
    tracks: int = 0
    markers: int = 0
    hot_cues: int = 0
    memory_cues: int = 0
    loops: int = 0
    unknown_markers: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_rekordbox_xml(path: str | Path) -> RekordboxXmlValidation:
    """Validate known cue marker shapes in a Rekordbox XML export."""
    tree = _safe_parse(str(path))
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        return RekordboxXmlValidation(errors=["No COLLECTION element in Rekordbox XML"])

    result = RekordboxXmlValidation()
    tracks = collection.findall("TRACK")
    result.tracks = len(tracks)
    for track in tracks:
        track_id = track.get("TrackID", "")
        for mark in track.findall("POSITION_MARK"):
            result.markers += 1
            _validate_marker(mark, track_id, result)
    return result


def _validate_marker(mark, track_id: str, result: RekordboxXmlValidation) -> None:
    mark_type = mark.get("Type", "0")
    num = mark.get("Num", "")
    start = _float_attr(mark, "Start")
    end = _float_attr(mark, "End")
    marker_ref = f"TrackID={track_id or '?'} Name={mark.get('Name', '')!r}"

    if start is None:
        result.errors.append(f"{marker_ref}: POSITION_MARK is missing numeric Start")
        return

    if mark_type == "0" and num not in {"", "-1"}:
        try:
            cue_num = int(num)
        except ValueError:
            result.errors.append(f"{marker_ref}: hot cue Num is not numeric: {num!r}")
            return
        if cue_num < 0 or cue_num > 7:
            result.errors.append(f"{marker_ref}: hot cue Num outside 0-7: {num!r}")
            return
        result.hot_cues += 1
        return

    if mark_type == "0" and num in {"", "-1"}:
        result.memory_cues += 1
        return

    if mark_type == "4":
        if end is None:
            result.errors.append(f"{marker_ref}: loop marker is missing numeric End")
            return
        if end <= start:
            result.errors.append(f"{marker_ref}: loop End must be greater than Start")
            return
        if num not in {"", "-1"}:
            result.errors.append(f"{marker_ref}: loop Num should be -1 or empty, got {num!r}")
            return
        result.loops += 1
        return

    result.unknown_markers += 1


def _float_attr(mark, name: str) -> float | None:
    value = mark.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
