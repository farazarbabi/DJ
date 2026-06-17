"""Rekordbox XML helpers shared by cue analysis and export."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from defusedxml.lxml import parse as _safe_parse

from ..adapters.rekordbox_xml import _decode_location
from ..models import FileRecord, SourceObservation
from ..store.csv_store import CsvStore


@dataclass
class RekordboxTrackRef:
    """One TRACK entry from a Rekordbox XML export."""

    rekordbox_track_id: str
    name: str
    artist: str
    location: str
    local_path: str
    size_bytes: int = 0
    total_time_sec: float = 0.0


@dataclass
class MatchedRekordboxTrack:
    """Rekordbox track matched to a registry FileRecord."""

    ref: RekordboxTrackRef
    file_record: FileRecord
    match_method: str

    @property
    def track_id(self) -> str:
        return self.file_record.track_id

    @property
    def file_id(self) -> str:
        return self.file_record.file_id


def parse_rekordbox_tracks(
    xml_path: str | Path,
    prefix_map: dict[str, str] | None = None,
    *,
    min_duration_sec: float = 0.0,
) -> list[RekordboxTrackRef]:
    """Parse TRACK refs from a Rekordbox XML export."""
    prefix_map = prefix_map or {}
    tree = _safe_parse(str(xml_path))
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        return []

    refs: list[RekordboxTrackRef] = []
    for track_el in collection.findall("TRACK"):
        total_time = _float_attr(track_el.get("TotalTime"))
        if total_time < min_duration_sec:
            continue
        location = track_el.get("Location", "")
        refs.append(RekordboxTrackRef(
            rekordbox_track_id=track_el.get("TrackID", ""),
            name=track_el.get("Name", ""),
            artist=track_el.get("Artist", ""),
            location=location,
            local_path=_decode_location(location, prefix_map) if location else "",
            size_bytes=_int_attr(track_el.get("Size")),
            total_time_sec=total_time,
        ))
    return refs


def match_rekordbox_tracks(
    refs: list[RekordboxTrackRef],
    store: CsvStore,
) -> list[MatchedRekordboxTrack]:
    """Match Rekordbox TRACK refs to registry files."""
    files = store.load_files()
    tracks = {track.track_id: track for track in store.load_tracks()}
    observations = store.load_observations()

    file_by_id = {f.file_id: f for f in files}
    files_by_path = {_path_key(f.path_abs): f for f in files if f.path_abs}
    files_by_name_size: dict[str, list[FileRecord]] = {}
    for frec in files:
        if frec.file_name and frec.size_bytes:
            key = f"{frec.file_name.lower()}|{frec.size_bytes}"
            files_by_name_size.setdefault(key, []).append(frec)

    rb_obs = _rekordbox_observations_by_object_id(observations)

    matches: list[MatchedRekordboxTrack] = []
    seen_file_ids: set[str] = set()
    for ref in refs:
        matched: FileRecord | None = None
        match_method = ""

        obs = rb_obs.get(ref.rekordbox_track_id)
        if obs:
            matched = file_by_id.get(obs.file_id)
            if not matched and obs.track_id in tracks:
                primary_id = tracks[obs.track_id].primary_file_id
                matched = file_by_id.get(primary_id)
            if matched:
                match_method = "rekordbox_observation"

        if not matched and ref.local_path:
            matched = files_by_path.get(_path_key(ref.local_path))
            if matched:
                match_method = "path"

        if not matched and ref.local_path and ref.size_bytes:
            name_size = f"{os.path.basename(ref.local_path).lower()}|{ref.size_bytes}"
            candidates = files_by_name_size.get(name_size, [])
            if len(candidates) == 1:
                matched = candidates[0]
                match_method = "filename_size"

        if matched and matched.file_id not in seen_file_ids:
            matches.append(MatchedRekordboxTrack(ref=ref, file_record=matched, match_method=match_method))
            seen_file_ids.add(matched.file_id)

    return matches


def _rekordbox_observations_by_object_id(observations: list[SourceObservation]) -> dict[str, SourceObservation]:
    return {
        obs.source_object_id: obs
        for obs in observations
        if obs.source_system == "rekordbox" and obs.source_object_id
    }


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)).lower()


def _int_attr(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def _float_attr(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0
