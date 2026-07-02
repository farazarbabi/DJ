"""Generate a Rekordbox XML collection from the registry.

Emits a single ``collection.xml`` (Rekordbox XML) with the full ``COLLECTION``
— every library track, carrying key/BPM/beatgrid/cues — plus a ``PLAYLISTS``
tree that mirrors the ``.m3u8`` files already written under
``outputs/playlists/`` (categorical, grouper, and Spotify playlists).

Workflow it enables: point Rekordbox's *rekordbox xml* bridge at this file
(Preferences -> Advanced -> Database -> rekordbox xml), drag the tree into the
collection — it imports **pre-analyzed** (key/BPM/beatgrid/cues carry over, so
Rekordbox does not re-analyze) — then Export to USB for CDJ/XDJ.

The playlist tree is built by walking the ``.m3u8`` files rather than
re-deriving buckets: the m3u8s are the single source of truth for playlist
membership, so the tree matches ``outputs/playlists/`` exactly (including the
grouper's ``groups*/`` and the Spotify ``spotify/`` playlists) with one uniform
mechanism.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from urllib.parse import quote

from lxml import etree

from ..cues.export_rekordbox import _append_position_mark
from ..models import CuePoint, FileRecord, LogicalTrack

logger = logging.getLogger(__name__)

_M3U_SUFFIXES = (".m3u8", ".m3u")


def _parse_float(value: str) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _year(release_date: str) -> str:
    """Extract a 4-digit year from a canonical release date string."""
    head = (release_date or "").strip()[:4]
    return head if head.isdigit() else ""


def _location_uri(path_abs: str) -> str:
    """Absolute path -> Rekordbox ``file://localhost/`` URI.

    Inverse of ``adapters.rekordbox_xml._decode_location``: forward slashes,
    URL-encoded, drive colon preserved.
    """
    forward = os.path.abspath(path_abs).replace(os.sep, "/")
    return "file://localhost/" + quote(forward, safe="/:")


def _cue_num(cue: CuePoint) -> str:
    """Rekordbox POSITION_MARK ``Num``: hot-cue slot index, else ``-1``."""
    if (cue.cue_kind or "").strip().lower() == "hot":
        return str(cue.rekordbox_num)
    return "-1"


def _append_track(
    collection: etree._Element,
    track: LogicalTrack,
    primary: FileRecord,
    track_id: int,
    cues: list[CuePoint],
) -> None:
    el = etree.SubElement(collection, "TRACK")
    el.set("TrackID", str(track_id))
    el.set(
        "Name",
        track.title_canonical or primary.embedded_title or Path(primary.path_abs).stem,
    )
    el.set("Artist", track.artist_canonical or primary.embedded_artist or "")
    if track.album_canonical:
        el.set("Album", track.album_canonical)
    genre = track.canonical_genre or track.genre
    if genre:
        el.set("Genre", genre)
    if track.label_canonical:
        el.set("Label", track.label_canonical)

    duration = int(round(track.duration_sec_canonical or primary.audio_duration_sec or 0.0))
    if duration > 0:
        el.set("TotalTime", str(duration))
    year = _year(track.release_date_canonical)
    if year:
        el.set("Year", year)

    bpm = _parse_float(track.canonical_bpm)
    if bpm and bpm > 0:
        el.set("AverageBpm", f"{bpm:.2f}")
    tonality = track.canonical_key_standard or track.canonical_key_camelot
    if tonality:
        el.set("Tonality", tonality)
    if primary.size_bytes:
        el.set("Size", str(primary.size_bytes))
    el.set("Location", _location_uri(primary.path_abs))

    # Beatgrid: a single constant-tempo grid anchored at the first detected
    # beat (falls back to 0.0 when unavailable). Correct for four-on-the-floor;
    # tracks with genuine tempo changes would need multiple TEMPO nodes.
    if bpm and bpm > 0:
        inizio = _parse_float(getattr(track, "tagger_first_beat_sec", "")) or 0.0
        tempo = etree.SubElement(el, "TEMPO")
        tempo.set("Inizio", f"{inizio:.3f}")
        tempo.set("Bpm", f"{bpm:.2f}")
        tempo.set("Metro", "4/4")
        tempo.set("Battito", "1")

    for cue in cues:
        _append_position_mark(el, cue, num=_cue_num(cue))


def _playlist_node(
    m3u8_path: Path,
    path_index: dict[str, int],
    counters: dict[str, int],
) -> etree._Element | None:
    """Build a playlist NODE from an m3u8, resolving paths to TrackIDs."""
    try:
        text = m3u8_path.read_text(encoding="utf-8-sig")
    except OSError:
        return None

    keys: list[int] = []
    seen: set[int] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        track_id = path_index.get(os.path.normcase(os.path.abspath(line)))
        if track_id is None:
            counters["unresolved"] += 1
            continue
        if track_id in seen:
            continue
        seen.add(track_id)
        keys.append(track_id)

    if not keys:
        return None
    node = etree.Element(
        "NODE", Name=m3u8_path.stem, Type="1", KeyType="0", Entries=str(len(keys))
    )
    for track_id in keys:
        etree.SubElement(node, "TRACK", Key=str(track_id))
    counters["playlists"] += 1
    counters["entries"] += len(keys)
    return node


def _folder_node(
    dir_path: Path,
    path_index: dict[str, int],
    counters: dict[str, int],
) -> etree._Element | None:
    """Build a folder NODE mirroring a directory; None if it has no playlists."""
    node = etree.Element("NODE", Type="0", Name=dir_path.name)
    children = 0
    for entry in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
        child: etree._Element | None = None
        if entry.is_dir():
            child = _folder_node(entry, path_index, counters)
        elif entry.suffix.lower() in _M3U_SUFFIXES:
            child = _playlist_node(entry, path_index, counters)
        if child is not None:
            node.append(child)
            children += 1
    if children == 0:
        return None
    node.set("Count", str(children))
    return node


def _build_playlists_element(
    playlists_root: str | None,
    path_index: dict[str, int],
    counters: dict[str, int],
) -> etree._Element:
    playlists_el = etree.Element("PLAYLISTS")
    root_node = etree.SubElement(playlists_el, "NODE", Type="0", Name="ROOT")
    children = 0
    root_dir = Path(playlists_root) if playlists_root else None
    if root_dir and root_dir.is_dir():
        for entry in sorted(root_dir.iterdir(), key=lambda p: p.name.lower()):
            child: etree._Element | None = None
            if entry.is_dir():
                child = _folder_node(entry, path_index, counters)
            elif entry.suffix.lower() in _M3U_SUFFIXES:
                child = _playlist_node(entry, path_index, counters)
            if child is not None:
                root_node.append(child)
                children += 1
    root_node.set("Count", str(children))
    return playlists_el


def generate_rekordbox_collection(
    tracks: list[LogicalTrack],
    files: list[FileRecord],
    cue_points: list[CuePoint],
    out_path: str,
    *,
    playlists_root: str | None = None,
) -> dict[str, int | str]:
    """Write a Rekordbox XML collection to ``out_path``.

    Includes every track with a resolvable primary file in ``COLLECTION`` (with
    TEMPO/POSITION_MARK), and a ``PLAYLISTS`` tree mirroring the ``.m3u8`` files
    under ``playlists_root``. Returns counts: ``tracks``, ``playlists``,
    ``entries``, ``unresolved``, and ``out_path``.
    """
    file_by_id = {f.file_id: f for f in files}
    cues_by_file: dict[str, list[CuePoint]] = {}
    for cue in cue_points:
        cues_by_file.setdefault(cue.file_id, []).append(cue)

    root = etree.Element("DJ_PLAYLISTS", Version="1.0.0")
    etree.SubElement(root, "PRODUCT", Name="rekordbox", Version="6.0.0", Company="AlphaTheta")
    collection = etree.SubElement(root, "COLLECTION")

    track_id_by_track: dict[str, int] = {}
    next_id = 1
    for track in tracks:
        primary = file_by_id.get(track.primary_file_id)
        if primary is None or not primary.path_abs:
            continue
        track_id = next_id
        next_id += 1
        track_id_by_track[track.track_id] = track_id
        _append_track(
            collection, track, primary, track_id, cues_by_file.get(track.primary_file_id, [])
        )
    included = len(track_id_by_track)
    collection.set("Entries", str(included))

    # Map every file (incl. non-primary) whose track made it into the collection
    # back to that track's TrackID, so m3u8 entries resolve regardless of which
    # file of a logical track a playlist happens to reference.
    path_index: dict[str, int] = {}
    for f in files:
        track_id = track_id_by_track.get(f.track_id)
        if track_id is None or not f.path_abs:
            continue
        path_index[os.path.normcase(os.path.abspath(f.path_abs))] = track_id

    counters = {"playlists": 0, "entries": 0, "unresolved": 0}
    root.append(_build_playlists_element(playlists_root, path_index, counters))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    etree.ElementTree(root).write(
        out_path, encoding="utf-8", xml_declaration=True, pretty_print=True
    )
    logger.info(
        "Rekordbox collection: %d tracks, %d playlists, %d entries (%d unresolved) -> %s",
        included,
        counters["playlists"],
        counters["entries"],
        counters["unresolved"],
        out_path,
    )
    return {
        "tracks": included,
        "playlists": counters["playlists"],
        "entries": counters["entries"],
        "unresolved": counters["unresolved"],
        "out_path": out_path,
    }
