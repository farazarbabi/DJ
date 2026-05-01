"""Rekordbox XML export parser."""

from __future__ import annotations

import logging
import os
import shutil
from urllib.parse import unquote, urlparse

from defusedxml.lxml import parse as _safe_parse

from ..config import RegistryConfig
from ..key_utils import parse_any_key
from ..models import SourceObservation, PayloadIndexEntry, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from ..store.obs_cache import ObsCache

logger = logging.getLogger(__name__)


def _decode_location(location: str, prefix_map: dict[str, str]) -> str:
    """Convert Rekordbox file URI to a local filesystem path.

    Example: 'file://localhost/C:/Users/Foo/Music/track.mp3' -> 'C:\\Users\\Foo\\Music\\track.mp3'
    """
    parsed = urlparse(location)
    # URL-decode the path
    path = unquote(parsed.path)

    # Remove leading slash before drive letter on Windows (e.g., '/C:/' -> 'C:/')
    if len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]

    # Normalize to OS-native separators
    path = path.replace("/", os.sep)

    # Apply drive letter prefix mapping
    for old_prefix, new_prefix in prefix_map.items():
        if path.upper().startswith(old_prefix.upper()):
            path = new_prefix + path[len(old_prefix):]
            break

    return os.path.normpath(path)


def ingest_rekordbox(
    config: RegistryConfig,
    store: CsvStore,
    obs_cache: ObsCache | None = None,
    *,
    show_progress: bool = False,
) -> int:
    """Parse Rekordbox XML and create source observations.

    Returns number of tracks matched and ingested.
    """
    xml_path = config.rekordbox_xml_path
    if not xml_path or not os.path.exists(xml_path):
        logger.warning("Rekordbox XML not found: %s", xml_path)
        return 0

    # Copy raw XML to raw directory
    raw_dir = os.path.join(config.raw_dir, "rekordbox_xml")
    os.makedirs(raw_dir, exist_ok=True)
    raw_copy = os.path.join(raw_dir, os.path.basename(xml_path))
    shutil.copy2(xml_path, raw_copy)
    payload_ref = f"rekordbox-xml-{now_iso().replace(':', '-')}"

    # Parse XML (defused: no entity expansion, no DTD, no XInclude)
    tree = _safe_parse(xml_path)
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        logger.error("No COLLECTION element in Rekordbox XML")
        return 0

    # Build file lookup from files_master
    files = store.load_files()
    files_by_path = {f.path_abs.lower(): f for f in files}
    files_by_name_size: dict[str, list[tuple[str, str]]] = {}
    for f in files:
        key = f"{f.file_name.lower()}|{f.size_bytes}"
        files_by_name_size.setdefault(key, []).append((f.file_id, f.track_id))

    # Delete previous rekordbox observations (always re-ingest)
    all_obs = store.load_observations()
    obs_keep = [o for o in all_obs if o.source_system != "rekordbox"]

    new_obs: list[SourceObservation] = []
    matched = 0
    track_elements = collection.findall("TRACK")
    progress = ProgressBar(len(track_elements), label="Rekordbox", enabled=show_progress)

    for index, track_el in enumerate(track_elements, start=1):
        rb_id = track_el.get("TrackID", "")
        name = track_el.get("Name", "")
        artist = track_el.get("Artist", "")
        location = track_el.get("Location", "")
        tonality = track_el.get("Tonality", "")
        bpm_str = track_el.get("AverageBpm", "")
        genre = track_el.get("Genre", "")
        label = track_el.get("Label", "")
        rating = track_el.get("Rating", "0")
        comments = track_el.get("Comments", "")
        total_time = track_el.get("TotalTime", "0")
        size = track_el.get("Size", "0")
        play_count = track_el.get("PlayCount", "0")
        date_added = track_el.get("DateAdded", "")
        year = track_el.get("Year", "")
        remixer = track_el.get("Remixer", "")

        # Skip very short tracks (samples, SFX)
        try:
            duration = int(total_time)
        except ValueError:
            duration = 0
        if duration < 30:
            progress.update(index, name, matched=matched)
            continue

        # Match to local file
        local_path = _decode_location(location, config.path_prefix_map)
        file_id = ""
        track_id = ""

        # Try exact path match
        frec_match = files_by_path.get(local_path.lower())
        if frec_match:
            file_id = frec_match.file_id
            track_id = frec_match.track_id

        # Try filename + size match
        if not file_id:
            fname = os.path.basename(local_path).lower()
            key = f"{fname}|{size}"
            candidates = files_by_name_size.get(key, [])
            if len(candidates) == 1:
                file_id, track_id = candidates[0]

        if not track_id:
            logger.debug("Rekordbox track not matched: %s - %s", artist, name)
            progress.update(index, name, matched=matched)
            continue

        matched += 1

        # Parse key
        key_std, key_cam = "", ""
        if tonality:
            parsed = parse_any_key(tonality)
            if parsed:
                key_std, key_cam = parsed

        # Parse BPM
        bpm_val = ""
        if bpm_str:
            try:
                bpm_val = str(round(float(bpm_str)))
            except ValueError:
                pass

        # One observation row per track with all available data
        obs = SourceObservation(
            observation_id=f"OBS-rb-{rb_id}",
            track_id=track_id,
            file_id=file_id,
            source_system="rekordbox",
            source_object_id=rb_id,
            key_standard=key_std,
            key_camelot=key_cam,
            key_confidence=1.0 if key_cam else 0.0,
            bpm=bpm_val,
            genre=genre,
            label=label,
            year=year if year and year != "0" else "",
            is_remix="true" if remixer else "",
            rating=rating if rating and rating != "0" else "",
            play_count=play_count if play_count and play_count != "0" else "",
            date_added=date_added,
            comments=comments,
            payload_ref=payload_ref,
            observed_at=now_iso(),
        )
        new_obs.append(obs)
        if obs_cache:
            obs_cache.put_by_file(local_path, float(total_time), "rekordbox", obs)
        progress.update(index, name, matched=matched)

    progress.finish()
    # Update label_canonical on tracks if we got label data
    tracks = store.load_tracks()
    track_by_id = {t.track_id: t for t in tracks}
    for o in new_obs:
        if o.label and o.track_id in track_by_id:
            t = track_by_id[o.track_id]
            if not t.label_canonical:
                t.label_canonical = o.label

    # Save
    obs_keep.extend(new_obs)
    store.save_observations(obs_keep)
    store.save_tracks(tracks)

    # Add payload index entry
    store.add_payload_entry(PayloadIndexEntry(
        payload_ref=payload_ref,
        source_system="rekordbox",
        source_type="xml",
        payload_path=os.path.relpath(raw_copy, config.output_dir),
        fetched_at=now_iso(),
    ))

    xml_total = len(track_elements)
    unmatched = xml_total - matched
    logger.info("Rekordbox: %d matched, %d unmatched (of %d XML tracks)", matched, unmatched, xml_total)
    return matched
