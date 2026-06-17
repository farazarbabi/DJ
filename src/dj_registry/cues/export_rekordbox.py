"""Export generated cue points back into a Rekordbox XML file."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from defusedxml.lxml import parse as _safe_parse
from lxml import etree

from ..config import RegistryConfig
from ..models import CuePoint, now_iso
from ..store.csv_store import CsvStore
from .rekordbox import match_rekordbox_tracks, parse_rekordbox_tracks

logger = logging.getLogger(__name__)

REVIEW_PLAYLIST_NAME = "AI Generated Cues - Review"


@dataclass
class CueExportStats:
    cues_total: int = 0
    inserted: int = 0
    skipped_conflict: int = 0
    unmatched: int = 0
    invalid: int = 0
    review_tracks: int = 0
    output_xml: str = ""
    report_path: str = ""


def export_rekordbox_cues(
    config: RegistryConfig,
    store: CsvStore,
    *,
    input_xml: str | Path,
    output_xml: str | Path,
) -> CueExportStats:
    """Write selected cue points into a copied Rekordbox XML export."""
    input_xml = str(input_xml)
    output_xml = str(output_xml)
    if not os.path.exists(input_xml):
        raise FileNotFoundError(f"Rekordbox XML not found: {input_xml}")

    tree = _safe_parse(input_xml)
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError("No COLLECTION element in Rekordbox XML")

    refs = parse_rekordbox_tracks(input_xml, config.path_prefix_map)
    matches = match_rekordbox_tracks(refs, store)
    file_to_rb_id = {match.file_id: match.ref.rekordbox_track_id for match in matches}
    rb_track_elements = {
        track_el.get("TrackID", ""): track_el
        for track_el in collection.findall("TRACK")
        if track_el.get("TrackID")
    }

    cue_points = store.load_cue_points()
    stats = CueExportStats(
        cues_total=len(cue_points),
        output_xml=output_xml,
        report_path=os.path.join(config.reports_dir, "cue_rekordbox_export_report.csv"),
    )
    report_rows: list[dict[str, str]] = []
    review_priority: dict[str, int] = {}

    for cue in cue_points:
        status, message, rb_id = _export_one_cue(cue, file_to_rb_id, rb_track_elements)
        cue.export_status = status
        cue.export_message = message
        cue.updated_at = now_iso()

        if status == "inserted":
            stats.inserted += 1
        elif status == "skipped_conflict":
            stats.skipped_conflict += 1
        elif status == "unmatched":
            stats.unmatched += 1
        else:
            stats.invalid += 1

        if rb_id and status in ("inserted", "skipped_conflict"):
            priority = 0 if cue.manual_review_required or status == "skipped_conflict" else 1
            review_priority[rb_id] = min(priority, review_priority.get(rb_id, 1))

        report_rows.append(_report_row(cue, rb_id, status, message))

    stats.review_tracks = len(review_priority)
    _replace_review_playlist(root, review_priority)
    store.save_cue_points(cue_points)
    _write_report(stats.report_path, report_rows)

    os.makedirs(os.path.dirname(output_xml) or ".", exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True, pretty_print=True)
    logger.info(
        "Cue export: %d inserted, %d conflicts, %d unmatched -> %s",
        stats.inserted,
        stats.skipped_conflict,
        stats.unmatched,
        output_xml,
    )
    return stats


def _export_one_cue(
    cue: CuePoint,
    file_to_rb_id: dict[str, str],
    rb_track_elements: dict[str, etree._Element],
) -> tuple[str, str, str]:
    if cue.cue_kind != "hot" or cue.rekordbox_type != "0":
        return "invalid", "v1 only exports hot cues with rekordbox_type=0", ""
    if cue.rekordbox_num < 0 or cue.rekordbox_num > 7:
        return "invalid", f"unsupported hot cue Num={cue.rekordbox_num}", ""

    rb_id = file_to_rb_id.get(cue.file_id)
    if not rb_id:
        return "unmatched", "cue file was not present in input XML", ""
    track_el = rb_track_elements.get(rb_id)
    if track_el is None:
        return "unmatched", "matched XML track element was not found", rb_id

    existing_nums = {
        mark.get("Num", "")
        for mark in track_el.findall("POSITION_MARK")
        if mark.get("Type", "0") == "0"
    }
    num = str(cue.rekordbox_num)
    if num in existing_nums:
        return "skipped_conflict", f"existing hot cue Num={num} preserved", rb_id

    mark = etree.Element("POSITION_MARK")
    mark.set("Name", cue.cue_name)
    mark.set("Type", cue.rekordbox_type)
    mark.set("Start", f"{cue.cue_time_sec:.3f}")
    mark.set("Num", num)
    mark.set("Red", str(cue.red))
    mark.set("Green", str(cue.green))
    mark.set("Blue", str(cue.blue))
    track_el.append(mark)
    return "inserted", "inserted", rb_id


def _replace_review_playlist(root: etree._Element, review_priority: dict[str, int]) -> None:
    playlists = root.find("PLAYLISTS")
    if playlists is None:
        playlists = etree.SubElement(root, "PLAYLISTS")

    root_node = playlists.find("NODE")
    if root_node is None:
        root_node = etree.SubElement(playlists, "NODE")
        root_node.set("Type", "0")
        root_node.set("Name", "ROOT")
        root_node.set("Count", "0")

    for existing in list(root_node.findall("NODE")):
        if existing.get("Name") == REVIEW_PLAYLIST_NAME:
            root_node.remove(existing)

    ordered_track_ids = [
        rb_id
        for rb_id, _priority in sorted(review_priority.items(), key=lambda item: (item[1], item[0]))
    ]
    playlist = etree.SubElement(root_node, "NODE")
    playlist.set("Name", REVIEW_PLAYLIST_NAME)
    playlist.set("Type", "1")
    playlist.set("KeyType", "0")
    playlist.set("Entries", str(len(ordered_track_ids)))
    for rb_id in ordered_track_ids:
        entry = etree.SubElement(playlist, "TRACK")
        entry.set("Key", rb_id)

    root_node.set("Count", str(len(root_node.findall("NODE"))))


def _write_report(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "cue_id",
        "track_id",
        "file_id",
        "rekordbox_track_id",
        "cue_slot",
        "rekordbox_num",
        "cue_name",
        "cue_time_sec",
        "confidence",
        "manual_review_required",
        "export_status",
        "export_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_row(cue: CuePoint, rb_id: str, status: str, message: str) -> dict[str, str]:
    return {
        "cue_id": cue.cue_id,
        "track_id": cue.track_id,
        "file_id": cue.file_id,
        "rekordbox_track_id": rb_id,
        "cue_slot": cue.cue_slot,
        "rekordbox_num": str(cue.rekordbox_num),
        "cue_name": cue.cue_name,
        "cue_time_sec": f"{cue.cue_time_sec:.3f}",
        "confidence": f"{cue.confidence:.3f}",
        "manual_review_required": str(cue.manual_review_required).lower(),
        "export_status": status,
        "export_message": message,
    }
