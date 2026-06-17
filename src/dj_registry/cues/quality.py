"""Cue quality reporting."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from ..config import RegistryConfig
from ..models import CuePoint, FileRecord
from ..store.csv_store import CsvStore


QUALITY_REPORT_NAME = "cue_quality_report.csv"
LOW_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class CueQualityStats:
    cues_total: int = 0
    manual_review: int = 0
    low_confidence: int = 0
    report_path: str = ""


def write_cue_quality_report(config: RegistryConfig, store: CsvStore) -> CueQualityStats:
    """Write a review-oriented cue quality CSV report."""
    cues = store.load_cue_points()
    files_by_id = {file_rec.file_id: file_rec for file_rec in store.load_files()}
    report_path = os.path.join(config.reports_dir, QUALITY_REPORT_NAME)

    rows = [_quality_row(cue, files_by_id.get(cue.file_id)) for cue in cues]
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    fieldnames = [
        "track_id",
        "file_id",
        "file_name",
        "cue_id",
        "cue_kind",
        "cue_role",
        "cue_time_sec",
        "cue_end_sec",
        "cue_bar_index",
        "confidence",
        "manual_review_required",
        "selection_reason",
        "quality_flags",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return CueQualityStats(
        cues_total=len(cues),
        manual_review=sum(1 for cue in cues if cue.manual_review_required),
        low_confidence=sum(1 for cue in cues if cue.confidence < LOW_CONFIDENCE_THRESHOLD),
        report_path=report_path,
    )


def quality_flags_for_cue(cue: CuePoint, file_rec: FileRecord | None = None) -> list[str]:
    """Return stable quality flags for one cue."""
    flags: list[str] = []
    reason = cue.selection_reason or ""
    if cue.confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("low_confidence")
    if "fallback" in reason or "no_bar_grid" in reason:
        flags.append("fallback_used")
    if cue.cue_role in {"peak", "outro_start"} and "fallback" in reason:
        flags.append("missing_section")
    if cue.cue_kind == "loop" and ("short_loop" in reason or cue.manual_review_required):
        flags.append("short_loop")
    if cue.cue_kind == "loop" and file_rec is not None and file_rec.audio_duration_sec > 0:
        if cue.cue_end_sec >= max(0.0, file_rec.audio_duration_sec - 10.0):
            flags.append("loop_near_track_end")
    if "adjusted" in reason:
        flags.append("cue_order_adjusted")
    if "possible_grid_offset" in reason:
        flags.append("possible_grid_offset")
    if cue.export_status == "skipped_conflict":
        flags.append("export_conflict")
    return list(dict.fromkeys(flags))


def _quality_row(cue: CuePoint, file_rec: FileRecord | None) -> dict[str, str]:
    return {
        "track_id": cue.track_id,
        "file_id": cue.file_id,
        "file_name": file_rec.file_name if file_rec is not None else "",
        "cue_id": cue.cue_id,
        "cue_kind": cue.cue_kind,
        "cue_role": cue.cue_role,
        "cue_time_sec": f"{cue.cue_time_sec:.3f}",
        "cue_end_sec": f"{cue.cue_end_sec:.3f}",
        "cue_bar_index": str(cue.cue_bar_index),
        "confidence": f"{cue.confidence:.3f}",
        "manual_review_required": str(cue.manual_review_required).lower(),
        "selection_reason": cue.selection_reason,
        "quality_flags": ";".join(quality_flags_for_cue(cue, file_rec)),
    }
