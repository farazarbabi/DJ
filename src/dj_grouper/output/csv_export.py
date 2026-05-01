"""CSV exports for groups, features, and recommendations."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..features.builder import TrackFeatures
from ..grouping.assignment import GroupAssignment
from ..recommend.neighbors import Recommendation

logger = logging.getLogger(__name__)


def export_groups_csv(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    path: str,
) -> None:
    """Export groups.csv with one row per track."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file", "group_id", "group_folder", "bpm", "energy", "key",
        "structure", "mood", "vibe", "vocal", "vocal_profile", "is_medoid",
    ]

    medoid_indices = {g.medoid_index for g in assignment.groups}

    idx_to_group: dict[int, str] = {}
    idx_to_folder: dict[int, str] = {}
    for group in assignment.groups:
        for idx in group.member_indices:
            idx_to_group[idx] = group.group_id
            idx_to_folder[idx] = group.folder_name

    rows = []
    for i, tf in enumerate(tracks):
        rows.append({
            "file": Path(tf.path).name,
            "group_id": idx_to_group.get(i, ""),
            "group_folder": idx_to_folder.get(i, ""),
            "bpm": tf.info.bpm or "",
            "energy": tf.info.energy or "",
            "key": tf.info.key or "",
            "structure": tf.info.structure or "",
            "mood": tf.info.vibe or "",
            "vibe": tf.info.vibe or "",
            "vocal": tf.info.vocal or "",
            "vocal_profile": tf.info.vocal or "",
            "is_medoid": i in medoid_indices,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported groups to %s (%d rows)", path, len(rows))


def export_recommendations_csv(
    recommendations: list[Recommendation],
    path: str,
) -> None:
    """Export recommendations.csv with one row per recommendation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file", "rank", "target_file", "score",
        "energy_delta", "bpm_delta", "key_compatible", "struct_compatible",
    ]

    rows = []
    for rec in recommendations:
        rows.append({
            "source_file": Path(rec.source_path).name,
            "rank": rec.rank,
            "target_file": Path(rec.target_path).name,
            "score": rec.score,
            "energy_delta": rec.energy_delta,
            "bpm_delta": rec.bpm_delta,
            "key_compatible": "yes" if rec.key_compatible else "no",
            "struct_compatible": "yes" if rec.struct_compatible else "no",
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d recommendations to %s", len(rows), path)


def export_features_csv(
    tracks: list[TrackFeatures],
    dsp_features: list[dict[str, float]],
    path: str,
) -> None:
    """Export features.csv for inspection."""
    if not dsp_features:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    base_fields = ["file", "energy", "key", "bpm", "structure", "mood", "vibe", "vocal", "vocal_profile"]
    dsp_fields = sorted(dsp_features[0].keys()) if dsp_features else []
    fieldnames = base_fields + dsp_fields

    rows = []
    for i, tf in enumerate(tracks):
        row: dict = {
            "file": Path(tf.path).name,
            "energy": tf.info.energy,
            "key": tf.info.key,
            "bpm": tf.info.bpm,
            "structure": tf.info.structure,
            "mood": tf.info.vibe,
            "vibe": tf.info.vibe,
            "vocal": tf.info.vocal,
            "vocal_profile": tf.info.vocal,
        }
        if i < len(dsp_features):
            row.update({k: round(v, 6) for k, v in dsp_features[i].items()})
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported features to %s (%d rows)", path, len(rows))
