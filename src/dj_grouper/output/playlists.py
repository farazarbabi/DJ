"""M3U8 playlist generation for Rekordbox."""

from __future__ import annotations

import logging
from pathlib import Path

from ..features.builder import TrackFeatures
from ..grouping.assignment import GroupAssignment
from ..recommend.neighbors import Recommendation

logger = logging.getLogger(__name__)


def generate_group_playlists(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    output_dir: str,
) -> None:
    """Generate one .m3u8 playlist per group."""
    out = Path(output_dir) / "groups"
    out.mkdir(parents=True, exist_ok=True)

    _illegal = str.maketrans({c: "_" for c in r'<>:"/\|?*'})
    for group in assignment.groups:
        playlist_path = out / f"{group.folder_name.translate(_illegal)}.m3u8"
        lines = ["#EXTM3U"]
        for idx in group.member_indices:
            tf = tracks[idx]
            lines.append(f"#EXTINF:-1,{Path(tf.path).stem}")
            lines.append(tf.path)

        playlist_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Generated %d group playlists in %s", len(assignment.groups), out)


def generate_recommendation_playlists(
    recommendations: list[Recommendation],
    output_dir: str,
) -> None:
    """Generate per-track .m3u8 recommendation playlists."""
    out = Path(output_dir) / "recommendations"
    out.mkdir(parents=True, exist_ok=True)

    # Group recommendations by source
    by_source: dict[str, list[Recommendation]] = {}
    for rec in recommendations:
        by_source.setdefault(rec.source_path, []).append(rec)

    count = 0
    for source_path, recs in by_source.items():
        stem = Path(source_path).stem
        safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem)[:80]
        playlist_path = out / f"{safe_stem}.m3u8"

        lines = ["#EXTM3U"]
        for rec in sorted(recs, key=lambda r: r.rank):
            lines.append(f"#EXTINF:-1,{Path(rec.target_path).stem}")
            lines.append(rec.target_path)

        playlist_path.write_text("\n".join(lines), encoding="utf-8")
        count += 1

    logger.info("Generated %d recommendation playlists in %s", count, out)
