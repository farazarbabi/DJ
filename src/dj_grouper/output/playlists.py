"""M3U8 group-playlist generation for Rekordbox."""

from __future__ import annotations

import logging
from pathlib import Path

from ..features.builder import TrackFeatures
from ..grouping.assignment import GroupAssignment

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
