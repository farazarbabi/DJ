"""M3U8 group-playlist generation for Rekordbox."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..features.builder import TrackFeatures
from ..grouping.assignment import GroupAssignment

logger = logging.getLogger(__name__)


def playlist_abs_path(path: str) -> str:
    """Absolute on-disk path for an M3U8 entry.

    Track paths are stored relative to the scan root (e.g. ``files\\track.mp3``
    for the default ``./files`` library). Rekordbox resolves relative M3U8
    entries against the *playlist file's* own directory, so emitting them
    verbatim produces empty playlists. Anchor relative paths to the current
    working directory (the scan root's parent); already-absolute paths are
    returned normalized and unchanged.
    """
    return os.path.abspath(path)


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
            lines.append(playlist_abs_path(tf.path))

        playlist_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Generated %d group playlists in %s", len(assignment.groups), out)
