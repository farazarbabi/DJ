"""Group folder creation with hard links (Windows NTFS)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ..features.builder import TrackFeatures
from ..grouping.assignment import GroupAssignment

logger = logging.getLogger(__name__)


def create_group_folders(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
    output_dir: str,
    *,
    dry_run: bool = False,
    use_copy: bool = False,
) -> None:
    """Create group folders with hard-linked (or copied) track files."""
    out = Path(output_dir)

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    fell_back_to_copy = False

    for group in assignment.groups:
        folder = out / group.folder_name
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)

        member_tracks = [tracks[i] for i in group.member_indices]
        for tf in member_tracks:
            src = Path(tf.path)
            dst = folder / src.name

            if dry_run:
                logger.info("[DRY RUN] %s -> %s", src, dst)
                continue

            if dst.exists():
                logger.debug("Already exists: %s", dst)
                continue

            if use_copy or fell_back_to_copy:
                shutil.copy2(str(src), str(dst))
                logger.debug("Copied: %s -> %s", src, dst)
            else:
                try:
                    _create_hard_link(str(dst), str(src))
                    logger.debug("Linked: %s -> %s", src, dst)
                except OSError as e:
                    if not fell_back_to_copy:
                        logger.info("Hard links not supported (%s), using copy", e)
                        fell_back_to_copy = True
                    shutil.copy2(str(src), str(dst))

        # Write group info file
        if not dry_run:
            _write_group_info(folder, group, member_tracks)

    n_folders = len(assignment.groups)
    n_files = sum(len(g.member_indices) for g in assignment.groups)
    action = "DRY RUN" if dry_run else ("copied" if use_copy or fell_back_to_copy else "linked")
    logger.info("Created %d folders with %d files (%s)", n_folders, n_files, action)


def _write_group_info(folder: Path, group, member_tracks) -> None:
    """Write a _group_info.txt summary file inside the group folder."""
    info_path = folder / "_group_info.txt"
    lines = [
        f"Group: {group.group_id}",
        f"Folder: {group.folder_name}",
        f"Tracks: {len(member_tracks)}",
        f"Key: {group.key}",
        f"Energy: E{group.energy}",
        f"Vibe: {group.vibe}",
        f"BPM: {group.bpm}",
        f"Structure: {group.structure}",
        f"Vocal: {group.vocal}",
        f"Medoid: {Path(member_tracks[0].path).name if member_tracks else 'N/A'}",
        "",
        "Members:",
    ]
    for tf in member_tracks:
        info = tf.info
        tag_parts = []
        if info.key:
            tag_parts.append(info.key)
        if info.energy:
            tag_parts.append(f"E{info.energy}")
        if info.bpm:
            tag_parts.append(str(info.bpm))
        if info.vibe:
            tag_parts.append(info.vibe)
        if info.vocal:
            tag_parts.append(info.vocal)
        if hasattr(tf, "role") and tf.role:
            tag_parts.append(f"[{tf.role}]")
        lines.append(f"  {Path(tf.path).name}  ({' | '.join(tag_parts)})")

    info_path.write_text("\n".join(lines), encoding="utf-8")


def _create_hard_link(link_path: str, target_path: str) -> None:
    """Create a hard link. Works on Windows NTFS without admin."""
    os.link(target_path, link_path)
