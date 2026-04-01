"""Review mode: inspect proposed groupings before writing."""

from __future__ import annotations

import logging
from pathlib import Path

from .features.builder import TrackFeatures
from .grouping.assignment import GroupAssignment

logger = logging.getLogger(__name__)


def print_review(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
) -> None:
    """Print proposed groups for review."""
    print(f"\n{'=' * 70}")
    print(f"  GROUPING REVIEW — {len(assignment.groups)} groups, {len(tracks)} tracks")
    print(f"{'=' * 70}\n")

    for group in assignment.groups:
        members = [tracks[i] for i in group.member_indices]
        medoid_path = tracks[group.medoid_index].path

        print(f"  {group.group_id}  {group.folder_name}")
        print(f"  {'─' * 60}")
        print(f"  Tracks: {len(members)}  |  Energy: E{group.energy}  |  Vibe: {group.vibe}")
        print(f"  BPM: {group.bpm}  |  Structure: {group.structure}  |  Vocal: {group.vocal}")
        print(f"  Medoid: {Path(medoid_path).name}")
        print()

        for idx in group.member_indices:
            tf = tracks[idx]
            info = tf.info
            marker = " *" if idx == group.medoid_index else "  "
            tag_parts = []
            if info.energy:
                tag_parts.append(f"E{info.energy}")
            if info.key:
                tag_parts.append(info.key)
            if info.bpm:
                tag_parts.append(str(info.bpm))
            if info.structure:
                tag_parts.append(info.structure)
            if info.vibe:
                tag_parts.append(info.vibe)
            if info.vocal:
                tag_parts.append(info.vocal)
            tag_str = " | ".join(tag_parts) if tag_parts else "(no tags)"
            print(f"  {marker} {Path(tf.path).name}")
            print(f"       {tag_str}")

        print()

    # Summary
    sizes = [len(g.member_indices) for g in assignment.groups]
    print(f"  {'─' * 60}")
    print(f"  Group sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={sorted(sizes)[len(sizes)//2]}")
    print()


def interactive_review(
    tracks: list[TrackFeatures],
    assignment: GroupAssignment,
) -> list[dict]:
    """Interactive review loop. Returns list of override actions."""
    print_review(tracks, assignment)

    overrides: list[dict] = []

    print("  Review commands:")
    print("    approve        — accept all groupings")
    print("    move N GID     — move track N to group GID")
    print("    merge G1 G2    — merge group G1 into G2")
    print("    done           — finish review")
    print()

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("approve", "done", ""):
            break
        elif cmd.startswith("move "):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    track_name = parts[1]
                    target_gid = parts[2].upper()
                    overrides.append({"action": "move", "track": track_name, "group": target_gid})
                    print(f"    Noted: move {track_name} -> {target_gid}")
                except (ValueError, IndexError):
                    print("    Usage: move <filename> <GID>")
            else:
                print("    Usage: move <filename> <GID>")
        elif cmd.startswith("merge "):
            parts = cmd.split()
            if len(parts) == 3:
                overrides.append({
                    "action": "merge",
                    "source": parts[1].upper(),
                    "target": parts[2].upper(),
                })
                print(f"    Noted: merge {parts[1].upper()} -> {parts[2].upper()}")
            else:
                print("    Usage: merge <GID1> <GID2>")
        else:
            print(f"    Unknown command: {cmd}")

    return overrides
