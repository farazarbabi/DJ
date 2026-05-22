"""Categorical M3U8 playlist generation from the registry.

Emits browsing playlists grouped by Camelot key and DJ taxonomy sub-genre.
Pure registry data — no clustering/grouper math.

A popularity-tier writer used to live here too, but Spotify dropped the
``popularity`` field from /v1/tracks/{id} for client-credentials apps in
late 2024 and Songstats stream-count endpoints are paywalled, so there
is no free popularity source to bucket on. Re-introduce when one is
available.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from ..models import FileRecord, LogicalTrack

logger = logging.getLogger(__name__)


_ILLEGAL_CHARS = str.maketrans({c: "_" for c in r'<>:"/\|?*'})


def _safe_filename(stem: str) -> str:
    return stem.replace(" ", "_").translate(_ILLEGAL_CHARS)


def _bpm_value(track: LogicalTrack) -> float:
    try:
        return float(track.canonical_bpm)
    except (ValueError, TypeError):
        return math.inf


def _track_label(track: LogicalTrack, fallback_path: str) -> str:
    if track.artist_canonical and track.title_canonical:
        return f"{track.artist_canonical} - {track.title_canonical}"
    return Path(fallback_path).stem


def _write_m3u8(path: Path, entries: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for label, abs_path in entries:
        lines.append(f"#EXTINF:-1,{label}")
        lines.append(abs_path)
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_paths(
    tracks: list[LogicalTrack], files: list[FileRecord]
) -> dict[str, str]:
    file_by_id = {f.file_id: f for f in files}
    out: dict[str, str] = {}
    for t in tracks:
        if not t.primary_file_id:
            continue
        f = file_by_id.get(t.primary_file_id)
        if not f or not f.path_abs:
            continue
        out[t.track_id] = f.path_abs
    return out


def _sorted_entries(
    bucket: list[LogicalTrack],
    path_by_id: dict[str, str],
) -> list[tuple[str, str]]:
    sorted_tracks = sorted(bucket, key=_bpm_value)
    entries: list[tuple[str, str]] = []
    for t in sorted_tracks:
        path = path_by_id.get(t.track_id)
        if not path:
            continue
        entries.append((_track_label(t, path), path))
    return entries


def _camelot_padded(key: str) -> str | None:
    """`3A` -> `03A`, `12B` -> `12B`. Returns None for unrecognized formats."""
    number = "".join(c for c in key if c.isdigit())
    letter = "".join(c for c in key if c.isalpha()).upper()
    if not number or letter not in ("A", "B"):
        return None
    try:
        n = int(number)
    except ValueError:
        return None
    if not 1 <= n <= 12:
        return None
    return f"{n:02d}{letter}"


def _write_by_key(
    out_dir: Path,
    tracks: list[LogicalTrack],
    path_by_id: dict[str, str],
) -> int:
    buckets: dict[str, list[LogicalTrack]] = {}
    for t in tracks:
        key = t.canonical_key_camelot.strip()
        if not key or t.track_id not in path_by_id:
            continue
        padded = _camelot_padded(key)
        if padded is None:
            logger.warning("Skipping unexpected Camelot value: %r", key)
            continue
        buckets.setdefault(padded, []).append(t)

    written = 0
    for padded, members in buckets.items():
        entries = _sorted_entries(members, path_by_id)
        if not entries:
            continue
        _write_m3u8(out_dir / f"{padded}.m3u8", entries)
        written += 1
    return written


def _write_by_subgenre(
    out_dir: Path,
    tracks: list[LogicalTrack],
    path_by_id: dict[str, str],
) -> int:
    buckets: dict[str, list[LogicalTrack]] = {}
    for t in tracks:
        label = t.dj_taxonomy_label.strip()
        if not label or t.track_id not in path_by_id:
            continue
        buckets.setdefault(label, []).append(t)

    written = 0
    for label, members in buckets.items():
        entries = _sorted_entries(members, path_by_id)
        if not entries:
            continue
        _write_m3u8(out_dir / f"{_safe_filename(label)}.m3u8", entries)
        written += 1
    return written


def generate_categorical_playlists(
    tracks: list[LogicalTrack],
    files: list[FileRecord],
    playlists_root: str,
) -> dict[str, int]:
    """Write by_key/ and by_subgenre/ M3U8 playlists.

    Returns counts of playlists written per category.
    """
    root = Path(playlists_root)
    path_by_id = _resolve_paths(tracks, files)

    counts = {
        "by_key": _write_by_key(root / "by_key", tracks, path_by_id),
        "by_subgenre": _write_by_subgenre(root / "by_subgenre", tracks, path_by_id),
    }
    logger.info(
        "Categorical playlists: by_key=%d, by_subgenre=%d",
        counts["by_key"],
        counts["by_subgenre"],
    )
    return counts
