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
    # utf-8-sig: Rekordbox requires a UTF-8 BOM on .m3u8 files, otherwise
    # entries with non-ASCII path characters fail to match and the playlist
    # imports empty.
    path.write_text("\n".join(lines), encoding="utf-8-sig")


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


def _coarse_key_bucket(padded: str) -> str | None:
    """Map padded Camelot `NNL` to a 4-key coarse window: `01A-02B`, ..., `11A-12B`.

    The Camelot wheel pairs by relative maj/min (same number) and by ±1 step
    (adjacent numbers) — both are standard harmonic-mix moves, so a 4-key
    neighborhood is musically coherent for browsing during a live set.
    """
    try:
        n = int(padded[:2])
    except (ValueError, IndexError):
        return None
    if not 1 <= n <= 12:
        return None
    lo = n if n % 2 == 1 else n - 1  # 1,3,5,7,9,11
    hi = lo + 1
    return f"{lo:02d}A-{hi:02d}B"


def _write_by_key_coarse(
    out_dir: Path,
    tracks: list[LogicalTrack],
    path_by_id: dict[str, str],
) -> int:
    """Bucket tracks into 6 coarse key windows: {1A,1B,2A,2B}, ..., {11A,11B,12A,12B}."""
    buckets: dict[str, list[LogicalTrack]] = {}
    for t in tracks:
        key = t.canonical_key_camelot.strip()
        if not key or t.track_id not in path_by_id:
            continue
        padded = _camelot_padded(key)
        if padded is None:
            continue
        bucket = _coarse_key_bucket(padded)
        if bucket is None:
            continue
        buckets.setdefault(bucket, []).append(t)

    written = 0
    for bucket, members in buckets.items():
        entries = _sorted_entries(members, path_by_id)
        if not entries:
            continue
        _write_m3u8(out_dir / f"{bucket}.m3u8", entries)
        written += 1
    return written


def _coarse_subgenre_bucket(label: str, family_first_words: set[str]) -> str:
    """Map a `dj_taxonomy_label` to a coarse bucket.

    Two-pass derivation:
      1. Strip everything after the first '/' so `Indie Dance / Acid` → `Indie Dance`.
      2. If the first word of the head appears as the first word of ≥2 distinct
         heads across the input set (`family_first_words`), collapse to that
         first word — `Acid Breaks` / `Acid House` / `Acid Techno` → `Acid`.

    Returns the original label if neither pass shortens it (no merge available).
    """
    head = label.split("/", 1)[0].strip()
    if not head:
        return label.strip()
    tokens = head.split()
    if not tokens:
        return head
    first = tokens[0]
    if first in family_first_words:
        return first
    return head


def _compute_family_first_words(labels: list[str]) -> set[str]:
    """First words that appear as the first token in ≥2 distinct slash-heads.

    Driven by the labels actually present in the input — no curated map.
    """
    heads: set[str] = set()
    for label in labels:
        head = label.split("/", 1)[0].strip()
        if head:
            heads.add(head)
    from collections import Counter

    first_words = Counter(
        h.split()[0] for h in heads if h.split() and len(h.split()) > 1
    )
    return {w for w, n in first_words.items() if n >= 2}


def _write_by_subgenre_coarse(
    out_dir: Path,
    tracks: list[LogicalTrack],
    path_by_id: dict[str, str],
) -> int:
    """Bucket subgenres via label-prefix merge: strip after `/`, then collapse
    heads sharing a first word that appears in ≥2 heads."""
    labels = [
        t.dj_taxonomy_label.strip()
        for t in tracks
        if t.dj_taxonomy_label.strip() and t.track_id in path_by_id
    ]
    family_first_words = _compute_family_first_words(labels)

    buckets: dict[str, list[LogicalTrack]] = {}
    for t in tracks:
        label = t.dj_taxonomy_label.strip()
        if not label or t.track_id not in path_by_id:
            continue
        bucket = _coarse_subgenre_bucket(label, family_first_words)
        buckets.setdefault(bucket, []).append(t)

    written = 0
    for bucket, members in buckets.items():
        entries = _sorted_entries(members, path_by_id)
        if not entries:
            continue
        _write_m3u8(out_dir / f"{_safe_filename(bucket)}.m3u8", entries)
        written += 1
    return written


def generate_categorical_playlists(
    tracks: list[LogicalTrack],
    files: list[FileRecord],
    playlists_root: str,
    *,
    fine: bool = True,
    coarse: bool = False,
) -> dict[str, int]:
    """Write categorical M3U8 playlists, gated independently by resolution.

    ``fine`` writes by_key/ and by_subgenre/; ``coarse`` writes by_key_coarse/
    and by_subgenre_coarse/. They are independent, so any combination
    (fine-only, coarse-only, or both) is valid.

    Returns counts of playlists written per category (only the keys actually
    generated are present).
    """
    root = Path(playlists_root)
    path_by_id = _resolve_paths(tracks, files)

    counts: dict[str, int] = {}
    if fine:
        counts["by_key"] = _write_by_key(root / "by_key", tracks, path_by_id)
        counts["by_subgenre"] = _write_by_subgenre(
            root / "by_subgenre", tracks, path_by_id
        )
    if coarse:
        counts["by_key_coarse"] = _write_by_key_coarse(
            root / "by_key_coarse", tracks, path_by_id
        )
        counts["by_subgenre_coarse"] = _write_by_subgenre_coarse(
            root / "by_subgenre_coarse", tracks, path_by_id
        )
    logger.info(
        "Categorical playlists: %s",
        ", ".join(f"{k}={v}" for k, v in counts.items()) or "none",
    )
    return counts
