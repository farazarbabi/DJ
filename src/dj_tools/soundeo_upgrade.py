"""Upgrade library tracks to better Soundeo cuts (``dj fetch-missing --upgrade-soundeo``).

For each Spotify-playlist track already in the library, ask Soundeo for its best
cut and download it only when it beats the file on disk: a lossless Soundeo AIFF
beats any marked ``[U]``/``[M]``/``[W]`` download, an Extended cut beats an
Original/plain one, and either beats a Radio Edit. The superseded file is deleted
when it was a marked tool download, or moved to ``<library>/outputs/fetch/replaced/``
when it was unmarked (curated or a real Soundeo AIFF) so the swap can be undone.
Tracks missing from the library are left to ``dj fetch-missing``; YouTube is never
consulted here.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from .soundeo import SoundeoClient, SoundeoError, SoundeoResult, _version_rank
from .spotify_fetch import (
    PlaylistTrack,
    _QuotaState,
    _record_soundeo_tags,
    classify_tracks,
    collect_playlists,
    download_from_soundeo,
    load_soundeo_tags_log,
    load_unique_tracks,
    prune_superseded_downloads,
    save_soundeo_tags_log,
    scan_library,
    soundeo_target_stem,
    strip_tool_marker,
)

logger = logging.getLogger(__name__)

REPLACED_DIRNAME = "replaced"
REPORT_NAME = "soundeo_upgrade.csv"

# Quality tiers, best first. Tool markers rank below every unmarked cut because
# they flag a lossy or converted source. A suffix-less title counts as the
# Original: Spotify and curated files usually omit "(Original Mix)".
TIER_EXTENDED, TIER_ORIGINAL, TIER_RADIO, TIER_WAV, TIER_MP3, TIER_YOUTUBE = range(6)
_MARKER_TIER = {"[W]": TIER_WAV, "[M]": TIER_MP3, "[U]": TIER_YOUTUBE}
TIER_LABEL = {
    TIER_EXTENDED: "Extended",
    TIER_ORIGINAL: "Original",
    TIER_RADIO: "Radio Edit",
    TIER_WAV: "Soundeo WAV [W]",
    TIER_MP3: "Soundeo MP3 [M]",
    TIER_YOUTUBE: "YouTube [U]",
}


def quality_tier(stem: str) -> int:
    """Quality tier of a library filename stem (lower is better)."""
    base, marker = strip_tool_marker(stem)
    if marker:
        return _MARKER_TIER[marker]
    # Rank the title only: an artist like "Radiohead" must not read as a Radio Edit.
    title = base.split(" - ", 1)[1] if " - " in base else base
    rank = _version_rank(title)
    if rank == 0:
        return TIER_EXTENDED
    if rank == 3:
        return TIER_RADIO
    return TIER_ORIGINAL


def is_upgrade(current_stem: str, target_stem: str, pick: SoundeoResult) -> bool:
    """True when downloading ``pick`` (landing as ``target_stem``) beats ``current_stem``.

    A Radio Edit is never a target: the short broadcast cut does not improve on
    a longer download, whatever the source quality.
    """
    if _version_rank(pick.title) == 3:
        return False
    return quality_tier(target_stem) < quality_tier(current_stem)


def retire_superseded(current: str, library: str, new_path: str) -> str:
    """Retire the file an upgrade replaced; returns a short note for the log.

    Marked tool downloads are deleted. An unmarked file was curated (or a real
    Soundeo AIFF), so it is moved to ``<library>/outputs/fetch/replaced/`` —
    out of both scanners' reach but recoverable.
    """
    if os.path.normcase(os.path.abspath(current)) == os.path.normcase(os.path.abspath(new_path)):
        return "overwritten in place"
    name = os.path.basename(current)
    if strip_tool_marker(Path(current).stem)[1]:
        os.remove(current)
        return f"removed {name}"
    replaced_dir = os.path.join(library, "outputs", "fetch", REPLACED_DIRNAME)
    os.makedirs(replaced_dir, exist_ok=True)
    os.replace(current, os.path.join(replaced_dir, name))
    return f"moved {name} to outputs/fetch/{REPLACED_DIRNAME}/"


def upgrade_soundeo(
    playlists: list[str],
    library: str,
    *,
    soundeo: SoundeoClient,
    audio_format: str = "aiff",
    threshold: float = 0.62,
    dry_run: bool = False,
) -> dict:
    """Upgrade present playlist tracks to better Soundeo cuts; see the module doc.

    Returns a summary dict of counts plus ``report_path`` (the CSV listing every
    checked track and what happened). ``dry_run`` searches and reports but
    downloads and moves nothing. Searching is free; the pass stops at the first
    download that hits the daily quota.
    """
    if not os.path.isdir(library):
        raise FileNotFoundError(f"library dir not found: {library}")

    pruned = prune_superseded_downloads(library, dry_run=dry_run)
    if pruned:
        logger.info("soundeo-upgrade: %s %d marked download(s) already superseded by curated originals",
                    "would remove" if dry_run else "removed", len(pruned))

    csv_paths = collect_playlists(playlists)
    tracks = load_unique_tracks(csv_paths)
    present, missing = classify_tracks(tracks, scan_library(library), threshold=threshold)
    if missing:
        logger.info("soundeo-upgrade: %d track(s) not in the library — skipped "
                    "(run `dj fetch-missing` to download them)", len(missing))

    report_dir = os.path.join(library, "outputs", "fetch")
    os.makedirs(report_dir, exist_ok=True)
    soundeo_log = load_soundeo_tags_log(report_dir)
    quota = _QuotaState()
    summary = {
        "checked": len(present), "missing": len(missing), "upgraded": 0, "would_upgrade": 0,
        "no_upgrade": 0, "not_found": 0, "failed": 0, "deferred": 0,
    }
    rows: list[dict[str, str]] = []
    total = len(present)

    def record(track: PlaylistTrack, current: str, target: str, action: str) -> None:
        rows.append({"artist": track.artist_display, "track": track.name,
                     "current": current, "soundeo": target, "action": action})

    try:
        for i, match in enumerate(present, 1):
            track, current = match.track, match.closest
            label = f"[{i}/{total}] {track.artist_display} - {track.name}"
            current_stem = Path(current).stem
            current_label = TIER_LABEL[quality_tier(current_stem)]

            if quality_tier(current_stem) == TIER_EXTENDED:
                logger.info("%s | %s | nothing better exists", label, current_label)
                summary["no_upgrade"] += 1
                record(track, current_label, "", "skip")
                continue
            try:
                results = soundeo.search(track)
                pick = soundeo.pick(track, results) if results else None
            except SoundeoError as exc:
                logger.warning("%s | %s | Soundeo error: %s", label, current_label, exc)
                summary["failed"] += 1
                record(track, current_label, "error", f"error: {exc}")
                continue
            if pick is None:
                logger.info("%s | %s | not on Soundeo", label, current_label)
                summary["not_found"] += 1
                record(track, current_label, "not found", "skip")
                continue

            target_stem, _ = soundeo_target_stem(pick, audio_format)
            target_label = TIER_LABEL[quality_tier(target_stem)]
            if not is_upgrade(current_stem, target_stem, pick):
                logger.info("%s | %s | Soundeo has %s | no upgrade", label, current_label, target_label)
                summary["no_upgrade"] += 1
                record(track, current_label, target_label, "skip")
                continue
            if dry_run:
                logger.info("%s | %s -> %s | would download", label, current_label, target_label)
                summary["would_upgrade"] += 1
                record(track, current_label, target_label, "would download")
                continue

            outcome = download_from_soundeo(soundeo, pick, library, quota=quota,
                                            audio_format=audio_format)
            if outcome.status == "quota_skip":
                summary["deferred"] = total - i + 1
                logger.warning("soundeo: daily download limit reached — stopping; %d track(s) "
                               "not checked, re-run after midnight CET", summary["deferred"])
                record(track, current_label, target_label, "deferred: quota")
                break
            if outcome.status != "ok":
                logger.warning("%s | %s -> %s | failed: %s", label, current_label, target_label,
                               outcome.detail)
                summary["failed"] += 1
                record(track, current_label, target_label, f"failed: {outcome.detail}")
                continue

            note = retire_superseded(current, library, outcome.outfile)
            if not strip_tool_marker(Path(outcome.outfile).stem)[1]:
                _record_soundeo_tags(soundeo_log, outcome.outfile,
                                     outcome.detail.split("soundeo:", 1)[-1])
            logger.info("%s | %s -> %s | downloaded %s; %s", label, current_label, target_label,
                        os.path.basename(outcome.outfile), note)
            summary["upgraded"] += 1
            record(track, current_label, target_label, f"upgraded; {note}")
    finally:
        save_soundeo_tags_log(report_dir, soundeo_log)

    report_path = os.path.join(report_dir, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "track", "current", "soundeo", "action"])
        writer.writeheader()
        writer.writerows(rows)
    summary["report_path"] = report_path

    done_key = "would_upgrade" if dry_run else "upgraded"
    logger.info(
        "soundeo-upgrade: done — checked=%d %s=%d no_upgrade=%d not_found=%d failed=%d "
        "deferred=%d missing=%d -> %s",
        summary["checked"], done_key, summary[done_key], summary["no_upgrade"],
        summary["not_found"], summary["failed"], summary["deferred"], summary["missing"],
        report_path,
    )
    return summary
