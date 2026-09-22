"""Audit Spotify playlists against Soundeo: find upgrade opportunities."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from .soundeo import SoundeoClient, SoundeoError
from .spotify_fetch import parse_playlist_csv, collect_playlists, scan_library, best_match

logger = logging.getLogger(__name__)


def audit_soundeo(
    csv_paths: list[str],
    library: str,
    *,
    soundeo: SoundeoClient | None = None,
    output_path: str | None = None,
) -> str:
    """Audit Spotify playlists against Soundeo; report upgrade opportunities.

    Compares what's in Spotify CSVs vs what's available on Soundeo and currently
    in the library, identifying upgrade opportunities (e.g., YouTube [U] → Extended,
    Radio Edit → Original).

    Returns path to the generated CSV report.
    """
    csv_paths = collect_playlists(csv_paths)
    if not csv_paths:
        raise FileNotFoundError("No Spotify playlist CSVs found")

    library_index = scan_library(library)
    output_path = output_path or os.path.join(library, "outputs", "fetch", "soundeo_audit.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    total_tracks = sum(len(parse_playlist_csv(p)) for p in csv_paths)

    for csv_path in csv_paths:
        playlist_tracks = parse_playlist_csv(csv_path)
        for idx, track in enumerate(playlist_tracks, 1):
            # Check library for existing version
            lib_path, lib_score = best_match(track, library_index)
            lib_status = "present" if lib_path else "missing"
            lib_version = _guess_version(Path(lib_path).stem) if lib_path else ""

            # Search Soundeo
            soundeo_available = ""
            soundeo_version = ""
            soundeo_duration = ""
            upgrade_opportunity = ""

            if soundeo is not None:
                try:
                    results = soundeo.search(track)
                    pick = soundeo.pick(track, results) if results else None
                    if pick:
                        soundeo_available = "yes"
                        soundeo_version = _guess_version(pick.title)
                        soundeo_duration = f"{int(pick.duration_sec or 0) // 60}:{int(pick.duration_sec or 0) % 60:02d}"

                        # Determine upgrade opportunity
                        if lib_status == "missing":
                            upgrade_opportunity = f"download → {soundeo_version}"
                        elif lib_version in ("YouTube [U]", "YouTube [W]", "YouTube [M]"):
                            # Upgrade marked downloads to Soundeo versions
                            if soundeo_version in ("Extended", "Original", "Remix", "Extended Remix"):
                                upgrade_opportunity = f"{lib_version} → {soundeo_version}"
                        elif lib_version == "Radio Edit":
                            if soundeo_version in ("Original", "Extended", "Remix", "Extended Remix"):
                                upgrade_opportunity = f"{lib_version} → {soundeo_version}"
                        elif lib_version == "Original":
                            # Upgrade Original to Extended if available
                            if soundeo_version == "Extended" or soundeo_version == "Extended Remix":
                                upgrade_opportunity = f"{lib_version} → {soundeo_version}"
                        elif lib_version == "Remix" or lib_version == "Extended Remix":
                            # Remixes are distinct but may have better Soundeo versions
                            if soundeo_version in ("Remix", "Extended Remix"):
                                upgrade_opportunity = f"{lib_version} (quality upgrade from Soundeo)"
                except SoundeoError:
                    soundeo_available = "error"

            rows.append({
                "playlist": os.path.basename(csv_path),
                "track": track.name,
                "artist": track.artist_display,
                "duration_spotify": f"{int(track.duration_sec or 0) // 60}:{int(track.duration_sec or 0) % 60:02d}",
                "library_status": lib_status,
                "library_version": lib_version,
                "soundeo_available": soundeo_available,
                "soundeo_version": soundeo_version,
                "soundeo_duration": soundeo_duration,
                "upgrade_opportunity": upgrade_opportunity,
            })

    # Write CSV report
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "playlist", "track", "artist", "duration_spotify",
            "library_status", "library_version",
            "soundeo_available", "soundeo_version", "soundeo_duration",
            "upgrade_opportunity",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    total = len(rows)
    upgradeable = len([r for r in rows if r["upgrade_opportunity"]])
    missing = len([r for r in rows if r["library_status"] == "missing"])

    logger.info(
        "soundeo-audit: %d total track(s), %d missing, %d upgradeable → %s",
        total, missing, upgradeable, output_path,
    )
    return output_path


def upgrade_soundeo(
    csv_paths: list[str],
    library: str,
    *,
    soundeo: SoundeoClient | None = None,
    audio_format: str = "aiff",
    dry_run: bool = False,
) -> str:
    """Download and replace tracks from Soundeo where better versions exist."""
    from .spotify_fetch import acquire_track

    csv_paths = collect_playlists(csv_paths)
    library_index = scan_library(library)
    output_path = os.path.join(library, "outputs", "fetch", "soundeo_upgrade.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    replaced = 0
    skipped = 0
    rows: list[dict[str, str]] = []

    all_tracks = []
    for csv_path in csv_paths:
        all_tracks.extend(parse_playlist_csv(csv_path))
    total = len(all_tracks)

    for idx, track in enumerate(all_tracks, 1):
        lib_path, lib_score = best_match(track, library_index)
        lib_version = _guess_version(Path(lib_path).stem) if lib_path else "missing"

        # Check if this is an upgrade opportunity: any tool-marked file or lower-quality version
        # (remixes are distinct tracks but may still benefit from upgraded source quality)
        should_upgrade = (
            lib_version in ("YouTube [U]", "YouTube [W]", "YouTube [M]", "Radio Edit", "Original", "Remix")
            or lib_version == "missing"
        )

        if not should_upgrade or not soundeo:
            logger.info(
                "[%d/%d] %s - %s | current: %s | skipped",
                idx, total, track.artist_display, track.name, lib_version
            )
            skipped += 1
            continue

        # Try to download from Soundeo
        try:
            results = soundeo.search(track)
            pick = soundeo.pick(track, results) if results else None
            soundeo_version = _guess_version(pick.title) if pick else "not found"

            if not pick:
                logger.info(
                    "[%d/%d] %s - %s | current: %s | soundeo: not found",
                    idx, total, track.artist_display, track.name, lib_version
                )
                rows.append({
                    "track": track.name,
                    "artist": track.artist_display,
                    "current": lib_version,
                    "soundeo": "not found",
                    "action": "skipped",
                })
                continue

            # Download the better version
            outcome = acquire_track(
                track, library,
                soundeo=soundeo,
                audio_format=audio_format,
            )

            if outcome.status == "ok":
                action = "downloaded & replaced" if lib_path else "downloaded"
                logger.info(
                    "[%d/%d] %s - %s | %s → %s ✓",
                    idx, total, track.artist_display, track.name,
                    lib_version, soundeo_version
                )
                downloaded += 1

                # Delete old version if it's marked [U]/[W]/[M]
                if lib_path and any(m in Path(lib_path).stem for m in ["[U]", "[W]", "[M]"]):
                    if not dry_run:
                        try:
                            Path(lib_path).unlink()
                            replaced += 1
                        except OSError as e:
                            logger.warning("could not delete %s: %s", lib_path, e)

                rows.append({
                    "track": track.name,
                    "artist": track.artist_display,
                    "current": lib_version,
                    "soundeo": soundeo_version,
                    "action": action,
                })
            else:
                logger.info(
                    "[%d/%d] %s - %s | %s → %s | failed: %s",
                    idx, total, track.artist_display, track.name,
                    lib_version, soundeo_version, outcome.status
                )
                rows.append({
                    "track": track.name,
                    "artist": track.artist_display,
                    "current": lib_version,
                    "soundeo": soundeo_version,
                    "action": f"failed: {outcome.status}",
                })
        except SoundeoError as e:
            logger.info(
                "[%d/%d] %s - %s | error: %s",
                idx, total, track.artist_display, track.name, str(e)
            )
            rows.append({
                "track": track.name,
                "artist": track.artist_display,
                "current": lib_version,
                "soundeo": "error",
                "action": "error",
            })

    # Write results CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "artist", "current", "soundeo", "action"])
        writer.writeheader()
        writer.writerows(rows)

    verb = "would download" if dry_run else "downloaded"
    logger.info(
        "soundeo-upgrade: %s %d, replaced %d, skipped %d → %s",
        verb, downloaded, replaced, skipped, output_path,
    )
    return output_path


def _guess_version(name: str) -> str:
    """Guess track version from filename/title.

    Check tool markers first so "Remix[U]" is classified as YouTube [U], not Remix.
    """
    n = name.lower()
    if "[u]" in n:
        return "YouTube [U]"
    if "[w]" in n:
        return "YouTube [W]"
    if "[m]" in n:
        return "YouTube [M]"
    if "extended" in n and "remix" in n:
        return "Extended Remix"
    if "extended" in n:
        return "Extended"
    if "remix" in n:
        return "Remix"
    if "radio" in n or "edit" in n:
        return "Radio Edit"
    if "original" in n:
        return "Original"
    return "Original"
