"""Spotify popularity adapter — fetch the 0-100 popularity score for ISRC tracks.

Spotify's Web API exposes a recency-weighted `popularity` integer on each track
object. We can't get raw stream counts on the free tier, but popularity is a
useful proxy for ranking and filtering.

Source priority for the spotify_id lookup:
  1. spotify_id already stored on a Songstats observation (no API call)
  2. Spotify ISRC search fallback (one /v1/search call per missing track)

The fetched popularity is stored on a 'spotify' SourceObservation, cached via
ObsCache (isrc:{isrc}|spotify in the raw cache).
"""

from __future__ import annotations

import logging
import os
import time

from ..models import SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from ..store.obs_cache import ObsCache
from .spotify_isrc import SpotifyClient, SpotifyTransientError

logger = logging.getLogger(__name__)

_SAVE_EVERY_N = 25
_NOT_FOUND_OBJECT_ID = "not_found"


def _existing_spotify_ids_by_track(store: CsvStore) -> dict[str, str]:
    """Map track_id -> spotify_id from existing Songstats observations."""
    out: dict[str, str] = {}
    for obs in store.load_observations():
        if obs.source_system == "songstats" and obs.spotify_id:
            out.setdefault(obs.track_id, obs.spotify_id)
    return out


def ingest_spotify_popularity(
    store: CsvStore,
    obs_cache: ObsCache | None = None,
    *,
    limit: int | None = None,
    show_progress: bool = False,
) -> dict:
    """Fetch Spotify popularity for tracks with ISRC.

    Returns stats dict: {"total", "cached", "fetched", "skipped"}.
    """
    stats = {"total": 0, "cached": 0, "fetched": 0, "skipped": 0, "candidates": 0}

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        logger.info("Spotify popularity: skipped (no credentials)")
        return stats

    tracks = store.load_tracks()
    candidates = [t for t in tracks if t.isrc_canonical]
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        logger.info("Spotify popularity: no tracks with ISRC")
        return stats

    stats["candidates"] = len(candidates)
    spotify_id_by_track = _existing_spotify_ids_by_track(store)
    client = SpotifyClient(client_id, client_secret)

    new_obs: list[SourceObservation] = []
    cache_hits = 0
    fetched = 0
    skipped = 0

    progress = ProgressBar(len(candidates), label="Spotify popularity", enabled=show_progress)
    transient_bail = False
    pending_writes = 0
    for index, track in enumerate(candidates, start=1):
        isrc = track.isrc_canonical

        # Cache: any prior observation (positive or negative) is definitive.
        # Negative entries (source_object_id == "not_found" or empty popularity)
        # mean we've already asked Spotify and there is no result for this ISRC.
        if obs_cache is not None:
            cached = obs_cache.get_by_isrc(isrc, "spotify")
            if cached is not None:
                if cached.popularity:
                    cached.track_id = track.track_id
                    cached.observation_id = f"OBS-spop-{isrc}"
                    new_obs.append(cached)
                cache_hits += 1
                progress.update(
                    index,
                    track.title_canonical or isrc,
                    cached=cache_hits,
                    fetched=fetched,
                    skipped=skipped,
                )
                continue

        try:
            spotify_id = spotify_id_by_track.get(track.track_id, "")
            if not spotify_id:
                spotify_id = client.find_id_by_isrc(isrc)
                time.sleep(0.1)
            track_obj = client.get_track(spotify_id) if spotify_id else None
            if track_obj is not None:
                time.sleep(0.1)
        except SpotifyTransientError as e:
            logger.warning("Spotify popularity: bailing out — %s", e)
            transient_bail = True
            break

        popularity = track_obj.get("popularity") if track_obj else None

        if not spotify_id or track_obj is None or popularity is None:
            # Definitive negative — cache so we never ask again.
            neg = SourceObservation(
                observation_id=f"OBS-spop-{isrc}",
                track_id=track.track_id,
                source_system="spotify",
                source_object_id=spotify_id or _NOT_FOUND_OBJECT_ID,
                spotify_id=spotify_id,
                popularity="",
                observed_at=now_iso(),
            )
            if obs_cache is not None:
                obs_cache.put_by_isrc(isrc, "spotify", neg)
                pending_writes += 1
            skipped += 1
        else:
            obs = SourceObservation(
                observation_id=f"OBS-spop-{isrc}",
                track_id=track.track_id,
                source_system="spotify",
                source_object_id=spotify_id,
                spotify_id=spotify_id,
                popularity=str(popularity),
                observed_at=now_iso(),
            )
            new_obs.append(obs)
            if obs_cache is not None:
                obs_cache.put_by_isrc(isrc, "spotify", obs)
                pending_writes += 1
            fetched += 1

        # Persist obs_cache periodically so an interruption later in the run
        # doesn't lose the negative entries we just learned.
        if obs_cache is not None and pending_writes >= _SAVE_EVERY_N:
            obs_cache.save()
            pending_writes = 0

        progress.update(
            index,
            track.title_canonical or isrc,
            cached=cache_hits,
            fetched=fetched,
            skipped=skipped,
        )

    progress.finish()
    if obs_cache is not None and pending_writes:
        obs_cache.save()
    if new_obs:
        store.add_observations(new_obs)

    stats["total"] = cache_hits + fetched
    stats["cached"] = cache_hits
    stats["fetched"] = fetched
    stats["skipped"] = skipped
    if transient_bail:
        logger.info(
            "Spotify popularity: bailed (%d cached, %d fetched, %d skipped) — re-run to continue",
            cache_hits,
            fetched,
            skipped,
        )
    elif cache_hits or skipped:
        logger.info(
            "Spotify popularity: %d total (%d cached, %d fetched, %d skipped)",
            stats["total"],
            cache_hits,
            fetched,
            skipped,
        )
    else:
        logger.info("Spotify popularity: %d fetched", fetched)
    return stats
