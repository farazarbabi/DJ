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
from .spotify_isrc import SpotifyClient

logger = logging.getLogger(__name__)


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
    for index, track in enumerate(candidates, start=1):
        isrc = track.isrc_canonical

        # Cache: full SourceObservation already present
        if obs_cache:
            cached = obs_cache.get_by_isrc(isrc, "spotify")
            if cached and cached.popularity:
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

        spotify_id = spotify_id_by_track.get(track.track_id, "")
        if not spotify_id:
            spotify_id = client.find_id_by_isrc(isrc)
            time.sleep(0.1)
        if not spotify_id:
            skipped += 1
            progress.update(
                index,
                track.title_canonical or isrc,
                cached=cache_hits,
                fetched=fetched,
                skipped=skipped,
            )
            continue

        track_obj = client.get_track(spotify_id)
        time.sleep(0.1)
        if not track_obj:
            skipped += 1
            progress.update(
                index,
                track.title_canonical or isrc,
                cached=cache_hits,
                fetched=fetched,
                skipped=skipped,
            )
            continue

        popularity = track_obj.get("popularity")
        if popularity is None:
            skipped += 1
            progress.update(
                index,
                track.title_canonical or isrc,
                cached=cache_hits,
                fetched=fetched,
                skipped=skipped,
            )
            continue

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
        if obs_cache:
            obs_cache.put_by_isrc(isrc, "spotify", obs)
        fetched += 1
        progress.update(
            index,
            track.title_canonical or isrc,
            cached=cache_hits,
            fetched=fetched,
            skipped=skipped,
        )

    progress.finish()
    if new_obs:
        store.add_observations(new_obs)

    stats["total"] = cache_hits + fetched
    stats["cached"] = cache_hits
    stats["fetched"] = fetched
    stats["skipped"] = skipped
    if cache_hits or skipped:
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
