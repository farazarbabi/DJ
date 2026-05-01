"""Spotify API client — look up ISRCs by artist+title search."""

from __future__ import annotations

import base64
import logging
import os
import time

import httpx

from dj_tagger.cache import load_cache, save_cache, cache_key

from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


class SpotifyClient:
    """Spotify Web API client for ISRC lookup."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str = ""
        self._token_expires: float = 0.0

    def _authenticate(self) -> None:
        """Get or refresh access token via client credentials flow."""
        if self._token and time.time() < self._token_expires - 60:
            return

        creds = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                TOKEN_URL,
                headers={"Authorization": f"Basic {creds}"},
                data={"grant_type": "client_credentials"},
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expires = time.time() + data.get("expires_in", 3600)

        logger.debug("Spotify token obtained")

    def search_track(
        self, artist: str, title: str, duration_sec: float = 0.0
    ) -> dict | None:
        """Search for a track and return the best match with ISRC.

        If duration_sec is provided, filters results to within 5 seconds.
        Returns dict with keys: isrc, spotify_id, name, artists, duration_sec, or None.
        """
        self._authenticate()

        import re

        # Clean title: strip parenthesized mix/version info, featuring, artist prefix
        clean_title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip()
        # Remove "feat./ft." and everything after
        clean_title = re.sub(r"\s+(feat\.?|ft\.?|featuring)\s+.*$", "", clean_title, flags=re.IGNORECASE)
        # Remove artist name from title if it appears as prefix (e.g., "Kosheen - Addict" -> "Addict")
        if artist and " - " in clean_title:
            after_dash = clean_title.split(" - ", 1)[1].strip()
            if after_dash:
                clean_title = after_dash

        # Use only first artist for comma-separated multi-artist credits
        # Keep & in artist names (e.g., "Eli & Fur" is one act, not two)
        clean_artist = artist
        if clean_artist and "," in clean_artist:
            clean_artist = clean_artist.split(",")[0].strip()

        # Build search queries to try (most specific first, then fallbacks)
        queries = []
        if clean_artist:
            queries.append(f"artist:{clean_artist} track:{clean_title}")
        queries.append(f"{clean_artist} {clean_title}".strip())  # plain text fallback

        items = []
        for query in queries:
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get(
                        SEARCH_URL,
                        headers={"Authorization": f"Bearer {self._token}"},
                        params={"q": query, "type": "track", "limit": 10},
                    )

                    if resp.status_code == 429:
                        retry = int(resp.headers.get("Retry-After", "5"))
                        logger.warning("Spotify rate limited, waiting %ds", retry)
                        time.sleep(retry)
                        resp = client.get(
                            SEARCH_URL,
                            headers={"Authorization": f"Bearer {self._token}"},
                            params={"q": query, "type": "track", "limit": 10},
                        )

                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("tracks", {}).get("items", [])
                    if items:
                        break

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.debug("Spotify search failed for '%s': %s", query, e)
                continue
        if not items:
            return None

        # Score candidates by duration match — strict first, then relaxed
        STRICT_TOLERANCE = 10.0  # seconds
        best = None
        best_diff = float("inf")

        for item in items:
            isrc = item.get("external_ids", {}).get("isrc", "")
            if not isrc:
                continue

            spotify_dur = item.get("duration_ms", 0) / 1000.0

            if duration_sec > 0 and spotify_dur > 0:
                diff = abs(duration_sec - spotify_dur)
                if diff < best_diff:
                    best_diff = diff
                    best = item
            elif best is None:
                best = item

        # If best match is way off on duration, still accept it but with a warning
        if best and duration_sec > 0 and best_diff > STRICT_TOLERANCE:
            logger.debug("Spotify: best match for '%s' has %ds duration diff", clean_title, int(best_diff))

        if not best:
            return None

        isrc = best.get("external_ids", {}).get("isrc", "")
        if not isrc:
            return None

        return {
            "isrc": isrc,
            "spotify_id": best.get("id", ""),
            "name": best.get("name", ""),
            "artists": ", ".join(a.get("name", "") for a in best.get("artists", [])),
            "duration_sec": best.get("duration_ms", 0) / 1000.0,
        }


TAGGER_CACHE_PATH = os.path.join("cache", "tagger_cache.pkl")


def enrich_isrcs(store: CsvStore, limit: int | None = None, *, show_progress: bool = False) -> int:
    """Look up ISRCs from Spotify for tracks missing them.

    Reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from environment.
    Writes ISRCs to both the registry CSV and the shared tagger cache.
    Returns number of tracks enriched.
    """
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.warning("Spotify credentials not configured (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)")
        return 0

    client = SpotifyClient(client_id, client_secret)
    tracks = store.load_tracks()
    files = store.load_files()

    # Build file lookup: track_id -> primary file
    file_by_track = {}
    for f in files:
        if f.is_primary_file and f.track_id:
            file_by_track[f.track_id] = f

    # Load shared tagger cache
    tagger_cache = load_cache(TAGGER_CACHE_PATH)

    # Check cache first — skip tracks whose file already has an ISRC cached
    candidates = []
    cache_hits = 0
    for t in tracks:
        if t.isrc_canonical:
            continue
        if not t.artist_canonical and not t.title_canonical:
            continue
        frec = file_by_track.get(t.track_id)
        if frec:
            # Check cache with duration-aware key
            dur = frec.audio_duration_sec if frec.audio_duration_sec > 0 else None
            ck = cache_key(frec.path_abs, dur)
            entry = tagger_cache.get(ck)
            if not entry:
                # Fall back to name-only key
                ck = cache_key(frec.path_abs)
                entry = tagger_cache.get(ck)
            if entry and entry.result.get("isrc"):
                t.isrc_canonical = entry.result["isrc"]
                cache_hits += 1
                continue
        candidates.append(t)

    if limit:
        candidates = candidates[:limit]

    if not candidates:
        if cache_hits:
            store.save_tracks(tracks)
            logger.info("ISRCs: %d from cache, 0 to fetch", cache_hits)
        return cache_hits

    logger.info("ISRCs: %d cached, %d to fetch from Spotify", cache_hits, len(candidates))

    enriched = 0
    cache_dirty = False
    progress = ProgressBar(len(candidates), label="Spotify ISRC", enabled=show_progress)
    for index, t in enumerate(candidates, start=1):
        frec = file_by_track.get(t.track_id)
        dur = frec.audio_duration_sec if frec and frec.audio_duration_sec > 0 else 0.0
        result = client.search_track(t.artist_canonical, t.title_canonical, duration_sec=dur)
        if result and result["isrc"]:
            t.isrc_canonical = result["isrc"]
            enriched += 1
            logger.debug("  ISRC %s <- %s - %s", result["isrc"], t.artist_canonical, t.title_canonical)

            # Write ISRC + spotify_id to tagger cache (both key formats)
            if frec:
                for ck in (cache_key(frec.path_abs, dur if dur > 0 else None), cache_key(frec.path_abs)):
                    entry = tagger_cache.get(ck)
                    if entry:
                        entry.result["isrc"] = result["isrc"]
                        entry.result["spotify_id"] = result["spotify_id"]
                        cache_dirty = True
        else:
            logger.debug("  No ISRC found for %s - %s", t.artist_canonical, t.title_canonical)

        progress.update(index, t.title_canonical, cached=cache_hits, enriched=enriched)
        time.sleep(0.1)

    progress.finish()
    store.save_tracks(tracks)
    if cache_dirty:
        save_cache(tagger_cache, TAGGER_CACHE_PATH)

    total = cache_hits + enriched
    not_found = len(candidates) - enriched
    if not_found:
        logger.info("ISRCs: %d found (%d cached + %d Spotify), %d not found", total, cache_hits, enriched, not_found)
    else:
        logger.info("ISRCs: %d found (%d cached + %d Spotify)", total, cache_hits, enriched)
    return total
