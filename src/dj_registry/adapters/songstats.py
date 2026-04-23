"""Songstats API client — ISRC-based metadata lookup."""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

from ..config import RegistryConfig
from ..key_utils import parse_any_key
from ..models import SourceObservation, PayloadIndexEntry, now_iso
from ..store.csv_store import CsvStore
from ..store.obs_cache import ObsCache

logger = logging.getLogger(__name__)


class SongstatsClient:
    """Songstats Enterprise API v1 client."""

    def __init__(self, config: RegistryConfig) -> None:
        self.base_url = config.songstats_base_url.rstrip("/")
        self.api_key = config.songstats_api_key
        self.rate_limit = config.songstats_rate_limit
        self._last_request_time = 0.0

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "",
            "apikey": self.api_key,
        }

    def _throttle(self) -> None:
        """Enforce rate limiting."""
        if self.rate_limit <= 0:
            return
        interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.time()

    def fetch_track_by_isrc(self, isrc: str) -> dict | None:
        """Fetch track info by ISRC. Returns parsed JSON or None."""
        if not self.api_key:
            logger.warning("Songstats API key not configured, skipping")
            return None

        self._throttle()

        url = f"{self.base_url}/tracks/info"
        params = {"isrc": isrc}

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=self._headers(), params=params)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    logger.warning("Rate limited, waiting %ds", retry_after)
                    time.sleep(retry_after)
                    resp = client.get(url, headers=self._headers(), params=params)

                if resp.status_code == 404:
                    logger.debug("No Songstats result for ISRC %s", isrc)
                    return None

                resp.raise_for_status()
                return resp.json()

        except httpx.HTTPStatusError as e:
            logger.warning("Songstats API error for ISRC %s: %s", isrc, e)
            return None
        except httpx.RequestError as e:
            logger.warning("Songstats request failed for ISRC %s: %s", isrc, e)
            return None


def ingest_songstats(
    config: RegistryConfig,
    store: CsvStore,
    obs_cache: ObsCache | None = None,
    *,
    limit: int | None = None,
) -> dict:
    """Fetch Songstats metadata for tracks with ISRC.

    Uses pkl cache to avoid redundant API calls and JSON re-parsing.
    Returns stats dict: {"total", "cached", "fetched", "candidates"}.
    """
    stats = {"total": 0, "cached": 0, "fetched": 0, "candidates": 0}

    config.load_env()
    if not config.songstats_api_key:
        logger.info("Songstats: skipped (no API key)")
        return stats

    client = SongstatsClient(config)
    tracks = store.load_tracks()

    candidates = [t for t in tracks if t.isrc_canonical]

    if limit:
        candidates = candidates[:limit]

    if not candidates:
        logger.info("Songstats: no tracks with ISRC")
        return stats

    stats["candidates"] = len(candidates)

    logger.debug("Songstats: fetching %d tracks", len(candidates))

    raw_dir = os.path.join(config.raw_dir, "songstats")
    os.makedirs(raw_dir, exist_ok=True)

    new_obs: list[SourceObservation] = []
    new_payloads: list[PayloadIndexEntry] = []
    success_count = 0

    cache_hits = 0
    for track in candidates:
        isrc = track.isrc_canonical

        # Check pkl cache first — no file I/O or API needed
        if obs_cache:
            cached = obs_cache.get_by_isrc(isrc, "songstats")
            if cached and (cached.key_camelot or cached.bpm or cached.genre):
                cached.track_id = track.track_id
                cached.observation_id = f"OBS-ss-{isrc}"
                new_obs.append(cached)
                cache_hits += 1
                success_count += 1
                continue

        payload_path = os.path.join(raw_dir, f"{isrc}.json")

        # Try cached raw payload to avoid API call
        result = None
        if os.path.exists(payload_path):
            try:
                with open(payload_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                logger.debug("Songstats: reusing cached payload for %s", isrc)
            except Exception:
                pass

        if result is None:
            result = client.fetch_track_by_isrc(isrc)
            if result is None:
                continue
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        payload_ref = f"songstats-{isrc}"
        new_payloads.append(PayloadIndexEntry(
            payload_ref=payload_ref,
            source_system="songstats",
            source_type="json",
            source_object_id=isrc,
            track_id=track.track_id,
            payload_path=os.path.relpath(payload_path, config.output_dir),
            fetched_at=now_iso(),
        ))

        # Extract fields from nested response
        track_info = result.get("track_info", {})
        audio_analysis = result.get("audio_analysis", [])

        ss_id = track_info.get("songstats_track_id", "")
        ss_release_date = track_info.get("release_date", "")

        # Genres: list of strings
        genres = track_info.get("genres") or []
        ss_genre = genres[0] if genres else ""

        # Labels: list of {"name": ..., "songstats_label_id": ...}
        labels = track_info.get("labels", [])
        ss_label = labels[0]["name"] if labels else ""

        # Audio analysis: list of {"key": field_name, "value": value}
        analysis_map = {item["key"]: item["value"] for item in audio_analysis}

        # Key: "key" has note (e.g., "F"), "mode" has 1.0=major/0.0=minor
        ss_key_note = analysis_map.get("key", "")
        ss_mode = analysis_map.get("mode", "")
        key_std, key_cam = "", ""
        if ss_key_note:
            mode_str = "major" if ss_mode in ("1.0", "1") else "minor"
            parsed = parse_any_key(f"{ss_key_note} {mode_str}")
            if parsed:
                key_std, key_cam = parsed

        # BPM from tempo
        ss_bpm = ""
        tempo = analysis_map.get("tempo", "")
        if tempo:
            try:
                ss_bpm = str(round(float(tempo)))
            except ValueError:
                pass

        # Platform IDs from links
        links = track_info.get("links", [])
        spotify_id = ""
        beatport_id = ""
        apple_music_id = ""
        for link in links:
            src = link.get("source", "")
            eid = link.get("external_id", "")
            if src == "spotify" and not spotify_id:
                spotify_id = eid
            elif src == "beatport" and not beatport_id:
                beatport_id = eid
            elif src == "apple_music" and not apple_music_id:
                apple_music_id = eid

        # One observation row per track with all available data
        obs = SourceObservation(
            observation_id=f"OBS-ss-{isrc}",
            track_id=track.track_id,
            source_system="songstats",
            source_object_id=ss_id,
            key_standard=key_std,
            key_camelot=key_cam,
            key_confidence=1.0 if key_cam else 0.0,
            bpm=ss_bpm,
            genre=ss_genre,
            genres_all="; ".join(genres),
            label=ss_label,
            release_date=ss_release_date,
            is_remix=str(track_info.get("is_remix", "")),
            acousticness=analysis_map.get("acousticness", ""),
            danceability=analysis_map.get("danceability", ""),
            energy=analysis_map.get("energy", ""),
            instrumentalness=analysis_map.get("instrumentalness", ""),
            liveness=analysis_map.get("liveness", ""),
            loudness=analysis_map.get("loudness", ""),
            speechiness=analysis_map.get("speechiness", ""),
            valence=analysis_map.get("valence", ""),
            time_signature=analysis_map.get("time_signature", ""),
            spotify_id=spotify_id,
            beatport_id=beatport_id,
            apple_music_id=apple_music_id,
            payload_ref=payload_ref,
            observed_at=now_iso(),
        )
        new_obs.append(obs)
        if obs_cache:
            obs_cache.put_by_isrc(isrc, "songstats", obs)

        success_count += 1

        # Save progress every 20 tracks
        if success_count % 20 == 0:
            store.add_observations(new_obs)
            new_obs = []
            all_payloads = store.load_payload_index()
            all_payloads.extend(new_payloads)
            store.save_payload_index(all_payloads)
            new_payloads = []
            logger.debug("Songstats: %d/%d", success_count, len(candidates))

    # Final save
    if new_obs:
        store.add_observations(new_obs)
    if new_payloads:
        all_payloads = store.load_payload_index()
        all_payloads.extend(new_payloads)
        store.save_payload_index(all_payloads)

    fetched = success_count - cache_hits
    stats["total"] = success_count
    stats["cached"] = cache_hits
    stats["fetched"] = fetched
    if cache_hits:
        logger.info("Songstats: %d tracks (%d cached, %d fetched)", success_count, cache_hits, fetched)
    else:
        logger.info("Songstats: %d tracks fetched", success_count)
    return stats
