"""Look up Spotify audio features for all ground truth tracks.

Uses the Spotify Web API (client credentials flow) to search for each
track by artist + title, then fetches audio features (key, mode, tempo,
energy, danceability, valence, etc.).

Outputs: outputs/spotify_features.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

GT_PATH = "files/ground_truth.csv"
OUT_PATH = "outputs/spotify_features.csv"
CACHE_PATH = "outputs/spotify_cache.json"

# Spotify key uses pitch class: 0=C, 1=C#, ..., 11=B
# Mode: 1=major, 0=minor
PITCH_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KEY_TO_CAMELOT = {
    (0, 1): "8B",   (0, 0): "5A",
    (1, 1): "3B",   (1, 0): "12A",
    (2, 1): "10B",  (2, 0): "7A",
    (3, 1): "5B",   (3, 0): "2A",
    (4, 1): "12B",  (4, 0): "9A",
    (5, 1): "7B",   (5, 0): "4A",
    (6, 1): "2B",   (6, 0): "11A",
    (7, 1): "9B",   (7, 0): "6A",
    (8, 1): "4B",   (8, 0): "1A",
    (9, 1): "11B",  (9, 0): "8A",
    (10, 1): "6B",  (10, 0): "3A",
    (11, 1): "1B",  (11, 0): "10A",
}


def load_credentials() -> tuple[str, str]:
    """Load Spotify credentials from .env file."""
    env_path = Path(".env")
    if not env_path.exists():
        print("No .env file found. Create one with client_id and client_secret.")
        sys.exit(1)

    creds = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("#") and not line.startswith("---"):
                key, val = line.split(":", 1)
                creds[key.strip()] = val.strip()

    client_id = creds.get("client_id", "")
    client_secret = creds.get("client_secret", "")
    if not client_id or not client_secret:
        print("Missing client_id or client_secret in .env")
        sys.exit(1)

    return client_id, client_secret


def search_track(sp: spotipy.Spotify, title: str, artist: str) -> dict | None:
    """Search Spotify for a track. Returns the best match or None."""
    # Clean up title: remove "(Original Mix)", "(Extended Mix)", etc.
    clean_title = title
    for suffix in ["(Original Mix)", "(Extended Mix)", "(Extended Remix)"]:
        clean_title = clean_title.replace(suffix, "").strip()

    # Try exact artist + title first
    query = f"track:{clean_title} artist:{artist}"
    results = sp.search(q=query, type="track", limit=5)
    tracks = results.get("tracks", {}).get("items", [])

    if tracks:
        return tracks[0]

    # Fallback: broader search
    query = f"{artist} {clean_title}"
    results = sp.search(q=query, type="track", limit=5)
    tracks = results.get("tracks", {}).get("items", [])

    if tracks:
        return tracks[0]

    # Last resort: just title
    results = sp.search(q=clean_title, type="track", limit=5)
    tracks = results.get("tracks", {}).get("items", [])

    return tracks[0] if tracks else None


def main():
    client_id, client_secret = load_credentials()

    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth)

    # Load ground truth
    gt_rows = []
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gt_rows.append(row)

    print(f"Loaded {len(gt_rows)} tracks from ground truth")

    # Load cache if exists
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached Spotify results")

    results = []
    for i, row in enumerate(gt_rows):
        fname = row["file_name"].strip()
        title = row["Track Title"].strip()
        artist = row["Artist"].strip()

        if fname in cache:
            print(f"  [{i+1}/{len(gt_rows)}] {fname[:50]} -- cached")
            results.append(cache[fname])
            continue

        print(f"  [{i+1}/{len(gt_rows)}] Searching: {artist} - {title}...", end="")

        track = search_track(sp, title, artist)

        if track is None:
            print(" NOT FOUND")
            result = {
                "file_name": fname,
                "spotify_found": False,
                "spotify_id": "",
                "spotify_name": "",
                "spotify_artist": "",
            }
            results.append(result)
            cache[fname] = result
            time.sleep(0.1)
            continue

        track_id = track["id"]
        sp_name = track["name"]
        sp_artist = ", ".join(a["name"] for a in track["artists"])

        # Get audio features
        features = sp.audio_features([track_id])
        feat = features[0] if features and features[0] else {}

        sp_key = feat.get("key", -1)  # 0-11 pitch class, -1 if unknown
        sp_mode = feat.get("mode", -1)  # 1=major, 0=minor

        if sp_key >= 0 and sp_mode >= 0:
            camelot = KEY_TO_CAMELOT.get((sp_key, sp_mode), "??")
            note = PITCH_TO_NOTE[sp_key]
            mode_str = "major" if sp_mode == 1 else "minor"
            key_str = f"{note} {mode_str}"
        else:
            camelot = "??"
            key_str = "unknown"

        result = {
            "file_name": fname,
            "spotify_found": True,
            "spotify_id": track_id,
            "spotify_name": sp_name,
            "spotify_artist": sp_artist,
            "spotify_key": sp_key,
            "spotify_mode": sp_mode,
            "spotify_camelot": camelot,
            "spotify_key_name": key_str,
            "spotify_tempo": feat.get("tempo", 0),
            "spotify_energy": feat.get("energy", 0),
            "spotify_danceability": feat.get("danceability", 0),
            "spotify_valence": feat.get("valence", 0),
            "spotify_loudness": feat.get("loudness", 0),
            "spotify_acousticness": feat.get("acousticness", 0),
            "spotify_instrumentalness": feat.get("instrumentalness", 0),
            "spotify_speechiness": feat.get("speechiness", 0),
            "spotify_liveness": feat.get("liveness", 0),
            "spotify_time_signature": feat.get("time_signature", 4),
            "spotify_duration_ms": feat.get("duration_ms", 0),
        }
        results.append(result)
        cache[fname] = result

        print(f" -> {sp_artist} - {sp_name}")
        print(f"     Key: {key_str} ({camelot})  BPM: {feat.get('tempo', '?'):.1f}  Energy: {feat.get('energy', '?'):.3f}")

        time.sleep(0.1)  # Rate limiting

    # Save cache
    os.makedirs("outputs", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCached {len(cache)} results to {CACHE_PATH}")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        # Ensure all results have all fields
        all_fields = set()
        for r in results:
            all_fields.update(r.keys())
        fieldnames = sorted(all_fields)

        with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Wrote {len(results)} rows to {OUT_PATH}")

    # Summary comparison with ground truth
    print(f"\n{'='*70}")
    print("  Spotify vs Rekordbox Key Comparison")
    print(f"{'='*70}")

    gt_map = {row["file_name"].strip(): row["Key_rekordbox"].strip() for row in gt_rows}
    agree = disagree = missing = 0

    for r in results:
        fname = r["file_name"]
        if not r.get("spotify_found") or r.get("spotify_camelot", "??") == "??":
            missing += 1
            continue
        sp_cam = r["spotify_camelot"]
        rb_cam = gt_map.get(fname, "??")
        if sp_cam == rb_cam:
            agree += 1
        else:
            disagree += 1
            print(f"  {fname[:50]:<50s}  RB={rb_cam:>3s}  SP={sp_cam:>3s}")

    found = agree + disagree
    print(f"\n  Found: {found}/{len(results)}, Agree: {agree}/{found}, Disagree: {disagree}/{found}, Not found: {missing}")


if __name__ == "__main__":
    main()
