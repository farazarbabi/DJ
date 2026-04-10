"""Look up Songstats audio analysis for all ground truth tracks.

Uses the Songstats Enterprise API (via Tunebat partner) to fetch
key, mode, tempo, energy, and other audio features for each track.

Prerequisites:
  1. Add your Songstats API key to .env:  songstats_api_key: YOUR_KEY
  2. Have Spotify IDs cached (run spotify_lookup.py first, or this will search)

Outputs: outputs/songstats_features.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

GT_PATH = "files/ground_truth.csv"
SPOTIFY_IDS_PATH = "outputs/spotify_ids.json"
SONGSTATS_CACHE_PATH = "outputs/songstats_cache.json"
OUT_PATH = "outputs/songstats_features.csv"

API_BASE = "https://api.songstats.com/enterprise/v1"

PITCH_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Map note name -> pitch class (handle both sharp and flat)
NOTE_TO_PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
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


def load_env() -> dict[str, str]:
    """Load key=value and key: value pairs from .env."""
    env = {}
    env_path = Path(".env")
    if not env_path.exists():
        return env
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            for sep in [":", "="]:
                if sep in line:
                    key, val = line.split(sep, 1)
                    env[key.strip()] = val.strip()
                    break
    return env


def load_spotify_ids() -> dict[str, dict]:
    """Load cached Spotify IDs, or return empty dict."""
    if os.path.exists(SPOTIFY_IDS_PATH):
        with open(SPOTIFY_IDS_PATH) as f:
            return json.load(f)
    return {}


def search_spotify_id(sp: spotipy.Spotify, title: str, artist: str) -> str | None:
    """Search Spotify for a track ID."""
    clean = title
    for suffix in ["(Original Mix)", "(Extended Mix)", "(Extended Remix)"]:
        clean = clean.replace(suffix, "").strip()

    query = f"track:{clean} artist:{artist}"
    results = sp.search(q=query, type="track", limit=3)
    tracks = results.get("tracks", {}).get("items", [])

    if not tracks:
        query = f"{artist} {clean}"
        results = sp.search(q=query, type="track", limit=3)
        tracks = results.get("tracks", {}).get("items", [])

    return tracks[0]["id"] if tracks else None


def fetch_songstats_track(api_key: str, spotify_id: str) -> dict | None:
    """Fetch track info from Songstats API."""
    url = f"{API_BASE}/tracks/info"
    headers = {
        "Accept": "application/json",
        "apikey": api_key,
    }
    params = {"spotify_track_id": spotify_id}

    r = requests.get(url, headers=headers, params=params, timeout=15)
    if r.status_code == 200:
        return r.json()
    else:
        return {"status": r.status_code, "message": r.text[:200]}


def parse_audio_analysis(data: dict) -> dict:
    """Parse Songstats response into flat feature dict."""
    result = {}

    # Track info
    track_info = data.get("track_info", {})
    result["songstats_title"] = track_info.get("title", "")
    artists = track_info.get("artists", [])
    result["songstats_artist"] = ", ".join(a.get("name", "") for a in artists)
    result["songstats_genres"] = "|".join(track_info.get("genres", []))

    # Audio analysis
    analysis = data.get("audio_analysis", [])
    for item in analysis:
        key = item.get("key", "")
        value = item.get("value", "")
        result[f"ss_{key}"] = value

    # Derive Camelot from key + mode
    ss_key = result.get("ss_key", "")
    ss_mode = result.get("ss_mode", "")

    if ss_key and ss_mode:
        pitch = NOTE_TO_PITCH.get(ss_key, -1)
        mode_int = int(float(ss_mode)) if ss_mode else -1
        if pitch >= 0 and mode_int >= 0:
            camelot = KEY_TO_CAMELOT.get((pitch, mode_int), "??")
            mode_str = "major" if mode_int == 1 else "minor"
            result["ss_camelot"] = camelot
            result["ss_key_name"] = f"{ss_key} {mode_str}"
        else:
            result["ss_camelot"] = "??"
            result["ss_key_name"] = "??"
    else:
        result["ss_camelot"] = "??"
        result["ss_key_name"] = "??"

    return result


def camelot_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    try:
        an, al = int(a[:-1]), a[-1]
        bn, bl = int(b[:-1]), b[-1]
    except (ValueError, IndexError):
        return 99
    if al == bl:
        return min(abs(an - bn), 12 - abs(an - bn))
    return min(abs(an - bn), 12 - abs(an - bn)) + 1


def main():
    env = load_env()

    # Songstats API key
    api_key = env.get("songstats_api_key", "")
    if not api_key:
        print("ERROR: Add 'songstats_api_key: YOUR_KEY' to .env")
        sys.exit(1)

    # Spotify credentials (for searching track IDs)
    sp_client_id = env.get("client_id", "")
    sp_client_secret = env.get("client_secret", "")

    # Load ground truth
    gt_rows = []
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt_rows.append(row)
    print(f"Loaded {len(gt_rows)} tracks from ground truth")

    # Load Spotify IDs
    spotify_ids = load_spotify_ids()

    # Set up Spotify client for any missing IDs
    sp = None
    if sp_client_id and sp_client_secret:
        auth = SpotifyClientCredentials(client_id=sp_client_id, client_secret=sp_client_secret)
        sp = spotipy.Spotify(auth_manager=auth)

    # Load Songstats cache
    ss_cache = {}
    if os.path.exists(SONGSTATS_CACHE_PATH):
        with open(SONGSTATS_CACHE_PATH) as f:
            ss_cache = json.load(f)
        print(f"Loaded {len(ss_cache)} cached Songstats results")

    results = []
    for i, row in enumerate(gt_rows):
        fname = row["file_name"].strip()
        title = row["Track Title"].strip()
        artist = row["Artist"].strip()

        # Check cache
        if fname in ss_cache and ss_cache[fname].get("ss_camelot", "??") != "??":
            print(f"  [{i+1:2d}/{len(gt_rows)}] {fname[:50]} -- cached")
            results.append(ss_cache[fname])
            continue

        # Get Spotify ID
        sp_data = spotify_ids.get(fname)
        sp_id = sp_data["id"] if sp_data else None

        if not sp_id and sp:
            sp_id = search_spotify_id(sp, title, artist)
            if sp_id:
                spotify_ids[fname] = {"id": sp_id, "name": title, "artist": artist}

        if not sp_id:
            print(f"  [{i+1:2d}/{len(gt_rows)}] {fname[:50]} -- no Spotify ID, skipping")
            results.append({"file_name": fname, "ss_camelot": "??", "ss_key_name": "??"})
            continue

        # Fetch from Songstats
        print(f"  [{i+1:2d}/{len(gt_rows)}] {fname[:50]}...", end="", flush=True)
        data = fetch_songstats_track(api_key, sp_id)

        if data and data.get("result") == "success":
            parsed = parse_audio_analysis(data)
            parsed["file_name"] = fname
            parsed["spotify_id"] = sp_id
            results.append(parsed)
            ss_cache[fname] = parsed
            print(f" -> {parsed.get('ss_key_name', '??')} ({parsed.get('ss_camelot', '??')}) "
                  f"BPM={parsed.get('ss_tempo', '?')} Energy={parsed.get('ss_energy', '?')}")
        else:
            status = data.get("status", "?") if data else "?"
            msg = data.get("message", "unknown error")[:100] if data else "no response"
            print(f" FAILED ({status}: {msg})")
            results.append({"file_name": fname, "ss_camelot": "??", "error": str(msg)})
            ss_cache[fname] = {"file_name": fname, "ss_camelot": "??", "error": str(msg)}

        # Save cache incrementally
        if (i + 1) % 5 == 0:
            with open(SONGSTATS_CACHE_PATH, "w") as f:
                json.dump(ss_cache, f, indent=2)

        time.sleep(0.3)  # Rate limiting

    # Save final cache
    os.makedirs("outputs", exist_ok=True)
    with open(SONGSTATS_CACHE_PATH, "w") as f:
        json.dump(ss_cache, f, indent=2)

    # Save Spotify IDs
    with open(SPOTIFY_IDS_PATH, "w") as f:
        json.dump(spotify_ids, f, indent=2)

    # Write CSV
    if results:
        all_fields = set()
        for r in results:
            all_fields.update(r.keys())
        fieldnames = ["file_name"] + sorted(all_fields - {"file_name"})

        with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nWrote {len(results)} rows to {OUT_PATH}")

    # Comparison summary
    gt_map = {row["file_name"].strip(): row["Key_rekordbox"].strip() for row in gt_rows}

    print(f"\n{'='*70}")
    print("  Songstats vs Rekordbox Key Comparison")
    print(f"{'='*70}")

    agree = disagree = compat = missing = 0
    for r in results:
        fname = r.get("file_name", "")
        ss_cam = r.get("ss_camelot", "??")
        rb_cam = gt_map.get(fname, "??")

        if ss_cam == "??":
            missing += 1
            continue

        dist = camelot_distance(ss_cam, rb_cam)
        if dist == 0:
            agree += 1
        elif dist <= 1:
            compat += 1
            print(f"  {fname[:50]:<50s}  RB={rb_cam:>3s}  SS={ss_cam:>3s}  (neighbour)")
        else:
            disagree += 1
            print(f"  {fname[:50]:<50s}  RB={rb_cam:>3s}  SS={ss_cam:>3s}  ** DISAGREE (dist={dist})")

    found = agree + compat + disagree
    print(f"\n  Found: {found}/{len(results)}")
    print(f"  Exact agree: {agree}/{found}")
    print(f"  Neighbour: {compat}/{found}")
    print(f"  Disagree: {disagree}/{found}")
    print(f"  Missing: {missing}")

    # Also compare our predictions vs Songstats
    print(f"\n{'='*70}")
    print("  Our Predictions vs Songstats (independent validation)")
    print(f"{'='*70}")

    our_exact = our_compat = our_disagree = 0
    for r in results:
        fname = r.get("file_name", "")
        ss_cam = r.get("ss_camelot", "??")
        if ss_cam == "??":
            continue

        # Load our prediction from ground truth CSV
        for row in gt_rows:
            if row["file_name"].strip() == fname:
                our_cam = row.get("pred_Key_v1", "??").strip()
                if our_cam and our_cam != "??":
                    dist = camelot_distance(our_cam, ss_cam)
                    if dist == 0:
                        our_exact += 1
                    elif dist <= 1:
                        our_compat += 1
                    else:
                        our_disagree += 1
                        print(f"  {fname[:50]:<50s}  Ours={our_cam:>3s}  SS={ss_cam:>3s}  (dist={dist})")
                break

    our_total = our_exact + our_compat + our_disagree
    if our_total > 0:
        print(f"\n  Exact: {our_exact}/{our_total}")
        print(f"  Neighbour: {our_compat}/{our_total}")
        print(f"  Disagree: {our_disagree}/{our_total}")


if __name__ == "__main__":
    main()
