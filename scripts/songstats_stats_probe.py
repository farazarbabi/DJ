"""Probe Songstats stats endpoints to see which fields your plan exposes.

Usage:
  python scripts/songstats_stats_probe.py
  python scripts/songstats_stats_probe.py --isrc USXYZ1234567
  python scripts/songstats_stats_probe.py --isrc USXYZ1234567 --source spotify
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dj_registry.config import RegistryConfig

CANDIDATE_ENDPOINTS = [
    "/tracks/stats",
    "/tracks/historic_stats",
    "/tracks/info",
]


def first_isrc_from_registry(config: RegistryConfig) -> str | None:
    path = config.tracks_master_path
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            isrc = (row.get("isrc_canonical") or "").strip()
            if isrc:
                return isrc
    return None


def probe(base_url: str, api_key: str, endpoint: str, isrc: str, source: str | None) -> None:
    url = f"{base_url.rstrip('/')}{endpoint}"
    params: dict[str, str] = {"isrc": isrc}
    if source:
        params["source"] = source
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "",
        "apikey": api_key,
    }

    print(f"\n=== GET {endpoint}  params={params} ===")
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers, params=params)
    except httpx.RequestError as e:
        print(f"request error: {e}")
        return

    print(f"status: {resp.status_code}")
    body_text = resp.text
    try:
        body = resp.json()
    except ValueError:
        print("non-JSON body:")
        print(body_text[:500])
        return

    pretty = json.dumps(body, indent=2)
    if len(pretty) > 4000:
        print(pretty[:4000])
        print(f"... [truncated, full length {len(pretty)} chars]")
    else:
        print(pretty)

    hits = _scan_for_play_count_keys(body)
    if hits:
        print("\nlikely play-count / stream fields found:")
        for path_, value in hits:
            print(f"  {path_} = {value!r}")
    else:
        print("\nno obvious stream/play-count fields in this response")


_PLAY_COUNT_KEY_HINTS = (
    "stream", "play", "listen", "popularity", "playlist", "follower", "audience",
)


def _scan_for_play_count_keys(node, prefix: str = "") -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            path_ = f"{prefix}.{k}" if prefix else k
            if any(hint in k.lower() for hint in _PLAY_COUNT_KEY_HINTS):
                if not isinstance(v, (dict, list)):
                    found.append((path_, v))
                else:
                    found.append((path_, f"<{type(v).__name__} len={len(v)}>"))
            found.extend(_scan_for_play_count_keys(v, path_))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(_scan_for_play_count_keys(v, f"{prefix}[{i}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isrc", help="ISRC to probe. Defaults to first ISRC in registry.")
    parser.add_argument(
        "--source",
        help="Optional ?source= filter (e.g. spotify, beatport, apple_music).",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        help="Limit to a specific endpoint path. Repeatable. "
             "Default probes /tracks/stats, /tracks/historic_stats, /tracks/info.",
    )
    args = parser.parse_args()

    config = RegistryConfig()
    config.load_env()
    if not config.songstats_api_key:
        print("ERROR: SONGSTATS_API_KEY not set (check .env)")
        return 1

    isrc = args.isrc or first_isrc_from_registry(config)
    if not isrc:
        print("ERROR: no ISRC provided and none found in tracks_master.csv")
        return 1

    print(f"probing Songstats with ISRC={isrc} source={args.source or '<none>'}")
    print(f"base_url={config.songstats_base_url}")

    endpoints = args.endpoint or CANDIDATE_ENDPOINTS
    for endpoint in endpoints:
        probe(config.songstats_base_url, config.songstats_api_key, endpoint, isrc, args.source)

    return 0


if __name__ == "__main__":
    sys.exit(main())
