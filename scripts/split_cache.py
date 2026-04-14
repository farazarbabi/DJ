"""Temporary script: split universal_cache.pkl into raw_cache.pkl + derived_cache.pkl.

Reads all existing cache files (universal_cache.pkl, tagger_cache.pkl,
features_cache.pkl, clap_cache.pkl, registry_cache.pkl), deduplicates,
and writes:
  - cache/raw_cache.pkl    — permanent raw layers (dsp, clap, raw_analysis, tag, rekordbox, songstats, etc.)
  - cache/derived_cache.pkl — derived layers (tagger) with current version

Run from project root:
    python scripts/split_cache.py

After verifying the new files, delete the old ones:
    del cache\\universal_cache.pkl cache\\tagger_cache.pkl cache\\features_cache.pkl cache\\clap_cache.pkl
"""

import os
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dj_tagger.universal_cache import (
    UniversalCache, CacheEntry, RAW_LAYERS, DERIVED_VERSIONS, CACHE_VERSION,
)

CACHE_DIR = "cache"
RAW_PATH = os.path.join(CACHE_DIR, "raw_cache.pkl")
DERIVED_PATH = os.path.join(CACHE_DIR, "derived_cache.pkl")


def _layer_from_key(key: str) -> str | None:
    """Extract the layer name from a cache key."""
    # ISRC keys: "isrc:XXX|songstats" → "songstats"
    if key.startswith("isrc:"):
        parts = key.split("|")
        return parts[-1] if len(parts) >= 2 else None
    # Track keys: "filename|duration|layer" or "filename|layer"
    parts = key.split("|")
    return parts[-1] if len(parts) >= 2 else None


def _is_raw(layer: str | None) -> bool:
    if layer is None:
        return False
    return layer in RAW_LAYERS


def main():
    # Load the universal cache (auto-migrates from legacy files)
    ucache_path = os.path.join(CACHE_DIR, "universal_cache.pkl")
    print(f"Loading universal cache from {ucache_path}...")
    ucache = UniversalCache(ucache_path)
    print(f"  {len(ucache)} total entries")

    raw_entries: dict[str, CacheEntry] = {}
    derived_entries: dict[str, CacheEntry] = {}
    skipped = 0

    for key, entry in ucache._entries.items():
        layer = _layer_from_key(key)
        if _is_raw(layer):
            raw_entries[key] = entry
        elif layer is not None:
            # Only keep entries with the current derived version
            expected_ver = DERIVED_VERSIONS.get(layer)
            if expected_ver is None or entry.version == expected_ver:
                derived_entries[key] = entry
            else:
                skipped += 1
        else:
            # Unknown format — put in raw to be safe
            raw_entries[key] = entry

    print(f"\nSplit results:")
    print(f"  Raw entries:     {len(raw_entries)}")
    print(f"  Derived entries: {len(derived_entries)}")
    print(f"  Skipped (stale): {skipped}")

    # Show layer breakdown
    raw_layers: dict[str, int] = {}
    for key in raw_entries:
        layer = _layer_from_key(key) or "unknown"
        raw_layers[layer] = raw_layers.get(layer, 0) + 1
    derived_layers: dict[str, int] = {}
    for key in derived_entries:
        layer = _layer_from_key(key) or "unknown"
        derived_layers[layer] = derived_layers.get(layer, 0) + 1

    print(f"\n  Raw layers:")
    for layer, count in sorted(raw_layers.items()):
        print(f"    {layer}: {count}")
    print(f"\n  Derived layers:")
    for layer, count in sorted(derived_layers.items()):
        print(f"    {layer}: {count}")

    # Write raw_cache.pkl
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RAW_PATH, "wb") as f:
        pickle.dump({"_version": CACHE_VERSION, "entries": raw_entries}, f)
    raw_size = os.path.getsize(RAW_PATH)
    print(f"\nWritten: {RAW_PATH} ({raw_size:,} bytes, {len(raw_entries)} entries)")

    # Write derived_cache.pkl
    with open(DERIVED_PATH, "wb") as f:
        pickle.dump({"_version": CACHE_VERSION, "entries": derived_entries}, f)
    derived_size = os.path.getsize(DERIVED_PATH)
    print(f"Written: {DERIVED_PATH} ({derived_size:,} bytes, {len(derived_entries)} entries)")

    print(f"\nTotal: {raw_size + derived_size:,} bytes")
    print(f"\nOld files you can now delete:")
    for name in ["universal_cache.pkl", "tagger_cache.pkl", "features_cache.pkl", "clap_cache.pkl", "registry_cache.pkl"]:
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
