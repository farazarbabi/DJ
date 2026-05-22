"""Backfill the vocal_stem cache layer for the library.

Runs Demucs on every track that has DSP cached but no vocal_stem layer yet.
After this completes, ``dj run --no-grouping --no-tags`` (or any re-derive
pass) will pick up stem signals and refresh vocal_profile classifications
without redoing DSP or other expensive analysis.

Cost: ~30-45s per track on CPU. One-time operation.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new stems (0 = all).")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--music-dir", default=str(REPO_ROOT / "Music"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from dj_tagger.analyzers.vocal_stem import (
        analyze_vocal_stem,
        is_demucs_available,
        result_to_dict,
    )
    from dj_tagger.audio import load_audio_features
    from dj_tagger.universal_cache import get_cache, quick_duration

    if not is_demucs_available():
        print("Demucs not installed. Run: pip install demucs", file=sys.stderr)
        return 1

    music_dir = Path(args.music_dir)
    ucache = get_cache(str(Path(args.cache_dir) / "universal_cache.pkl"))

    files = sorted([p for p in music_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".mp3", ".aiff", ".aif", ".flac", ".m4a", ".wav"}])
    print(f"Found {len(files)} audio files in {music_dir}")

    todo: list[tuple[Path, float]] = []
    cached = 0
    for path in files:
        dur = quick_duration(str(path)) or 0.0
        if dur < 30.0:
            continue
        existing = ucache.get_track(path.name, dur, "vocal_stem")
        if isinstance(existing, dict) and "vocal_stem_mix_ratio_db" in existing:
            cached += 1
            continue
        todo.append((path, dur))

    print(f"  {cached} already have stem cached")
    print(f"  {len(todo)} need extraction")
    if args.limit and len(todo) > args.limit:
        todo = todo[: args.limit]
        print(f"  Limited to {args.limit} per --limit")

    if not todo:
        print("Nothing to do.")
        return 0

    t0 = time.time()
    n_failed = 0
    save_every = 10
    for idx, (path, dur) in enumerate(todo, 1):
        ts = time.time()
        try:
            track_audio = load_audio_features(str(path))
            result = analyze_vocal_stem(track_audio)
        except Exception as e:
            n_failed += 1
            print(f"  [{idx}/{len(todo)}] FAIL  {path.name}: {e}", flush=True)
            continue
        if result is None:
            n_failed += 1
            print(f"  [{idx}/{len(todo)}] SKIP  {path.name} (analyzer returned None)", flush=True)
            continue

        data = result_to_dict(result)
        ucache.put_track(path.name, dur, "vocal_stem", data)
        print(
            f"  [{idx}/{len(todo)}] OK  "
            f"ratio={data['vocal_stem_mix_ratio_db']:6.2f}  "
            f"act={data['vocal_stem_activity_frac']:.2f}  "
            f"({time.time() - ts:.1f}s)  {path.name[:60]}",
            flush=True,
        )

        if idx % save_every == 0:
            ucache.save()

    ucache.save()
    elapsed = time.time() - t0
    print(f"\nDone. {len(todo) - n_failed} succeeded, {n_failed} failed/skipped in {elapsed:.1f}s "
          f"({elapsed/max(len(todo), 1):.1f}s/track avg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
