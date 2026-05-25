"""Populate the CLAP and Demucs vocal_stem cache layers over the library.

One-time compute job to fill the raw-cache layers that the DJ subgenre
classifier wants as dense features. Both layers are identity-keyed by
filename + duration in the universal cache, so this is safe to re-run —
already-cached tracks are skipped.

Why this exists: the cache currently has DSP for ~100% of the library but
CLAP for only ~17% and Demucs vocal_stem for ~4% (because both analyzers
were added after many tracks were first analyzed and the raw layers are
never invalidated by signature changes). Without backfilling, the
classifier's CLAP/vocal_stem helpers no-op on most rows.

Cost:
  - CLAP : ~5 minutes for 350 tracks on CPU (batches of 20).
  - Demucs: ~30-45 seconds per track on CPU; faster on GPU.

Usage:
  python scripts/populate_audio_embeddings.py --files "D:\\Music"
  python scripts/populate_audio_embeddings.py --files "D:\\Music" --skip-demucs
  python scripts/populate_audio_embeddings.py --files "D:\\Music" --limit 10
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

# Silence the huggingface_hub auth-missing warning (we use only public
# weights so the warning is just noise). Must happen BEFORE any transitive
# HF import — _populate_clap pulls in laion_clap → transformers → hf_hub.
import os  # noqa: E402
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".aiff", ".aif", ".wav", ".flac", ".m4a"}


def _discover_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def _populate_clap(files: list[Path], cache_dir: Path, limit: int = 0) -> None:
    """Extract CLAP embeddings via the existing grouper helper.

    ``extract_clap_incremental`` already handles batching, caching to the
    universal cache under the ``clap`` layer, and skipping files that are
    already cached. We just wrap the discover/limit logic around it.
    """
    from dj_grouper.features.embeddings import extract_clap_incremental, is_clap_available
    from dj_tagger.universal_cache import get_cache, quick_duration

    if not is_clap_available():
        print("CLAP not installed (laion_clap + torch missing). Skipping.")
        return

    ucache_path = str(cache_dir / "raw_cache.pkl")
    ucache = get_cache(ucache_path)

    todo: list[Path] = []
    already_cached = 0
    for path in files:
        dur = quick_duration(str(path))
        existing = ucache.get_track(path.name, dur, "clap")
        if existing is None:
            todo.append(path)
        else:
            already_cached += 1

    unlimited_todo = len(todo)
    limited = bool(limit and unlimited_todo > limit)
    if limited:
        todo = todo[:limit]

    pieces = [f"CLAP: {already_cached} already cached", f"{unlimited_todo} pending"]
    if limited:
        pieces.append(f"running first {limit} (--limit)")
    print("  ".join(pieces))
    if not todo:
        return

    t0 = time.time()
    # extract_clap_incremental writes to the same ucache via get_cache(); pass
    # the full set so its internal "already cached" check is authoritative.
    extract_clap_incremental(
        [str(p) for p in todo],
        cache_path=ucache_path,
        force=False,
    )
    ucache.save()
    print(f"  CLAP done in {time.time() - t0:.1f}s")


def _populate_vocal_stem(
    files: list[Path],
    cache_dir: Path,
    limit: int = 0,
    only_legacy: bool = False,
) -> None:
    """Run Demucs over each file, write per-stem metrics to the cache.

    Mirrors scripts/refresh_vocal_stem.py but scoped to a caller-supplied file
    list (so we can target ground-truth rows or a specific directory rather
    than always scanning the project's local Music/ folder).

    When ``only_legacy=True`` the pass targets only tracks whose cache entry
    exists with the old vocals-only schema (vocal_stem_* keys present but no
    stem_drums_* keys). Useful when you've already let Demucs run partway and
    want to upgrade those entries to the full per-stem schema before kicking
    off a much longer "never-cached" sweep.
    """
    from dj_tagger.analyzers.vocal_stem import (
        analyze_vocal_stem,
        is_demucs_available,
        result_to_dict,
    )
    from dj_tagger.audio import load_audio_features
    from dj_tagger.universal_cache import get_cache, quick_duration

    if not is_demucs_available():
        print("Demucs not installed (demucs + torch missing). Skipping.")
        return

    ucache_path = str(cache_dir / "raw_cache.pkl")
    ucache = get_cache(ucache_path)

    todo: list[tuple[Path, float]] = []
    already_cached = 0
    legacy_only = 0
    never_cached = 0
    too_short = 0
    for path in files:
        dur = quick_duration(str(path)) or 0.0
        if dur < 30.0:
            # Demucs analyzer rejects clips shorter than 30s; skip silently.
            too_short += 1
            continue
        existing = ucache.get_track(path.name, dur, "vocal_stem")
        is_legacy = (
            isinstance(existing, dict)
            and "vocal_stem_mix_ratio_db" in existing
            and not any(k.startswith("stem_drums_") for k in existing)
        )
        if isinstance(existing, dict) and any(k.startswith("stem_drums_") for k in existing):
            already_cached += 1
            continue
        if is_legacy:
            legacy_only += 1
            todo.append((path, dur))
        else:
            never_cached += 1
            if not only_legacy:
                todo.append((path, dur))

    unlimited_todo = len(todo)
    limited = bool(limit and unlimited_todo > limit)
    if limited:
        todo = todo[:limit]

    pieces = [
        f"Demucs stems: {already_cached} fully cached",
        f"{unlimited_todo} pending",
    ]
    if legacy_only:
        pieces.append(f"{legacy_only} have legacy schema only (re-extracting)")
    if only_legacy and never_cached:
        pieces.append(f"{never_cached} never cached (skipped, --only-legacy-stems)")
    if too_short:
        pieces.append(f"{too_short} too short to analyze")
    if limited:
        pieces.append(f"running first {limit} (--limit)")
    print("  ".join(pieces))
    if not todo:
        return

    from dj_registry.progress import ProgressBar

    t0 = time.time()
    failed = 0
    bar = ProgressBar(len(todo), label="Demucs vocal_stem")
    # Pre-publish so the bar appears on screen before the first slow track —
    # otherwise the user stares at a blank line for ~90 seconds.
    bar.update(0, path_label(todo[0][0]), ok=0, failed=0)
    for idx, (path, dur) in enumerate(todo, start=1):
        ts = time.time()
        try:
            audio = load_audio_features(str(path))
            result = analyze_vocal_stem(audio)
        except Exception as exc:
            failed += 1
            logger.warning("Demucs FAIL %s: %s", path.name, exc)
            bar.update(idx, path_label(path), ok=idx - failed, failed=failed)
            continue
        if result is None:
            failed += 1
            logger.info("Demucs SKIP %s (analyzer returned None)", path.name)
            bar.update(idx, path_label(path), ok=idx - failed, failed=failed)
            continue
        data = result_to_dict(result)
        ucache.put_track(path.name, dur, "vocal_stem", data)
        if idx % 10 == 0:
            ucache.save()
        logger.info(
            "Demucs OK %s  ratio=%.2f act=%.2f (%.1fs)",
            path.name, data["vocal_stem_mix_ratio_db"],
            data["vocal_stem_activity_frac"], time.time() - ts,
        )
        bar.update(idx, path_label(path), ok=idx - failed, failed=failed)

    bar.finish("complete")
    ucache.save()
    elapsed = time.time() - t0
    print(
        f"  Demucs done in {elapsed:.1f}s "
        f"({elapsed / max(1, len(todo)):.1f}s/track avg, {failed} failed/skipped)"
    )


def path_label(path: Path) -> str:
    """Truncate a filename for the progress-bar postfix."""
    name = path.name
    return name if len(name) <= 40 else name[:37] + "..."


def _print_coverage(files: list[Path], cache_dir: Path) -> None:
    """One-line per-layer coverage report after the run."""
    from dj_tagger.universal_cache import get_cache, quick_duration

    ucache = get_cache(str(cache_dir / "raw_cache.pkl"))

    layers = ("dsp", "clap", "vocal_stem")
    counts = {layer: 0 for layer in layers}
    for path in files:
        dur = quick_duration(str(path))
        for layer in layers:
            if ucache.get_track(path.name, dur, layer) is not None:
                counts[layer] += 1
    n = len(files)
    print("\nCoverage:")
    for layer in layers:
        pct = (counts[layer] * 100) // n if n else 0
        print(f"  {layer:11s} {counts[layer]:4d}/{n:<4d}  {pct:>3}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True,
                    help="Directory to scan recursively for audio files (mp3/aiff/wav/flac/m4a).")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"),
                    help="Directory containing raw_cache.pkl (default: ./cache).")
    ap.add_argument("--skip-clap", action="store_true", help="Skip CLAP extraction.")
    ap.add_argument("--skip-demucs", action="store_true", help="Skip Demucs vocal_stem extraction.")
    ap.add_argument(
        "--only-legacy-stems",
        action="store_true",
        help="Demucs pass: only re-extract tracks that have the old vocals-only "
             "schema (vocal_stem_* keys without stem_drums_*). Never-cached tracks "
             "are left alone. Useful for upgrading partial coverage without "
             "kicking off the full overnight sweep.",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N new extractions per layer (0 = all). For smoke-testing.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.files)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = _discover_audio_files(root)
    print(f"Found {len(files)} audio files in {root}")
    if not files:
        return 1

    if not args.skip_clap:
        _populate_clap(files, cache_dir, limit=args.limit)
    if not args.skip_demucs:
        _populate_vocal_stem(
            files, cache_dir, limit=args.limit, only_legacy=args.only_legacy_stems,
        )

    _print_coverage(files, cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
