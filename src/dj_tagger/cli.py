"""Command-line interface for dj-tagger."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import __version__
from .constants import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def find_audio_files(
    paths: list[str], recursive: bool, exclude_dirs: set[str] | None = None,
) -> list[Path]:
    """Discover audio files from the given paths."""
    exclude = {d.lower() for d in exclude_dirs} if exclude_dirs else set()
    results: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(p)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    if exclude and any(part.lower() in exclude for part in child.relative_to(p).parts):
                        continue
                    results.append(child)
        else:
            logger.warning("Skipping %s (not a file or directory)", p)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dj-tagger",
        description="Batch-analyze DJ music files and write structured comment tags.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["files"],
        metavar="PATH",
        help="Audio files or directories to process (default: ./files)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        default=True,
        help="Analyze without writing tags (default)",
    )
    parser.add_argument(
        "--write-tags",
        action="store_true",
        help="Actually write tags to files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing comment tags",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=max(1, os.cpu_count() // 2) if os.cpu_count() else 1,
        help="Number of parallel workers (default: cpu_count // 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N files",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output results as JSON lines",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-analyzer detail",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )
    parser.add_argument(
        "--use-essentia",
        action="store_true",
        help="Use Essentia for key detection if installed",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip analysis cache (force full re-analysis)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cached results and re-analyze from scratch",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip registry key/BPM resolution (use analysis values directly)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if verbose
        else "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _quick_duration(path: str) -> float | None:
    """Get audio duration cheaply via mutagen (no audio decoding)."""
    try:
        import mutagen
        m = mutagen.File(path)
        if m and m.info:
            return getattr(m.info, "length", None)
    except Exception:
        pass
    return None


def _resolve_cache_path() -> str:
    return os.path.join("cache", "tagger_cache.pkl")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    """Render a simple progress bar like [=========>          ] 45%."""
    frac = done / total if total else 0
    filled = int(width * frac)
    bar = "=" * filled
    if filled < width:
        bar += ">"
        bar += " " * (width - filled - 1)
    pct = int(frac * 100)
    return f"[{bar}] {pct:>3}%"


def _format_result_tag(
    result: dict,
    *,
    camelot: str | None = None,
    bpm=None,
    category: str | None = None,
) -> str:
    """Build a current tag string from cached or freshly analyzed result fields."""
    from .formats import format_tag

    bpm_val = bpm if bpm is not None else result.get("bpm")
    if bpm_val is not None:
        try:
            bpm_rounded = int(round(float(bpm_val)))
        except (ValueError, TypeError):
            bpm_rounded = None
    else:
        bpm_rounded = None
    return format_tag(
        energy=result.get("energy"),
        camelot=camelot or result.get("camelot"),
        bpm=bpm_rounded,
        vibe=result.get("vibe"),
        has_vocals=result.get("has_vocals"),
        vocal_profile=result.get("vocal_profile", result.get("vocal")),
        category=category or result.get("category"),
    )


def _tag_from_registry_result(result: dict, registry_result: dict) -> str:
    """Prefer the registry-built COMMENT tag, falling back to canonical key/BPM."""
    comment_tag = registry_result.get("comment_tag")
    if comment_tag:
        return comment_tag

    canon_key = registry_result.get("canonical_key_camelot")
    canon_bpm = registry_result.get("canonical_bpm")
    category = registry_result.get("category")
    if canon_key or canon_bpm or category:
        return _format_result_tag(result, camelot=canon_key, bpm=canon_bpm, category=category)
    return result.get("tag") or _format_result_tag(result)


def _resolve_via_registry(
    file_paths: list[str],
    tagger_results: dict[str, dict],
) -> dict[str, dict]:
    """Run minimal registry pipeline to get canonical key/BPM/category.

    Returns {abs_path: {"canonical_key_camelot": str, "canonical_bpm": str, ...}} or
    empty dict on failure.
    """
    try:
        from dj_registry.adapters.file_scanner import scan_files
        from dj_registry.adapters.local_analysis import (
            _extract_tagger_features,
            _apply_tagger_to_track,
            _build_observation,
        )
        from dj_registry.config import RegistryConfig
        from dj_registry.identity.matcher import link_files_to_tracks
        from dj_registry.resolver.bpm_resolver import resolve_all_bpms
        from dj_registry.resolver.key_resolver import resolve_all_keys
        from dj_registry.store.csv_store import CsvStore
        from dj_registry.store.obs_cache import ObsCache
        from dj_registry.sync.tag_writer import _build_comment_tag, _category_code_for_tag
    except ImportError:
        logger.warning("Registry not available (install with pip install -e '.[registry]')")
        return {}

    config = RegistryConfig()
    # Derive library roots from file paths
    roots = list({str(Path(p).parent) for p in file_paths})
    config.library_roots = roots

    store = CsvStore(config.output_dir)
    obs_cache = ObsCache()

    # Scan + link (ensures files and tracks exist in registry)
    scan_files(config, store, obs_cache=obs_cache)
    link_files_to_tracks(config, store)

    # Ingest tagger results as observations
    files = store.load_files()
    tracks = store.load_tracks()
    track_by_id = {t.track_id: t for t in tracks}
    file_by_path = {}
    for f in files:
        file_by_path[f.path_abs] = f
        # Also index by filename for matching
        file_by_path[f.file_name] = f

    all_obs = store.load_observations()
    # Remove old analysis observations (will be replaced)
    processed_track_ids = set()

    for abs_path, result in tagger_results.items():
        frec = file_by_path.get(abs_path) or file_by_path.get(Path(abs_path).name)
        if not frec or not frec.track_id:
            continue

        features = _extract_tagger_features(result)
        if not features:
            continue

        processed_track_ids.add(frec.track_id)

        # Build observation
        obs = _build_observation(frec.track_id, frec.file_id, "analysis_librosa", features)
        if obs:
            all_obs.append(obs)

        # Update LogicalTrack with tagger features
        track = track_by_id.get(frec.track_id)
        if track:
            _apply_tagger_to_track(track, features)

    # Remove old analysis obs for tracks we just processed
    all_obs = [
        o for o in all_obs
        if not (o.source_system.startswith("analysis_") and o.track_id in processed_track_ids)
        or o in all_obs[-len(processed_track_ids):]  # keep the new ones
    ]
    # Simpler: rebuild without old analysis for processed tracks, then add new
    kept = [
        o for o in all_obs
        if not (o.source_system.startswith("analysis_") and o.track_id in processed_track_ids)
    ]
    for abs_path, result in tagger_results.items():
        frec = file_by_path.get(abs_path) or file_by_path.get(Path(abs_path).name)
        if not frec or not frec.track_id:
            continue
        features = _extract_tagger_features(result)
        obs = _build_observation(frec.track_id, frec.file_id, "analysis_librosa", features)
        if obs:
            kept.append(obs)
    store.save_observations(kept)

    store.save_tracks(tracks)
    obs_cache.save()

    # Resolve canonical key + BPM
    resolve_all_keys(config, store, force=True)
    resolve_all_bpms(config, store, force=True)

    # Populate DJ category labels so dry-run/write output uses the same final
    # COMMENT tag as the registry sync path.
    try:
        from dj_registry.taxonomy.dj_model import classify_all_dj_taxonomies

        classify_all_dj_taxonomies(store, primary_model="internal")
    except FileNotFoundError as exc:
        logger.info("DJ taxonomy: skipped (%s)", exc)

    # Build result mapping: file path -> canonical values
    tracks = store.load_tracks()
    files = store.load_files()
    track_by_id = {t.track_id: t for t in tracks}
    canonical: dict[str, dict] = {}
    for f in files:
        track = track_by_id.get(f.track_id)
        if track and f.path_abs in tagger_results:
            canonical[f.path_abs] = {
                "canonical_key_camelot": track.canonical_key_camelot,
                "canonical_bpm": track.canonical_bpm,
                "category": _category_code_for_tag(track),
                "category_label": track.dj_taxonomy_internal_label or track.dj_taxonomy_label,
                "comment_tag": _build_comment_tag(track),
            }

    return canonical


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --write-tags overrides the default --dry-run
    dry_run = not args.write_tags

    _setup_logging(args.verbose, args.quiet)

    # --- Cache setup ---
    use_cache = not args.no_cache
    cache_path = _resolve_cache_path()

    if args.clear_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Cleared tagger cache at %s", cache_path)

    cache: dict = {}
    if use_cache:
        from .cache import load_cache
        cache = load_cache(cache_path)

    files = find_audio_files(args.paths, args.recursive)
    if not files:
        logger.info("No audio files found.")
        return 0

    if args.limit:
        files = files[: args.limit]

    total = len(files)
    logger.info(
        "Found %d file%s. %s",
        total,
        "" if total == 1 else "s",
        "DRY RUN (no tags will be written)" if dry_run else "WRITE MODE",
    )

    # Import here to avoid top-level heavy imports for --help / --version
    from .pipeline import analyze_track, AnalysisConfig
    from .cache import get_cached, put_cached, save_cache
    from .metadata import write_tag

    # Always analyze with dry_run=True — tags are written later after registry resolution
    analysis_config = AnalysisConfig(
        dry_run=True,
        overwrite=args.overwrite,
        use_essentia=args.use_essentia,
        verbose=args.verbose,
    )

    results: list[dict] = []
    n_ok = 0
    n_fail = 0
    n_cached = 0
    t_start = time.perf_counter()

    # --- Split files into cache hits and misses ---
    hits: list[tuple[Path, dict]] = []
    misses: list[Path] = []

    if use_cache:
        for fpath in files:
            try:
                mtime = os.path.getmtime(str(fpath))
            except OSError:
                misses.append(fpath)
                continue
            dur = _quick_duration(str(fpath))
            cached_result = get_cached(cache, str(fpath), mtime, duration=dur)
            if cached_result is not None:
                hits.append((fpath, cached_result))
            else:
                misses.append(fpath)
    else:
        misses = list(files)

    # --- Phase 1: Collect results (analysis only, no tag writing) ---
    done_count = 0

    # Process cache hits
    for fpath, result in hits:
        done_count += 1
        n_ok += 1
        n_cached += 1
        result = {**result, "file": str(fpath)}

        result["tag"] = _format_result_tag(result)

        results.append(result)
        if not args.quiet and not args.json_output:
            elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
            print(
                f"{_progress_bar(done_count, total)} "
                f"{fpath.name} -> {result['tag']}  [cached]  ({elapsed_total})"
            )

    # Process cache misses
    new_results_count = 0

    def _handle_completed(fpath: Path, result: dict, elapsed: float | None = None) -> None:
        nonlocal n_ok, done_count, new_results_count
        done_count += 1
        n_ok += 1
        new_results_count += 1
        if elapsed is not None:
            result["elapsed"] = round(elapsed, 2)
        results.append(result)
        # Store in cache
        if use_cache:
            try:
                mtime = os.path.getmtime(str(fpath))
                dur = _quick_duration(str(fpath))
                put_cached(cache, str(fpath), mtime, result, duration=dur)
            except OSError:
                pass
            if new_results_count % 20 == 0:
                save_cache(cache, cache_path)
        if not args.quiet and not args.json_output:
            track_str = f"({_fmt_elapsed(elapsed)})" if elapsed is not None else ""
            elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
            print(
                f"{_progress_bar(done_count, total)} "
                f"{fpath.name} -> {result.get('tag', '??')}  {track_str}  ({elapsed_total})"
            )

    if args.workers == 1:
        for fpath in misses:
            t0 = time.perf_counter()
            try:
                result = analyze_track(str(fpath), analysis_config)
                elapsed = time.perf_counter() - t0
                _handle_completed(fpath, result, elapsed)
            except Exception as exc:
                n_fail += 1
                done_count += 1
                elapsed = time.perf_counter() - t0
                if not args.quiet:
                    elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
                    print(
                        f"{_progress_bar(done_count, total)} "
                        f"{fpath.name} -> FAILED: {exc}  ({_fmt_elapsed(elapsed)})  ({elapsed_total})"
                    )
                logger.debug("Traceback for %s", fpath, exc_info=True)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fpath in misses:
                fut = pool.submit(analyze_track, str(fpath), analysis_config)
                futures[fut] = fpath

            for fut in as_completed(futures):
                fpath = futures[fut]
                try:
                    result = fut.result()
                    _handle_completed(fpath, result)
                except Exception as exc:
                    n_fail += 1
                    done_count += 1
                    if not args.quiet:
                        elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
                        print(
                            f"{_progress_bar(done_count, total)} "
                            f"{fpath.name} -> FAILED: {exc}  ({elapsed_total})"
                        )
                    logger.debug("Traceback for %s", fpath, exc_info=True)

    # Save cache
    if use_cache and new_results_count > 0:
        save_cache(cache, cache_path)

    # --- Phase 2: Registry resolution (canonical key/BPM) ---
    canonical: dict[str, dict] = {}
    if not args.no_registry and results:
        if not args.quiet and not args.json_output:
            print("\nResolving canonical key/BPM via registry...")
        tagger_results = {r["file"]: r for r in results if "file" in r and "error" not in r}
        canonical = _resolve_via_registry(list(tagger_results.keys()), tagger_results)
        if canonical and not args.quiet and not args.json_output:
            print(f"  {len(canonical)} tracks resolved")

    # --- Phase 3: Rebuild tags with canonical values and write ---
    for result in results:
        fpath = result.get("file")
        if not fpath:
            continue

        canon = canonical.get(fpath, {})
        # Rebuild tag from registry output when available. This includes
        # canonical key/BPM and DJ taxonomy category labels.
        if canon:
            result["tag"] = _tag_from_registry_result(result, canon)

        # Write tag to file
        if not dry_run and result.get("tag"):
            try:
                write_tag(fpath, result["tag"], dry_run=False)
            except Exception:
                logger.debug("Failed to write tag for %s", fpath, exc_info=True)

    # Output JSON results (after tag rebuild)
    if args.json_output:
        for result in results:
            print(json.dumps(result))

    # Summary
    if not args.quiet and not args.json_output:
        parts = [f"{n_ok} processed"]
        if n_cached:
            parts.append(f"{n_cached} cached")
        if n_fail:
            parts.append(f"{n_fail} failed")
        elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
        print(f"\nDone: {', '.join(parts)} out of {total} files in {elapsed_total}.")

    # CSV export
    if args.csv_path and results:
        _write_csv(args.csv_path, results)
        if not args.quiet:
            print(f"Results written to {args.csv_path}")

    if n_fail == total:
        return 2
    if n_fail > 0:
        return 1
    return 0


def _write_csv(path: str, results: list[dict]) -> None:
    fieldnames = [
        "file", "tag", "bpm", "energy", "key", "camelot",
        "structure", "vibe", "vocal", "vocal_profile", "key_confidence", "vocal_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
