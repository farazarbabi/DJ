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


def _resolve_cache_path() -> str:
    return os.path.join("outputs", "tagger_cache.pkl")


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

    config = AnalysisConfig(
        dry_run=dry_run,
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
            cached_result = get_cached(cache, str(fpath), mtime)
            if cached_result is not None:
                hits.append((fpath, cached_result))
            else:
                misses.append(fpath)
    else:
        misses = list(files)

    # --- Process cache hits ---
    done_count = 0
    for fpath, result in hits:
        done_count += 1
        n_ok += 1
        n_cached += 1
        # Update the file path in the result to match current location
        result = {**result, "file": str(fpath)}
        if not dry_run:
            try:
                write_tag(str(fpath), result["tag"], dry_run=False)
            except Exception:
                logger.debug("Failed to write tag for cached %s", fpath, exc_info=True)
        results.append(result)
        if args.json_output:
            print(json.dumps(result))
        elif not args.quiet:
            elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
            print(
                f"{_progress_bar(done_count, total)} "
                f"{fpath.name} -> {result['tag']}  [cached]  ({elapsed_total})"
            )

    # --- Process cache misses ---
    new_results_count = 0

    def _handle_completed(fpath: Path, result: dict, elapsed: float | None = None) -> None:
        nonlocal n_ok, n_cached, done_count, new_results_count
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
                put_cached(cache, str(fpath), mtime, result)
            except OSError:
                pass
            if new_results_count % 20 == 0:
                save_cache(cache, cache_path)
        if args.json_output:
            print(json.dumps(result))
        elif not args.quiet:
            track_str = f"({_fmt_elapsed(elapsed)})" if elapsed is not None else ""
            elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
            print(
                f"{_progress_bar(done_count, total)} "
                f"{fpath.name} -> {result['tag']}  {track_str}  ({elapsed_total})"
            )

    if args.workers == 1:
        for fpath in misses:
            t0 = time.perf_counter()
            try:
                result = analyze_track(str(fpath), config)
                elapsed = time.perf_counter() - t0
                _handle_completed(fpath, result, elapsed)
            except Exception as exc:
                n_fail += 1
                done_count += 1
                elapsed = time.perf_counter() - t0
                if args.json_output:
                    print(json.dumps({"file": str(fpath), "error": str(exc)}))
                elif not args.quiet:
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
                fut = pool.submit(analyze_track, str(fpath), config)
                futures[fut] = fpath

            for fut in as_completed(futures):
                fpath = futures[fut]
                try:
                    result = fut.result()
                    _handle_completed(fpath, result)
                except Exception as exc:
                    n_fail += 1
                    done_count += 1
                    if args.json_output:
                        print(json.dumps({"file": str(fpath), "error": str(exc)}))
                    elif not args.quiet:
                        elapsed_total = _fmt_elapsed(time.perf_counter() - t_start)
                        print(
                            f"{_progress_bar(done_count, total)} "
                            f"{fpath.name} -> FAILED: {exc}  ({elapsed_total})"
                        )
                    logger.debug("Traceback for %s", fpath, exc_info=True)

    # --- Save cache ---
    if use_cache and new_results_count > 0:
        save_cache(cache, cache_path)

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
        "structure", "vibe", "vocal", "key_confidence", "vocal_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
