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


def find_audio_files(paths: list[str], recursive: bool) -> list[Path]:
    """Discover audio files from the given paths."""
    results: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(p)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
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
        nargs="+",
        metavar="PATH",
        help="Audio files or directories to process",
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


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --write-tags overrides the default --dry-run
    dry_run = not args.write_tags

    _setup_logging(args.verbose, args.quiet)

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

    config = AnalysisConfig(
        dry_run=dry_run,
        overwrite=args.overwrite,
        use_essentia=args.use_essentia,
        verbose=args.verbose,
    )

    results: list[dict] = []
    n_ok = 0
    n_fail = 0

    # Use ProcessPoolExecutor for parallelism, but fall back to serial
    # if workers == 1 (easier to debug).
    if args.workers == 1:
        for i, fpath in enumerate(files, 1):
            t0 = time.perf_counter()
            try:
                result = analyze_track(str(fpath), config)
                elapsed = time.perf_counter() - t0
                result["elapsed"] = round(elapsed, 2)
                results.append(result)
                n_ok += 1
                if args.json_output:
                    print(json.dumps(result))
                elif not args.quiet:
                    print(
                        f"[{i:>{len(str(total))}}/{total}] "
                        f"{fpath.name} -> {result['tag']}  ({elapsed:.1f}s)"
                    )
            except Exception as exc:
                n_fail += 1
                elapsed = time.perf_counter() - t0
                if args.json_output:
                    print(json.dumps({"file": str(fpath), "error": str(exc)}))
                elif not args.quiet:
                    print(
                        f"[{i:>{len(str(total))}}/{total}] "
                        f"{fpath.name} -> FAILED: {exc}  ({elapsed:.1f}s)"
                    )
                logger.debug("Traceback for %s", fpath, exc_info=True)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fpath in files:
                fut = pool.submit(analyze_track, str(fpath), config)
                futures[fut] = fpath

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                fpath = futures[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    n_ok += 1
                    if args.json_output:
                        print(json.dumps(result))
                    elif not args.quiet:
                        print(
                            f"[{done_count:>{len(str(total))}}/{total}] "
                            f"{fpath.name} -> {result['tag']}"
                        )
                except Exception as exc:
                    n_fail += 1
                    if args.json_output:
                        print(json.dumps({"file": str(fpath), "error": str(exc)}))
                    elif not args.quiet:
                        print(
                            f"[{done_count:>{len(str(total))}}/{total}] "
                            f"{fpath.name} -> FAILED: {exc}"
                        )
                    logger.debug("Traceback for %s", fpath, exc_info=True)

    # Summary
    if not args.quiet and not args.json_output:
        print(f"\nDone: {n_ok} processed, {n_fail} failed out of {total} files.")

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
