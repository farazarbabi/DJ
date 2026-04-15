"""Unified DJ tools CLI — single command for the full pipeline.

Orchestrates: registry (scan/link/ingest) -> analysis -> resolve -> tag -> group.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

from . import __version__

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if verbose
        else "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)
    # Suppress noisy HTTP loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dj",
        description="Unified DJ tools: analyze, resolve, tag, and group your music library.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("vibe-audit", help="Audit vibe score distributions from cache")

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument(
        "paths", nargs="*", default=["./files"], metavar="PATH",
        help="Audio files or directories to process (default: ./files)",
    )
    p_run.add_argument("--rekordbox-xml", dest="rekordbox_xml", default=None,
                       help="Rekordbox XML path (default: auto-detect latest .xml in library dir)")
    p_run.add_argument("--no-songstats", action="store_true", help="Skip Songstats metadata fetch")
    p_run.add_argument("--no-tags", action="store_true", help="Skip writing tags to files")
    p_run.add_argument("--no-grouping", action="store_true", help="Skip grouping phase")
    p_run.add_argument("--no-clap", action="store_true", help="Disable CLAP embeddings in grouper")
    p_run.add_argument("--no-essentia", action="store_true", help="Skip essentia key analysis")
    p_run.add_argument("-w", "--workers", type=int, default=1, help="Analysis workers (default: 1)")
    p_run.add_argument("--force-extract", action="store_true", help="Force re-extraction in grouper")
    p_run.add_argument("--output", default="./outputs/registry", help="Registry output dir")

    return parser


def _find_latest_xml(paths: list[str]) -> str:
    """Find the most recently modified .xml file in the given directories."""
    from pathlib import Path

    candidates: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            candidates.extend(p.glob("*.xml"))
    if not candidates:
        return ""
    latest = max(candidates, key=lambda f: f.stat().st_mtime)
    return str(latest)


def _run_songstats(config, store, obs_cache) -> dict:
    """Try Songstats ingest. Returns summary dict. Gracefully handles failures."""
    summary = {"isrcs_enriched": 0, "songstats_total": 0, "songstats_cached": 0, "songstats_fetched": 0}

    if not config.songstats_api_key:
        logger.info("Songstats: skipped (no API key, set SONGSTATS_API_KEY)")
        return summary

    try:
        from dj_registry.adapters.spotify_isrc import enrich_isrcs
        summary["isrcs_enriched"] = enrich_isrcs(store)
    except Exception:
        logger.warning("Songstats: ISRC enrichment failed, continuing", exc_info=True)

    try:
        from dj_registry.adapters.songstats import ingest_songstats
        stats = ingest_songstats(config, store, obs_cache=obs_cache)
        summary["songstats_total"] = stats["total"]
        summary["songstats_cached"] = stats["cached"]
        summary["songstats_fetched"] = stats["fetched"]
    except Exception:
        logger.warning("Songstats: metadata fetch failed, continuing", exc_info=True)

    return summary


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable hh:mm:ss or mm:ss or Ns."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _run_vibe_audit() -> int:
    """Audit vibe score distributions from cached DSP + Songstats data."""
    import numpy as np

    from dj_tagger.derive import derive_vibe
    from dj_tagger.universal_cache import get_cache

    ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))

    # Collect all DSP entries and their songstats (if available)
    tracks: list[tuple[str, dict, dict[str, float] | None]] = []  # (name, vibe_result, audio_features)
    for key, entry in ucache._entries.items():
        if not key.endswith("|dsp"):
            continue
        if not isinstance(entry.data, dict):
            continue
        name = key.rsplit("|", 2)[0]
        # Look up songstats by scanning for matching ISRC entries
        # (we don't have the ISRC→filename mapping here, so try all songstats entries)
        tracks.append((name, entry.data, None))

    # Try to load Songstats features via registry overview CSV
    overview_path = os.path.join("outputs", "registry", "registry_overview.csv")
    ss_by_name: dict[str, dict[str, float]] = {}
    if os.path.exists(overview_path):
        import csv
        with open(overview_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("file_name", "")
                if not fn:
                    continue
                af: dict[str, float] = {}
                for ss_key, col in [
                    ("valence", "ss_valence"), ("energy", "ss_energy"),
                    ("instrumentalness", "ss_instrumentalness"),
                    ("liveness", "ss_liveness"), ("acousticness", "ss_acousticness"),
                ]:
                    val = row.get(col, "")
                    if val:
                        try:
                            af[ss_key] = float(val)
                        except (ValueError, TypeError):
                            pass
                if af:
                    ss_by_name[fn] = af

    # Run derive_vibe on each track
    labels: list[str] = []
    all_scores: dict[str, list[float]] = {v: [] for v in ["MEL", "DRK", "HYPN", "TRIB", "DEEP", "ATM", "RAW", "ACID"]}
    near_misses: dict[str, int] = {v: 0 for v in all_scores}  # non-MEL vibes that lost to MEL by <0.10
    ss_count = 0

    for name, dsp, _ in tracks:
        af = ss_by_name.get(name)
        if af:
            ss_count += 1
        result = derive_vibe(dsp, audio_features=af)
        labels.append(result["vibe"])
        for v, score in result["vibe_scores"].items():
            all_scores[v].append(score)

        # Near-miss: non-MEL vibe scored within 0.10 of MEL but lost
        if result["vibe"] == "MEL":
            mel_score = result["vibe_scores"]["MEL"]
            for v, score in result["vibe_scores"].items():
                if v != "MEL" and mel_score - score < 0.10:
                    near_misses[v] += 1

    if not tracks:
        print("No cached DSP entries found. Run 'dj run' first.")
        return 1

    n = len(tracks)
    print(f"\nVibe Audit: {n} tracks ({ss_count} with Songstats data)\n")

    # Label distribution
    print("Label Distribution:")
    from collections import Counter
    counts = Counter(labels)
    for v in ["MEL", "DRK", "HYPN", "TRIB", "DEEP", "ATM", "RAW", "ACID"]:
        c = counts.get(v, 0)
        pct = 100.0 * c / n
        bar = "#" * int(pct / 2)
        print(f"  {v:5s}  {c:4d}  ({pct:5.1f}%)  {bar}")

    # Score stats
    print("\nScore Statistics (mean / p25 / p50 / p75 / max):")
    for v in ["MEL", "DRK", "HYPN", "TRIB", "DEEP", "ATM", "RAW", "ACID"]:
        arr = np.array(all_scores[v])
        if len(arr) == 0:
            continue
        print(f"  {v:5s}  {np.mean(arr):.3f} / {np.percentile(arr, 25):.3f} / "
              f"{np.percentile(arr, 50):.3f} / {np.percentile(arr, 75):.3f} / {np.max(arr):.3f}")

    # Near-miss analysis
    mel_count = counts.get("MEL", 0)
    if mel_count > 0:
        print(f"\nNear-Misses (non-MEL vibes that lost to MEL by <0.10):")
        for v in ["DRK", "HYPN", "TRIB", "DEEP", "ATM", "RAW", "ACID"]:
            c = near_misses[v]
            if c > 0:
                print(f"  {v:5s}  {c:4d}  ({100.0 * c / mel_count:.1f}% of MEL tracks)")

    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full unified pipeline."""
    import time

    from dj_registry.adapters.file_scanner import scan_files
    from dj_registry.adapters.local_analysis import run_analysis
    from dj_registry.config import RegistryConfig
    from dj_registry.identity.matcher import link_files_to_tracks
    from dj_registry.resolver.bpm_resolver import resolve_all_bpms
    from dj_registry.resolver.key_resolver import resolve_all_keys
    from dj_registry.review.queue_builder import build_review_queue
    from dj_registry.store.csv_store import CsvStore
    from dj_registry.store.obs_cache import ObsCache
    from dj_registry.sync.export import generate_reports
    from dj_registry.sync.tag_writer import sync_tags

    t_start = time.perf_counter()

    # Build config
    config = RegistryConfig()
    config.library_roots = args.paths
    config.analysis_workers = args.workers
    if args.rekordbox_xml:
        config.rekordbox_xml_path = args.rekordbox_xml
    else:
        # Auto-detect: use the most recently modified .xml in library roots
        config.rekordbox_xml_path = _find_latest_xml(args.paths)
        if config.rekordbox_xml_path:
            logger.info("Rekordbox XML: auto-detected %s", config.rekordbox_xml_path)
    config.output_dir = args.output
    config.load_env()

    run_id = uuid.uuid4().hex[:8]
    store = CsvStore(config.output_dir)
    obs_cache = ObsCache()
    store.snapshot(run_id)

    # Clear stale observations — rebuilt from cache each run
    store.save_observations([])

    # Phase 1: Scan + Link
    t0 = time.perf_counter()
    files = scan_files(config, store, obs_cache=obs_cache)
    link_files_to_tracks(config, store)
    tracks = store.load_tracks()
    logger.info("Pipeline: scan + link done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 2: Ingest external sources
    t0 = time.perf_counter()
    if config.rekordbox_xml_path:
        from dj_registry.adapters.rekordbox_xml import ingest_rekordbox
        ingest_rekordbox(config, store, obs_cache=obs_cache)
    else:
        logger.info("Rekordbox: skipped (no .xml found in library dir)")

    if not args.no_songstats:
        _run_songstats(config, store, obs_cache)
    else:
        logger.info("Songstats: skipped (--no-songstats)")

    obs_cache.save()
    logger.info("Pipeline: ingest done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 3: Analyze (full tagger pipeline)
    t0 = time.perf_counter()
    run_analysis(config, store, no_essentia=args.no_essentia)
    logger.info("Pipeline: analysis done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 4: Resolve canonical key + BPM
    t0 = time.perf_counter()
    from dj_registry.pipelines.orchestrator import _enrich_observations
    _enrich_observations(store)

    resolve_all_keys(config, store, force=True)
    resolve_all_bpms(config, store, force=True)
    build_review_queue(config, store)
    logger.info("Pipeline: resolve done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 5: Write tags
    t0 = time.perf_counter()
    if not args.no_tags:
        sync_tags(store, dry_run=False)
        logger.info("Pipeline: tags written in %s\n", _fmt_elapsed(time.perf_counter() - t0))
    else:
        logger.info("Tags: skipped (--no-tags)\n")

    # Phase 6: Reports
    generate_reports(store, config.reports_dir)

    # Phase 7: Grouping
    if not args.no_grouping:
        t0 = time.perf_counter()
        try:
            from dj_grouper.cli import main as grouper_main
            grouper_argv = list(args.paths)
            if args.no_clap:
                grouper_argv.append("--no-clap")
            if args.force_extract:
                grouper_argv.append("--force-extract")
            if args.workers > 1:
                grouper_argv.extend(["-w", str(args.workers)])
            grouper_main(grouper_argv)
            logger.info("Pipeline: grouping done in %s", _fmt_elapsed(time.perf_counter() - t0))
        except Exception:
            logger.error("Grouping failed", exc_info=True)
    else:
        logger.info("Grouping: skipped (--no-grouping)")

    logger.info("Pipeline: total %s", _fmt_elapsed(time.perf_counter() - t_start))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    raw = argv if argv is not None else sys.argv[1:]
    # Handle --version and --help at top level
    if raw and raw[0] in ("--version", "--help", "-h"):
        args = parser.parse_args(raw)
        return 0
    # Default to "run" when no subcommand is given
    known_commands = {"run", "vibe-audit"}
    if not raw or raw[0] not in known_commands:
        raw = ["run"] + list(raw)

    args = parser.parse_args(raw)
    _setup_logging(
        getattr(args, "verbose", False),
        getattr(args, "quiet", False),
    )

    if args.command == "vibe-audit":
        try:
            return _run_vibe_audit()
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    if args.command == "run":
        try:
            return _run_pipeline(args)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
