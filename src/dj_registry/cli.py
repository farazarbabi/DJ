"""CLI entry point for dj-registry."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import RegistryConfig
from .store.csv_store import CsvStore

logger = logging.getLogger("dj_registry")


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    # Suppress noisy HTTP request logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_config(args: argparse.Namespace) -> RegistryConfig:
    """Build RegistryConfig from parsed CLI args."""
    config = RegistryConfig()

    if hasattr(args, "paths") and args.paths:
        config.library_roots = args.paths
    if hasattr(args, "rekordbox_xml") and args.rekordbox_xml:
        config.rekordbox_xml_path = args.rekordbox_xml
    if hasattr(args, "output") and args.output:
        config.output_dir = args.output
    if hasattr(args, "workers") and args.workers:
        config.analysis_workers = args.workers

    config.load_env()

    return config


def cmd_scan(args: argparse.Namespace) -> int:
    from .adapters.file_scanner import scan_files
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    files = scan_files(config, store)
    print(f"Scanned {len(files)} files")
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    from .identity.matcher import link_files_to_tracks
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    link_files_to_tracks(config, store)
    tracks = store.load_tracks()
    print(f"Linked to {len(tracks)} tracks")
    return 0


def cmd_ingest_rekordbox(args: argparse.Namespace) -> int:
    from .adapters.rekordbox_xml import ingest_rekordbox
    config = _build_config(args)
    if not config.rekordbox_xml_path:
        print("Error: --xml required")
        return 1
    store = CsvStore(config.output_dir)
    count = ingest_rekordbox(config, store)
    print(f"Ingested {count} Rekordbox tracks")
    return 0


def cmd_enrich_isrcs(args: argparse.Namespace) -> int:
    from .adapters.spotify_isrc import enrich_isrcs
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    limit = getattr(args, "limit", None)
    count = enrich_isrcs(store, limit=limit)
    print(f"Enriched {count} tracks with ISRCs")
    return 0


def cmd_ingest_songstats(args: argparse.Namespace) -> int:
    from .adapters.songstats import ingest_songstats
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    limit = getattr(args, "limit", None)
    only_missing = getattr(args, "only_missing", False)
    count = ingest_songstats(config, store, limit=limit, only_missing=only_missing)
    print(f"Fetched {count} tracks from Songstats")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .adapters.local_analysis import run_analysis
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    count = run_analysis(
        config, store,
        no_essentia=getattr(args, "no_essentia", False),
    )
    print(f"Analyzed {count} tracks")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from .resolver.bpm_resolver import resolve_all_bpms
    from .resolver.key_resolver import resolve_all_keys
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    force = getattr(args, "force", False)
    resolved, review = resolve_all_keys(config, store, force=force)
    print(f"Resolved {resolved} keys, {review} need review")
    bpm_resolved, bpm_missing = resolve_all_bpms(config, store, force=force)
    print(f"Resolved {bpm_resolved} BPMs, {bpm_missing} missing")
    return 0


def cmd_review_queue(args: argparse.Namespace) -> int:
    from .review.queue_builder import build_review_queue
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    count = build_review_queue(config, store)
    print(f"Review queue: {count} items")
    return 0


def cmd_import_reviews(args: argparse.Namespace) -> int:
    from .review.importer import import_reviews
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    accepted, overridden, skipped = import_reviews(store)
    print(f"Imported: {accepted} accepted, {overridden} overridden, {skipped} skipped")
    return 0


def cmd_sync_tags(args: argparse.Namespace) -> int:
    from .sync.tag_writer import sync_tags
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    written, skipped, errors = sync_tags(
        store,
        dry_run=args.dry_run,
        only_changed=getattr(args, "only_changed", True),
    )
    print(f"Tags: {written} written, {skipped} skipped, {errors} errors")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .sync.export import generate_reports
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    generate_reports(store, config.reports_dir)
    print("Reports generated")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipelines.orchestrator import run_full_pipeline
    config = _build_config(args)
    summary = run_full_pipeline(
        config,
        dry_run=args.dry_run,
        write_tags=getattr(args, "write_tags", False),
        include_rekordbox=bool(config.rekordbox_xml_path),
        include_songstats=getattr(args, "songstats", False),
        songstats_limit=getattr(args, "songstats_limit", None),
        no_essentia=getattr(args, "no_essentia", False),
        analysis_workers=config.analysis_workers,
    )
    print("\nPipeline summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dj-registry",
        description="Track metadata registry — single source of truth for music metadata",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # scan
    p_scan = sub.add_parser("scan", help="Scan library files")
    p_scan.add_argument("paths", nargs="*", default=["./files"])
    p_scan.add_argument("--dry-run", action="store_true")
    p_scan.add_argument("--output", default="./outputs/registry")

    # link
    p_link = sub.add_parser("link", help="Link files to tracks")
    p_link.add_argument("--output", default="./outputs/registry")

    # ingest-rekordbox
    p_rb = sub.add_parser("ingest-rekordbox", help="Ingest Rekordbox XML")
    p_rb.add_argument("--xml", dest="rekordbox_xml", required=True)
    p_rb.add_argument("--output", default="./outputs/registry")

    # enrich-isrcs
    p_isrc = sub.add_parser("enrich-isrcs", help="Look up ISRCs via Spotify")
    p_isrc.add_argument("--limit", type=int)
    p_isrc.add_argument("--output", default="./outputs/registry")

    # ingest-songstats
    p_ss = sub.add_parser("ingest-songstats", help="Fetch Songstats metadata")
    p_ss.add_argument("--limit", type=int)
    p_ss.add_argument("--only-missing", action="store_true")
    p_ss.add_argument("--output", default="./outputs/registry")

    # analyze
    p_an = sub.add_parser("analyze", help="Run local key analysis")
    p_an.add_argument("--no-essentia", action="store_true")
    p_an.add_argument("-w", "--workers", type=int, default=1)
    p_an.add_argument("--output", default="./outputs/registry")

    # resolve
    p_res = sub.add_parser("resolve", help="Resolve canonical keys")
    p_res.add_argument("--force", action="store_true")
    p_res.add_argument("--output", default="./outputs/registry")

    # review-queue
    p_rq = sub.add_parser("review-queue", help="Generate review queue")
    p_rq.add_argument("--output", default="./outputs/registry")

    # import-reviews
    p_ir = sub.add_parser("import-reviews", help="Import review decisions")
    p_ir.add_argument("--output", default="./outputs/registry")

    # sync-tags
    p_st = sub.add_parser("sync-tags", help="Write canonical key to file tags")
    p_st.add_argument("--dry-run", action="store_true", default=True)
    p_st.add_argument("--write", dest="dry_run", action="store_false")
    p_st.add_argument("--only-changed", action="store_true", default=True)
    p_st.add_argument("--output", default="./outputs/registry")

    # export
    p_ex = sub.add_parser("export", help="Generate reports")
    p_ex.add_argument("--output", default="./outputs/registry")

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("paths", nargs="*", default=["./files"])
    p_run.add_argument("--rekordbox-xml", dest="rekordbox_xml")
    p_run.add_argument("--songstats", action="store_true")
    p_run.add_argument("--songstats-limit", type=int)
    p_run.add_argument("--dry-run", action="store_true", default=True)
    p_run.add_argument("--write-tags", action="store_true")
    p_run.add_argument("--no-essentia", action="store_true")
    p_run.add_argument("-w", "--workers", type=int, default=1)
    p_run.add_argument("--output", default="./outputs/registry")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)

    if not args.command:
        # Default to run
        args.command = "run"
        args.paths = ["./files"]
        args.rekordbox_xml = None
        args.songstats = False
        args.songstats_limit = None
        args.dry_run = True
        args.write_tags = False
        args.no_essentia = False
        args.workers = 1
        args.output = "./outputs/registry"

    commands = {
        "scan": cmd_scan,
        "link": cmd_link,
        "ingest-rekordbox": cmd_ingest_rekordbox,
        "enrich-isrcs": cmd_enrich_isrcs,
        "ingest-songstats": cmd_ingest_songstats,
        "analyze": cmd_analyze,
        "resolve": cmd_resolve,
        "review-queue": cmd_review_queue,
        "import-reviews": cmd_import_reviews,
        "sync-tags": cmd_sync_tags,
        "export": cmd_export,
        "run": cmd_run,
    }

    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.error("Fatal error", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
