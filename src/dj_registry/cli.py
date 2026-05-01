"""CLI entry point for dj-registry."""

from __future__ import annotations

import argparse
import json
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


def _show_progress(args: argparse.Namespace) -> bool:
    return not getattr(args, "quiet", False) and not getattr(args, "no_progress", False)


def cmd_scan(args: argparse.Namespace) -> int:
    from .adapters.file_scanner import scan_files
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    files = scan_files(config, store, show_progress=_show_progress(args))
    print(f"Scanned {len(files)} files")
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    from .identity.matcher import link_files_to_tracks
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    link_files_to_tracks(config, store, show_progress=_show_progress(args))
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
    count = ingest_rekordbox(config, store, show_progress=_show_progress(args))
    print(f"Ingested {count} Rekordbox tracks")
    return 0


def cmd_enrich_isrcs(args: argparse.Namespace) -> int:
    from .adapters.spotify_isrc import enrich_isrcs
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    limit = getattr(args, "limit", None)
    count = enrich_isrcs(store, limit=limit, show_progress=_show_progress(args))
    print(f"Enriched {count} tracks with ISRCs")
    return 0


def cmd_ingest_songstats(args: argparse.Namespace) -> int:
    from .adapters.songstats import ingest_songstats
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    limit = getattr(args, "limit", None)
    only_missing = getattr(args, "only_missing", False)
    stats = ingest_songstats(config, store, limit=limit, only_missing=only_missing, show_progress=_show_progress(args))
    print(f"Songstats: {stats['total']} tracks ({stats['cached']} cached, {stats['fetched']} fetched)")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .adapters.local_analysis import run_analysis
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    stats = run_analysis(
        config, store,
        no_essentia=getattr(args, "no_essentia", False),
        show_progress=_show_progress(args),
    )
    print(f"Analysis: {stats['total']} tracks ({stats['cached']} cached, {stats['analyzed']} analyzed, {stats['failed']} failed)")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from .resolver.bpm_resolver import resolve_all_bpms
    from .resolver.key_resolver import resolve_all_keys
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    force = getattr(args, "force", False)
    resolved, review = resolve_all_keys(config, store, force=force, show_progress=_show_progress(args))
    print(f"Resolved {resolved} keys, {review} need review")
    bpm_resolved, bpm_missing = resolve_all_bpms(config, store, force=force, show_progress=_show_progress(args))
    print(f"Resolved {bpm_resolved} BPMs, {bpm_missing} missing")
    return 0


def cmd_review_queue(args: argparse.Namespace) -> int:
    from .review.queue_builder import build_review_queue
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    count = build_review_queue(config, store, show_progress=_show_progress(args))
    print(f"Review queue: {count} items")
    return 0


def cmd_import_reviews(args: argparse.Namespace) -> int:
    from .review.importer import import_reviews
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    accepted, overridden, skipped = import_reviews(store, show_progress=_show_progress(args))
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
        write_key_tag=getattr(args, "write_key_tag", False),
        show_progress=_show_progress(args),
    )
    print(f"Tags: {written} written, {skipped} skipped, {errors} errors")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .sync.export import generate_reports
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    generate_reports(store, config.reports_dir, show_progress=_show_progress(args))
    print("Reports generated")
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    config = _build_config(args)
    store = CsvStore(config.output_dir)

    taxonomy_command = getattr(args, "taxonomy_command", None) or "classify"
    if taxonomy_command == "train-model":
        from .taxonomy.model import train_taxonomy_model

        stats = train_taxonomy_model(
            store,
            getattr(args, "labels"),
            taxonomy_path=getattr(args, "taxonomy", None),
            model_dir=getattr(args, "model_dir", None),
            show_progress=_show_progress(args),
        )
        print(
            "Taxonomy model trained: "
            f"{stats.examples} examples, {stats.subgenre_classes} subgenre classes -> {stats.model_dir}"
        )
        return 0

    if taxonomy_command == "evaluate":
        from .taxonomy.model import evaluate_taxonomy_model

        metrics = evaluate_taxonomy_model(
            store,
            getattr(args, "labels"),
            model_dir=getattr(args, "model_dir"),
            taxonomy_path=getattr(args, "taxonomy", None),
            show_progress=_show_progress(args),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if taxonomy_command == "generate-ground-truth":
        from .taxonomy.ground_truth import generate_ground_truth_csv

        stats = generate_ground_truth_csv(
            store,
            files_dir=getattr(args, "files", "./files"),
            output_path=getattr(args, "out", "files/taxonomy_ground_truth.csv"),
            taxonomy_path=getattr(args, "taxonomy", None),
            model=getattr(args, "model", None),
            limit=getattr(args, "limit", None),
            force=getattr(args, "force", False),
            fail_fast=not getattr(args, "keep_going", False),
            show_progress=_show_progress(args),
            cache_dir=getattr(args, "cache_dir", None),
        )
        print(
            "Ground truth generated: "
            f"{stats.rows_written} rows ({stats.generated} new, {stats.reused} reused, {stats.errors} errors) "
            f"-> {stats.output_path}"
        )
        if stats.error_message:
            print(f"Ground truth error: {stats.error_message}")
        return 1 if stats.errors else 0

    if taxonomy_command == "test-api":
        from .taxonomy.ground_truth import test_api_connection

        result = test_api_connection(model=getattr(args, "model", None))
        if result.ok:
            print(f"Taxonomy API connection ok: provider={result.provider} model={result.model}")
            return 0
        print(
            "Taxonomy API connection failed: "
            f"provider={result.provider} model={result.model} error={result.error_message}"
        )
        return 1

    from .taxonomy.classifier import classify_all_taxonomies
    from .sync.export import generate_reports

    count = classify_all_taxonomies(
        store,
        taxonomy_path=getattr(args, "taxonomy", None),
        model_dir=getattr(args, "model_dir", None),
        use_model=not getattr(args, "no_model", False),
        show_progress=_show_progress(args),
    )
    generate_reports(store, config.reports_dir, show_progress=_show_progress(args))
    print(f"Genre taxonomy: {count} tracks classified; reports regenerated")
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
        show_progress=_show_progress(args),
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
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars")

    sub = parser.add_subparsers(dest="command")

    def add_no_progress(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars")

    # scan
    p_scan = sub.add_parser("scan", help="Scan library files")
    p_scan.add_argument("paths", nargs="*", default=["./files"])
    p_scan.add_argument("--dry-run", action="store_true")
    p_scan.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_scan)

    # link
    p_link = sub.add_parser("link", help="Link files to tracks")
    p_link.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_link)

    # ingest-rekordbox
    p_rb = sub.add_parser("ingest-rekordbox", help="Ingest Rekordbox XML")
    p_rb.add_argument("--xml", dest="rekordbox_xml", required=True)
    p_rb.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_rb)

    # enrich-isrcs
    p_isrc = sub.add_parser("enrich-isrcs", help="Look up ISRCs via Spotify")
    p_isrc.add_argument("--limit", type=int)
    p_isrc.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_isrc)

    # ingest-songstats
    p_ss = sub.add_parser("ingest-songstats", help="Fetch Songstats metadata")
    p_ss.add_argument("--limit", type=int)
    p_ss.add_argument("--only-missing", action="store_true")
    p_ss.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_ss)

    # analyze
    p_an = sub.add_parser("analyze", help="Run local key analysis")
    p_an.add_argument("--no-essentia", action="store_true")
    p_an.add_argument("-w", "--workers", type=int, default=1)
    p_an.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_an)

    # resolve
    p_res = sub.add_parser("resolve", help="Resolve canonical keys")
    p_res.add_argument("--force", action="store_true")
    p_res.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_res)

    # review-queue
    p_rq = sub.add_parser("review-queue", help="Generate review queue")
    p_rq.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_rq)

    # import-reviews
    p_ir = sub.add_parser("import-reviews", help="Import review decisions")
    p_ir.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_ir)

    # sync-tags
    p_st = sub.add_parser("sync-tags", help="Write canonical key to file tags")
    p_st.add_argument("--dry-run", action="store_true", default=True)
    p_st.add_argument("--write", dest="dry_run", action="store_false")
    p_st.add_argument("--only-changed", action="store_true", default=True)
    p_st.add_argument("--write-key-tag", action="store_true", default=False,
                      help="Also write canonical key to TKEY/InitialKey field (off by default)")
    p_st.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_st)

    # export
    p_ex = sub.add_parser("export", help="Generate reports")
    p_ex.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_ex)

    # taxonomy
    p_tax = sub.add_parser("taxonomy", help="Classify tracks into 3-level genre taxonomy")
    p_tax.add_argument("--taxonomy", default=None, help="Optional taxonomy JSON path")
    p_tax.add_argument("--model-dir", default=None, help="Optional trained taxonomy model directory")
    p_tax.add_argument("--no-model", action="store_true", help="Disable trained taxonomy model even if present")
    p_tax.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_tax)
    tax_sub = p_tax.add_subparsers(dest="taxonomy_command")

    p_tax_classify = tax_sub.add_parser("classify", help="Classify tracks")
    p_tax_classify.add_argument("--taxonomy", default=None, help="Optional taxonomy JSON path")
    p_tax_classify.add_argument("--model-dir", default=None, help="Optional trained taxonomy model directory")
    p_tax_classify.add_argument("--no-model", action="store_true", help="Disable trained taxonomy model")
    p_tax_classify.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_tax_classify)

    p_tax_train = tax_sub.add_parser("train-model", help="Train learned taxonomy model from labels CSV")
    p_tax_train.add_argument("--labels", required=True, help="External taxonomy labels CSV")
    p_tax_train.add_argument("--taxonomy", default=None, help="Optional taxonomy JSON path")
    p_tax_train.add_argument("--model-dir", default=None, help="Output model directory")
    p_tax_train.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_tax_train)

    p_tax_eval = tax_sub.add_parser("evaluate", help="Evaluate trained taxonomy model against labels CSV")
    p_tax_eval.add_argument("--labels", required=True, help="External taxonomy labels CSV")
    p_tax_eval.add_argument("--model-dir", required=True, help="Trained taxonomy model directory")
    p_tax_eval.add_argument("--taxonomy", default=None, help="Optional taxonomy JSON path")
    p_tax_eval.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_tax_eval)

    p_tax_gt = tax_sub.add_parser("generate-ground-truth", help="Generate GPT-5 seeded taxonomy labels")
    p_tax_gt.add_argument("--files", default="./files", help="Folder of audio files to label")
    p_tax_gt.add_argument("--out", default="files/taxonomy_ground_truth.csv", help="Output labels CSV path")
    p_tax_gt.add_argument("--model", default=None, help="OpenAI model to use (defaults to OPENAI_MODEL or gpt-5)")
    p_tax_gt.add_argument("--taxonomy", default=None, help="Optional taxonomy JSON path")
    p_tax_gt.add_argument("--limit", type=int, help="Maximum number of files to label")
    p_tax_gt.add_argument("--force", action="store_true", help="Regenerate rows and cache entries")
    p_tax_gt.add_argument("--keep-going", action="store_true", help="Continue after per-track API failures and write error rows")
    p_tax_gt.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    p_tax_gt.add_argument("--cache-dir", default=None, help="Optional response cache directory")
    p_tax_gt.add_argument("--output", default="./outputs/registry")

    p_tax_api = tax_sub.add_parser("test-api", help="Test OpenAI/Azure OpenAI taxonomy labeling connection")
    p_tax_api.add_argument("--model", default=None, help="OpenAI model to use (defaults to OPENAI_MODEL or gpt-5)")
    p_tax_api.add_argument("--output", default="./outputs/registry")

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
    add_no_progress(p_run)

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
        "taxonomy": cmd_taxonomy,
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
