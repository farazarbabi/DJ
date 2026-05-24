"""CLI entry point for dj-registry."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
    try:
        from .progress import install_tqdm_log_handler
        install_tqdm_log_handler()
    except ImportError:
        pass
    # Suppress noisy HTTP request logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_config(args: argparse.Namespace) -> RegistryConfig:
    """Build RegistryConfig from parsed CLI args.

    Output-dir resolution order (matches `dj run`):
      1. --output (explicit)
      2. <paths[0]>/outputs/registry if --paths set (auto-derived from library location)
      3. ./outputs/registry (default)
    """
    config = RegistryConfig()

    if hasattr(args, "paths") and args.paths:
        config.library_roots = args.paths
    if hasattr(args, "rekordbox_xml") and args.rekordbox_xml:
        config.rekordbox_xml_path = args.rekordbox_xml
    if hasattr(args, "output") and args.output:
        config.output_dir = args.output
    elif hasattr(args, "paths") and args.paths:
        # Auto-derive from the first library path so models / overview / etc.
        # land next to the tracks rather than in cwd.
        first = args.paths[0] if isinstance(args.paths, list) else args.paths
        first = str(first).rstrip("/").rstrip("\\")
        config.output_dir = f"{first}/outputs/registry"
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
    from .sync.tag_writer import load_group_ids_by_file, sync_tags
    config = _build_config(args)
    store = CsvStore(config.output_dir)
    group_ids = load_group_ids_by_file(getattr(args, "groups_csv", "")) if getattr(args, "groups_csv", "") else None
    written, skipped, errors = sync_tags(
        store,
        dry_run=args.dry_run,
        only_changed=getattr(args, "only_changed", True),
        write_key_tag=getattr(args, "write_key_tag", False),
        group_ids_by_file=group_ids,
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


def cmd_dj_taxonomy(args: argparse.Namespace) -> int:
    config = _build_config(args)
    store = CsvStore(config.output_dir)

    command = getattr(args, "dj_taxonomy_command", None) or "classify"
    if command == "generate-ground-truth":
        from .taxonomy.dj_ground_truth import generate_dj_ground_truth_csv

        stats = generate_dj_ground_truth_csv(
            store,
            config=config,
            files_dir=getattr(args, "files", "./files"),
            output_path=getattr(args, "out", None),
            taxonomy_path=getattr(args, "taxonomy", None),
            model=getattr(args, "model", None),
            limit=getattr(args, "limit", None),
            force=getattr(args, "force", False),
            fail_fast=not getattr(args, "keep_going", False),
            show_progress=_show_progress(args),
            cache_dir=getattr(args, "cache_dir", None),
            collect=not getattr(args, "no_collect", False),
            include_external=not getattr(args, "no_external", False),
            include_songstats=not getattr(args, "no_songstats", False),
            songstats_limit=getattr(args, "songstats_limit", None),
            rekordbox_xml=getattr(args, "rekordbox_xml", None),
            no_essentia=getattr(args, "no_essentia", False),
            workers=getattr(args, "workers", None),
        )
        print(
            "DJ taxonomy ground truth generated: "
            f"{stats.rows_written} rows ({stats.generated} new, {stats.reused} reused, {stats.errors} errors) "
            f"-> {stats.output_path}"
        )
        if stats.collection_summary:
            print("Collection summary:")
            for key, value in sorted(stats.collection_summary.items()):
                print(f"  {key}: {value}")
        if stats.error_message:
            print(f"DJ taxonomy ground truth error: {stats.error_message}")
        return 1 if stats.errors else 0

    if command == "train-models":
        model_type = getattr(args, "model", "lr")
        if model_type == "legacy-dual":
            from .taxonomy.dj_model import train_dj_taxonomy_models

            result = train_dj_taxonomy_models(
                store,
                getattr(args, "labels"),
                taxonomy_path=getattr(args, "taxonomy", None),
                model_dir=getattr(args, "model_dir", None),
                validation_split=getattr(args, "validation_split", 0.2),
                seed=getattr(args, "seed", 42),
                show_progress=_show_progress(args),
            )
            _print_dj_taxonomy_locations(result["model_dir"], include_training=True)
            return 0
        return _cmd_train_unified(args, store)

    if command == "evaluate":
        from .taxonomy.dj_model import evaluate_dj_taxonomy_models

        metrics = evaluate_dj_taxonomy_models(
            store,
            getattr(args, "labels"),
            model_dir=getattr(args, "model_dir"),
            taxonomy_path=getattr(args, "taxonomy", None),
            show_progress=_show_progress(args),
        )
        _print_dj_taxonomy_locations(getattr(args, "model_dir"), include_training=False)
        return 0

    if command == "test-api":
        from .taxonomy.dj_ground_truth import test_dj_api_connection

        result = test_dj_api_connection(model=getattr(args, "model", None))
        if result.ok:
            print(f"DJ taxonomy API connection ok: provider={result.provider} model={result.model}")
            return 0
        print(
            "DJ taxonomy API connection failed: "
            f"provider={result.provider} model={result.model} error={result.error_message}"
        )
        return 1

    if command == "report":
        return _cmd_dj_taxonomy_report(args, store)

    from .sync.export import generate_reports
    from .taxonomy.dj_model import classify_all_dj_taxonomies

    count = classify_all_dj_taxonomies(
        store,
        taxonomy_path=getattr(args, "taxonomy", None),
        model_dir=getattr(args, "model_dir", None),
        show_progress=_show_progress(args),
    )
    generate_reports(store, config.reports_dir, show_progress=_show_progress(args))
    print(f"DJ taxonomy: {count} tracks classified by internal and external models; reports regenerated")
    return 0


def _print_dj_taxonomy_locations(model_dir: str, *, include_training: bool) -> None:
    base = Path(model_dir)
    if include_training:
        print(f"DJ taxonomy models trained -> {base}")
        print(f"Internal report -> {base / 'internal' / 'training_report.json'}")
        print(f"External report -> {base / 'external' / 'training_report.json'}")
    else:
        print(f"DJ taxonomy evaluation complete -> {base}")
    print(f"Comparison metrics -> {base / 'model_comparison.json'}")
    print(f"Per-track comparison -> {base / 'model_comparison.csv'}")


def _cmd_train_unified(args: argparse.Namespace, store: CsvStore) -> int:
    """Train LR / XGB / both via the unified entry point and print a comparison."""
    from .taxonomy.dj_model import train_dj_taxonomy_unified

    model_type = getattr(args, "model", "lr")
    result = train_dj_taxonomy_unified(
        store,
        getattr(args, "labels"),
        model_type=model_type,
        taxonomy_path=getattr(args, "taxonomy", None),
        model_dir=getattr(args, "model_dir", None),
        validation_split=getattr(args, "validation_split", 0.2),
        seed=getattr(args, "seed", 42),
        tune_lr=getattr(args, "tune_lr", False),
        xgb_n_iter=getattr(args, "xgb_n_iter", 30),
        show_progress=_show_progress(args),
    )

    base_dir = Path(result["model_dir"])
    lr_result = result.get("lr")
    xgb_result = result.get("xgb")

    print()
    print("Training complete.")
    if lr_result:
        print(f"  LR  -> {lr_result['model_dir']}")
    if xgb_result:
        print(f"  XGB -> {xgb_result['model_dir']}")

    # Comparison report when both were trained
    if model_type == "both":
        from .taxonomy.dj_model import run_library_distribution_check
        from .taxonomy.dj_model_comparison import format_full_report

        # Library distribution checks for each model variant
        lr_check = None
        xgb_check = None
        try:
            lr_check = _distribution_check_with_subdir(store, base_dir / "lr", args)
        except Exception as exc:
            print(f"  (LR library distribution check failed: {exc})")
        try:
            xgb_check = _distribution_check_with_subdir(store, base_dir / "xgb", args)
        except Exception as exc:
            print(f"  (XGB library distribution check failed: {exc})")

        examples_total = result.get("examples_total")
        classes = lr_result.get("classes") if lr_result else (xgb_result.get("classes") if xgb_result else None)
        print()
        print(format_full_report(
            lr_result, xgb_result,
            lr_check=lr_check, xgb_check=xgb_check,
            examples_total=examples_total, classes=classes,
        ))
        print()
        if "comparison_path" in result:
            print(f"Comparison JSON -> {result['comparison_path']}")

    return 0


def _distribution_check_with_subdir(store: CsvStore, model_dir: Path, args: argparse.Namespace) -> dict:
    """Helper: run run_library_distribution_check by pretending the per-variant
    dir (lr/ or xgb/) is a 'mode' subdir under the base model dir.

    run_library_distribution_check expects a base dir containing internal/ or
    external/ subdirs; for unified models the subdir IS the model dir, so we
    point it at the parent and let load_dj_taxonomy_model_if_available fall
    through to the available mode.
    """
    from .taxonomy.dj_model import run_library_distribution_check
    # The unified model lives at model_dir/model.pkl; run_library_distribution_check
    # expects model_dir/{internal,external}/model.pkl. We restructure by passing
    # the model_dir's parent and letting it fall through.
    # Simpler: temporarily load the model directly here for the check.
    from .taxonomy.dj_model import (
        load_dj_taxonomy_model_if_available,
        UNCLASSIFIED_ID,
        UNCLASSIFIED_THRESHOLD,
    )
    from .taxonomy.dj_schema import load_dj_taxonomy

    taxonomy = load_dj_taxonomy(getattr(args, "taxonomy", None))
    # Try LR loader first; if model is XGB, use the XGB loader.
    model = load_dj_taxonomy_model_if_available(model_dir, taxonomy_path=getattr(args, "taxonomy", None))
    if model is None:
        from .taxonomy.dj_model_xgb import load_xgb_model
        model = load_xgb_model(model_dir, taxonomy_path=getattr(args, "taxonomy", None))
    if model is None:
        raise FileNotFoundError(f"No model found at {model_dir}")

    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()
    file_by_track = {f.track_id: f for f in files if f.track_id and f.is_primary_file}
    obs_by_track: dict[str, list] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    from .taxonomy.dj_model import _resolve_cache
    ucache = _resolve_cache()
    counts: dict[str, int] = {}
    for track in tracks:
        track_obs = obs_by_track.get(track.track_id, [])
        file_record = file_by_track.get(track.track_id)
        prediction = model.predict(track, track_obs, file_record, taxonomy, ucache=ucache)
        if prediction.category_id and prediction.confidence < UNCLASSIFIED_THRESHOLD:
            cat_id = UNCLASSIFIED_ID
        else:
            cat_id = prediction.category_id or UNCLASSIFIED_ID
        counts[cat_id] = counts.get(cat_id, 0) + 1

    total = max(1, len(tracks))
    largest_id = max(counts, key=counts.get) if counts else ""
    largest_count = counts.get(largest_id, 0)
    largest_share = largest_count / total
    return {
        "total_tracks": len(tracks),
        "predictions": counts,
        "largest_bucket_id": largest_id,
        "largest_bucket_share": round(largest_share, 4),
        "passes_cap": largest_share <= 0.20,
        "max_bucket_share": 0.20,
    }


def _cmd_dj_taxonomy_report(args: argparse.Namespace, store: CsvStore) -> int:
    """Pretty-print training metrics, model comparison, and (optionally) library distribution."""
    model_dir = Path(getattr(args, "model_dir", None) or Path(store.output_dir) / "dj_taxonomy_model")
    if not model_dir.exists():
        print(f"No trained models found at {model_dir}.")
        print("Run `dj-registry dj-taxonomy train-models --labels <ground-truth.csv>` first.")
        return 1

    print(f"DJ Taxonomy Model Report")
    print("=" * 64)
    print(f"Model dir: {model_dir}")
    print()

    # 1. Per-mode training metrics
    print("── Training metrics (per mode, on held-out validation) ─────────")
    metrics_by_mode: dict[str, dict] = {}
    for mode in ("internal", "external"):
        report_path = model_dir / mode / "training_report.json"
        if not report_path.exists():
            print(f"  {mode}: report not found ({report_path})")
            continue
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        metrics_by_mode[mode] = metrics
        validation_kind = metrics.get("validation_kind", "?")
        validation_n = metrics.get("validation_examples", "?")
        top1 = metrics.get("top1_accuracy", 0.0)
        top3 = metrics.get("top3_accuracy", 0.0)
        mf1 = metrics.get("macro_f1", 0.0)
        wf1 = metrics.get("weighted_f1", 0.0)
        conf = metrics.get("average_confidence", 0.0)
        examples = report.get("examples", "?")
        classes = report.get("classes", "?")
        print(f"  {mode:9s}  examples={examples}  classes={classes}  validation={validation_kind} ({validation_n})")
        print(f"             top-1={top1:.1%}   top-3={top3:.1%}   macro-F1={mf1:.3f}   weighted-F1={wf1:.3f}   avg-conf={conf:.2f}")
        warnings = report.get("warnings", [])
        for warning in warnings:
            print(f"             ! {warning}")
    print()

    # 2. §9.4 gate check
    print("── §9.4 gates ──────────────────────────────────────────────────")
    gates = (
        ("top-1 ≥ 50%", "top1_accuracy", 0.50, lambda v: f"{v:.1%}"),
        ("top-3 ≥ 75%", "top3_accuracy", 0.75, lambda v: f"{v:.1%}"),
        ("macro F1 ≥ 0.30", "macro_f1", 0.30, lambda v: f"{v:.3f}"),
    )
    for label, key, threshold, fmt in gates:
        cells = []
        for mode in ("internal", "external"):
            value = metrics_by_mode.get(mode, {}).get(key, 0.0)
            mark = "✓" if value >= threshold else "✗"
            cells.append(f"{mode}={mark} {fmt(value)}")
        print(f"  {label:18s} {' | '.join(cells)}")
    print()

    # 3. Cross-model comparison
    comparison_path = model_dir / "model_comparison.json"
    if comparison_path.exists():
        with comparison_path.open("r", encoding="utf-8") as f:
            comparison = json.load(f)
        print("── Cross-model comparison ──────────────────────────────────────")
        print(f"  Examples evaluated: {comparison.get('examples', 0)}")
        agreement = comparison.get("agreement_rate", 0.0)
        print(f"  Agreement rate:     {agreement:.1%}")
        print(f"  External wins:      {comparison.get('external_improved', 0)} (correct where internal wrong)")
        print(f"  Internal wins:      {comparison.get('external_worsened', 0)} (correct where external wrong)")
        print()

    # 4. Live library distribution (if a model is present)
    print("── Library bucket distribution (current `dj run` predictions) ──")
    try:
        from .taxonomy.dj_model import run_library_distribution_check
        check = run_library_distribution_check(
            store,
            model_dir=str(model_dir),
            taxonomy_path=getattr(args, "taxonomy", None),
            show_progress=_show_progress(args),
        )
    except FileNotFoundError as exc:
        print(f"  Skipped: {exc}")
        return 0

    total = check.get("total_tracks", 0)
    largest_id = check.get("largest_bucket_id", "")
    largest_share = check.get("largest_bucket_share", 0.0)
    cap = check.get("max_bucket_share", 0.20)
    passes = check.get("passes_cap", False)
    cap_mark = "✓" if passes else "✗"
    print(f"  Total tracks:   {total}")
    print(f"  Largest bucket: {largest_id} ({largest_share:.1%})  [cap ≤{cap:.0%}: {cap_mark}]")
    counts = check.get("predictions", {})
    if counts:
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"  Top 10 predicted buckets:")
        for cat_id, count in top:
            share = count / total if total else 0.0
            print(f"    {count:4d}  ({share:5.1%})  {cat_id}")
    print()
    if not passes:
        print(f"  §9.4 bucket cap: ✗ FAILED — {largest_id!r} at {largest_share:.1%} exceeds {cap:.0%}.")
        print(f"  Consider retraining via train_with_collapse_guard, or tightening features.")
    else:
        print(f"  §9.4 bucket cap: ✓ PASSED")

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
    p_st.add_argument("--groups-csv", default="", help="Optional grouper groups.csv to append G### group IDs to COMMENT tags")
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

    # dj-taxonomy
    p_dj_tax = sub.add_parser("dj-taxonomy", help="Train and apply flat DJ-functional taxonomy models")
    p_dj_tax.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax.add_argument("--model-dir", default=None, help="Optional DJ taxonomy model directory")
    p_dj_tax.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_dj_tax)
    dj_tax_sub = p_dj_tax.add_subparsers(dest="dj_taxonomy_command")

    p_dj_tax_classify = dj_tax_sub.add_parser("classify", help="Classify tracks with both DJ taxonomy models")
    p_dj_tax_classify.add_argument("paths", nargs="*", help="Library paths (auto-derives --output)")
    p_dj_tax_classify.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax_classify.add_argument("--model-dir", default=None, help="Optional DJ taxonomy model directory")
    p_dj_tax_classify.add_argument("--output", default=None, help="Registry output dir (default: <paths>/outputs/registry or ./outputs/registry)")
    add_no_progress(p_dj_tax_classify)

    p_dj_tax_gt = dj_tax_sub.add_parser("generate-ground-truth", help="Generate GPT-5 seeded DJ taxonomy labels")
    p_dj_tax_gt.add_argument("--files", default="./files", help="Folder of audio files to label")
    p_dj_tax_gt.add_argument(
        "--out",
        default=None,
        help="Output labels CSV path (defaults to outputs/dj_taxonomy_ground_truth.csv)",
    )
    p_dj_tax_gt.add_argument("--model", default=None, help="OpenAI model to use (defaults to OPENAI_MODEL or gpt-5)")
    p_dj_tax_gt.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax_gt.add_argument("--limit", type=int, help="Maximum number of files to label")
    p_dj_tax_gt.add_argument("--force", action="store_true", help="Regenerate rows and cache entries")
    p_dj_tax_gt.add_argument("--keep-going", action="store_true", help="Continue after per-track API failures and write error rows")
    p_dj_tax_gt.add_argument("--cache-dir", default=None, help="Optional response cache directory")
    p_dj_tax_gt.add_argument("--no-collect", action="store_true", help="Do not run registry/tagger collection before GPT labeling")
    p_dj_tax_gt.add_argument("--no-external", action="store_true", help="Collect internal registry/tagger data only")
    p_dj_tax_gt.add_argument("--no-songstats", action="store_true", help="Skip Songstats enrichment even when configured")
    p_dj_tax_gt.add_argument("--rekordbox-xml", dest="rekordbox_xml", help="Optional Rekordbox XML path to ingest before labeling")
    p_dj_tax_gt.add_argument("--songstats-limit", type=int, help="Maximum Songstats tracks to fetch during collection")
    p_dj_tax_gt.add_argument("--no-essentia", action="store_true")
    p_dj_tax_gt.add_argument("-w", "--workers", type=int, default=1)
    p_dj_tax_gt.add_argument("--output", default="./outputs/registry")
    add_no_progress(p_dj_tax_gt)

    p_dj_tax_train = dj_tax_sub.add_parser("train-models", help="Train DJ taxonomy models (LR and/or XGB)")
    p_dj_tax_train.add_argument("paths", nargs="*", help="Library paths (auto-derives --output)")
    p_dj_tax_train.add_argument("--labels", required=True, help="DJ taxonomy ground-truth labels CSV")
    p_dj_tax_train.add_argument(
        "--model",
        choices=["lr", "xgb", "both", "legacy-dual"],
        default="lr",
        help="lr / xgb / both — train one or both. legacy-dual = old internal+external LR pipeline (back-compat).",
    )
    p_dj_tax_train.add_argument("--tune-lr", action="store_true", help="Enable RandomizedSearchCV hyperparameter tuning for LR")
    p_dj_tax_train.add_argument("--xgb-n-iter", type=int, default=30, help="RandomizedSearchCV iterations for XGB (default 30)")
    p_dj_tax_train.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax_train.add_argument("--model-dir", default=None, help="Output model directory (default: <output>/dj_taxonomy_model)")
    p_dj_tax_train.add_argument("--validation-split", type=float, default=0.2)
    p_dj_tax_train.add_argument("--seed", type=int, default=42)
    p_dj_tax_train.add_argument("--output", default=None, help="Registry output dir (default: <paths>/outputs/registry or ./outputs/registry)")
    add_no_progress(p_dj_tax_train)

    p_dj_tax_eval = dj_tax_sub.add_parser("evaluate", help="Evaluate both DJ taxonomy models against labels CSV")
    p_dj_tax_eval.add_argument("paths", nargs="*", help="Library paths (auto-derives --output)")
    p_dj_tax_eval.add_argument("--labels", required=True, help="DJ taxonomy ground-truth labels CSV")
    p_dj_tax_eval.add_argument("--model-dir", required=True, help="Trained DJ taxonomy model directory")
    p_dj_tax_eval.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax_eval.add_argument("--output", default=None, help="Registry output dir (default: <paths>/outputs/registry or ./outputs/registry)")
    add_no_progress(p_dj_tax_eval)

    p_dj_tax_api = dj_tax_sub.add_parser("test-api", help="Test OpenAI/Azure OpenAI DJ taxonomy labeling connection")
    p_dj_tax_api.add_argument("--model", default=None, help="OpenAI model to use (defaults to OPENAI_MODEL or gpt-5)")
    p_dj_tax_api.add_argument("--output", default="./outputs/registry")

    p_dj_tax_report = dj_tax_sub.add_parser(
        "report",
        help="Pretty-print training metrics, model comparison, gate checks, and library distribution",
    )
    p_dj_tax_report.add_argument("paths", nargs="*", help="Library paths (auto-derives --output)")
    p_dj_tax_report.add_argument("--model-dir", default=None, help="Trained model directory (default: <output>/dj_taxonomy_model)")
    p_dj_tax_report.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path")
    p_dj_tax_report.add_argument("--output", default=None, help="Registry output dir (default: <paths>/outputs/registry or ./outputs/registry)")
    add_no_progress(p_dj_tax_report)

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
        "dj-taxonomy": cmd_dj_taxonomy,
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
