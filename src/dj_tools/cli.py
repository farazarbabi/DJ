"""Unified DJ tools CLI — single command for the full pipeline.

Orchestrates: registry (scan/link/ingest) -> analysis -> resolve -> tag -> group.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

from dj_tagger import DEFAULT_LIBRARY_DIR
from dj_tagger.universal_cache import cache_dir_for

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
    # Route log output through tqdm.write so warnings (e.g. Spotify rate
    # limits) don't break active progress bars.
    try:
        from dj_registry.progress import install_tqdm_log_handler
        install_tqdm_log_handler()
    except ImportError:
        pass
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
    parser.add_argument("--no-progress", action="store_true", help="Disable registry progress bars")

    sub = parser.add_subparsers(dest="command")

    p_va = sub.add_parser("vibe-audit", help="Audit vibe score distributions from cache")
    p_va.add_argument("path", nargs="?", default=DEFAULT_LIBRARY_DIR, metavar="PATH",
                      help="Library root (default: D:\Music)")
    p_va.add_argument("--output", default=None, help="Registry output dir (default: <library>/outputs/registry)")

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument(
        "paths", nargs="*", default=[DEFAULT_LIBRARY_DIR], metavar="PATH",
        help="Audio files or directories to process (default: D:\Music)",
    )
    p_run.add_argument("--rekordbox-xml", dest="rekordbox_xml", default=None,
                       help="Rekordbox XML path (default: auto-detect latest .xml in library dir)")
    p_run.add_argument("--no-songstats", action="store_true", help="Skip Songstats metadata fetch")
    p_run.add_argument("--no-tags", action="store_true", help="Skip writing tags to files")
    p_run.add_argument("--write-key-tag", action="store_true",
                       help="Also write canonical key to TKEY/InitialKey field (off by default)")
    p_run.add_argument("--no-grouping", action="store_true", help="Skip grouping phase")
    p_run.add_argument("--cache-only", dest="cache_only", action="store_true",
                       help="Tag and group only tracks already analyzed (in the cache); "
                            "skip analysis of un-cached tracks entirely (never decodes "
                            "audio). Un-analyzed tracks are left out of tags/grouping.")
    p_run.add_argument("--no-clap", action="store_true", help="Disable CLAP embeddings in grouper")
    p_run.add_argument("--no-essentia", action="store_true", help="Skip essentia key analysis")
    p_run.add_argument("-w", "--workers", type=int, default=0,
                       help="Analysis workers (default: 0 = auto — ~cores-2, RAM-capped; pass 1 for serial)")
    p_run.add_argument("--force-extract", action="store_true", help="Force re-extraction in grouper")
    p_run.add_argument("--fine-playlists", action="store_true",
                       help="Also write fine-grained playlists (by_key/, by_subgenre/, groups/) alongside the coarse, half-resolution ones written by default")
    p_run.add_argument("--rekordbox-collection", dest="rekordbox_collection",
                       default="", metavar="XML",
                       help="Override the output path for the Rekordbox XML collection "
                            "(tracks + playlist tree + key/BPM/beatgrid/cues) written by "
                            "default for the rekordbox-xml import bridge "
                            "(default: <playlists>/collection.xml)")
    p_run.add_argument("--no-rekordbox-collection", dest="no_rekordbox_collection",
                       action="store_true",
                       help="Skip writing the Rekordbox XML collection (written by default)")
    p_run.add_argument("--output", default=None, help="Registry output dir (default: <library>/outputs/registry)")
    p_run.add_argument("--no-progress", action="store_true", help="Disable registry progress bars")
    p_run.add_argument(
        "--fetch-missing", dest="fetch_missing", nargs="*", default=None, metavar="CSV",
        help="Before analysis, download tracks from these Spotify playlist CSV(s)/dir. "
             "By default dj run uses <library>/spotify-playlists; pass CSV(s)/dir "
             "here to override that default.",
    )
    p_run.add_argument("--no-fetch-missing", dest="no_fetch_missing", action="store_true",
                       help="Skip the default Spotify playlist gap-fill phase")
    p_run.add_argument("--fetch-format", dest="fetch_format", choices=["aiff", "wav"],
                       default="aiff", help="Format for --fetch-missing downloads (default: aiff)")
    p_run.add_argument("--no-soundeo", dest="no_soundeo", action="store_true",
                       help="With --fetch-missing, force YouTube-only (skip the Soundeo source)")
    p_run.add_argument("--force-lookup", dest="force_lookup", action="store_true",
                       help="With --fetch-missing, re-search tracks previously cached as "
                            "not found on Soundeo or YouTube")
    p_run.add_argument("--check-marked-upgrades", dest="check_marked_upgrades",
                       action="store_true",
                       help="With --fetch-missing, also check existing [U]/[W]/[M] "
                            "downloads for Soundeo upgrades")
    p_run.add_argument("--cues", action="store_true",
                       help="Generate Rekordbox cue points during the run")
    p_run.add_argument("--cue-profile", choices=["v1", "v2", "v3-default"],
                       default="v3-default", help="Cue generation profile (default: v3-default)")
    p_run.add_argument("--cue-profile-file", default=None,
                       help="JSON/YAML cue profile file")
    p_run.add_argument("--cue-loop-bars", type=int, default=None,
                       help="Loop length in bars for generated cue loops")
    p_run.add_argument("--cue-force", action="store_true",
                       help="Regenerate existing auto_* cue rows for matched files")
    p_run.add_argument("--cue-force-analysis", action="store_true",
                       help="Recompute and rewrite cached cue_analysis grids from audio")
    p_run.add_argument("--cue-limit", type=int, default=None,
                       help="Limit cue analysis to this many matched Rekordbox tracks")
    p_run.add_argument("--cue-export-xml", default=None,
                       help="Write generated cues into this copied Rekordbox XML export")
    p_run.add_argument("--cue-export-policy",
                       choices=["preserve", "replace-generated", "replace-empty-slot", "review-only"],
                       default="preserve", help="Cue XML export policy")
    p_run.add_argument("--cue-export-dry-run", action="store_true",
                       help="Preview cue XML export statuses without mutating cue rows or markers")
    p_run.add_argument("--cue-quality-report", action="store_true",
                       help="Write cue quality report as part of the run")
    p_run.add_argument("--cue-validate-xml", action="store_true",
                       help="Validate Rekordbox cue marker shapes before cue work")

    p_fetch = sub.add_parser(
        "fetch-missing",
        help="Download tracks from Spotify playlist CSVs that aren't in your library yet",
    )
    p_fetch.add_argument(
        "playlists", nargs="*", metavar="CSV",
        help="Exportify-style Spotify playlist CSV file(s) or a directory of them. "
             "Defaults to <library>/spotify-playlists when omitted "
             "(not required with --prune-only)",
    )
    p_fetch.add_argument(
        "--library", default=DEFAULT_LIBRARY_DIR, metavar="DIR",
        help=f"Library dir to check for existing tracks and download into (default: {DEFAULT_LIBRARY_DIR})",
    )
    p_fetch.add_argument(
        "--prune-only", action="store_true",
        help="Only delete superseded [U]/[W]/[M] downloads whose curated original "
             "now exists in the library, then exit. No matching, downloads, or "
             "playlists. Honors --dry-run (report without deleting).",
    )
    p_fetch.add_argument(
        "--refix-soundeo-tags", dest="refix_soundeo_tags", action="store_true",
        help="Restore genuine Soundeo tags on library files that came from Soundeo "
             "(re-downloads the owned cut — free, no quota — matching each file's "
             "duration so no re-analysis is triggered), then exit. Leaves curated "
             "originals and marked fallback files untouched. Honors --dry-run.",
    )
    p_fetch.add_argument("--format", dest="audio_format", choices=["aiff", "wav"],
                         default="aiff", help="Download format (default: aiff)")
    p_fetch.add_argument("--no-soundeo", dest="no_soundeo", action="store_true",
                         help="Force YouTube-only; skip the Soundeo source even if "
                              "SOUNDEO_USER/SOUNDEO_PASS are configured")
    p_fetch.add_argument("--force-lookup", dest="force_lookup", action="store_true",
                         help="Re-search tracks previously cached as not found on "
                              "Soundeo or YouTube (default: skip them)")
    p_fetch.add_argument("--check-marked-upgrades", dest="check_marked_upgrades",
                         action="store_true",
                         help="Also check existing [U]/[W]/[M] downloads for Soundeo "
                              "upgrades (default: skip present marked files)")
    p_fetch.add_argument("--forget-cached", dest="forget_cached", nargs="+", metavar="QUERY",
                         help="Drop matching entries from the not-found cache and exit "
                              "(match a substring of 'Artist - Title', or 'all' to clear "
                              "it). Honors --dry-run.")
    p_fetch.add_argument("--threshold", type=float, default=0.62,
                         help="Match score threshold; below this counts as missing (default: 0.62)")
    p_fetch.add_argument("--duration-tolerance", dest="tolerance", type=float, default=3.0,
                         help="Accept a YouTube result only within this many seconds of the "
                              "Spotify track length (default: 3)")
    p_fetch.add_argument("--max-attempts", dest="max_attempts", type=int, default=3,
                         help="Candidate downloads to try per track before reporting it "
                              "unmatched (default: 3)")
    p_fetch.add_argument("--max-duration", type=int, default=900,
                         help="Reject YouTube results longer than this many seconds (default: 900)")
    p_fetch.add_argument("--min-duration", type=int, default=30,
                         help="Reject YouTube results shorter than this many seconds (default: 30)")
    p_fetch.add_argument("--upgrade-soundeo", dest="upgrade_soundeo", action="store_true",
                         help="Upgrade playlist tracks already in the library to better Soundeo "
                              "cuts, then exit: lossless AIFF over [U]/[M]/[W] downloads, Extended "
                              "over Original, either over a Radio Edit. Marked old copies are "
                              "deleted; unmarked ones move to outputs/fetch/replaced/. Needs "
                              "Soundeo credentials, never uses YouTube, skips missing tracks. "
                              "--dry-run only reports. Writes outputs/fetch/soundeo_upgrade.csv.")
    p_fetch.add_argument("--dry-run", action="store_true",
                         help="Only report matched/missing; do not download")

    p_rbx = sub.add_parser(
        "export-rekordbox",
        help="Generate a Rekordbox XML collection (tracks + playlist tree + "
             "key/BPM/beatgrid/cues) from an existing registry, for the "
             "rekordbox-xml import bridge",
    )
    p_rbx.add_argument(
        "paths", nargs="*", default=[DEFAULT_LIBRARY_DIR], metavar="PATH",
        help="Library root(s) (default: D:\Music) — used to locate outputs/registry "
             "and outputs/playlists",
    )
    p_rbx.add_argument("--output", default=None,
                       help="Registry output dir (default: <library>/outputs/registry)")
    p_rbx.add_argument("--out", dest="rekordbox_collection_out", default=None, metavar="XML",
                       help="Output XML path (default: <playlists>/collection.xml)")

    p_merge = sub.add_parser(
        "merge-cache",
        help="Fold another cache directory's raw/derived caches into the library cache",
    )
    p_merge.add_argument("source", metavar="DIR",
                         help="Cache directory to merge from (e.g. an old ./cache); left unchanged")
    p_merge.add_argument("--into", default=cache_dir_for(DEFAULT_LIBRARY_DIR), metavar="DIR",
                         help="Destination cache directory (default: %(default)s). For duplicate "
                              "keys the entry from the newer file wins (current tagger version "
                              "wins for derived entries); destination files are backed up as *.bak")

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


def _run_songstats(config, store, obs_cache, *, show_progress: bool = False) -> dict:
    """Try Songstats ingest. Returns summary dict. Gracefully handles failures."""
    summary = {"isrcs_enriched": 0, "songstats_total": 0, "songstats_cached": 0, "songstats_fetched": 0}

    if not config.songstats_api_key:
        logger.info("Songstats: skipped (no API key, set SONGSTATS_API_KEY)")
        return summary

    try:
        from dj_registry.adapters.spotify_isrc import enrich_isrcs
        summary["isrcs_enriched"] = enrich_isrcs(
            store, show_progress=show_progress, cache_path=config.raw_cache_path,
        )
    except Exception:
        logger.warning("Songstats: ISRC enrichment failed, continuing", exc_info=True)

    try:
        from dj_registry.adapters.songstats import ingest_songstats
        stats = ingest_songstats(config, store, obs_cache=obs_cache, show_progress=show_progress)
        summary["songstats_total"] = stats["total"]
        summary["songstats_cached"] = stats["cached"]
        summary["songstats_fetched"] = stats["fetched"]
    except Exception:
        logger.warning("Songstats: metadata fetch failed, continuing", exc_info=True)

    return summary


def _dj_taxonomy_model_dir_for_tags(store) -> str | None:
    """Prefer the active registry model, then the project-level trained model."""
    from pathlib import Path

    active = Path(store.output_dir) / "dj_taxonomy_model"
    if (active / "xgb" / "model.pkl").exists():
        return str(active)
    fallback = Path("outputs") / "registry" / "dj_taxonomy_model"
    if (fallback / "xgb" / "model.pkl").exists():
        return str(fallback)
    return None


def _run_dj_taxonomy_for_tags(store, *, show_progress: bool = False) -> int:
    """Populate DJ category fields before COMMENT tag sync when a model exists."""
    try:
        from dj_registry.taxonomy.dj_model import classify_all_dj_taxonomies

        model_dir = _dj_taxonomy_model_dir_for_tags(store)
        if not model_dir:
            logger.info("DJ taxonomy: skipped (no model found)")
            return 0
        return classify_all_dj_taxonomies(
            store,
            model_dir=model_dir,
            show_progress=show_progress,
        )
    except FileNotFoundError as exc:
        logger.info("DJ taxonomy: skipped (%s)", exc)
        return 0


def _load_group_ids_by_file(groups_csv: str) -> dict[str, str]:
    """Load filename -> group ID from the grouper output."""
    from dj_registry.sync.tag_writer import load_group_ids_by_file

    return load_group_ids_by_file(groups_csv)


def _grouper_groups_csv_path(paths: list[str]) -> str:
    """Return the groups.csv path that dj-grouper will write for these paths."""
    from pathlib import Path

    input_path = Path(paths[0] if paths else DEFAULT_LIBRARY_DIR).resolve()
    project_dir = Path.cwd().resolve()
    try:
        input_path.relative_to(project_dir)
        return str(Path("outputs") / "groups.csv")
    except ValueError:
        return str(input_path / "outputs" / "groups.csv")


def _playlists_dir(paths: list[str]) -> str:
    """Return the playlists/ directory that mirrors dj-grouper's --playlists default."""
    from pathlib import Path

    input_path = Path(paths[0] if paths else DEFAULT_LIBRARY_DIR).resolve()
    project_dir = Path.cwd().resolve()
    try:
        input_path.relative_to(project_dir)
        return str(Path("outputs") / "playlists")
    except ValueError:
        return str(input_path / "outputs" / "playlists")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable hh:mm:ss or mm:ss or Ns."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _cue_work_requested(args: argparse.Namespace) -> bool:
    return any((
        getattr(args, "cues", False),
        bool(getattr(args, "cue_export_xml", None)),
        getattr(args, "cue_quality_report", False),
        getattr(args, "cue_validate_xml", False),
    ))


def _run_cue_work(config, store, args: argparse.Namespace, *, show_progress: bool = False) -> dict:
    """Run cue-point analysis/export/reporting requested through `dj run`."""
    summary: dict = {}
    xml_path = getattr(config, "rekordbox_xml_path", "")
    if not xml_path:
        logger.info("Cues: skipped (no Rekordbox XML path)")
        return summary

    if getattr(args, "cue_validate_xml", False):
        from dj_registry.cues.validate_rekordbox import validate_rekordbox_xml

        validation = validate_rekordbox_xml(xml_path)
        summary["cue_xml_markers"] = validation.markers
        summary["cue_xml_hot"] = validation.hot_cues
        summary["cue_xml_memory"] = validation.memory_cues
        summary["cue_xml_loops"] = validation.loops
        if not validation.ok:
            for error in validation.errors:
                logger.error("Cues XML validation: %s", error)
            raise RuntimeError("Rekordbox XML cue validation failed")
        logger.info(
            "Cues: XML validation passed (%d markers: %d hot, %d memory, %d loops)",
            validation.markers,
            validation.hot_cues,
            validation.memory_cues,
            validation.loops,
        )

    should_analyze = getattr(args, "cues", False) or bool(getattr(args, "cue_export_xml", None))
    if should_analyze:
        from dj_registry.cues.analysis import analyze_rekordbox_cues

        stats = analyze_rekordbox_cues(
            config,
            store,
            limit=getattr(args, "cue_limit", None),
            force=getattr(args, "cue_force", False),
            profile=getattr(args, "cue_profile", "v3-default"),
            profile_file=getattr(args, "cue_profile_file", None),
            loop_bars=getattr(args, "cue_loop_bars", None),
            force_analysis=getattr(args, "cue_force_analysis", False),
            show_progress=show_progress,
        )
        summary["cue_tracks_analyzed"] = stats.analyzed
        summary["cue_tracks_cached"] = stats.cached
        summary["cue_points_written"] = stats.cues_written
        summary["cue_tracks_skipped"] = stats.skipped_existing
        summary["cue_tracks_failed"] = stats.failed
        logger.info(
            "Cues: analyzed=%d cached=%d written=%d skipped=%d failed=%d",
            stats.analyzed,
            stats.cached,
            stats.cues_written,
            stats.skipped_existing,
            stats.failed,
        )

    if getattr(args, "cue_export_xml", None):
        from dj_registry.cues.export_rekordbox import export_rekordbox_cues

        export_stats = export_rekordbox_cues(
            config,
            store,
            input_xml=xml_path,
            output_xml=getattr(args, "cue_export_xml"),
            policy=getattr(args, "cue_export_policy", "preserve"),
            dry_run=getattr(args, "cue_export_dry_run", False),
        )
        summary["cue_export_inserted"] = export_stats.inserted
        summary["cue_export_replaced"] = export_stats.replaced
        summary["cue_export_would_insert"] = export_stats.would_insert
        summary["cue_export_would_replace"] = export_stats.would_replace
        summary["cue_export_conflicts"] = export_stats.skipped_conflict
        summary["cue_export_invalid"] = export_stats.invalid
        summary["cue_export_report"] = export_stats.report_path
        logger.info(
            "Cues: export inserted=%d replaced=%d would_insert=%d conflicts=%d invalid=%d -> %s",
            export_stats.inserted,
            export_stats.replaced,
            export_stats.would_insert,
            export_stats.skipped_conflict,
            export_stats.invalid,
            export_stats.output_xml,
        )

    if (
        getattr(args, "cue_quality_report", False)
        or getattr(args, "cues", False)
        or bool(getattr(args, "cue_export_xml", None))
    ):
        from dj_registry.cues.quality import write_cue_quality_report

        quality = write_cue_quality_report(config, store)
        summary["cue_quality_report"] = quality.report_path
        summary["cue_quality_review"] = quality.manual_review
        logger.info(
            "Cues: quality report %d cues, %d review -> %s",
            quality.cues_total,
            quality.manual_review,
            quality.report_path,
        )

    return summary


def _run_vibe_audit(output_dir: str, cache_path: str) -> int:
    """Audit the same registry-backed vibe truth that `dj run` writes."""
    from collections import Counter

    import numpy as np

    from dj_registry.adapters.local_analysis import _lookup_audio_features
    from dj_registry.store.csv_store import CsvStore
    from dj_tagger.derive import derive_vibe
    from dj_tagger.moods import MOOD_LABELS, normalize_mood_code
    from dj_tagger.universal_cache import DERIVED_VERSIONS, get_cache, quick_duration

    ucache = get_cache(cache_path)
    store = CsvStore(output_dir)
    tracks = store.load_tracks()
    files = store.load_files()
    file_by_id = {f.file_id: f for f in files}

    registry_rows = []
    for track in tracks:
        if not track.primary_file_id:
            continue
        frec = file_by_id.get(track.primary_file_id)
        if not frec:
            continue
        cache_dur = quick_duration(frec.path_abs) or frec.audio_duration_sec
        registry_rows.append((track, frec, cache_dur))

    if not registry_rows:
        print("No registry tracks found. Run 'dj run' first.")
        return 1

    current_ver = DERIVED_VERSIONS.get("tagger", "?")
    print(f"\nDerived version (current): {current_ver}")

    canonical_labels: list[str] = []
    cached_vibes: list[str] = []
    all_scores: dict[str, list[float]] = {v: [] for v in MOOD_LABELS}
    near_misses: dict[str, int] = {v: 0 for v in all_scores}
    mismatches: list[tuple[str, str, str]] = []
    n_tagger_hit = 0
    n_tagger_stale = 0
    n_tagger_miss = 0
    ss_count = 0
    dsp_missing = 0

    for track, frec, cache_dur in registry_rows:
        filename = frec.file_name or os.path.basename(frec.path_abs)
        tagger_key = ucache.track_key(filename, cache_dur, "tagger")
        tagger_entry = ucache._entries.get(tagger_key)
        if tagger_entry is None:
            n_tagger_miss += 1
        elif tagger_entry.version != current_ver:
            n_tagger_stale += 1
        else:
            n_tagger_hit += 1
            if isinstance(tagger_entry.data, dict):
                cached_vibes.append(normalize_mood_code(tagger_entry.data.get("mood") or tagger_entry.data.get("vibe", "?")))

        if track.tagger_vibe:
            canonical_labels.append(normalize_mood_code(track.tagger_vibe))

        audio_features = _lookup_audio_features(ucache, track.isrc_canonical)
        if audio_features:
            ss_count += 1

        dsp_data = ucache.get_track(filename, cache_dur, "dsp")
        if not dsp_data or not isinstance(dsp_data, dict):
            dsp_missing += 1
            continue

        result = derive_vibe(dsp_data, audio_features=audio_features)
        derived_label = result["vibe"]
        for vibe, score in result["vibe_scores"].items():
            all_scores[vibe].append(score)

        stored_label = normalize_mood_code(track.tagger_vibe)
        if stored_label and stored_label != derived_label:
            mismatches.append((filename, stored_label, derived_label))

        if derived_label == "MEL":
            mel_score = result["vibe_scores"]["MEL"]
            for vibe, score in result["vibe_scores"].items():
                if vibe != "MEL" and mel_score - score < 0.10:
                    near_misses[vibe] += 1

    print(f"Tagger cache: {n_tagger_hit} current, {n_tagger_stale} stale, {n_tagger_miss} missing")
    if cached_vibes:
        print(f"Cached tagger moods/vibes: {dict(Counter(cached_vibes).most_common())}")

    print(f"\nMood/Vibe Audit: {len(registry_rows)} registry tracks ({ss_count} with Songstats data)")
    if dsp_missing:
        print(f"DSP missing for {dsp_missing} track(s); drift checks skipped for those entries.")

    print("\nCanonical Label Distribution (registry / dj run):")
    counts = Counter(canonical_labels)
    for vibe in MOOD_LABELS:
        count = counts.get(vibe, 0)
        pct = 100.0 * count / len(registry_rows)
        bar = "#" * int(pct / 2)
        print(f"  {vibe:5s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    if mismatches:
        print(f"\nDrift Check (registry vs fresh derive): {len(mismatches)} mismatch(es)")
        for name, cached, derived in mismatches[:10]:
            print(f"  {name}: registry={cached}, derive={derived}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("\nDrift Check (registry vs fresh derive): 0 mismatches")

    print("\nScore Statistics (mean / p25 / p50 / p75 / max):")
    for vibe in MOOD_LABELS:
        arr = np.array(all_scores[vibe])
        if len(arr) == 0:
            continue
        print(
            f"  {vibe:5s}  {np.mean(arr):.3f} / {np.percentile(arr, 25):.3f} / "
            f"{np.percentile(arr, 50):.3f} / {np.percentile(arr, 75):.3f} / {np.max(arr):.3f}"
        )

    mel_count = counts.get("MEL", 0)
    if mel_count > 0:
        print("\nNear-Misses (non-MEL vibes that lost to MEL by <0.10):")
        for vibe in MOOD_LABELS:
            if vibe == "MEL":
                continue
            count = near_misses[vibe]
            if count > 0:
                print(f"  {vibe:5s}  {count:4d}  ({100.0 * count / mel_count:.1f}% of MEL tracks)")

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
    from dj_registry.taxonomy.classifier import classify_all_taxonomies

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
    if args.output:
        config.output_dir = args.output
    else:
        config.output_dir = os.path.join(args.paths[0], "outputs", "registry")
    config.load_env()

    run_id = uuid.uuid4().hex[:8]
    store = CsvStore(config.output_dir)
    obs_cache = ObsCache(config.raw_cache_path)
    store.snapshot(run_id)

    # Clear stale observations — rebuilt from cache each run
    store.save_observations([])

    # Phase 0: Fetch missing tracks from Spotify playlists into the library.
    # Enabled by default for dj run; --fetch-missing CSV... overrides the
    # default <library>/spotify-playlists input, and --no-fetch-missing skips it.
    if not getattr(args, "no_fetch_missing", False):
        from .spotify_fetch import default_playlists_dir, fetch_missing, resolve_library_dir

        t0 = time.perf_counter()
        library_dir = resolve_library_dir(args.paths[0] if args.paths else DEFAULT_LIBRARY_DIR)
        explicit_playlists = getattr(args, "fetch_missing", None)
        default_playlists = default_playlists_dir(library_dir)
        playlists = explicit_playlists or [default_playlists]
        if explicit_playlists is None and not os.path.isdir(default_playlists):
            logger.info(
                "Pipeline: fetch-missing skipped (default playlists dir not found: %s)",
                default_playlists,
            )
        else:
            try:
                summary = fetch_missing(
                    playlists,
                    library_dir,
                    audio_format=getattr(args, "fetch_format", "aiff"),
                    use_soundeo=not getattr(args, "no_soundeo", False),
                    force_lookup=getattr(args, "force_lookup", False),
                    check_marked_upgrades=getattr(args, "check_marked_upgrades", False),
                )
                logger.info(
                    "Pipeline: fetch-missing done in %s (downloaded=%d skipped=%d failed=%d pruned=%d)",
                    _fmt_elapsed(time.perf_counter() - t0),
                    summary["downloaded"], summary["skipped"], summary["failed"],
                    summary["pruned"],
                )
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error("Pipeline: fetch-missing failed (%s); continuing without it", exc)

    # Phase 1: Scan + Link
    t0 = time.perf_counter()
    show_progress = not getattr(args, "quiet", False) and not getattr(args, "no_progress", False)
    files = scan_files(config, store, obs_cache=obs_cache, show_progress=show_progress)
    link_files_to_tracks(config, store, show_progress=show_progress)
    tracks = store.load_tracks()
    logger.info("Pipeline: scan + link done in %s", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 2: Ingest external sources
    t0 = time.perf_counter()
    if config.rekordbox_xml_path:
        from dj_registry.adapters.rekordbox_xml import ingest_rekordbox
        ingest_rekordbox(config, store, obs_cache=obs_cache, show_progress=show_progress)
    else:
        logger.info("Rekordbox: skipped (no .xml found in library dir)")

    if not args.no_songstats:
        _run_songstats(config, store, obs_cache, show_progress=show_progress)
    else:
        logger.info("Songstats: skipped (--no-songstats)")

    obs_cache.save()
    logger.info("Pipeline: ingest done in %s", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 3: Analyze (full tagger pipeline). --cache-only reuses cached
    # analysis only and skips decoding audio for un-cached tracks.
    t0 = time.perf_counter()
    cache_only = getattr(args, "cache_only", False)
    if cache_only:
        logger.info("Pipeline: --cache-only — tagging/grouping the already-analyzed subset only")
    run_analysis(config, store, no_essentia=args.no_essentia,
                 show_progress=show_progress, cache_only=cache_only)
    logger.info("Pipeline: analysis done in %s", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 4: Resolve canonical key + BPM
    t0 = time.perf_counter()
    from dj_registry.pipelines.orchestrator import _enrich_observations
    _enrich_observations(store)

    resolve_all_keys(config, store, force=True, show_progress=show_progress)
    resolve_all_bpms(config, store, force=True, show_progress=show_progress)
    build_review_queue(config, store, show_progress=show_progress)
    logger.info("Pipeline: resolve done in %s", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 5: DJ taxonomy category for tags
    t0 = time.perf_counter()
    classified = _run_dj_taxonomy_for_tags(store, show_progress=show_progress)
    if classified:
        logger.info("Pipeline: DJ taxonomy classified %d tracks in %s", classified, _fmt_elapsed(time.perf_counter() - t0))

    # Phase 6: Write tags
    t0 = time.perf_counter()
    if not args.no_tags and args.no_grouping:
        sync_tags(
            store,
            dry_run=False,
            write_key_tag=getattr(args, "write_key_tag", False),
            show_progress=show_progress,
            cache_path=config.raw_cache_path,
        )
        logger.info("Pipeline: tags written in %s", _fmt_elapsed(time.perf_counter() - t0))
    elif not args.no_tags:
        logger.info("Tags: delayed until after grouping so group IDs can be included")
    else:
        logger.info("Tags: skipped (--no-tags)")

    # Phase 7: Reports
    classify_all_taxonomies(store, show_progress=show_progress)
    generate_reports(store, config.reports_dir, show_progress=show_progress)

    # Phase 7b: Categorical playlists (by key / BPM / sub-genre)
    try:
        from dj_registry.sync.playlists import generate_categorical_playlists
        playlists_root = _playlists_dir(args.paths)
        cat_counts = generate_categorical_playlists(
            store.load_tracks(),
            store.load_files(),
            playlists_root,
            fine=getattr(args, "fine_playlists", False),
            coarse=True,
        )
        logger.info(
            "Pipeline: categorical playlists — %s in %s",
            ", ".join(f"{k}={v}" for k, v in cat_counts.items()) or "none",
            playlists_root,
        )
    except Exception:
        logger.warning("Categorical playlists: generation failed, continuing", exc_info=True)

    # Phase 7c: Cue points
    if _cue_work_requested(args):
        t0 = time.perf_counter()
        _run_cue_work(config, store, args, show_progress=show_progress)
        logger.info("Pipeline: cue work done in %s", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 8: Grouping
    if not args.no_grouping:
        t0 = time.perf_counter()
        try:
            from dj_grouper.cli import main as grouper_main
            grouper_argv = list(args.paths)
            groups_csv = _grouper_groups_csv_path(args.paths)
            grouper_argv.extend(["--registry-dir", config.output_dir])
            grouper_argv.extend(["--csv", groups_csv])
            if args.no_clap:
                grouper_argv.append("--no-clap")
            if cache_only:
                grouper_argv.append("--cache-only")
            if args.force_extract:
                grouper_argv.append("--force-extract")
            if args.workers > 1:
                grouper_argv.extend(["-w", str(args.workers)])
            if getattr(args, "fine_playlists", False):
                grouper_argv.append("--fine-playlists")
            grouper_main(grouper_argv)
            logger.info("Pipeline: grouping done in %s", _fmt_elapsed(time.perf_counter() - t0))
            if not args.no_tags:
                tag_t0 = time.perf_counter()
                group_ids = _load_group_ids_by_file(groups_csv)
                sync_tags(
                    store,
                    dry_run=False,
                    write_key_tag=getattr(args, "write_key_tag", False),
                    group_ids_by_file=group_ids,
                    show_progress=show_progress,
                    cache_path=config.raw_cache_path,
                )
                logger.info("Pipeline: final tags written in %s", _fmt_elapsed(time.perf_counter() - tag_t0))
        except Exception:
            logger.error("Grouping failed", exc_info=True)
    else:
        logger.info("Grouping: skipped (--no-grouping)")

    # Phase 9: Rekordbox XML collection (on by default) — runs last so its
    # playlist tree captures the categorical, grouper, and Spotify playlists
    # together. Skip with --no-rekordbox-collection.
    if not getattr(args, "no_rekordbox_collection", False):
        try:
            from dj_registry.sync.rekordbox_export import generate_rekordbox_collection
            playlists_root = _playlists_dir(args.paths)
            out_path = getattr(args, "rekordbox_collection", "") or os.path.join(
                playlists_root, "collection.xml"
            )
            res = generate_rekordbox_collection(
                store.load_tracks(),
                store.load_files(),
                store.load_cue_points(),
                out_path,
                playlists_root=playlists_root,
            )
            logger.info(
                "Pipeline: rekordbox collection — %s tracks, %s playlists, %s entries -> %s",
                res["tracks"], res["playlists"], res["entries"], res["out_path"],
            )
        except Exception:
            logger.warning("Rekordbox collection: generation failed, continuing", exc_info=True)

    logger.info("Pipeline: total %s", _fmt_elapsed(time.perf_counter() - t_start))
    return 0


def _fetch_playlists(args: argparse.Namespace) -> list[str] | None:
    """Playlist inputs for fetch-missing, defaulting to ``<library>/spotify-playlists``.

    Returns None (after logging the error) when nothing was given and the
    default directory does not exist.
    """
    if args.playlists:
        return args.playlists
    from .spotify_fetch import default_playlists_dir

    default_dir = default_playlists_dir(args.library)
    if not os.path.isdir(default_dir):
        logger.error(
            "fetch-missing: no playlists given and default %s not found; "
            "pass CSV(s)/dir or use --prune-only", default_dir,
        )
        return None
    logger.info("fetch-missing: no playlists given, using default %s", default_dir)
    return [default_dir]


def _run_fetch_missing(args: argparse.Namespace) -> int:
    """Match Spotify playlist CSVs against the library and download what's missing."""
    from .spotify_fetch import fetch_missing, prune_superseded_downloads

    if getattr(args, "forget_cached", None):
        from .spotify_fetch import forget_not_found
        removed = forget_not_found(args.library, args.forget_cached, dry_run=args.dry_run)
        verb = "would forget" if args.dry_run else "forgot"
        if removed:
            logger.info("fetch-missing: %s %d cached not-found entry(ies)", verb, len(removed))
            for label in removed:
                logger.info("  %s %s", verb, label)
        else:
            logger.info("fetch-missing: no cached not-found entries matched %s",
                        args.forget_cached)
        return 0

    if getattr(args, "refix_soundeo_tags", False):
        from .spotify_fetch import refix_soundeo_tags
        try:
            refix_soundeo_tags(args.library, dry_run=args.dry_run,
                               audio_format=args.audio_format)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("fetch-missing: %s", exc)
            return 1
        return 0

    if getattr(args, "upgrade_soundeo", False):
        from .soundeo import SoundeoClient, SoundeoError
        from .soundeo_upgrade import upgrade_soundeo

        if getattr(args, "no_soundeo", False):
            logger.error("fetch-missing: --upgrade-soundeo cannot be combined with --no-soundeo")
            return 1
        playlists = _fetch_playlists(args)
        if playlists is None:
            return 1
        soundeo = SoundeoClient.from_env(audio_format=args.audio_format)
        if soundeo is None:
            logger.error("fetch-missing: --upgrade-soundeo needs SOUNDEO_USER/SOUNDEO_PASS in .env")
            return 1
        try:
            upgrade_soundeo(
                playlists, args.library, soundeo=soundeo, audio_format=args.audio_format,
                threshold=args.threshold, dry_run=args.dry_run,
            )
        except (FileNotFoundError, RuntimeError, SoundeoError) as exc:
            logger.error("fetch-missing: %s", exc)
            return 1
        finally:
            soundeo.close()
        return 0

    if args.prune_only:
        if not os.path.isdir(args.library):
            logger.error("fetch-missing: library dir not found: %s", args.library)
            return 1
        removed = prune_superseded_downloads(args.library, dry_run=args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        logger.info("fetch-missing: %s %d superseded marked download(s)", verb, len(removed))
        for p in removed:
            logger.info("  %s %s", verb, os.path.basename(p))
        return 0

    playlists = _fetch_playlists(args)
    if playlists is None:
        return 1

    try:
        fetch_missing(
            playlists,
            args.library,
            audio_format=args.audio_format,
            threshold=args.threshold,
            tolerance=args.tolerance,
            max_attempts=args.max_attempts,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            dry_run=args.dry_run,
            use_soundeo=not getattr(args, "no_soundeo", False),
            force_lookup=getattr(args, "force_lookup", False),
            check_marked_upgrades=getattr(args, "check_marked_upgrades", False),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("fetch-missing: %s", exc)
        return 1
    return 0


def _run_merge_cache(args: argparse.Namespace) -> int:
    """Fold another cache directory into the library cache."""
    from dj_tagger.universal_cache import merge_cache_dirs

    if not os.path.isdir(args.source):
        logger.error("merge-cache: source dir not found: %s", args.source)
        return 1
    stats = merge_cache_dirs(args.source, args.into)
    for name, s in stats.items():
        logger.info("merge-cache: %s — added %d, replaced %d, now %d entries",
                    name, s["added"], s["replaced"], s["total"])
    logger.info("merge-cache: %s merged into %s (previous files kept as *.bak)",
                args.source, args.into)
    return 0


def _run_export_rekordbox(args: argparse.Namespace) -> int:
    """Generate a Rekordbox XML collection from an existing registry."""
    from dj_registry.store.csv_store import CsvStore
    from dj_registry.sync.rekordbox_export import generate_rekordbox_collection

    paths = getattr(args, "paths", None) or [DEFAULT_LIBRARY_DIR]
    output_dir = args.output or os.path.join(paths[0], "outputs", "registry")
    store = CsvStore(output_dir)
    playlists_root = _playlists_dir(paths)
    out_path = getattr(args, "rekordbox_collection_out", None) or os.path.join(
        playlists_root, "collection.xml"
    )
    res = generate_rekordbox_collection(
        store.load_tracks(),
        store.load_files(),
        store.load_cue_points(),
        out_path,
        playlists_root=playlists_root,
    )
    logger.info(
        "Rekordbox collection: %s tracks, %s playlists, %s entries (%s unresolved) -> %s",
        res["tracks"], res["playlists"], res["entries"], res["unresolved"], res["out_path"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    raw = argv if argv is not None else sys.argv[1:]
    # Handle --version and --help at top level
    if raw and raw[0] in ("--version", "--help", "-h"):
        args = parser.parse_args(raw)
        return 0
    # Default to "run" when no subcommand is given
    known_commands = {"run", "vibe-audit", "fetch-missing", "export-rekordbox", "merge-cache"}
    if not raw or raw[0] not in known_commands:
        raw = ["run"] + list(raw)

    args = parser.parse_args(raw)
    _setup_logging(
        getattr(args, "verbose", False),
        getattr(args, "quiet", False),
    )

    if args.command == "vibe-audit":
        try:
            library = getattr(args, "path", DEFAULT_LIBRARY_DIR)
            output = getattr(args, "output", None) or os.path.join(library, "outputs", "registry")
            return _run_vibe_audit(output, os.path.join(cache_dir_for(library), "raw_cache.pkl"))
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    if args.command == "merge-cache":
        return _run_merge_cache(args)

    if args.command == "run":
        try:
            return _run_pipeline(args)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    if args.command == "fetch-missing":
        try:
            return _run_fetch_missing(args)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    if args.command == "export-rekordbox":
        try:
            return _run_export_rekordbox(args)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.error("Fatal error", exc_info=True)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
