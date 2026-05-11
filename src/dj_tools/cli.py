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
    parser.add_argument("--no-progress", action="store_true", help="Disable registry progress bars")

    sub = parser.add_subparsers(dest="command")

    p_va = sub.add_parser("vibe-audit", help="Audit vibe score distributions from cache")
    p_va.add_argument("path", nargs="?", default="./files", metavar="PATH",
                      help="Library root (default: ./files)")
    p_va.add_argument("--output", default=None, help="Registry output dir (default: <library>/outputs/registry)")

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument(
        "paths", nargs="*", default=["./files"], metavar="PATH",
        help="Audio files or directories to process (default: ./files)",
    )
    p_run.add_argument("--rekordbox-xml", dest="rekordbox_xml", default=None,
                       help="Rekordbox XML path (default: auto-detect latest .xml in library dir)")
    p_run.add_argument("--no-songstats", action="store_true", help="Skip Songstats metadata fetch")
    p_run.add_argument("--no-tags", action="store_true", help="Skip writing tags to files")
    p_run.add_argument("--write-key-tag", action="store_true",
                       help="Also write canonical key to TKEY/InitialKey field (off by default)")
    p_run.add_argument("--no-grouping", action="store_true", help="Skip grouping phase")
    p_run.add_argument("--no-clap", action="store_true", help="Disable CLAP embeddings in grouper")
    p_run.add_argument("--no-essentia", action="store_true", help="Skip essentia key analysis")
    p_run.add_argument("-w", "--workers", type=int, default=1, help="Analysis workers (default: 1)")
    p_run.add_argument("--force-extract", action="store_true", help="Force re-extraction in grouper")
    p_run.add_argument("--output", default=None, help="Registry output dir (default: <library>/outputs/registry)")
    p_run.add_argument("--no-progress", action="store_true", help="Disable registry progress bars")

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
        summary["isrcs_enriched"] = enrich_isrcs(store, show_progress=show_progress)
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

    try:
        from dj_registry.adapters.spotify_popularity import ingest_spotify_popularity
        pop_stats = ingest_spotify_popularity(store, obs_cache=obs_cache, show_progress=show_progress)
        summary["spotify_popularity_total"] = pop_stats["total"]
        summary["spotify_popularity_cached"] = pop_stats["cached"]
        summary["spotify_popularity_fetched"] = pop_stats["fetched"]
    except Exception:
        logger.warning("Spotify popularity: fetch failed, continuing", exc_info=True)

    return summary


def _dj_taxonomy_model_dir_for_tags(store) -> str | None:
    """Prefer the active registry model, then the project-level trained model."""
    from pathlib import Path

    active = Path(store.output_dir) / "dj_taxonomy_model"
    if (active / "internal").exists() or (active / "external").exists():
        return str(active)
    fallback = Path("outputs") / "registry" / "dj_taxonomy_model"
    if (fallback / "internal").exists() or (fallback / "external").exists():
        return str(fallback)
    return None


def _run_dj_taxonomy_for_tags(store, *, show_progress: bool = False) -> int:
    """Populate internal DJ category fields before COMMENT tag sync when models exist."""
    try:
        from dj_registry.taxonomy.dj_model import classify_all_dj_taxonomies

        model_dir = _dj_taxonomy_model_dir_for_tags(store)
        if not model_dir:
            logger.info("DJ taxonomy: skipped (no model found)")
            return 0
        return classify_all_dj_taxonomies(
            store,
            model_dir=model_dir,
            primary_model="internal",
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

    input_path = Path(paths[0] if paths else "./files").resolve()
    project_dir = Path.cwd().resolve()
    try:
        input_path.relative_to(project_dir)
        return str(Path("outputs") / "groups.csv")
    except ValueError:
        return str(input_path / "outputs" / "groups.csv")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable hh:mm:ss or mm:ss or Ns."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _run_vibe_audit(output_dir: str = "./files/outputs/registry") -> int:
    """Audit canonical vibe output for the current registry and compare it to fresh derivation."""
    return _run_vibe_audit_canonical(output_dir)

    # Collect all DSP entries
    dsp_entries: list[tuple[str, str, dict]] = []  # (key_prefix, name, dsp_data)
    for key, entry in ucache._entries.items():
        if not key.endswith("|dsp"):
            continue
        if not isinstance(entry.data, dict):
            continue
        # key format: "filename|duration|dsp"
        parts = key.rsplit("|", 1)  # ["filename|duration", "dsp"]
        prefix = parts[0]  # "filename|duration"
        name_parts = prefix.split("|")
        name = name_parts[0] if name_parts else prefix
        dsp_entries.append((prefix, name, entry.data))

    from dj_tagger.moods import MOOD_LABELS, normalize_mood_code

    # Check tagger cache state for each DSP entry
    n_tagger_hit = 0
    n_tagger_stale = 0
    n_tagger_miss = 0
    cached_vibes: list[str] = []
    for prefix, name, dsp in dsp_entries:
        tagger_key = f"{prefix}|tagger"
        tagger_entry = ucache._entries.get(tagger_key)
        if tagger_entry is None:
            n_tagger_miss += 1
        elif tagger_entry.version != current_ver:
            n_tagger_stale += 1
        else:
            n_tagger_hit += 1
            if isinstance(tagger_entry.data, dict):
                cached_vibes.append(normalize_mood_code(tagger_entry.data.get("mood") or tagger_entry.data.get("vibe", "?")))

    print(f"Tagger cache: {n_tagger_hit} current, {n_tagger_stale} stale, {n_tagger_miss} missing")
    if cached_vibes:
        from collections import Counter as C
        print(f"Cached tagger moods/vibes: {dict(C(cached_vibes).most_common())}")

    # Load Songstats features from raw cache (same path as pipeline)
    # Build filename → ISRC mapping from registry, then look up songstats by ISRC
    ss_by_name: dict[str, dict[str, float]] = {}
    try:
        from dj_registry.store.csv_store import CsvStore
        store = CsvStore(output_dir)
        tracks_list = store.load_tracks()
        files_list = store.load_files()
        file_by_id = {f.file_id: f for f in files_list}
        for t in tracks_list:
            if not t.isrc_canonical or not t.primary_file_id:
                continue
            frec = file_by_id.get(t.primary_file_id)
            if not frec:
                continue
            data = ucache.get(f"isrc:{t.isrc_canonical}|songstats")
            if not data or not isinstance(data, dict):
                continue
            af: dict[str, float] = {}
            for feat_key in ("valence", "instrumentalness", "energy", "liveness", "acousticness"):
                val = data.get(feat_key, "")
                if val != "" and val is not None:
                    try:
                        af[feat_key] = float(val)
                    except (ValueError, TypeError):
                        pass
            if af:
                ss_by_name[frec.file_name] = af
    except Exception as e:
        print(f"  (Songstats lookup failed: {e})")

    # Run derive_vibe on each track (fresh, from DSP)
    labels: list[str] = []
    all_scores: dict[str, list[float]] = {v: [] for v in MOOD_LABELS}
    near_misses: dict[str, int] = {v: 0 for v in all_scores}
    ss_count = 0
    mismatches: list[tuple[str, str, str]] = []  # (name, cached_vibe, derived_vibe)

    for prefix, name, dsp in dsp_entries:
        af = ss_by_name.get(name)
        if af:
            ss_count += 1
        result = derive_vibe(dsp, audio_features=af)
        derived_label = result["vibe"]
        labels.append(derived_label)
        for v, score in result["vibe_scores"].items():
            all_scores[v].append(score)

        # Check if cached tagger mood/vibe matches
        tagger_key = f"{prefix}|tagger"
        tagger_entry = ucache._entries.get(tagger_key)
        if tagger_entry and isinstance(tagger_entry.data, dict):
            cached_label = normalize_mood_code(tagger_entry.data.get("mood") or tagger_entry.data.get("vibe", ""))
            if cached_label and cached_label != derived_label:
                mismatches.append((name, cached_label, derived_label))

        if result["vibe"] == "MEL":
            mel_score = result["vibe_scores"]["MEL"]
            for v, score in result["vibe_scores"].items():
                if v != "MEL" and mel_score - score < 0.10:
                    near_misses[v] += 1

    if not dsp_entries:
        print("No cached DSP entries found. Run 'dj run' first.")
        return 1

    n = len(dsp_entries)
    print(f"\nMood/Vibe Audit: {n} tracks ({ss_count} with Songstats data)\n")

    # Label distribution (from fresh derive_vibe)
    print("Label Distribution (fresh derive_vibe):")
    from collections import Counter
    counts = Counter(labels)
    for v in MOOD_LABELS:
        c = counts.get(v, 0)
        pct = 100.0 * c / n
        bar = "#" * int(pct / 2)
        print(f"  {v:5s}  {c:4d}  ({pct:5.1f}%)  {bar}")

    # Mismatches between cached tagger and fresh derive
    if mismatches:
        print(f"\nMismatches (cached tagger vs fresh derive): {len(mismatches)}")
        for name, cached, derived in mismatches[:10]:
            print(f"  {name}: cached={cached}, derive={derived}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    # Score stats
    print("\nScore Statistics (mean / p25 / p50 / p75 / max):")
    for v in MOOD_LABELS:
        arr = np.array(all_scores[v])
        if len(arr) == 0:
            continue
        print(f"  {v:5s}  {np.mean(arr):.3f} / {np.percentile(arr, 25):.3f} / "
              f"{np.percentile(arr, 50):.3f} / {np.percentile(arr, 75):.3f} / {np.max(arr):.3f}")

    # Near-miss analysis
    mel_count = counts.get("MEL", 0)
    if mel_count > 0:
        print(f"\nNear-Misses (non-MEL vibes that lost to MEL by <0.10):")
        for v in MOOD_LABELS:
            if v == "MEL":
                continue
            c = near_misses[v]
            if c > 0:
                print(f"  {v:5s}  {c:4d}  ({100.0 * c / mel_count:.1f}% of MEL tracks)")

    return 0


def _run_vibe_audit_canonical(output_dir: str = "./files/outputs/registry") -> int:
    """Audit the same registry-backed vibe truth that `dj run` writes."""
    from collections import Counter

    import numpy as np

    from dj_registry.adapters.local_analysis import _lookup_audio_features
    from dj_registry.store.csv_store import CsvStore
    from dj_tagger.derive import derive_vibe
    from dj_tagger.moods import MOOD_LABELS, normalize_mood_code
    from dj_tagger.universal_cache import DERIVED_VERSIONS, get_cache, quick_duration

    ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))
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
    obs_cache = ObsCache()
    store.snapshot(run_id)

    # Clear stale observations — rebuilt from cache each run
    store.save_observations([])

    # Phase 1: Scan + Link
    t0 = time.perf_counter()
    show_progress = not getattr(args, "quiet", False) and not getattr(args, "no_progress", False)
    files = scan_files(config, store, obs_cache=obs_cache, show_progress=show_progress)
    link_files_to_tracks(config, store, show_progress=show_progress)
    tracks = store.load_tracks()
    logger.info("Pipeline: scan + link done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

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
    logger.info("Pipeline: ingest done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 3: Analyze (full tagger pipeline)
    t0 = time.perf_counter()
    run_analysis(config, store, no_essentia=args.no_essentia, show_progress=show_progress)
    logger.info("Pipeline: analysis done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 4: Resolve canonical key + BPM
    t0 = time.perf_counter()
    from dj_registry.pipelines.orchestrator import _enrich_observations
    _enrich_observations(store)

    resolve_all_keys(config, store, force=True, show_progress=show_progress)
    resolve_all_bpms(config, store, force=True, show_progress=show_progress)
    build_review_queue(config, store, show_progress=show_progress)
    logger.info("Pipeline: resolve done in %s\n", _fmt_elapsed(time.perf_counter() - t0))

    # Phase 5: DJ taxonomy category for tags
    t0 = time.perf_counter()
    classified = _run_dj_taxonomy_for_tags(store, show_progress=show_progress)
    if classified:
        logger.info("Pipeline: DJ taxonomy classified %d tracks in %s\n", classified, _fmt_elapsed(time.perf_counter() - t0))

    # Phase 6: Write tags
    t0 = time.perf_counter()
    if not args.no_tags and args.no_grouping:
        sync_tags(store, dry_run=False, write_key_tag=getattr(args, "write_key_tag", False), show_progress=show_progress)
        logger.info("Pipeline: tags written in %s\n", _fmt_elapsed(time.perf_counter() - t0))
    elif not args.no_tags:
        logger.info("Tags: delayed until after grouping so group IDs can be included\n")
    else:
        logger.info("Tags: skipped (--no-tags)\n")

    # Phase 7: Reports
    classify_all_taxonomies(store, show_progress=show_progress)
    generate_reports(store, config.reports_dir, show_progress=show_progress)

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
            if args.force_extract:
                grouper_argv.append("--force-extract")
            if args.workers > 1:
                grouper_argv.extend(["-w", str(args.workers)])
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
                )
                logger.info("Pipeline: final tags written in %s", _fmt_elapsed(time.perf_counter() - tag_t0))
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
            library = getattr(args, "path", "./files")
            output = getattr(args, "output", None) or os.path.join(library, "outputs", "registry")
            return _run_vibe_audit(output)
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
