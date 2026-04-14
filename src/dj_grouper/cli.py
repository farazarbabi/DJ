"""CLI for dj-grouper: run, extract, cluster, review, apply, recommend, feedback."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import GrouperConfig

logger = logging.getLogger(__name__)

_DEFAULTS = GrouperConfig()


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if verbose
        else "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dj-grouper",
        description="DJ track grouping and recommendation system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # --- run (all-in-one) ---
    p_run = sub.add_parser("run", help="Run full pipeline: extract -> cluster -> recommend -> output")
    p_run.add_argument("paths", nargs="*", default=[_DEFAULTS.input_dir], metavar="PATH", help="Library paths (default: ./files)")
    p_run.add_argument("-r", "--recursive", action="store_true")
    p_run.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers for extraction (default: 1)")
    p_run.add_argument("--output", default=_DEFAULTS.grouped_dir, help="Folder output dir")
    p_run.add_argument("--write-tags", action="store_true", help="Write tags to file metadata")
    p_run.add_argument("--dry-run", action="store_true", help="Preview without writing anything")
    p_run.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking")
    p_run.add_argument("--no-clap", action="store_true", help="Disable CLAP embeddings")
    p_run.add_argument("--playlists", default=_DEFAULTS.playlists_dir, help="Playlists dir")
    p_run.add_argument("--csv", default=_DEFAULTS.groups_file)
    p_run.add_argument("--recommendations-csv", default=_DEFAULTS.recommendations_file)
    p_run.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_run.add_argument("--clap-cache", default=_DEFAULTS.clap_cache_file)
    p_run.add_argument("--feedback", default=_DEFAULTS.feedback_file)
    p_run.add_argument("--force-extract", action="store_true", help="Clear DSP cache + outputs, re-extract")
    p_run.add_argument("--force-clap", action="store_true", help="Clear CLAP cache, re-extract embeddings")
    p_run.add_argument("--rebuild", action="store_true", help="Force full re-clustering (ignore existing groups)")
    p_run.add_argument("--clean", action="store_true", help="Delete all outputs and caches, then exit")

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Extract features for all tracks")
    p_extract.add_argument("paths", nargs="*", default=[_DEFAULTS.input_dir], metavar="PATH", help="Library paths (default: ./files)")
    p_extract.add_argument("-r", "--recursive", action="store_true")
    p_extract.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers (default: 1)")
    p_extract.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_extract.add_argument("--features-csv", default=None)
    p_extract.add_argument("--force", action="store_true", help="Regenerate features even if cache exists")

    # --- cluster ---
    p_cluster = sub.add_parser("cluster", help="Cluster tracks into groups")
    p_cluster.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_cluster.add_argument("--feedback", default=_DEFAULTS.feedback_file)
    p_cluster.add_argument("--output-csv", default=_DEFAULTS.groups_file)

    # --- review ---
    p_review = sub.add_parser("review", help="Review proposed groupings")
    p_review.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_review.add_argument("--feedback", default=_DEFAULTS.feedback_file)

    # --- apply ---
    p_apply = sub.add_parser("apply", help="Write tags and create folders")
    p_apply.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_apply.add_argument("--output", default=_DEFAULTS.grouped_dir)
    p_apply.add_argument("--write-tags", action="store_true")
    p_apply.add_argument("--copy", action="store_true")
    p_apply.add_argument("--dry-run", action="store_true")
    p_apply.add_argument("--playlists", default=_DEFAULTS.playlists_dir)

    # --- recommend ---
    p_rec = sub.add_parser("recommend", help="Compute recommendations")
    p_rec.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_rec.add_argument("--csv", default=_DEFAULTS.recommendations_file)
    p_rec.add_argument("--playlists", default=_DEFAULTS.playlists_dir)

    # --- feedback ---
    p_fb = sub.add_parser("feedback", help="Add feedback entries")
    p_fb.add_argument("--good", nargs=2, metavar=("TRACK_A", "TRACK_B"))
    p_fb.add_argument("--bad", nargs=2, metavar=("TRACK_A", "TRACK_B"))
    p_fb.add_argument("--override", nargs=2, metavar=("TRACK", "GID"))
    p_fb.add_argument("--file", default=_DEFAULTS.feedback_file)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate recommendations against known pairs")
    p_eval.add_argument("--cache", default=_DEFAULTS.cache_file)
    p_eval.add_argument("--eval-file", required=True, help="CSV with track_a, track_b, label columns")

    return parser


_SUBCOMMANDS = {"run", "extract", "cluster", "review", "apply", "recommend", "feedback", "evaluate"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    # Default to "run" when no subcommand is given
    raw = argv if argv is not None else sys.argv[1:]
    if not any(arg in _SUBCOMMANDS for arg in raw):
        raw = ["run"] + list(raw)
    args = parser.parse_args(raw)
    _setup_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))

    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "extract":
            return _cmd_extract(args)
        elif args.command == "cluster":
            return _cmd_cluster(args)
        elif args.command == "review":
            return _cmd_review(args)
        elif args.command == "apply":
            return _cmd_apply(args)
        elif args.command == "recommend":
            return _cmd_recommend(args)
        elif args.command == "feedback":
            return _cmd_feedback(args)
        elif args.command == "evaluate":
            return _cmd_evaluate(args)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=getattr(args, "verbose", False))
        return 1

    return 0


# ─── Parallel extraction worker ──────────────────────────────────────────────


def _extract_worker(track_path: str, needs_analysis: bool) -> dict:
    """Run audio loading + analysis + DSP extraction for one track.

    Returns a dict with all results. Must be a top-level function for pickling.
    """
    from dj_tagger.audio import load_audio_features
    from dj_tagger.analyzers.energy import analyze_energy
    from dj_tagger.analyzers.key import analyze_key
    from dj_tagger.analyzers.sections import analyze_sections
    from dj_tagger.analyzers.structure import analyze_structure
    from dj_tagger.analyzers.vibe import analyze_vibe
    from dj_tagger.analyzers.vocal import analyze_vocal
    from .features.dsp import extract_dsp_features, extract_section_dsp

    audio = load_audio_features(track_path)
    result = {}

    if needs_analysis:
        energy_result = analyze_energy(audio)
        result["energy"] = energy_result.level
        result["energy_conf"] = energy_result.confidence

        key_result = analyze_key(audio)
        result["key"] = key_result.camelot
        result["key_conf"] = key_result.confidence

        structure_result = analyze_structure(audio)
        result["structure"] = structure_result.formatted
        result["intro_bars"] = structure_result.intro_bars
        result["flow_type"] = structure_result.flow_type
        result["structure_conf"] = structure_result.confidence

    result["dsp"] = extract_dsp_features(audio)
    result["tempo"] = audio.tempo

    section_map = analyze_sections(audio)
    result["section_dsp"] = extract_section_dsp(audio, section_map)

    vibe_result = analyze_vibe(audio)
    result["vibe"] = vibe_result.label
    result["vibe_scores"] = vibe_result.scores
    result["vibe_conf"] = vibe_result.confidence

    vocal_result = analyze_vocal(audio)
    result["vocal"] = "V" if vocal_result.has_vocals else "NV"
    result["vocal_conf"] = vocal_result.confidence
    result["has_vocals"] = vocal_result.has_vocals

    return result


# ─── Shared extraction service ──────────────────────────────────────────────


class _ExtractionStats:
    __slots__ = ("n_cached", "n_extracted", "n_analyzed", "n_failed", "n_removed")

    def __init__(self) -> None:
        self.n_cached = 0
        self.n_extracted = 0
        self.n_analyzed = 0
        self.n_failed = 0
        self.n_removed = 0

    def summary_parts(self) -> list[str]:
        parts = []
        if self.n_cached:
            parts.append(f"{self.n_cached} cached")
        if self.n_analyzed:
            parts.append(f"{self.n_analyzed} analyzed")
        if self.n_extracted:
            parts.append(f"{self.n_extracted} extracted")
        if self.n_failed:
            parts.append(f"{self.n_failed} failed")
        if self.n_removed:
            parts.append(f"{self.n_removed} removed")
        return parts


def _run_extraction(
    tracks,
    cache_path: str,
    *,
    force: bool = False,
    workers: int = 1,
    analyze_untagged: bool = False,
    write_tags: bool = False,
) -> tuple[dict, _ExtractionStats]:
    """Shared extraction logic for run and extract commands.

    Args:
        tracks: scanned track list from scan_library.
        cache_path: path to the raw features pickle cache.
        force: clear cache and re-extract everything.
        workers: parallel worker count.
        analyze_untagged: if True, run full analysis (energy/key/structure)
            on tracks missing tags. If False, only extract DSP/vibe/vocal.
        write_tags: if True and analysis ran, write tags to file metadata.

    Returns (raw_cache, stats).
    """
    import os
    from pathlib import Path
    from .features.builder import RawCacheEntry, load_raw_cache, save_raw_cache

    stats = _ExtractionStats()
    raw_cache = {} if force else load_raw_cache(cache_path)

    # Separate cached vs needs-extraction
    to_extract: list[tuple[int, object, float, bool]] = []
    for i, t in enumerate(tracks):
        mtime = os.path.getmtime(t.path)
        cached = raw_cache.get(t.path)
        if cached and cached.mtime == mtime:
            cached.info = t
            stats.n_cached += 1
        else:
            needs_analysis = analyze_untagged and (t.energy is None or t.key is None)
            to_extract.append((i, t, mtime, needs_analysis))

    if stats.n_cached:
        print(f"  {stats.n_cached} tracks loaded from cache")

    # Tag writing imports (only when needed)
    _format_tag = None
    _write_tag = None
    if write_tags:
        from dj_tagger.formats import format_tag as _format_tag
        from dj_tagger.metadata import write_tag as _write_tag

    def _apply_result(t, mtime, needs_analysis, result):
        """Apply worker result to track object and cache."""
        if needs_analysis:
            t.energy = result["energy"]
            t.confidences["energy"] = result["energy_conf"]
            t.key = result["key"]
            t.confidences["key"] = result["key_conf"]
            t.structure = result["structure"]
            t.intro_bars = result["intro_bars"]
            t.flow_type = result["flow_type"]
            t.confidences["structure"] = result["structure_conf"]
            stats.n_analyzed += 1

        if t.bpm is None and result["tempo"] > 0:
            t.bpm = round(result["tempo"])
        t.vibe = result["vibe"]
        t.vibe_scores = result["vibe_scores"]
        t.confidences["vibe"] = result["vibe_conf"]
        t.vocal = result["vocal"]
        t.confidences["vocal"] = result["vocal_conf"]

        if needs_analysis and _format_tag and _write_tag:
            base_tag = _format_tag(
                energy=t.energy, camelot=t.key, bpm=t.bpm,
                structure=t.structure, vibe=t.vibe,
                has_vocals=result["has_vocals"],
            )
            _write_tag(t.path, base_tag, dry_run=False)

        raw_cache[t.path] = RawCacheEntry(
            mtime=mtime, info=t, dsp=result["dsp"], section_dsp=result["section_dsp"],
        )
        stats.n_extracted += 1

    # Run extraction
    n_todo = len(to_extract)
    actual_workers = workers if workers > 0 else 1

    if n_todo == 0:
        pass
    elif actual_workers <= 1 or n_todo == 1:
        for j, (i, t, mtime, needs_analysis) in enumerate(to_extract):
            done = j + 1
            print(f"  [{done:>{len(str(n_todo))}}/{n_todo}] {Path(t.path).name}", end="", flush=True)
            try:
                result = _extract_worker(t.path, needs_analysis)
                _apply_result(t, mtime, needs_analysis, result)
                status = "[analyzed + extracted]" if needs_analysis else "[extracted]"
                print(f"  {status}")
            except Exception as e:
                raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                stats.n_failed += 1
                print(f"  FAILED: {e}")
            if done % 20 == 0:
                save_raw_cache(raw_cache, cache_path)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  Using {actual_workers} workers for {n_todo} tracks...")
        futures = {}
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            for i, t, mtime, needs_analysis in to_extract:
                fut = pool.submit(_extract_worker, t.path, needs_analysis)
                futures[fut] = (i, t, mtime, needs_analysis)

            done_count = 0
            for fut in as_completed(futures):
                i, t, mtime, needs_analysis = futures[fut]
                done_count += 1
                try:
                    result = fut.result()
                    _apply_result(t, mtime, needs_analysis, result)
                    status = "[analyzed + extracted]" if needs_analysis else "[extracted]"
                    print(f"  [{done_count:>{len(str(n_todo))}}/{n_todo}] {Path(t.path).name}  {status}")
                except Exception as e:
                    raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                    stats.n_failed += 1
                    print(f"  [{done_count:>{len(str(n_todo))}}/{n_todo}] {Path(t.path).name}  FAILED: {e}")
                if done_count % 20 == 0:
                    save_raw_cache(raw_cache, cache_path)

    # Remove deleted files from cache
    current_paths = {t.path for t in tracks}
    removed = [p for p in raw_cache if p not in current_paths]
    for p in removed:
        del raw_cache[p]
    stats.n_removed = len(removed)

    save_raw_cache(raw_cache, cache_path)
    return raw_cache, stats


# ─── Soft vocal partitioning helper ──────────────────────────────────────────

def _cluster_with_soft_vocal(feature_tracks, distance_matrix, config):
    """Cluster with soft vocal partitioning and BPM validation.

    Only hard-split vocal tracks when confidence is high.
    Uncertain tracks are clustered with everyone.
    Passes BPM values for post-clustering validation.
    """
    import numpy as np
    from .grouping.clustering import cluster_tracks

    n = len(feature_tracks)
    all_bpms = [t.info.bpm for t in feature_tracks]
    all_energies = [t.info.energy for t in feature_tracks]
    all_keys = [t.info.key for t in feature_tracks]
    conf_threshold = config.vocal_confidence_threshold

    confident_v = [i for i in range(n) if feature_tracks[i].info.vocal == "V"
                   and feature_tracks[i].info.confidences.get("vocal", 1.0) >= conf_threshold]
    confident_nv = [i for i in range(n) if feature_tracks[i].info.vocal != "V"
                    or feature_tracks[i].info.confidences.get("vocal", 1.0) < conf_threshold]

    if len(confident_v) < 2:
        return cluster_tracks(distance_matrix, config, bpms=all_bpms, energies=all_energies, keys=all_keys)

    labels = np.zeros(n, dtype=np.intp)

    if confident_nv:
        sub = distance_matrix[np.ix_(confident_nv, confident_nv)]
        sub_bpms = [all_bpms[i] for i in confident_nv]
        sub_energies = [all_energies[i] for i in confident_nv]
        sub_keys = [all_keys[i] for i in confident_nv]
        sub_labels = cluster_tracks(sub, config, bpms=sub_bpms, energies=sub_energies, keys=sub_keys)
        for i, idx in enumerate(confident_nv):
            labels[idx] = sub_labels[i]

    label_offset = int(np.max(labels)) + 1 if confident_nv else 0
    sub_v = distance_matrix[np.ix_(confident_v, confident_v)]
    sub_v_bpms = [all_bpms[i] for i in confident_v]
    sub_v_energies = [all_energies[i] for i in confident_v]
    sub_v_keys = [all_keys[i] for i in confident_v]
    sub_v_labels = cluster_tracks(sub_v, config, bpms=sub_v_bpms, energies=sub_v_energies, keys=sub_v_keys)
    for i, idx in enumerate(confident_v):
        labels[idx] = sub_v_labels[i] + label_offset

    print(f"  NV/uncertain: {len(confident_nv)} tracks, V (confident): {len(confident_v)} tracks")
    return labels


# ─── Delete-target validation ───────────────────────────────────────────────

def _safe_rmtree(target: str, allowed_roots: list[str]) -> bool:
    """Remove a directory tree only if it's under one of the allowed roots.

    Returns True if deleted, False if skipped.
    """
    import shutil
    from pathlib import Path

    resolved = Path(target).resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).resolve())
            if resolved.exists():
                shutil.rmtree(str(resolved))
            return True
        except ValueError:
            continue
    logger.warning("Refusing to delete %s — not under allowed roots %s", target, allowed_roots)
    return False


def _safe_unlink(target: str, allowed_roots: list[str]) -> bool:
    """Remove a file only if it's under one of the allowed roots."""
    from pathlib import Path

    resolved = Path(target).resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).resolve())
            if resolved.exists():
                resolved.unlink()
            return True
        except ValueError:
            continue
    logger.warning("Refusing to delete %s — not under allowed roots %s", target, allowed_roots)
    return False


# ─── run: all-in-one pipeline ───────────────────────────────────────────────

def _cmd_run(args) -> int:
    from pathlib import Path
    from .scanner import scan_library
    from .features.builder import build_features_from_raw
    from .features.embeddings import (
        is_clap_available, extract_clap_incremental, fit_pca,
    )
    from .grouping.distance import compute_distance_matrix
    from .grouping.assignment import (
        assign_group_ids, save_assignment, load_assignment, assign_new_tracks,
    )
    from .feedback.store import load_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .output.csv_export import export_groups_csv, export_recommendations_csv
    from .output.folders import create_group_folders
    from .output.playlists import generate_group_playlists, generate_recommendation_playlists
    from .recommend.neighbors import compute_recommendations
    from .config import GrouperConfig
    from dj_tagger.formats import format_tag
    from dj_tagger.metadata import read_existing_tag, write_tag

    config = GrouperConfig()

    # ── Resolve output directory ──
    # If input is outside the project directory, write outputs next to the input
    input_path = Path(args.paths[0]).resolve()
    project_dir = Path.cwd().resolve()

    try:
        input_path.relative_to(project_dir)
        is_external = False
    except ValueError:
        is_external = True

    if is_external:
        ext_out = str(input_path / "outputs")
        args.cache = str(input_path / "cache" / "features_cache.pkl")
        args.clap_cache = str(input_path / "cache" / "clap_cache.pkl")
        args.csv = str(input_path / "outputs" / "groups.csv")
        args.recommendations_csv = str(input_path / "outputs" / "recommendations.csv")
        args.output = str(input_path / "outputs" / "Grouped")
        args.playlists = str(input_path / "outputs" / "playlists")
        args.feedback = str(input_path / "outputs" / "feedback.csv")
        out_dir = Path(ext_out)
        print(f"External input detected — outputs will be in {ext_out}/")
    else:
        out_dir = Path(_DEFAULTS.output_dir)

    # Allowed roots for destructive operations
    _allowed = [str(out_dir), str(Path(args.cache).parent)]

    # ── Handle --clean ──
    if args.clean:
        _safe_rmtree(str(out_dir), _allowed)
        print(f"Cleaned {out_dir}/")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Handle --force-extract: clear DSP cache + outputs (NOT clap cache) ──
    if args.force_extract:
        for f in [args.cache, args.cache.replace(".pkl", "_assignment.pkl")]:
            _safe_unlink(f, _allowed)
        for d in [args.output, args.playlists]:
            _safe_rmtree(d, _allowed)
        for f in [args.csv, args.recommendations_csv]:
            _safe_unlink(f, _allowed)
        print("Cleared DSP cache and outputs (CLAP cache preserved)")

    # ── Handle --force-clap: clear CLAP cache only ──
    if args.force_clap:
        _safe_unlink(args.clap_cache, _allowed)
        print("Cleared CLAP cache")

    import time as _time

    # ── Step 1: Scan + analyze + extract ──
    print("\n[1/5] Scanning library...", end="", flush=True)
    t_step = _time.perf_counter()
    tracks = scan_library(args.paths, args.recursive)
    if not tracks:
        print(" no tracks found.")
        return 0

    total_tracks = len(tracks)
    n_tagged = sum(1 for t in tracks if t.energy is not None)
    print(f" {total_tracks} tracks found ({n_tagged} tagged, {total_tracks - n_tagged} untagged) ({_fmt_elapsed(_time.perf_counter() - t_step)})")

    print("  Analyzing and extracting features...")
    raw_cache, ext_stats = _run_extraction(
        tracks, args.cache,
        force=args.force_extract,
        workers=getattr(args, "workers", 1) or 1,
        analyze_untagged=True,
        write_tags=args.write_tags,
    )
    elapsed_step = _time.perf_counter() - t_step
    print(f"  Done: {', '.join(ext_stats.summary_parts())} ({_fmt_elapsed(elapsed_step)})")

    # ── Step 1b: CLAP audio embeddings ──
    track_order = [t.path for t in tracks]
    clap_embeddings = None

    if not args.no_clap and is_clap_available():
        print(f"\n  Extracting CLAP audio embeddings ({total_tracks} tracks)...")
        t_clap = _time.perf_counter()
        try:
            raw_clap = extract_clap_incremental(
                track_order, args.clap_cache, force=args.force_clap,
            )
            clap_embeddings, _ = fit_pca(raw_clap, config.clap_pca_dims)
            print(f"  CLAP done ({_fmt_elapsed(_time.perf_counter() - t_clap)})")
        except Exception as e:
            print(f"  CLAP failed: {e} -- continuing without CLAP")
    elif not args.no_clap:
        print("\n  CLAP not installed -- skipping audio embeddings")

    cache = build_features_from_raw(raw_cache, track_order, clap_embeddings, config)
    feature_tracks = cache.tracks

    # ── Step 2: Cluster or incrementally assign ──
    assignment_path = args.cache.replace(".pkl", "_assignment.pkl")
    existing_assignment = None

    if not args.rebuild and not args.force_extract and Path(assignment_path).exists():
        try:
            existing_assignment = load_assignment(assignment_path)
        except Exception:
            existing_assignment = None

    if existing_assignment is not None:
        # Incremental: assign new tracks to existing groups
        known = set(existing_assignment.track_to_group.keys())
        current = {t.path for t in feature_tracks}
        n_new = len(current - known)
        n_del = len(known - current)

        if n_new == 0 and n_del == 0:
            print(f"\n[2/5] Grouping -- all {len(feature_tracks)} tracks already assigned")
            assignment = existing_assignment
        else:
            print(f"\n[2/5] Grouping -- incremental: {n_new} new, {n_del} removed...")
            assignment = assign_new_tracks(feature_tracks, existing_assignment, config)
    else:
        # Full re-clustering
        n_ft = len(feature_tracks)
        print(f"\n[2/5] Grouping {n_ft} tracks by similarity...")
        _t = _time.perf_counter()
        distance_matrix = compute_distance_matrix(feature_tracks, config)
        print(f"  Computing {n_ft}x{n_ft} distance matrix... ({_fmt_elapsed(_time.perf_counter() - _t)})")

        feedback = load_feedback(args.feedback)
        if feedback:
            distance_matrix = apply_feedback_to_distances(distance_matrix, feature_tracks, feedback, config)

        _t = _time.perf_counter()
        labels = _cluster_with_soft_vocal(feature_tracks, distance_matrix, config)
        print(f"  Clustering + validation done ({_fmt_elapsed(_time.perf_counter() - _t)})")

        _t = _time.perf_counter()
        assignment = assign_group_ids(feature_tracks, labels, distance_matrix)

    save_assignment(assignment, assignment_path)

    # ── Step 3: Groups ──
    n_groups = len(assignment.groups)
    sizes = sorted([len(g.member_indices) for g in assignment.groups], reverse=True)
    print(f"\n[3/5] {n_groups} groups formed (sizes: {', '.join(str(s) for s in sizes)})")
    for group in assignment.groups:
        print(f"  {group.folder_name}: {len(group.member_indices)} tracks")

    # ── Step 4: Recommendations ──
    print(f"\n[4/5] Scoring track-to-track recommendations ({len(feature_tracks)} tracks)...")
    _t = _time.perf_counter()
    recommendations = compute_recommendations(feature_tracks, config)
    n_recs = len(recommendations)
    print(f"  {n_recs} recommendations computed ({_fmt_elapsed(_time.perf_counter() - _t)})")

    # ── Step 5: Output ──
    print("\n[5/5] Writing output files...")

    # Atomic CSV: write to temp, then rename
    _atomic_write_csv(export_groups_csv, feature_tracks, assignment, args.csv)
    print(f"  CSV: {args.csv}")

    _atomic_write_csv(export_recommendations_csv, recommendations, args.recommendations_csv)
    print(f"  CSV: {args.recommendations_csv}")

    if not args.dry_run:
        # Clean and recreate group folders (hard links are instant)
        _safe_rmtree(args.output, _allowed)
        create_group_folders(
            feature_tracks, assignment, args.output,
            dry_run=False, use_copy=args.copy,
        )
        print(f"  Folders: {args.output}/ ({n_groups} groups)")

        # Incremental tag writing: skip files that already have the correct tag
        if args.write_tags:
            n_written = 0
            n_skipped = 0
            for group in assignment.groups:
                for idx in group.member_indices:
                    tf = feature_tracks[idx]
                    info = tf.info
                    new_tag = format_tag(
                        energy=info.energy, camelot=info.key, bpm=info.bpm,
                        structure=info.structure, vibe=info.vibe,
                        has_vocals=info.vocal == "V",
                        group_id=group.group_id,
                    )
                    existing = read_existing_tag(tf.path)
                    if existing == new_tag:
                        n_skipped += 1
                    else:
                        write_tag(tf.path, new_tag, dry_run=False)
                        n_written += 1
            print(f"  Tags: {n_written} written, {n_skipped} unchanged")
    else:
        print("  (dry run -- no files modified)")

    if args.playlists:
        # Clean and recreate playlists
        _safe_rmtree(args.playlists, _allowed)
        generate_group_playlists(feature_tracks, assignment, args.playlists)
        generate_recommendation_playlists(recommendations, args.playlists)
        print(f"  Playlists: {args.playlists}/")

    print(f"\nDone in {_fmt_elapsed(_time.perf_counter() - t_step)}.")
    return 0


def _atomic_write_csv(write_fn, *write_args):
    """Write a CSV via temp file + os.replace for crash safety."""
    import os
    import tempfile
    from pathlib import Path

    # The last positional arg is the output path
    out_path = write_args[-1]
    tmp_dir = str(Path(out_path).parent)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".csv.tmp")
    os.close(tmp_fd)
    try:
        write_fn(*write_args[:-1], tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ─── Individual subcommands ─────────────────────────────────────────────────

def _cmd_extract(args) -> int:
    from pathlib import Path
    from .scanner import scan_library
    from .features.builder import build_features_from_raw
    from .config import GrouperConfig

    config = GrouperConfig()
    Path(_DEFAULTS.output_dir).mkdir(parents=True, exist_ok=True)

    tracks = scan_library(args.paths, args.recursive)
    if not tracks:
        print("No tracks found.")
        return 0

    print(f"Processing {len(tracks)} tracks...")
    raw_cache, stats = _run_extraction(
        tracks, args.cache,
        force=args.force,
        workers=getattr(args, "workers", 1) or 1,
        analyze_untagged=False,
    )
    print(f"  {', '.join(stats.summary_parts())}")

    if args.features_csv:
        track_order = [t.path for t in tracks]
        cache = build_features_from_raw(raw_cache, track_order, config=config)
        dsp_dicts = [raw_cache[p].dsp for p in track_order]
        from .output.csv_export import export_features_csv
        export_features_csv(cache.tracks, dsp_dicts, args.features_csv)
        print(f"Features exported to {args.features_csv}")

    return 0


def _cmd_cluster(args) -> int:
    from .features.builder import load_raw_cache, build_features_from_raw
    from .grouping.distance import compute_distance_matrix
    from .grouping.assignment import assign_group_ids, save_assignment
    from .feedback.store import load_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .output.csv_export import export_groups_csv
    from .config import GrouperConfig

    config = GrouperConfig()
    raw_cache = load_raw_cache(args.cache)
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=config)
    tracks = cache.tracks

    print(f"Clustering {len(tracks)} tracks...")
    distance_matrix = compute_distance_matrix(tracks, config)

    feedback = load_feedback(args.feedback)
    if feedback:
        distance_matrix = apply_feedback_to_distances(distance_matrix, tracks, feedback, config)

    labels = _cluster_with_soft_vocal(tracks, distance_matrix, config)

    assignment = assign_group_ids(tracks, labels, distance_matrix)
    save_assignment(assignment, args.cache.replace(".pkl", "_assignment.pkl"))

    export_groups_csv(tracks, assignment, args.output_csv)
    print(f"Groups exported to {args.output_csv}")

    for group in assignment.groups:
        print(f"  {group.folder_name}: {len(group.member_indices)} tracks")

    return 0


def _cmd_review(args) -> int:
    from .features.builder import load_raw_cache, build_features_from_raw
    from .grouping.distance import compute_distance_matrix
    from .grouping.assignment import assign_group_ids
    from .feedback.store import load_feedback, add_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .review import interactive_review
    from .config import GrouperConfig

    config = GrouperConfig()
    raw_cache = load_raw_cache(args.cache)
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=config)
    tracks = cache.tracks

    distance_matrix = compute_distance_matrix(tracks, config)
    feedback = load_feedback(args.feedback)
    if feedback:
        distance_matrix = apply_feedback_to_distances(distance_matrix, tracks, feedback, config)

    labels = _cluster_with_soft_vocal(tracks, distance_matrix, config)

    assignment = assign_group_ids(tracks, labels, distance_matrix)
    overrides = interactive_review(tracks, assignment)

    for override in overrides:
        if override["action"] == "move":
            add_feedback(args.feedback, override["track"], "", "group_override", override["group"])

    if overrides:
        print(f"  {len(overrides)} overrides saved to {args.feedback}")

    return 0


def _cmd_apply(args) -> int:
    from .features.builder import load_raw_cache, build_features_from_raw
    from .grouping.assignment import load_assignment
    from .output.folders import create_group_folders
    from .output.playlists import generate_group_playlists
    from dj_tagger.formats import format_tag
    from dj_tagger.metadata import write_tag
    from .config import GrouperConfig

    raw_cache = load_raw_cache(args.cache)
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=GrouperConfig())
    tracks = cache.tracks
    assignment = load_assignment(args.cache.replace(".pkl", "_assignment.pkl"))

    print(f"Creating group folders in {args.output}...")
    create_group_folders(
        tracks, assignment, args.output,
        dry_run=args.dry_run, use_copy=args.copy,
    )

    if args.write_tags and not args.dry_run:
        print("Writing tags with group IDs to file metadata...")
        for group in assignment.groups:
            for idx in group.member_indices:
                tf = tracks[idx]
                info = tf.info
                tag = format_tag(
                    energy=info.energy, camelot=info.key, bpm=info.bpm,
                    structure=info.structure, vibe=info.vibe,
                    has_vocals=info.vocal == "V",
                    group_id=group.group_id,
                )
                write_tag(tf.path, tag, dry_run=False)
        print(f"  Tags written to {len(tracks)} files")

    if args.playlists:
        generate_group_playlists(tracks, assignment, args.playlists)
        print(f"  Playlists generated in {args.playlists}")

    return 0


def _cmd_recommend(args) -> int:
    from .features.builder import load_raw_cache, build_features_from_raw
    from .recommend.neighbors import compute_recommendations
    from .output.csv_export import export_recommendations_csv
    from .output.playlists import generate_recommendation_playlists
    from .config import GrouperConfig

    config = GrouperConfig()
    raw_cache = load_raw_cache(args.cache)
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=config)
    tracks = cache.tracks

    print(f"Computing recommendations for {len(tracks)} tracks...")
    recommendations = compute_recommendations(tracks, config)

    export_recommendations_csv(recommendations, args.csv)
    print(f"Recommendations exported to {args.csv}")

    if args.playlists:
        generate_recommendation_playlists(recommendations, args.playlists)
        print(f"Recommendation playlists generated in {args.playlists}")

    return 0


def _cmd_feedback(args) -> int:
    from pathlib import Path
    from .feedback.store import add_feedback

    Path(args.file).parent.mkdir(parents=True, exist_ok=True)

    if args.good:
        add_feedback(args.file, args.good[0], args.good[1], "good_pair")
        print(f"Added good_pair: {args.good[0]} <-> {args.good[1]}")
    elif args.bad:
        add_feedback(args.file, args.bad[0], args.bad[1], "bad_pair")
        print(f"Added bad_pair: {args.bad[0]} <-> {args.bad[1]}")
    elif args.override:
        add_feedback(args.file, args.override[0], "", "group_override", args.override[1])
        print(f"Added override: {args.override[0]} -> {args.override[1]}")
    else:
        print("Specify --good, --bad, or --override")
        return 1

    return 0


def _cmd_evaluate(args) -> int:
    from .features.builder import load_raw_cache, build_features_from_raw
    from .evaluation import load_eval_pairs, evaluate
    from .config import GrouperConfig

    config = GrouperConfig()
    raw_cache = load_raw_cache(args.cache)
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=config)

    pairs = load_eval_pairs(args.eval_file)
    if not pairs:
        print(f"No evaluation pairs found in {args.eval_file}")
        return 1

    print(f"Evaluating {len(pairs)} pairs...")
    metrics = evaluate(cache.tracks, pairs, config)

    print(f"\n  Good pairs:        {metrics['n_good']}")
    print(f"  Bad pairs:         {metrics['n_bad']}")
    print(f"  Good avg score:    {metrics['good_avg_score']:.4f}")
    print(f"  Bad avg score:     {metrics['bad_avg_score']:.4f}")
    print(f"  Good avg rank:     {metrics['good_avg_rank']:.1f}")
    if metrics['good_above_bad'] is not None:
        print(f"  Good above bad:    {metrics['good_above_bad']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
