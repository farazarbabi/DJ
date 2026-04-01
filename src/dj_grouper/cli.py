"""CLI for dj-grouper: run, extract, cluster, review, apply, recommend, feedback."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import GrouperConfig

logger = logging.getLogger(__name__)

_DEFAULTS = GrouperConfig()


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

    sub = parser.add_subparsers(dest="command", required=True)

    # --- run (all-in-one) ---
    p_run = sub.add_parser("run", help="Run full pipeline: extract -> cluster -> recommend -> output")
    p_run.add_argument("--input", nargs="+", default=[_DEFAULTS.input_dir], help="Library paths (default: ./files)")
    p_run.add_argument("-r", "--recursive", action="store_true")
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
    p_run.add_argument("--clean", action="store_true", help="Delete all outputs and caches, then exit")

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Extract features for all tracks")
    p_extract.add_argument("--input", nargs="+", default=[_DEFAULTS.input_dir], help="Library paths (default: ./files)")
    p_extract.add_argument("-r", "--recursive", action="store_true")
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    conf_threshold = config.vocal_confidence_threshold

    confident_v = [i for i in range(n) if feature_tracks[i].info.vocal == "V"
                   and feature_tracks[i].info.confidences.get("vocal", 1.0) >= conf_threshold]
    confident_nv = [i for i in range(n) if feature_tracks[i].info.vocal != "V"
                    or feature_tracks[i].info.confidences.get("vocal", 1.0) < conf_threshold]

    if len(confident_v) < 2:
        return cluster_tracks(distance_matrix, config, bpms=all_bpms, energies=all_energies)

    labels = np.zeros(n, dtype=np.intp)

    if confident_nv:
        sub = distance_matrix[np.ix_(confident_nv, confident_nv)]
        sub_bpms = [all_bpms[i] for i in confident_nv]
        sub_energies = [all_energies[i] for i in confident_nv]
        sub_labels = cluster_tracks(sub, config, bpms=sub_bpms, energies=sub_energies)
        for i, idx in enumerate(confident_nv):
            labels[idx] = sub_labels[i]

    label_offset = int(np.max(labels)) + 1 if confident_nv else 0
    sub_v = distance_matrix[np.ix_(confident_v, confident_v)]
    sub_v_bpms = [all_bpms[i] for i in confident_v]
    sub_v_energies = [all_energies[i] for i in confident_v]
    sub_v_labels = cluster_tracks(sub_v, config, bpms=sub_v_bpms, energies=sub_v_energies)
    for i, idx in enumerate(confident_v):
        labels[idx] = sub_v_labels[i] + label_offset

    print(f"  NV/uncertain: {len(confident_nv)} tracks, V (confident): {len(confident_v)} tracks")
    return labels


# ─── run: all-in-one pipeline ───────────────────────────────────────────────

def _cmd_run(args) -> int:
    import numpy as np
    import shutil
    import tempfile
    from pathlib import Path
    from .scanner import scan_library
    from .features.builder import (
        RawCacheEntry, load_raw_cache, save_raw_cache,
        build_features_from_raw,
    )
    from .features.dsp import extract_dsp_features, extract_section_dsp
    from .features.embeddings import (
        is_clap_available, extract_clap_incremental, fit_pca,
    )
    from .grouping.distance import compute_distance_matrix
    from .grouping.clustering import cluster_tracks
    from .grouping.assignment import assign_group_ids, save_assignment
    from dj_tagger.analyzers.sections import analyze_sections
    from dj_tagger.analyzers.vibe import analyze_vibe
    from dj_tagger.analyzers.vocal import analyze_vocal
    from .feedback.store import load_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .output.csv_export import export_groups_csv, export_recommendations_csv
    from .output.folders import create_group_folders
    from .output.playlists import generate_group_playlists, generate_recommendation_playlists
    from .recommend.neighbors import compute_recommendations
    from .config import GrouperConfig
    from dj_tagger.audio import load_audio_features
    from dj_tagger.formats import format_tag
    from dj_tagger.metadata import read_existing_tag, write_tag
    from dj_tagger.formats import parse_tag
    import os

    config = GrouperConfig()
    out_dir = Path(_DEFAULTS.output_dir)

    # ── Handle --clean ──
    if args.clean:
        if out_dir.exists():
            shutil.rmtree(str(out_dir))
            print(f"Cleaned {out_dir}/")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Handle --force-extract: clear DSP cache + outputs (NOT clap cache) ──
    if args.force_extract:
        for f in [args.cache, args.cache.replace(".pkl", "_assignment.pkl")]:
            if Path(f).exists():
                Path(f).unlink()
        for d in [args.output, args.playlists]:
            if Path(d).exists():
                shutil.rmtree(d)
        for f in [args.csv, args.recommendations_csv]:
            if Path(f).exists():
                Path(f).unlink()
        print("Cleared DSP cache and outputs (CLAP cache preserved)")

    # ── Handle --force-clap: clear CLAP cache only ──
    if args.force_clap:
        clap_path = Path(args.clap_cache)
        if clap_path.exists():
            clap_path.unlink()
        print("Cleared CLAP cache")

    # ── Step 1: Incremental DSP extraction ──
    tracks = scan_library(args.input, args.recursive)
    if not tracks:
        print("No tracks found.")
        return 0

    raw_cache = load_raw_cache(args.cache)

    n_cached = 0
    n_extracted = 0
    n_failed = 0
    print(f"\n[1/5] Processing {len(tracks)} tracks...")

    for i, t in enumerate(tracks):
        mtime = os.path.getmtime(t.path)
        cached = raw_cache.get(t.path)
        if cached and cached.mtime == mtime:
            cached.info = t
            n_cached += 1
            print(f"  [{i+1}/{len(tracks)}] {Path(t.path).name} [cached]")
        else:
            print(f"  [{i+1}/{len(tracks)}] {Path(t.path).name}", end="", flush=True)
            try:
                audio = load_audio_features(t.path)
                feats = extract_dsp_features(audio)
                if t.bpm is None and audio.tempo > 0:
                    t.bpm = round(audio.tempo)
                section_map = analyze_sections(audio)
                sec_dsp = extract_section_dsp(audio, section_map)
                vibe_result = analyze_vibe(audio)
                t.vibe_scores = vibe_result.scores
                t.confidences["vibe"] = vibe_result.confidence
                vocal_result = analyze_vocal(audio)
                t.confidences["vocal"] = vocal_result.confidence
                raw_cache[t.path] = RawCacheEntry(
                    mtime=mtime, info=t, dsp=feats, section_dsp=sec_dsp,
                )
                n_extracted += 1
                print(" OK")
            except Exception as e:
                raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                n_failed += 1
                print(f" FAILED: {e}")

        # Save cache periodically (every 20 tracks) for crash resilience
        if (i + 1) % 20 == 0:
            save_raw_cache(raw_cache, args.cache)

    # Remove deleted files
    current_paths = {t.path for t in tracks}
    removed = [p for p in raw_cache if p not in current_paths]
    for p in removed:
        del raw_cache[p]

    save_raw_cache(raw_cache, args.cache)
    print(f"  {n_cached} cached, {n_extracted} extracted, {n_failed} failed, {len(removed)} removed")

    # ── Step 1b: Incremental CLAP extraction (separate cache) ──
    track_order = [t.path for t in tracks]
    clap_embeddings = None

    if not args.no_clap and is_clap_available():
        print("  CLAP embeddings (incremental)...")
        try:
            raw_clap = extract_clap_incremental(
                track_order, args.clap_cache, force=args.force_clap,
            )
            clap_embeddings, _ = fit_pca(raw_clap, config.clap_pca_dims)
        except Exception as e:
            print(f"  CLAP failed: {e}")

    cache = build_features_from_raw(raw_cache, track_order, clap_embeddings, config)
    feature_tracks = cache.tracks

    # ── Step 2: Cluster ──
    print(f"\n[2/5] Clustering {len(feature_tracks)} tracks...")
    distance_matrix = compute_distance_matrix(feature_tracks, config)

    feedback = load_feedback(args.feedback)
    if feedback:
        distance_matrix = apply_feedback_to_distances(distance_matrix, feature_tracks, feedback, config)

    labels = _cluster_with_soft_vocal(feature_tracks, distance_matrix, config)

    assignment = assign_group_ids(feature_tracks, labels, distance_matrix)
    save_assignment(assignment, args.cache.replace(".pkl", "_assignment.pkl"))

    # ── Step 3: Groups ──
    print(f"\n[3/5] Groups:")
    for group in assignment.groups:
        print(f"  {group.folder_name}: {len(group.member_indices)} tracks")

    # ── Step 4: Recommendations ──
    print(f"\n[4/5] Computing recommendations...")
    recommendations = compute_recommendations(feature_tracks, config)

    # ── Step 5: Output (atomic CSVs, clean folder recreation, incremental tags) ──
    print(f"\n[5/5] Writing output...")

    # Atomic CSV: write to temp, then rename
    _atomic_write_csv(export_groups_csv, feature_tracks, assignment, args.csv)
    print(f"  {args.csv}")

    _atomic_write_csv(export_recommendations_csv, recommendations, args.recommendations_csv)
    print(f"  {args.recommendations_csv}")

    if not args.dry_run:
        # Clean and recreate group folders (hard links are instant)
        if Path(args.output).exists():
            shutil.rmtree(args.output)
        create_group_folders(
            feature_tracks, assignment, args.output,
            dry_run=False, use_copy=args.copy,
        )
        print(f"  {args.output}/")

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
                        has_vocals=info.vocal == "V", group_id=group.group_id,
                    )
                    existing = read_existing_tag(tf.path)
                    if existing == new_tag:
                        n_skipped += 1
                    else:
                        write_tag(tf.path, new_tag, dry_run=False)
                        n_written += 1
            print(f"  Tags: {n_written} written, {n_skipped} unchanged")
    else:
        print("  (dry run — no files modified)")

    if args.playlists:
        # Clean and recreate playlists
        if Path(args.playlists).exists():
            shutil.rmtree(args.playlists)
        generate_group_playlists(feature_tracks, assignment, args.playlists)
        generate_recommendation_playlists(recommendations, args.playlists)
        print(f"  Playlists in {args.playlists}/")

    print("\nDone.")
    return 0


def _atomic_write_csv(write_fn, *write_args):
    """Write a CSV via temp file + rename for crash safety."""
    import tempfile
    from pathlib import Path

    # The last positional arg is the output path
    out_path = write_args[-1]
    tmp_dir = str(Path(out_path).parent)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".csv.tmp")
    import os
    os.close(tmp_fd)
    try:
        write_fn(*write_args[:-1], tmp_path)
        # Atomic rename (same filesystem)
        if Path(out_path).exists():
            Path(out_path).unlink()
        Path(tmp_path).rename(out_path)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise


# ─── Individual subcommands ─────────────────────────────────────────────────

def _cmd_extract(args) -> int:
    from pathlib import Path
    from .scanner import scan_library
    from .features.dsp import extract_dsp_features, extract_section_dsp
    from .features.builder import (
        RawCacheEntry, load_raw_cache, save_raw_cache,
        build_features_from_raw,
    )
    from .features.embeddings import is_clap_available, extract_clap_embeddings, fit_pca
    from .config import GrouperConfig
    from dj_tagger.audio import load_audio_features
    from dj_tagger.analyzers.sections import analyze_sections
    from dj_tagger.analyzers.vibe import analyze_vibe
    from dj_tagger.analyzers.vocal import analyze_vocal
    import os

    config = GrouperConfig()
    Path(_DEFAULTS.output_dir).mkdir(parents=True, exist_ok=True)

    tracks = scan_library(args.input, args.recursive)
    if not tracks:
        print("No tracks found.")
        return 0

    raw_cache = {} if args.force else load_raw_cache(args.cache)

    n_cached = 0
    n_extracted = 0
    print(f"Processing {len(tracks)} tracks...")

    for i, t in enumerate(tracks):
        mtime = os.path.getmtime(t.path)
        cached = raw_cache.get(t.path)
        if cached and cached.mtime == mtime:
            cached.info = t
            n_cached += 1
            print(f"  [{i+1}/{len(tracks)}] {Path(t.path).name} [cached]")
        else:
            print(f"  [{i+1}/{len(tracks)}] {Path(t.path).name}", end="", flush=True)
            try:
                audio = load_audio_features(t.path)
                feats = extract_dsp_features(audio)
                if t.bpm is None and audio.tempo > 0:
                    t.bpm = round(audio.tempo)
                section_map = analyze_sections(audio)
                sec_dsp = extract_section_dsp(audio, section_map)
                vibe_result = analyze_vibe(audio)
                t.vibe_scores = vibe_result.scores
                t.confidences["vibe"] = vibe_result.confidence
                vocal_result = analyze_vocal(audio)
                t.confidences["vocal"] = vocal_result.confidence
                raw_cache[t.path] = RawCacheEntry(
                    mtime=mtime, info=t, dsp=feats, section_dsp=sec_dsp,
                )
                n_extracted += 1
                print(" OK")
            except Exception as e:
                raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                print(f" FAILED: {e}")

    # Remove deleted files
    current_paths = {t.path for t in tracks}
    removed = [p for p in raw_cache if p not in current_paths]
    for p in removed:
        del raw_cache[p]

    save_raw_cache(raw_cache, args.cache)
    print(f"  {n_cached} cached, {n_extracted} extracted, {len(removed)} removed")

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
    from .grouping.clustering import cluster_tracks
    from .grouping.assignment import assign_group_ids, save_assignment
    from .feedback.store import load_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .output.csv_export import export_groups_csv
    from .config import GrouperConfig
    import numpy as np

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
    from .grouping.clustering import cluster_tracks
    from .grouping.assignment import assign_group_ids
    from .feedback.store import load_feedback, add_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .review import interactive_review
    from .config import GrouperConfig
    import numpy as np

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
        print("Writing group IDs to file metadata...")
        for group in assignment.groups:
            for idx in group.member_indices:
                tf = tracks[idx]
                info = tf.info
                tag = format_tag(
                    energy=info.energy, camelot=info.key, bpm=info.bpm,
                    structure=info.structure, vibe=info.vibe,
                    has_vocals=info.vocal == "V", group_id=group.group_id,
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
