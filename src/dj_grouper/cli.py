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
    try:
        from dj_registry.progress import install_tqdm_log_handler
        install_tqdm_log_handler()
    except ImportError:
        pass  # Older dj_registry — keep going with default StreamHandler.


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
    p_run.add_argument("paths", nargs="*", default=[_DEFAULTS.input_dir], metavar="PATH", help=f"Library paths (default: {_DEFAULTS.input_dir})")
    p_run.add_argument("-r", "--recursive", action="store_true")
    p_run.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers for extraction (default: 1)")
    p_run.add_argument("--write-tags", action="store_true", help="Write tags to file metadata")
    p_run.add_argument("--dry-run", action="store_true", help="Preview without writing anything")
    p_run.add_argument("--no-clap", action="store_true", help="Disable CLAP embeddings")
    p_run.add_argument("--algorithm", choices=["constrained", "agglomerative"], default=_DEFAULTS.clustering_method, help="Clustering algorithm (default: constrained)")
    p_run.add_argument("--no-registry", action="store_true", help="Skip loading registry enrichment")
    p_run.add_argument("--registry-dir", default=None, help="Registry output dir (default: <library>/outputs/registry)")
    p_run.add_argument("--playlists", default=_DEFAULTS.playlists_dir, help="Playlists dir")
    p_run.add_argument("--csv", default=_DEFAULTS.groups_file)
    p_run.add_argument("--recommendations-csv", default=_DEFAULTS.recommendations_file)
    p_run.add_argument("--cache-dir", default=_DEFAULTS.cache_dir, help="Cache directory")
    p_run.add_argument("--feedback", default=_DEFAULTS.feedback_file)
    p_run.add_argument("--force-extract", action="store_true", help="Clear cache + outputs, re-extract")
    p_run.add_argument("--cache-only", dest="cache_only", action="store_true",
                       help="Group only tracks that are already analyzed (in the cache); "
                            "skip any un-analyzed track instead of decoding/extracting it "
                            "(never loads audio, incl. CLAP)")
    p_run.add_argument("--force-clap", action="store_true", help="Re-extract CLAP embeddings")
    p_run.add_argument("--rebuild", action="store_true", help="Force full re-clustering (ignore existing groups)")
    p_run.add_argument("--fine-playlists", action="store_true", help="Also write full-resolution group playlists to groups/ (groups_coarse/ is written by default)")
    p_run.add_argument("--clean", action="store_true", help="Delete all outputs and caches, then exit")

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Extract features for all tracks")
    p_extract.add_argument("paths", nargs="*", default=[_DEFAULTS.input_dir], metavar="PATH", help=f"Library paths (default: {_DEFAULTS.input_dir})")
    p_extract.add_argument("-r", "--recursive", action="store_true")
    p_extract.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers (default: 1)")
    p_extract.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_extract.add_argument("--features-csv", default=None)
    p_extract.add_argument("--force", action="store_true", help="Regenerate features even if cache exists")

    # --- cluster ---
    p_cluster = sub.add_parser("cluster", help="Cluster tracks into groups")
    p_cluster.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_cluster.add_argument("--feedback", default=_DEFAULTS.feedback_file)
    p_cluster.add_argument("--output-csv", default=_DEFAULTS.groups_file)

    # --- review ---
    p_review = sub.add_parser("review", help="Review proposed groupings")
    p_review.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_review.add_argument("--feedback", default=_DEFAULTS.feedback_file)

    # --- apply ---
    p_apply = sub.add_parser("apply", help="Write tags and generate playlists")
    p_apply.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_apply.add_argument("--write-tags", action="store_true")
    p_apply.add_argument("--dry-run", action="store_true")
    p_apply.add_argument("--playlists", default=_DEFAULTS.playlists_dir)

    # --- recommend ---
    p_rec = sub.add_parser("recommend", help="Compute recommendations (CSV only)")
    p_rec.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_rec.add_argument("--csv", default=_DEFAULTS.recommendations_file)

    # --- feedback ---
    p_fb = sub.add_parser("feedback", help="Add feedback entries")
    p_fb.add_argument("--good", nargs=2, metavar=("TRACK_A", "TRACK_B"))
    p_fb.add_argument("--bad", nargs=2, metavar=("TRACK_A", "TRACK_B"))
    p_fb.add_argument("--override", nargs=2, metavar=("TRACK", "GID"))
    p_fb.add_argument("--file", default=_DEFAULTS.feedback_file)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate recommendations against known pairs")
    p_eval.add_argument("--cache-dir", default=_DEFAULTS.cache_dir)
    p_eval.add_argument("--eval-file", required=True, help="CSV with track_a, track_b, label columns")

    return parser


_SUBCOMMANDS = {"run", "extract", "cluster", "review", "apply", "recommend", "feedback", "evaluate"}


def _cache_paths(cache_dir: str) -> dict[str, str]:
    """Derive all cache paths from the cache directory."""
    from pathlib import Path
    d = Path(cache_dir)
    return {
        "raw": str(d / "raw_cache.pkl"),
        "features": str(d / "features_cache.pkl"),  # legacy, for migration
        "clap": str(d / "clap_cache.pkl"),  # legacy, for migration
        "assignment": str(d / "features_cache_assignment.pkl"),
    }


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


def _extract_worker(
    track_path: str,
    needs_analysis: bool,
    cached_dsp: dict[str, float] | None = None,
) -> dict:
    """Run the canonical tagger artifact pipeline for one track."""
    from dj_tagger.audio import load_audio_features
    from dj_tagger.raw_features import compute_tagger_artifacts

    del needs_analysis  # kept for ProcessPoolExecutor/API compatibility.
    audio = load_audio_features(track_path)
    return compute_tagger_artifacts(audio, dsp=cached_dsp, use_essentia=False)


def _extract_dsp_worker(track_path: str) -> dict:
    """Extract only the raw DSP layers needed by grouper."""
    from dj_tagger.audio import load_audio_features
    from dj_tagger.analyzers.sections import analyze_sections
    from .features.dsp import extract_dsp_features, extract_section_dsp

    audio = load_audio_features(track_path)
    section_map = analyze_sections(audio)
    return {
        "dsp": extract_dsp_features(audio),
        "section_dsp": extract_section_dsp(audio, section_map),
    }


def _cached_raw_layer(ucache, filename: str, duration: float | None, layer: str):
    """Return an identity-keyed raw layer, ignoring version metadata."""
    data = ucache.get_track(filename, duration, layer)
    if data is not None:
        return data

    data = ucache.get_track_any_version(filename, duration, layer)
    if not _raw_layer_schema_matches(layer, data):
        return None

    ucache.put_track(filename, duration, layer, data)
    logger.debug("Reused cached %s layer for %s", layer, filename)
    return data


def _raw_layer_schema_matches(layer: str, data) -> bool:
    if not isinstance(data, dict):
        return False
    if layer == "dsp":
        from .features.dsp import DSP_CURATED_NAMES

        return all(name in data for name in DSP_CURATED_NAMES)
    if layer == "section_dsp":
        return True
    return False


# ─── Shared extraction service ──────────────────────────────────────────────


class _ExtractionStats:
    __slots__ = ("n_cached", "n_extracted", "n_analyzed", "n_failed", "n_removed",
                 "n_skipped_uncached")

    def __init__(self) -> None:
        self.n_cached = 0
        self.n_extracted = 0
        self.n_analyzed = 0
        self.n_failed = 0
        self.n_removed = 0
        self.n_skipped_uncached = 0

    def summary_parts(self) -> list[str]:
        parts = []
        if self.n_cached:
            parts.append(f"{self.n_cached} cached")
        if self.n_analyzed:
            parts.append(f"{self.n_analyzed} analyzed")
        if self.n_extracted:
            parts.append(f"{self.n_extracted} extracted")
        if self.n_skipped_uncached:
            parts.append(f"{self.n_skipped_uncached} skipped (uncached)")
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
    cache_only: bool = False,
) -> tuple[dict, _ExtractionStats]:
    """Shared extraction logic for run and extract commands.

    Uses raw + derived cache so that tracks analyzed by dj-tagger or
    dj-registry are never re-analyzed. Only extracts DSP features when
    the tagger analysis is already cached.

    When ``cache_only`` is set, tracks without complete cached analysis are
    **skipped** (counted in ``stats.n_skipped_uncached``) instead of extracted —
    so grouping runs over just the already-analyzed subset and never decodes
    audio. Mutually exclusive in spirit with ``force`` (which re-extracts all).

    Returns (raw_cache, stats).
    """
    import os
    from pathlib import Path
    from .features.builder import RawCacheEntry, load_raw_cache, save_raw_cache
    from .scanner import is_unknown_key
    from dj_tagger.universal_cache import get_cache as get_universal_cache, quick_duration
    from dj_tagger.tagger_cache import (
        hydrate_tagger_result,
        merge_rederived_tagger,
        tagger_core_metadata_matches,
        tagger_metadata_matches,
    )

    stats = _ExtractionStats()
    raw_cache = {} if force else load_raw_cache(cache_path)

    # Get cache for cross-module lookup
    ucache_path = str(Path(cache_path).parent / "raw_cache.pkl")
    ucache = get_universal_cache(ucache_path)

    # Pre-compute durations for cache keys (cheap mutagen read, no audio decoding)
    _durations: dict[str, float | None] = {}
    for t in tracks:
        _durations[t.path] = quick_duration(t.path)

    # Separate into: fully cached, needs DSP only, needs everything
    to_extract: list[tuple[int, object, float, bool, dict[str, float] | None]] = []
    to_dsp_only: list[tuple[int, object, float]] = []

    for i, t in enumerate(tracks):
        mtime = os.path.getmtime(t.path)
        cached = raw_cache.get(t.path)
        if cached and cached.mtime == mtime:
            cached.info = t
            stats.n_cached += 1
            continue

        filename = Path(t.path).name
        dur = _durations.get(t.path)

        # When force=True, skip cache and re-extract everything (unless
        # cache_only, which never decodes audio — an un-cached track is skipped).
        if force:
            if cache_only:
                stats.n_skipped_uncached += 1
                continue
            needs_analysis = analyze_untagged and (t.energy is None or is_unknown_key(t.key))
            to_extract.append((i, t, mtime, needs_analysis, None))
            continue

        # Check raw layers in universal cache. Raw/data collection is
        # identity-keyed; downstream signatures do not force recollection.
        dsp_data = _cached_raw_layer(ucache, filename, dur, "dsp")
        section_dsp_data = _cached_raw_layer(ucache, filename, dur, "section_dsp")
        raw_analysis = ucache.get_track(filename, dur, "raw_analysis")
        # Merge cached vocal_stem scalars so re-derivation picks them up
        # without re-running Demucs.
        stem_data = ucache.get_track(filename, dur, "vocal_stem")
        if isinstance(raw_analysis, dict) and isinstance(stem_data, dict):
            raw_analysis = {
                **raw_analysis,
                **{k: v for k, v in stem_data.items() if k.startswith("vocal_stem_")},
            }
        tagger_key = ucache.track_key(filename, dur, "tagger")
        any_tagger_entry = ucache._entries.get(tagger_key)
        collected_tagger = any_tagger_entry.data if any_tagger_entry and isinstance(any_tagger_entry.data, dict) else None

        # Check derived layer (versioned; returns None if stale)
        tagger_data = ucache.get_track(filename, dur, "tagger")
        usable_tagger = None
        if isinstance(tagger_data, dict):
            if tagger_metadata_matches(tagger_data, None):
                usable_tagger = tagger_data
            elif tagger_data.get("_tagger_audio_features_sig") and tagger_core_metadata_matches(tagger_data):
                # Grouper has no Songstats context. Preserve and reuse a richer current entry
                # from registry/tagger instead of downgrading it to DSP-only derivation.
                usable_tagger = tagger_data

        # Auto-recompute: if raw is cached but derived is stale, re-derive.
        # Raw collection layers are identity-keyed; downstream signature changes
        # must not force audio extraction when these payloads exist.
        if dsp_data and isinstance(dsp_data, dict) and raw_analysis and isinstance(raw_analysis, dict):
            if usable_tagger is None:
                # Raw available, derived stale: re-derive without loading audio.
                # Grouper has no Songstats features, so this path is DSP-only on purpose.
                from dj_tagger.derive import derive_all
                tagger_data = merge_rederived_tagger(collected_tagger, derive_all(dsp_data, raw_analysis))
                ucache.put_track(filename, dur, "tagger", tagger_data, mtime=mtime)
                logger.debug("Auto-recomputed derived analysis for %s from cached raw", filename)
                usable_tagger = tagger_data

        if usable_tagger is None and isinstance(collected_tagger, dict):
            # Fall back to collected tagger facts even if derived metadata is
            # stale; do not re-extract audio solely because signatures changed.
            usable_tagger = collected_tagger

        if usable_tagger is not None:
            _apply_tagger_result(t, usable_tagger, analyze_untagged)

        # Check if critical fields are still set
        missing_analysis = t.vibe is None or t.vocal is None

        if dsp_data and isinstance(dsp_data, dict) and not missing_analysis:
            # Fully cached — both raw DSP and derived analysis are complete
            raw_cache[t.path] = RawCacheEntry(
                mtime=mtime, info=t, dsp=dsp_data,
                section_dsp=section_dsp_data if isinstance(section_dsp_data, dict) else {},
            )
            stats.n_cached += 1
            continue

        # Need extraction — at least DSP, possibly full analysis. Under
        # cache_only we never decode audio, so an incompletely-cached track is
        # dropped from this run instead (grouping the already-analyzed subset).
        if cache_only:
            stats.n_skipped_uncached += 1
            continue
        needs_analysis = analyze_untagged and (t.energy is None or is_unknown_key(t.key))
        if usable_tagger is not None:
            to_dsp_only.append((i, t, mtime))
        elif dsp_data and isinstance(dsp_data, dict):
            to_extract.append((i, t, mtime, needs_analysis, dsp_data))
        else:
            to_extract.append((i, t, mtime, needs_analysis, None))

    if stats.n_cached:
        logger.info("  %d tracks loaded from cache", stats.n_cached)
    if stats.n_skipped_uncached:
        logger.info("  %d tracks skipped (not yet analyzed; --cache-only)",
                    stats.n_skipped_uncached)
    if to_dsp_only:
        logger.info("  %d tracks need DSP extraction only (analysis cached)", len(to_dsp_only))

    # Tag writing imports (only when needed)
    _format_tag = None
    _write_tag = None
    if write_tags:
        from dj_tagger.formats import format_tag as _format_tag
        from dj_tagger.metadata import write_tag as _write_tag

    def _apply_result(t, mtime, needs_analysis, result):
        """Apply worker result to track object and cache."""
        tagger_result = hydrate_tagger_result(result["tagger_result"])

        if needs_analysis:
            stats.n_analyzed += 1
        _apply_tagger_result(t, tagger_result, needs_analysis)

        if needs_analysis and _format_tag and _write_tag:
            base_tag = _format_tag(
                energy=t.energy, camelot=t.key, bpm=t.bpm,
                vibe=t.vibe,
                has_vocals=tagger_result.get("has_vocals"),
                vocal_profile=tagger_result.get("vocal_profile", tagger_result.get("vocal")),
            )
            _write_tag(t.path, base_tag, dry_run=False)

        raw_cache[t.path] = RawCacheEntry(
            mtime=mtime, info=t, dsp=result["dsp"], section_dsp=result["section_dsp"],
        )
        # Store in cache: raw layers (identity-keyed) + derived layer (versioned)
        filename = Path(t.path).name
        dur = _durations.get(t.path)
        ucache.put_track(filename, dur, "dsp", result["dsp"], mtime=mtime)
        ucache.put_track(filename, dur, "section_dsp", result["section_dsp"], mtime=mtime)
        if "raw_analysis" in result:
            ucache.put_track(filename, dur, "raw_analysis", result["raw_analysis"], mtime=mtime)
        if result.get("vocal_stem"):
            ucache.put_track(filename, dur, "vocal_stem", result["vocal_stem"], mtime=mtime)
        # Preserve a richer Songstats-aware tagger entry from registry/tagger if it exists.
        existing_key = ucache.track_key(filename, dur, "tagger")
        existing_entry = ucache._entries.get(existing_key)
        existing_tagger = existing_entry.data if existing_entry and isinstance(existing_entry.data, dict) else None
        preserve_existing = bool(
            existing_tagger
            and existing_tagger.get("_tagger_audio_features_sig")
            and tagger_core_metadata_matches(existing_tagger)
        )
        if not preserve_existing:
            ucache.put_track(filename, dur, "tagger", tagger_result, mtime=mtime)
        stats.n_extracted += 1

    def _apply_dsp_result(t, mtime, result):
        """Apply DSP-only worker result to cache without touching tagger analysis."""
        raw_cache[t.path] = RawCacheEntry(
            mtime=mtime, info=t, dsp=result["dsp"], section_dsp=result["section_dsp"],
        )
        filename = Path(t.path).name
        dur = _durations.get(t.path)
        ucache.put_track(filename, dur, "dsp", result["dsp"], mtime=mtime)
        ucache.put_track(filename, dur, "section_dsp", result["section_dsp"], mtime=mtime)
        stats.n_extracted += 1

    from dj_registry.progress import ProgressBar

    # ── Phase 1: DSP-only extraction (tagger analysis already cached) ──
    if to_dsp_only:
        bar = ProgressBar(len(to_dsp_only), label="DSP extract")
        extracted = 0
        failed = 0
        for j, (i, t, mtime) in enumerate(to_dsp_only):
            try:
                result = _extract_dsp_worker(t.path)
                _apply_dsp_result(t, mtime, result)
                extracted += 1
            except Exception as e:
                raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                stats.n_failed += 1
                failed += 1
                logger.debug("DSP extract failed for %s: %s", t.path, e)
            bar.update(j + 1, Path(t.path).name, extracted=extracted, failed=failed)
        bar.finish()

    # ── Phase 2: Full analysis + extraction ──
    n_todo = len(to_extract)
    actual_workers = workers if workers > 0 else 1

    if n_todo == 0:
        pass
    elif actual_workers <= 1 or n_todo == 1:
        bar = ProgressBar(n_todo, label="Extract")
        analyzed = 0
        extracted = 0
        failed = 0
        for j, (i, t, mtime, needs_analysis, cached_dsp) in enumerate(to_extract):
            done = j + 1
            try:
                result = _extract_worker(t.path, needs_analysis, cached_dsp)
                _apply_result(t, mtime, needs_analysis, result)
                if needs_analysis:
                    analyzed += 1
                else:
                    extracted += 1
            except Exception as e:
                raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                stats.n_failed += 1
                failed += 1
                logger.debug("Extract failed for %s: %s", t.path, e)
            bar.update(done, Path(t.path).name, analyzed=analyzed, extracted=extracted, failed=failed)
            if done % 20 == 0:
                save_raw_cache(raw_cache, cache_path)
        bar.finish()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info("  Using %d workers for %d tracks...", actual_workers, n_todo)
        bar = ProgressBar(n_todo, label="Extract")
        analyzed = 0
        extracted = 0
        failed = 0
        futures = {}
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            for i, t, mtime, needs_analysis, cached_dsp in to_extract:
                fut = pool.submit(_extract_worker, t.path, needs_analysis, cached_dsp)
                futures[fut] = (i, t, mtime, needs_analysis)

            done_count = 0
            for fut in as_completed(futures):
                i, t, mtime, needs_analysis = futures[fut]
                done_count += 1
                try:
                    result = fut.result()
                    _apply_result(t, mtime, needs_analysis, result)
                    if needs_analysis:
                        analyzed += 1
                    else:
                        extracted += 1
                except Exception as e:
                    raw_cache[t.path] = RawCacheEntry(mtime=mtime, info=t, dsp={})
                    stats.n_failed += 1
                    failed += 1
                    logger.debug("Extract failed for %s: %s", t.path, e)
                bar.update(done_count, Path(t.path).name, analyzed=analyzed, extracted=extracted, failed=failed)
                if done_count % 20 == 0:
                    save_raw_cache(raw_cache, cache_path)
        bar.finish()

    # Remove deleted files from cache
    current_paths = {t.path for t in tracks}
    removed = [p for p in raw_cache if p not in current_paths]
    for p in removed:
        del raw_cache[p]
    stats.n_removed = len(removed)

    if stats.n_extracted > 0 or stats.n_removed > 0:
        save_raw_cache(raw_cache, cache_path)
        ucache.save()
    return raw_cache, stats


def _apply_tagger_result(t, tagger_data: dict, analyze_untagged: bool) -> None:
    """Apply cached tagger analysis result to a TrackInfo object.

    The tagger result dict has this structure:
      energy: int, camelot: str, bpm: float, structure: str,
      vibe/mood: str, vocal: taxonomy profile code, vibe_scores: dict, vocal_ratio: float,
      confidences: {energy: float, key: float, structure: float, vibe: float, vocal: float}
    """
    from dj_tagger.moods import normalize_mood_code, normalize_mood_scores
    from dj_tagger.vocals import normalize_vocal_profile, normalize_vocal_profile_scores

    from .scanner import is_unknown_key

    if analyze_untagged and (t.energy is None or is_unknown_key(t.key)):
        if "energy" in tagger_data:
            t.energy = tagger_data.get("energy")
        if "camelot" in tagger_data:
            t.key = tagger_data.get("camelot")
        if "bpm" in tagger_data and t.bpm is None:
            bpm = tagger_data.get("bpm")
            if bpm and bpm > 0:
                t.bpm = round(bpm) if isinstance(bpm, float) else bpm
        if "structure" in tagger_data:
            t.structure = tagger_data.get("structure")
            if t.structure and len(t.structure) >= 2:
                try:
                    t.intro_bars = int(t.structure[:-1])
                    t.flow_type = t.structure[-1]
                except (ValueError, IndexError):
                    pass

    # Vibe and vocal — always apply (these are analyzed regardless of tags)
    if "vibe" in tagger_data or "mood" in tagger_data:
        t.vibe = normalize_mood_code(tagger_data.get("mood") or tagger_data.get("vibe"))
    if "vibe_scores" in tagger_data:
        t.vibe_scores = normalize_mood_scores(tagger_data.get("vibe_scores", {}))
    elif "mood_scores" in tagger_data:
        t.vibe_scores = normalize_mood_scores(tagger_data.get("mood_scores", {}))
    if "vocal" in tagger_data:
        vocal = normalize_vocal_profile(tagger_data.get("vocal_profile") or tagger_data.get("vocal"))
        if vocal:
            t.vocal = vocal
    elif "vocal_profile" in tagger_data:
        vocal = normalize_vocal_profile(tagger_data.get("vocal_profile"))
        if vocal:
            t.vocal = vocal
    if "vocal_scores" in tagger_data:
        t.vocal_scores = normalize_vocal_profile_scores(tagger_data.get("vocal_scores", {}))

    # Confidences — tagger stores as nested dict {"energy": 0.9, "key": 0.85, ...}
    confs = tagger_data.get("confidences")
    if isinstance(confs, dict):
        for conf_key in ("energy", "key", "structure", "vibe", "vocal"):
            val = confs.get(conf_key)
            if val is not None:
                t.confidences[conf_key] = val
    # Also check flat key_confidence (legacy format)
    elif "key_confidence" in tagger_data:
        t.confidences["key"] = tagger_data["key_confidence"]


# ─── Soft vocal partitioning helper ──────────────────────────────────────────

def _cluster_with_soft_vocal(feature_tracks, distance_matrix, config):
    """Cluster with soft vocal partitioning and BPM validation.

    Only hard-split vocal tracks when confidence is high.
    Uncertain tracks are clustered with everyone.
    Passes BPM values for post-clustering validation.
    """
    import numpy as np
    from dj_tagger.vocals import has_vocal_content

    from .grouping.clustering import cluster_tracks

    n = len(feature_tracks)
    all_bpms = [t.info.bpm for t in feature_tracks]
    all_energies = [t.info.energy for t in feature_tracks]
    all_keys = [t.info.key for t in feature_tracks]
    conf_threshold = config.vocal_confidence_threshold

    confident_v = [i for i in range(n) if has_vocal_content(feature_tracks[i].info.vocal)
                   and feature_tracks[i].info.confidences.get("vocal", 1.0) >= conf_threshold]
    confident_nv = [i for i in range(n) if not has_vocal_content(feature_tracks[i].info.vocal)
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

    print(f"  non-vocal/uncertain: {len(confident_nv)} tracks, vocal-profile (confident): {len(confident_v)} tracks")
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
    from .grouping.distance import compute_distance_matrix, unified_distance_matrix
    from .grouping.assignment import (
        assign_group_ids,
        assign_new_tracks,
        load_assignment,
        refresh_group_descriptors,
        save_assignment,
    )
    from .feedback.store import load_feedback
    from .feedback.apply import apply_feedback_to_distances
    from .output.csv_export import export_groups_csv, export_recommendations_csv
    from .output.playlists import generate_group_playlists
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
        args.cache_dir = str(input_path / "cache")
        args.csv = str(input_path / "outputs" / "groups.csv")
        args.recommendations_csv = str(input_path / "outputs" / "recommendations.csv")
        args.playlists = str(input_path / "outputs" / "playlists")
        args.feedback = str(input_path / "outputs" / "feedback.csv")
        out_dir = Path(ext_out)
        print(f"External input detected — outputs will be in {ext_out}/")
    else:
        out_dir = Path(_DEFAULTS.output_dir)

    # Registry dir: explicit flag > auto-derived from library path > default
    if getattr(args, "registry_dir", None):
        config.registry_dir = args.registry_dir
    elif is_external:
        config.registry_dir = str(input_path / "outputs" / "registry")

    cpaths = _cache_paths(args.cache_dir)

    # Allowed roots for destructive operations
    _allowed = [str(out_dir), args.cache_dir]

    # ── Handle --clean ──
    if args.clean:
        _safe_rmtree(str(out_dir), _allowed)
        print(f"Cleaned {out_dir}/")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Handle --force-extract: clear DSP cache + outputs (NOT clap cache) ──
    if args.force_extract:
        for f in [cpaths["features"], cpaths["assignment"], cpaths["raw"]]:
            _safe_unlink(f, _allowed)
        # Only clear group playlists; categorical playlists (by_key, by_subgenre,
        # by_popularity) are owned by the registry pipeline.
        _safe_rmtree(str(Path(args.playlists) / "groups"), _allowed)
        _safe_rmtree(str(Path(args.playlists) / "groups_coarse"), _allowed)
        for f in [args.csv, args.recommendations_csv]:
            _safe_unlink(f, _allowed)
        # Reset cache singleton so it reloads fresh
        from dj_tagger.universal_cache import reset_cache
        reset_cache()
        print("Cleared cache and outputs")

    # ── Handle --force-clap: clear CLAP entries from raw cache ──
    if args.force_clap:
        _safe_unlink(cpaths["clap"], _allowed)
        print("Cleared legacy CLAP cache")

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
    cache_only = getattr(args, "cache_only", False)
    raw_cache, ext_stats = _run_extraction(
        tracks, cpaths["features"],
        force=args.force_extract,
        workers=getattr(args, "workers", 1) or 1,
        analyze_untagged=True,
        write_tags=args.write_tags,
        cache_only=cache_only,
    )
    elapsed_step = _time.perf_counter() - t_step
    print(f"  Done: {', '.join(ext_stats.summary_parts())} ({_fmt_elapsed(elapsed_step)})")

    if cache_only:
        # Drop tracks that had no cached analysis so downstream steps (CLAP,
        # feature build, clustering) run only over the already-analyzed subset
        # and never decode audio. build_features_from_raw indexes raw_cache by
        # path, so track_order must not contain a skipped path.
        kept = [t for t in tracks if t.path in raw_cache]
        if len(kept) != len(tracks):
            print(f"  --cache-only: grouping {len(kept)} analyzed track(s), "
                  f"skipping {len(tracks) - len(kept)} not yet analyzed")
        tracks = kept
        total_tracks = len(tracks)
        if not tracks:
            print("  No analyzed tracks to group.")
            return 0

    # ── Step 1b: CLAP audio embeddings ──
    track_order = [t.path for t in tracks]
    clap_embeddings = None

    if not args.no_clap:
        # Check if all CLAP embeddings are already cached before importing torch/laion_clap
        from dj_tagger.universal_cache import get_cache as _get_ucache, quick_duration as _qd
        _uc = _get_ucache(str(Path(cpaths["features"]).parent / "raw_cache.pkl"))
        all_clap_cached = all(
            _uc.get_track(Path(p).name, _qd(p), "clap") is not None
            for p in track_order
        )

        if all_clap_cached or cache_only:
            # Load cached embeddings without importing CLAP/torch. Under
            # --cache-only we never decode audio, so any track lacking a cached
            # embedding is simply zero-filled (same as build_unified_vector's
            # missing-CLAP handling) rather than extracted.
            n_have = sum(1 for p in track_order
                         if _uc.get_track(Path(p).name, _qd(p), "clap") is not None)
            print(f"\n  CLAP embeddings: {n_have}/{total_tracks} cached"
                  + ("" if all_clap_cached else " (zero-filling the rest, --cache-only)"))
            import numpy as _np
            raw_clap = _np.zeros((len(track_order), 512), dtype=_np.float32)
            for i, p in enumerate(track_order):
                emb = _uc.get_track(Path(p).name, _qd(p), "clap")
                if emb is not None:
                    raw_clap[i] = emb
            clap_embeddings, _ = fit_pca(raw_clap, config.clap_pca_dims)
        elif is_clap_available():
            print(f"\n  Extracting CLAP audio embeddings ({total_tracks} tracks)...")
            t_clap = _time.perf_counter()
            try:
                raw_clap = extract_clap_incremental(
                    track_order, cpaths["clap"], force=args.force_clap,
                )
                clap_embeddings, _ = fit_pca(raw_clap, config.clap_pca_dims)
                print(f"  CLAP done ({_fmt_elapsed(_time.perf_counter() - t_clap)})")
            except Exception as e:
                print(f"  CLAP failed: {e} -- continuing without CLAP")
        else:
            print("\n  CLAP not installed -- skipping audio embeddings")

    # ── Step 1c: Registry enrichment ──
    registry_enrichments = None
    algorithm = getattr(args, "algorithm", config.clustering_method)
    if config.use_registry and not getattr(args, "no_registry", False):
        from .features.registry_bridge import RegistryBridge, match_enrichments
        bridge = RegistryBridge(config.registry_dir)
        raw_enrichments = bridge.load()
        if raw_enrichments:
            registry_enrichments = match_enrichments(track_order, raw_enrichments)
            n_matched = len(registry_enrichments)
            print(f"  Registry: {n_matched}/{total_tracks} tracks enriched")

    cache = build_features_from_raw(raw_cache, track_order, clap_embeddings, config, registry=registry_enrichments)
    feature_tracks = cache.tracks

    # ── Step 2: Cluster or incrementally assign ──
    assignment_path = cpaths["assignment"]
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
        print(f"\n[2/5] Grouping {n_ft} tracks by similarity (algorithm={algorithm})...")
        feedback = load_feedback(args.feedback)

        if algorithm == "constrained":
            from .features.fusion import fit_unified_pca
            from .grouping.constraints import build_constraints
            from .grouping.constrained import cop_kmedoids

            _t = _time.perf_counter()
            reduced_vectors, pca_params = fit_unified_pca(feature_tracks, config.pca_target_variance)
            n_dims = reduced_vectors.shape[1]
            print(f"  PCA fusion: {feature_tracks[0].unified_vector.shape[0]} -> {n_dims} dims ({_fmt_elapsed(_time.perf_counter() - _t)})")

            _t = _time.perf_counter()
            distance_matrix = unified_distance_matrix(reduced_vectors)
            print(f"  Computing {n_ft}x{n_ft} distance matrix... ({_fmt_elapsed(_time.perf_counter() - _t)})")

            if feedback:
                distance_matrix = apply_feedback_to_distances(distance_matrix, feature_tracks, feedback, config)

            constraints = build_constraints(feature_tracks, config, feedback)

            _t = _time.perf_counter()
            labels = cop_kmedoids(distance_matrix, constraints, config)
            print(f"  Clustering done ({_fmt_elapsed(_time.perf_counter() - _t)})")
        else:
            _t = _time.perf_counter()
            distance_matrix = compute_distance_matrix(feature_tracks, config)
            print(f"  Computing {n_ft}x{n_ft} distance matrix... ({_fmt_elapsed(_time.perf_counter() - _t)})")

            if feedback:
                distance_matrix = apply_feedback_to_distances(distance_matrix, feature_tracks, feedback, config)

            _t = _time.perf_counter()
            labels = _cluster_with_soft_vocal(feature_tracks, distance_matrix, config)
            print(f"  Clustering + validation done ({_fmt_elapsed(_time.perf_counter() - _t)})")

        _t = _time.perf_counter()
        assignment = assign_group_ids(feature_tracks, labels, distance_matrix)

    assignment = refresh_group_descriptors(feature_tracks, assignment)
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
        # Group folders intentionally not materialized — playlists are the
        # supported deliverable. See output/playlists.py.

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
                        vibe=info.vibe,
                        group_id=group.group_id,
                        vocal_profile=info.vocal,
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
        # Clean and recreate group playlists. Categorical playlists
        # (by_key/by_bpm/by_subgenre) live under the same root but are
        # written by the registry pipeline; leave them untouched here.
        # Coarse, half-resolution group playlists are the default; full-
        # resolution groups/ are opt-in via --fine-playlists.
        from .output.coarse_groups import generate_coarse_group_playlists
        coarse_subdir = str(Path(args.playlists) / "groups_coarse")
        _safe_rmtree(coarse_subdir, _allowed)
        n_coarse = generate_coarse_group_playlists(
            feature_tracks, assignment, config, args.playlists
        )
        print(f"  Playlists: {args.playlists}/groups_coarse/ ({n_coarse} coarse groups)")
        if getattr(args, "fine_playlists", False):
            groups_subdir = str(Path(args.playlists) / "groups")
            _safe_rmtree(groups_subdir, _allowed)
            generate_group_playlists(feature_tracks, assignment, args.playlists)
            print(f"  Playlists: {args.playlists}/groups/")

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
    cpaths = _cache_paths(args.cache_dir)
    raw_cache, stats = _run_extraction(
        tracks, cpaths["features"],
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
    cpaths = _cache_paths(args.cache_dir)
    raw_cache = load_raw_cache(cpaths["features"])
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
    save_assignment(assignment, cpaths["assignment"])

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
    cpaths = _cache_paths(args.cache_dir)
    raw_cache = load_raw_cache(cpaths["features"])
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
    from .output.playlists import generate_group_playlists
    from dj_tagger.formats import format_tag
    from dj_tagger.metadata import write_tag
    from .config import GrouperConfig

    cpaths = _cache_paths(args.cache_dir)
    raw_cache = load_raw_cache(cpaths["features"])
    if not raw_cache:
        print("No feature cache found. Run 'extract' first.")
        return 1
    track_order = list(raw_cache.keys())
    cache = build_features_from_raw(raw_cache, track_order, config=GrouperConfig())
    tracks = cache.tracks
    assignment = load_assignment(cpaths["assignment"])

    if args.write_tags and not args.dry_run:
        print("Writing tags with group IDs to file metadata...")
        for group in assignment.groups:
            for idx in group.member_indices:
                tf = tracks[idx]
                info = tf.info
                tag = format_tag(
                    energy=info.energy, camelot=info.key, bpm=info.bpm,
                    vibe=info.vibe,
                    group_id=group.group_id,
                    vocal_profile=info.vocal,
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
    from .config import GrouperConfig

    config = GrouperConfig()
    cpaths = _cache_paths(args.cache_dir)
    raw_cache = load_raw_cache(cpaths["features"])
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
    cpaths = _cache_paths(args.cache_dir)
    raw_cache = load_raw_cache(cpaths["features"])
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
