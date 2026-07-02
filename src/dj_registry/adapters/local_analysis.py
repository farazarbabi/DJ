"""Local audio analysis adapter — runs the full tagger pipeline.

Uses raw cache (cache/raw_cache.pkl) and derived cache (cache/derived_cache.pkl)
so files analyzed by any module are never re-analyzed.

Runs the full tagger analysis (energy, mood/vibe, vocal, structure, key) and stores
all results in the registry as SourceObservations + LogicalTrack tagger fields.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from dj_tagger.moods import normalize_mood_code, normalize_mood_scores
from dj_tagger.universal_cache import quick_duration
from dj_tagger.vocals import (
    normalize_vocal_profile,
    normalize_vocal_profile_scores,
    vocal_profile_from_has_vocals,
)

from ..config import RegistryConfig
from ..key_utils import parse_any_key
from ..models import LogicalTrack, SourceObservation, PayloadIndexEntry, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def _default_analysis_workers() -> int:
    """Pick a sensible default worker count for cache-miss analysis.

    Demucs (~90% of per-track cost) is memory-bandwidth-bound on CPU: measured
    throughput plateaus around ~4 concurrent inferences (4 workers ≈ 1.85x,
    10 workers ≈ 1.72x — more workers only inflate the startup spike and RAM).
    So target a modest ~cpu/4 workers rather than saturating every core.

    Each worker holds its own Demucs model + decoded audio, so RAM (not cores) is
    the real ceiling — budget ~2 GB per worker. Callers additionally cap this by
    the number of cache-miss tracks so small/incremental runs stay on the fast
    single-worker (all-threads, one model load) path.
    """
    cpu = os.cpu_count() or 1
    workers = max(1, cpu // 4)  # bandwidth-bound: ~4 is the throughput sweet spot
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        ram_cap = max(1, int(avail_gb // 2.0))  # ~2 GB per Demucs worker
        workers = min(workers, ram_cap)
    except Exception:
        workers = min(workers, 8)
    # Hard cap — beyond ~4 the bandwidth-bound Demucs sees no throughput gain,
    # only more startup/model-load overhead and RAM.
    return max(1, min(workers, 8))


def _pool_worker_init(threads: int) -> None:
    """Pin per-worker CPU threads so N parallel processes don't oversubscribe.

    Demucs/BLAS default to using every core for a single inference; with N
    worker processes that means N×cores threads fighting over the CPU. We run N
    independent tracks in parallel instead, each capped to a slice of the cores.
    Env vars are set before torch is imported here so OMP/MKL pick them up.
    """
    import os as _os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ[var] = str(threads)
    try:
        import torch

        torch.set_num_threads(max(1, threads))
    except Exception:
        pass


def _lookup_audio_features(ucache, isrc: str) -> dict[str, float] | None:
    """Look up Songstats audio features from cache by ISRC.

    Returns dict with float values for valence, instrumentalness, energy,
    liveness, acousticness — or None if no Songstats data is cached.
    """
    if not isrc:
        return None
    data = ucache.get(f"isrc:{isrc}|songstats")
    if not data or not isinstance(data, dict):
        return None
    features: dict[str, float] = {}
    for key in ("valence", "instrumentalness", "energy", "liveness", "acousticness", "speechiness"):
        val = data.get(key, "")
        if val != "" and val is not None:
            try:
                features[key] = float(val)
            except (ValueError, TypeError):
                pass
    return features if features else None


def _hydrate_tagger_result(result: dict, audio_features: dict[str, float] | None) -> dict:
    """Attach cache-only metadata used to detect stale vibe results."""
    from dj_tagger.tagger_cache import hydrate_tagger_result

    return hydrate_tagger_result(result, audio_features)


def _merge_rederived_tagger(existing: dict | None, derived: dict, audio_features: dict[str, float] | None) -> dict:
    """Merge re-derived fields into the richest available tagger record."""
    from dj_tagger.tagger_cache import merge_rederived_tagger

    return merge_rederived_tagger(existing, derived, audio_features)


def _analyze_full(
    path: str,
    use_essentia: bool,
    audio_features: dict[str, float] | None = None,
    vocal_stem: dict | None = None,
) -> dict:
    """Run full tagger analysis + optional essentia key. Top-level for pickling.

    ``vocal_stem`` may be passed in by the caller (loaded from the universal
    cache) to skip Demucs separation when the raw layer is already present.
    """
    from dj_tagger.audio import load_audio_features
    from dj_tagger.raw_features import compute_tagger_artifacts

    result: dict = {
        "path": path,
        "tagger_result": None,
        "dsp": None,
        "raw_analysis": None,
        "section_dsp": None,
        "vocal_stem": None,
        "essentia": None,
    }

    # Canonical tagger pipeline: raw extraction + settings-driven derivation + key analysis
    try:
        audio = load_audio_features(path)
        artifacts = compute_tagger_artifacts(
            audio,
            audio_features=audio_features,
            vocal_stem=vocal_stem,
            use_essentia=False,
        )
        result["tagger_result"] = artifacts["tagger_result"]
        result["dsp"] = artifacts["dsp"]
        result["raw_analysis"] = artifacts["raw_analysis"]
        result["section_dsp"] = artifacts["section_dsp"]
        result["vocal_stem"] = artifacts.get("vocal_stem") or {}
    except Exception as e:
        result["error"] = str(e)
        return result

    # Optional essentia key analysis (separate run)
    if use_essentia:
        try:
            from dj_tagger.analyzers.key import analyze_key

            kr = analyze_key(audio, use_essentia=True)
            result["essentia"] = {
                "camelot": kr.camelot,
                "key_name": kr.key_name,
                "confidence": kr.confidence,
            }
        except Exception as e:
            result["essentia_error"] = str(e)

    return result


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
        from dj_grouper.features.dsp import DSP_CURATED_NAMES

        return all(name in data for name in DSP_CURATED_NAMES)
    if layer == "raw_analysis":
        required = {
            "bar_energies",
            "n_bars",
            "tempo",
            "vocal_ratio",
            "vocal_temporal_bonus",
            "onset_rate",
        }
        return required.issubset(data)
    if layer == "section_dsp":
        return True
    return False


def _essentia_available() -> bool:
    """Check if Essentia is installed."""
    try:
        import essentia  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_tagger_features(result: dict) -> dict:
    """Extract tagger features from a cache result dict.

    Handles multiple cache entry formats (analyze_track, derive_all, grouper inline).
    Returns a dict with normalized field names, or empty dict on failure.
    """
    features: dict = {}

    # Energy
    energy = result.get("energy")
    if energy is not None:
        features["energy"] = str(energy)

    # Mood/vibe. The registry field name is kept as tagger_vibe for compatibility.
    vibe = result.get("mood") or result.get("vibe")
    if vibe:
        features["vibe"] = normalize_mood_code(vibe)

    # Vocal — handle both formats
    vocal = normalize_vocal_profile(result.get("vocal_profile") or result.get("vocal"))
    if vocal:
        features["vocal"] = vocal
    else:
        fallback = vocal_profile_from_has_vocals(result.get("has_vocals"))
        if fallback:
            features["vocal"] = fallback

    # Structure
    structure = result.get("structure")
    if structure:
        features["structure"] = structure

    # BPM
    bpm = result.get("bpm")
    if bpm is not None:
        features["bpm"] = str(round(float(bpm)))

    # Beatgrid anchor (first detected beat, seconds)
    first_beat = result.get("first_beat_sec")
    if first_beat is not None:
        features["first_beat_sec"] = f"{float(first_beat):.3f}"

    # Key
    camelot = result.get("camelot")
    if camelot:
        features["camelot"] = camelot
        features["key_name"] = result.get("key", "")
        features["key_confidence"] = result.get("key_confidence", 0.0)

    # Mood/vibe scores
    vibe_scores = result.get("mood_scores") or result.get("vibe_scores")
    if vibe_scores and isinstance(vibe_scores, dict):
        normalized_scores = normalize_mood_scores(vibe_scores)
        features["vibe_scores"] = json.dumps({k: round(v, 4) for k, v in normalized_scores.items()})

    vocal_scores = result.get("vocal_scores")
    if vocal_scores and isinstance(vocal_scores, dict):
        normalized_vocal_scores = normalize_vocal_profile_scores(vocal_scores)
        features["vocal_scores"] = json.dumps({k: round(v, 4) for k, v in normalized_vocal_scores.items()})

    # Confidences
    confidences = result.get("confidences")
    if confidences and isinstance(confidences, dict):
        features["confidences"] = json.dumps({k: round(v, 3) for k, v in confidences.items()})

    # Provenance signatures
    for src_key, feature_key in [
        ("_tagger_version", "tagger_version"),
        ("_tagger_raw_sig", "tagger_raw_signature"),
        ("_tagger_derived_sig", "tagger_derived_signature"),
        ("_tagger_key_sig", "tagger_key_signature"),
        ("_tagger_audio_features_sig", "tagger_audio_features_signature"),
    ]:
        value = result.get(src_key)
        if value:
            features[feature_key] = str(value)

    return features


def _apply_tagger_to_track(track: LogicalTrack, features: dict) -> None:
    """Write tagger features onto a LogicalTrack."""
    if "energy" in features:
        track.tagger_energy = features["energy"]
    if "vibe" in features:
        track.tagger_vibe = features["vibe"]
    if "vocal" in features:
        track.tagger_vocal = features["vocal"]
    if "structure" in features:
        track.tagger_structure = features["structure"]
    if "bpm" in features:
        track.tagger_bpm = features["bpm"]
    if "first_beat_sec" in features:
        track.tagger_first_beat_sec = features["first_beat_sec"]
    if "vibe_scores" in features:
        track.tagger_vibe_scores = features["vibe_scores"]
    if "vocal_scores" in features and hasattr(track, "tagger_vocal_scores"):
        track.tagger_vocal_scores = features["vocal_scores"]
    if "confidences" in features:
        track.tagger_confidences = features["confidences"]
    if "tagger_version" in features:
        track.tagger_version = features["tagger_version"]
    if "tagger_raw_signature" in features:
        track.tagger_raw_signature = features["tagger_raw_signature"]
    if "tagger_derived_signature" in features:
        track.tagger_derived_signature = features["tagger_derived_signature"]
    if "tagger_key_signature" in features:
        track.tagger_key_signature = features["tagger_key_signature"]
    if "tagger_audio_features_signature" in features:
        track.tagger_audio_features_signature = features["tagger_audio_features_signature"]


def _build_observation(
    track_id: str,
    file_id: str,
    source_system: str,
    features: dict,
) -> SourceObservation | None:
    """Build a SourceObservation from extracted features.

    Emits a row whenever any tagger field is present, even when key
    analysis didn't produce a camelot. The librosa observation is the
    canonical carrier for energy/vibe/vocal/structure — gating it on
    key extraction drops the whole row when only the key analyzer fails.
    """
    camelot = features.get("camelot")
    parsed = parse_any_key(camelot) if camelot else None
    std, cam = parsed if parsed else ("", "")

    has_any_tagger = any(
        features.get(k)
        for k in ("energy", "vibe", "vocal", "structure", "bpm")
    )
    if not cam and not has_any_tagger:
        return None

    obs = SourceObservation(
        observation_id=f"OBS-{source_system.replace('analysis_', '')}-{file_id}",
        track_id=track_id,
        file_id=file_id,
        source_system=source_system,
        key_standard=std,
        key_camelot=cam,
        key_confidence=features.get("key_confidence", 0.0),
        bpm=features.get("bpm", ""),
        observed_at=now_iso(),
    )

    # Attach tagger fields to the observation
    obs.tagger_energy = features.get("energy", "")
    obs.tagger_vibe = features.get("vibe", "")
    obs.tagger_vocal = features.get("vocal", "")
    obs.tagger_structure = features.get("structure", "")
    obs.tagger_vibe_scores = features.get("vibe_scores", "")
    obs.tagger_vocal_scores = features.get("vocal_scores", "")
    obs.tagger_confidences = features.get("confidences", "")

    return obs


def run_analysis(
    config: RegistryConfig,
    store: CsvStore,
    *,
    track_ids: list[str] | None = None,
    no_essentia: bool = False,
    show_progress: bool = False,
) -> dict:
    """Run full tagger analysis on tracks (energy, vibe, vocal, structure, key).

    Uses raw cache + derived cache. Files already analyzed by any module
    are read from cache instantly — no audio loading needed.
    New analysis results are written back so all modules can reuse them.

    Stores results as SourceObservations and writes tagger features to LogicalTrack.

    Returns stats dict: {"total", "cached", "analyzed", "failed"}.
    """
    use_essentia = config.run_essentia and not no_essentia and _essentia_available()
    if config.run_essentia and not no_essentia and not _essentia_available():
        logger.debug("Essentia not installed, running librosa only")

    files = store.load_files()
    tracks = store.load_tracks()

    file_by_id = {f.file_id: f for f in files}
    track_by_id = {t.track_id: t for t in tracks}

    candidates: list[tuple[str, str, str, float, float]] = []  # (track_id, file_id, path, mtime, duration)
    for t in tracks:
        if track_ids and t.track_id not in track_ids:
            continue
        if not t.primary_file_id:
            continue
        frec = file_by_id.get(t.primary_file_id)
        if not frec:
            continue
        try:
            mtime = os.path.getmtime(frec.path_abs)
        except OSError:
            continue
        candidates.append((t.track_id, frec.file_id, frec.path_abs, mtime, frec.audio_duration_sec))

    if not candidates:
        logger.info("Analysis: no candidates to process")
        return {"total": 0, "cached": 0, "analyzed": 0, "failed": 0}

    # Load shared caches
    from dj_tagger.universal_cache import get_cache as get_ucache
    from dj_tagger.tagger_cache import tagger_metadata_matches
    ucache = get_ucache(os.path.join("cache", "raw_cache.pkl"))

    # Split into cache hits and misses.
    # Uses universal cache directly (version-checked, no mtime check) so that
    # entries survive tag write-back (which changes mtime but not audio content).
    # Fallback: dsp + raw_analysis → derive_all.
    cache_hits: list[tuple[str, str, str, dict]] = []  # (track_id, file_id, path, features)
    cache_misses: list[tuple[str, str, str, float, float]] = []

    cache_progress = ProgressBar(len(candidates), label="Analysis cache", enabled=show_progress)
    for index, (track_id, file_id, path, mtime, duration) in enumerate(candidates, start=1):
        filename = os.path.basename(path)
        features: dict = {}
        track = track_by_id.get(track_id)

        # Use mutagen duration for cache lookups — matches how tagger/grouper store entries.
        # FileRecord.audio_duration_sec comes from soundfile which can differ slightly.
        cache_dur = quick_duration(path) or duration
        tagger_key = ucache.track_key(filename, cache_dur, "tagger")
        any_tagger_entry = ucache._entries.get(tagger_key)
        cached_tagger = any_tagger_entry.data if any_tagger_entry and isinstance(any_tagger_entry.data, dict) else None
        audio_features = _lookup_audio_features(ucache, track.isrc_canonical if track else "")
        dsp_data = _cached_raw_layer(ucache, filename, cache_dur, "dsp")
        raw_analysis = _cached_raw_layer(ucache, filename, cache_dur, "raw_analysis")

        # Try tagger layer in universal cache (version-checked, no mtime check).
        tagger_data = ucache.get_track(filename, cache_dur, "tagger")
        if tagger_metadata_matches(tagger_data if isinstance(tagger_data, dict) else None, audio_features):
            features = _extract_tagger_features(tagger_data)

        raw_layers_available = dsp_data and isinstance(dsp_data, dict) and raw_analysis and isinstance(raw_analysis, dict)
        raw_signature_changed = False
        if raw_layers_available and isinstance(cached_tagger, dict):
            try:
                from dj_tagger.settings import raw_version
                raw_signature_changed = cached_tagger.get("_tagger_raw_sig") != raw_version()
            except Exception:
                raw_signature_changed = False

        # Fill in missing/stale tagger features from raw cache via derive_all.
        # Raw collection layers are identity-keyed; downstream signature changes
        # must not force audio analysis when these payloads exist.
        if raw_layers_available and (raw_signature_changed or not (features.get("energy") and features.get("vibe"))):
            from dj_tagger.derive import derive_all
            derived = derive_all(dsp_data, raw_analysis, audio_features=audio_features)
            merged = _merge_rederived_tagger(cached_tagger, derived, audio_features)
            ucache.put_track(filename, cache_dur, "tagger", merged)
            derived_features = _extract_tagger_features(merged)
            # Merge: derived fills gaps, and raw provenance changes refresh
            # derived fields without recollecting audio.
            for k, v in derived_features.items():
                if raw_signature_changed or k not in features or not features[k]:
                    features[k] = v

        if not (features.get("energy") and features.get("vibe")) and cached_tagger:
            # Last resort: use the collected tagger facts by filename+duration
            # even when their derived metadata is stale. This preserves the
            # no-recollection guarantee; later runs can rederive when raw
            # layers are available.
            features = _extract_tagger_features(cached_tagger)

        has_key = bool(features.get("camelot"))
        has_tagger = bool(features.get("energy") and features.get("vibe"))

        if has_key and has_tagger:
            cache_hits.append((track_id, file_id, path, features))
        elif has_tagger:
            # Have tagger features but no key — still a hit, key will be blank
            cache_hits.append((track_id, file_id, path, features))
        else:
            cache_misses.append((track_id, file_id, path, mtime, duration))
        cache_progress.update(index, os.path.basename(path), cached=len(cache_hits), analyze=len(cache_misses))
    cache_progress.finish()

    if cache_misses:
        logger.info("Analysis: %d tracks (%d cached, %d to analyze)", len(candidates), len(cache_hits), len(cache_misses))
    else:
        logger.info("Analysis: %d tracks (all cached)", len(candidates))

    # Run full analysis on cache misses
    fresh_results: list[tuple[str, str, str, dict]] = []  # (track_id, file_id, path, raw_result)
    if cache_misses:
        paths_to_analyze = [path for _, _, path, _, _ in cache_misses]
        n_total = len(paths_to_analyze)

        # Pre-load cached Demucs vocal_stem scalars so the worker can skip the
        # expensive separation when the raw layer is already present (e.g.,
        # written by a prior grouper or registry run).
        cached_vocal_stems: list[dict | None] = []
        for _, _, path, _, duration in cache_misses:
            fname = os.path.basename(path)
            stem = ucache.get_track(fname, duration, "vocal_stem")
            cached_vocal_stems.append(stem if isinstance(stem, dict) and stem else None)

        # Persist each analyzed track to the cache as soon as it finishes, and
        # flush to disk every CHECKPOINT_EVERY tracks. Audio analysis is slow
        # per track, so checkpointing bounds how much work an interrupted run
        # can lose to at most CHECKPOINT_EVERY tracks instead of the whole run.
        CHECKPOINT_EVERY = 10
        since_checkpoint = 0

        def _persist_result(misses_idx: int, result: dict | None) -> None:
            """Store one analysis result in the cache and checkpoint periodically.

            Saves tagger (derived) + dsp/raw_analysis/section_dsp/vocal_stem (raw)
            so future settings.toml changes can re-derive without re-analyzing
            audio. Survives across runs — keyed by filename + duration, not mtime.
            """
            nonlocal since_checkpoint
            track_id, file_id, path, mtime, duration = cache_misses[misses_idx]
            if result and "error" not in result and result.get("tagger_result"):
                track = track_by_id.get(track_id)
                audio_features = _lookup_audio_features(ucache, track.isrc_canonical if track else "")
                tagger_result = _hydrate_tagger_result(result["tagger_result"], audio_features)
                store_dur = quick_duration(path) or duration
                fname = os.path.basename(path)
                ucache.put_track(fname, store_dur, "tagger", tagger_result, mtime=mtime)
                if isinstance(result.get("dsp"), dict):
                    ucache.put_track(fname, store_dur, "dsp", result["dsp"], mtime=mtime)
                if isinstance(result.get("raw_analysis"), dict):
                    ucache.put_track(fname, store_dur, "raw_analysis", result["raw_analysis"], mtime=mtime)
                if isinstance(result.get("section_dsp"), dict):
                    ucache.put_track(fname, store_dur, "section_dsp", result["section_dsp"], mtime=mtime)
                if isinstance(result.get("vocal_stem"), dict) and result["vocal_stem"]:
                    ucache.put_track(fname, store_dur, "vocal_stem", result["vocal_stem"], mtime=mtime)
                result["tagger_result"] = tagger_result
                fresh_results.append((track_id, file_id, path, result))
            else:
                err = result.get("error", "unknown") if result else "unknown"
                logger.warning("Analysis failed for %s: %s", path, err)

            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                ucache.save()
                logger.info("Checkpoint: saved cache after %d analyzed tracks", since_checkpoint)
                since_checkpoint = 0

        # Resolve worker count: <= 0 means auto-detect (memory-aware). A single
        # worker keeps the in-process serial path (torch may still use all cores
        # for the one Demucs inference — no oversubscription there).
        workers = config.analysis_workers
        if workers is None or workers <= 0:
            workers = _default_analysis_workers()
        # Never spawn more workers than there are tracks to analyze. With only a
        # few cache misses (the common incremental case) this keeps us on the
        # fast single-worker path — all threads, one model load — instead of
        # paying N cold starts + N model loads + thread-pinned contention that
        # would make a small run slower than serial.
        workers = max(1, min(workers, n_total))
        if workers > 1:
            logger.info("Analysis: using %d parallel workers", workers)

        if workers <= 1:
            analysis_progress = ProgressBar(n_total, label="Analyze audio", enabled=show_progress)
            for i, p in enumerate(paths_to_analyze):
                track_id = cache_misses[i][0]
                track = track_by_id.get(track_id)
                audio_features = _lookup_audio_features(ucache, track.isrc_canonical if track else "")
                result = _analyze_full(p, use_essentia, audio_features, cached_vocal_stems[i])
                _persist_result(i, result)
                fname = os.path.basename(p)
                if not show_progress:
                    logger.info("  [%d/%d] %s", i + 1, n_total, fname)
                analysis_progress.update(i + 1, fname)
            analysis_progress.finish()
        else:
            done = 0
            analysis_progress = ProgressBar(n_total, label="Analyze audio", enabled=show_progress)
            threads_per_worker = max(1, (os.cpu_count() or workers) // workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_pool_worker_init,
                initargs=(threads_per_worker,),
            ) as pool:
                futures = {
                    pool.submit(
                        _analyze_full,
                        p,
                        use_essentia,
                        _lookup_audio_features(
                            ucache,
                            track_by_id.get(cache_misses[i][0]).isrc_canonical if track_by_id.get(cache_misses[i][0]) else "",
                        ),
                        cached_vocal_stems[i],
                    ): i
                    for i, p in enumerate(paths_to_analyze)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    done += 1
                    try:
                        result = future.result()
                        fname = os.path.basename(paths_to_analyze[idx])
                        if not show_progress:
                            logger.info("  [%d/%d] %s", done, n_total, fname)
                        analysis_progress.update(done, fname)
                    except Exception as e:
                        result = {"path": paths_to_analyze[idx], "error": str(e)}
                        fname = os.path.basename(paths_to_analyze[idx])
                        if not show_progress:
                            logger.info("  [%d/%d] %s FAILED", done, n_total, fname)
                        analysis_progress.update(done, f"{fname} FAILED")
                    # Persist as each future completes so an interrupted run
                    # keeps everything finished up to the last checkpoint.
                    _persist_result(idx, result)
            analysis_progress.finish()

        # Flush any tracks analyzed since the last checkpoint.
        if since_checkpoint:
            ucache.save()

    if ucache.dirty:
        ucache.save()

    # Build observations and update tracks
    raw_dir = os.path.join(config.raw_dir, "analysis")
    os.makedirs(raw_dir, exist_ok=True)

    new_obs: list[SourceObservation] = []
    new_payloads: list[PayloadIndexEntry] = []
    processed = 0

    # Process cache hits
    for track_id, file_id, path, features in cache_hits:
        obs = _build_observation(track_id, file_id, "analysis_librosa", features)
        if obs:
            new_obs.append(obs)

        # Update LogicalTrack with tagger features
        track = track_by_id.get(track_id)
        if track:
            _apply_tagger_to_track(track, features)

        processed += 1

    # Process fresh results
    for track_id, file_id, path, result in fresh_results:
        tagger_result = result["tagger_result"]
        features = _extract_tagger_features(tagger_result)

        # Save raw payload
        payload_path = os.path.join(raw_dir, f"{file_id}_analysis.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        # Librosa observation (from full tagger result)
        obs = _build_observation(track_id, file_id, "analysis_librosa", features)
        if obs:
            obs.payload_ref = f"analysis-librosa-{file_id}"
            new_obs.append(obs)

        new_payloads.append(PayloadIndexEntry(
            payload_ref=f"analysis-librosa-{file_id}",
            source_system="analysis_librosa",
            source_type="json",
            track_id=track_id,
            file_id=file_id,
            payload_path=os.path.relpath(payload_path, config.output_dir),
            fetched_at=now_iso(),
        ))

        # Essentia observation (if available)
        er = result.get("essentia")
        if er:
            parsed = parse_any_key(er["camelot"])
            if parsed:
                std, cam = parsed
                new_obs.append(SourceObservation(
                    observation_id=f"OBS-essentia-{file_id}",
                    track_id=track_id,
                    file_id=file_id,
                    source_system="analysis_essentia",
                    key_standard=std,
                    key_camelot=cam,
                    key_confidence=er["confidence"],
                    payload_ref=f"analysis-essentia-{file_id}",
                    observed_at=now_iso(),
                ))
            new_payloads.append(PayloadIndexEntry(
                payload_ref=f"analysis-essentia-{file_id}",
                source_system="analysis_essentia",
                source_type="json",
                track_id=track_id,
                file_id=file_id,
                payload_path=os.path.relpath(payload_path, config.output_dir),
                fetched_at=now_iso(),
            ))

        # Update LogicalTrack with tagger features
        track = track_by_id.get(track_id)
        if track:
            _apply_tagger_to_track(track, features)

        processed += 1

    # Save observations (replace old analysis obs to avoid duplicates)
    if new_obs:
        all_obs = store.load_observations()
        track_ids_processed = {o.track_id for o in new_obs}
        all_obs = [
            o for o in all_obs
            if not (o.source_system.startswith("analysis_") and o.track_id in track_ids_processed)
        ]
        all_obs.extend(new_obs)
        store.save_observations(all_obs)

    if new_payloads:
        all_payloads = store.load_payload_index()
        all_payloads.extend(new_payloads)
        store.save_payload_index(all_payloads)

    # Save tracks with tagger features
    store.save_tracks(tracks)

    n_failed = len(cache_misses) - len(fresh_results)
    logger.info("Analysis: %d processed (%d cached, %d analyzed, %d failed)",
                processed, len(cache_hits), len(fresh_results), n_failed)
    return {
        "total": processed,
        "cached": len(cache_hits),
        "analyzed": len(fresh_results),
        "failed": n_failed,
    }
