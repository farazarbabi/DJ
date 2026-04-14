"""Local audio analysis adapter — wraps dj_tagger key analyzers.

Shares the tagger cache (outputs/tagger_cache.pkl) so files analyzed by
dj-tagger are never re-analyzed by the registry and vice versa.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from dj_tagger.cache import (
    ANALYZER_VERSION,
    TaggerCache,
    cache_key,
    get_cached,
    load_cache,
    put_cached,
    save_cache,
)

from ..config import RegistryConfig
from ..key_utils import parse_any_key
from ..models import SourceObservation, PayloadIndexEntry, now_iso
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)

TAGGER_CACHE_PATH = os.path.join("cache", "tagger_cache.pkl")


def _analyze_single(path: str, use_essentia: bool) -> dict:
    """Run key analysis on a single file. Top-level for pickling."""
    from dj_tagger.audio import load_audio_features
    from dj_tagger.analyzers.key import analyze_key

    result: dict = {"path": path, "librosa": None, "essentia": None}

    try:
        audio = load_audio_features(path)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Librosa analysis (always)
    try:
        kr = analyze_key(audio, use_essentia=False)
        result["librosa"] = {
            "camelot": kr.camelot,
            "key_name": kr.key_name,
            "confidence": kr.confidence,
        }
    except Exception as e:
        result["librosa_error"] = str(e)

    # Essentia analysis (optional)
    if use_essentia:
        try:
            kr = analyze_key(audio, use_essentia=True)
            result["essentia"] = {
                "camelot": kr.camelot,
                "key_name": kr.key_name,
                "confidence": kr.confidence,
            }
        except Exception as e:
            result["essentia_error"] = str(e)

    return result


def _essentia_available() -> bool:
    """Check if Essentia is installed."""
    try:
        import essentia  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_key_from_tagger_result(result: dict) -> dict | None:
    """Extract key analysis data from a tagger cache result dict."""
    camelot = result.get("camelot")
    key_name = result.get("key")
    confidence = result.get("key_confidence", 0.0)
    if not camelot:
        return None
    return {
        "camelot": camelot,
        "key_name": key_name or "",
        "confidence": confidence if confidence else 0.0,
    }


def run_analysis(
    config: RegistryConfig,
    store: CsvStore,
    *,
    track_ids: list[str] | None = None,
    no_essentia: bool = False,
) -> int:
    """Run local key analysis on tracks.

    Shares the tagger cache (outputs/tagger_cache.pkl) with dj-tagger.
    Files already analyzed by dj-tagger are read from cache instantly.
    New analysis results are written back so dj-tagger can reuse them too.

    Returns number of tracks processed.
    """
    use_essentia = config.run_essentia and not no_essentia and _essentia_available()
    if config.run_essentia and not no_essentia and not _essentia_available():
        logger.debug("Essentia not installed, running librosa only")

    files = store.load_files()
    tracks = store.load_tracks()

    file_by_id = {f.file_id: f for f in files}

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
        logger.info("Analysis: all tracks cached")
        return 0

    # Load shared tagger cache
    tagger_cache = load_cache(TAGGER_CACHE_PATH)

    # Split into cache hits and misses
    cache_hits: list[tuple[str, str, str, dict]] = []
    cache_misses: list[tuple[str, str, str, float, float]] = []

    for track_id, file_id, path, mtime, duration in candidates:
        cached_result = get_cached(tagger_cache, path, mtime, duration=duration)
        if cached_result:
            key_data = _extract_key_from_tagger_result(cached_result)
            if key_data:
                cache_hits.append((track_id, file_id, path, key_data))
                continue
        cache_misses.append((track_id, file_id, path, mtime, duration))

    if cache_misses:
        logger.info("Analysis: %d tracks (%d cached, %d to analyze)", len(candidates), len(cache_hits), len(cache_misses))
    else:
        logger.info("Analysis: %d tracks (all cached)", len(candidates))

    # Run analysis on cache misses
    fresh_results: list[tuple[str, str, str, dict | None]] = []
    if cache_misses:
        paths_to_analyze = [path for _, _, path, _ in cache_misses]

        if config.analysis_workers <= 1:
            raw_results = [_analyze_single(p, use_essentia) for p in paths_to_analyze]
        else:
            raw_results = [None] * len(cache_misses)
            with ProcessPoolExecutor(max_workers=config.analysis_workers) as pool:
                futures = {
                    pool.submit(_analyze_single, p, use_essentia): i
                    for i, p in enumerate(paths_to_analyze)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as e:
                        raw_results[idx] = {"path": paths_to_analyze[idx], "error": str(e)}

        # Store new results in tagger cache
        for (track_id, file_id, path, mtime, duration), result in zip(cache_misses, raw_results):
            if result and "error" not in result:
                # Build a tagger-compatible result dict for the cache
                lr = result.get("librosa")
                if lr:
                    tagger_result = {
                        "camelot": lr["camelot"],
                        "key": lr["key_name"],
                        "key_confidence": lr["confidence"],
                    }
                    put_cached(tagger_cache, path, mtime, tagger_result, duration=duration)
                fresh_results.append((track_id, file_id, path, result))
            else:
                err = result.get("error", "unknown") if result else "unknown"
                logger.warning("Analysis failed for %s: %s", path, err)

        save_cache(tagger_cache, TAGGER_CACHE_PATH)

    # Build observations from all results
    raw_dir = os.path.join(config.raw_dir, "analysis")
    os.makedirs(raw_dir, exist_ok=True)

    new_obs: list[SourceObservation] = []
    new_payloads: list[PayloadIndexEntry] = []
    processed = 0

    # Process cache hits (librosa only — tagger cache is librosa-based)
    for track_id, file_id, path, key_data in cache_hits:
        parsed = parse_any_key(key_data["camelot"])
        if parsed:
            std, cam = parsed
            new_obs.append(SourceObservation(
                observation_id=f"OBS-librosa-{file_id}",
                track_id=track_id,
                file_id=file_id,
                source_system="analysis_librosa",
                key_standard=std,
                key_camelot=cam,
                key_confidence=key_data["confidence"],
                payload_ref=f"analysis-librosa-{file_id}",
                observed_at=now_iso(),
            ))
        processed += 1

    # Process fresh results
    for track_id, file_id, path, result in fresh_results:
        payload_path = os.path.join(raw_dir, f"{file_id}_analysis.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        lr = result.get("librosa")
        if lr:
            payload_ref = f"analysis-librosa-{file_id}"
            parsed = parse_any_key(lr["camelot"])
            if parsed:
                std, cam = parsed
                new_obs.append(SourceObservation(
                    observation_id=f"OBS-librosa-{file_id}",
                    track_id=track_id,
                    file_id=file_id,
                    source_system="analysis_librosa",
                    key_standard=std,
                    key_camelot=cam,
                    key_confidence=lr["confidence"],
                    payload_ref=payload_ref,
                    observed_at=now_iso(),
                ))
            new_payloads.append(PayloadIndexEntry(
                payload_ref=payload_ref,
                source_system="analysis_librosa",
                source_type="json",
                track_id=track_id,
                file_id=file_id,
                payload_path=os.path.relpath(payload_path, config.output_dir),
                fetched_at=now_iso(),
            ))

        er = result.get("essentia")
        if er:
            payload_ref = f"analysis-essentia-{file_id}"
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
                    payload_ref=payload_ref,
                    observed_at=now_iso(),
                ))
            new_payloads.append(PayloadIndexEntry(
                payload_ref=payload_ref,
                source_system="analysis_essentia",
                source_type="json",
                track_id=track_id,
                file_id=file_id,
                payload_path=os.path.relpath(payload_path, config.output_dir),
                fetched_at=now_iso(),
            ))

        processed += 1

    # Save observations
    if new_obs:
        # Remove old analysis observations first to avoid duplicates on re-run
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

    logger.debug("Analysis done: %d processed (%d cached, %d fresh)", processed, len(cache_hits), len(fresh_results))
    return processed
