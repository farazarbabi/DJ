"""CLI-facing cue analysis orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from dj_tagger import universal_cache as cache_mod
from dj_tagger.analyzers.sections import analyze_sections
from dj_tagger.audio import load_audio_features
from dj_tagger.raw_features import extract_raw_analysis

from ..config import RegistryConfig
from ..models import CuePoint
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .profiles import load_cue_profile
from .rekordbox import match_rekordbox_tracks, parse_rekordbox_tracks
from .selection import CueGrid, CueSection, cue_grid_from_audio, select_profile_cues

logger = logging.getLogger(__name__)

CUE_ANALYSIS_CACHE_LAYER = "cue_analysis"
CUE_ANALYSIS_CHECKPOINT_EVERY = 10


@dataclass
class CueAnalysisStats:
    xml_tracks: int = 0
    matched: int = 0
    analyzed: int = 0
    cached: int = 0
    skipped_existing: int = 0
    failed: int = 0
    cues_written: int = 0
    unmatched: int = 0
    cue_points_path: str = ""


def analyze_rekordbox_cues(
    config: RegistryConfig,
    store: CsvStore,
    *,
    limit: int | None = None,
    force: bool = False,
    profile: str = "v1",
    profile_file: str | None = None,
    include_memory: bool = False,
    include_loops: bool = False,
    loop_bars: int | None = 16,
    force_analysis: bool = False,
    show_progress: bool = False,
) -> CueAnalysisStats:
    """Generate cue points for registry tracks present in a Rekordbox XML export."""
    cue_profile = load_cue_profile(
        profile,
        profile_file=profile_file,
        include_memory=include_memory,
        include_loops=include_loops,
        loop_bars=loop_bars,
    )
    xml_path = config.rekordbox_xml_path
    if not xml_path or not os.path.exists(xml_path):
        raise FileNotFoundError(f"Rekordbox XML not found: {xml_path}")

    refs = parse_rekordbox_tracks(xml_path, config.path_prefix_map, min_duration_sec=30.0)
    all_matches = match_rekordbox_tracks(refs, store)
    matches = all_matches
    if limit is not None:
        matches = matches[:max(0, limit)]

    selected_file_ids = {match.file_id for match in matches}
    existing_cues = store.load_cue_points()
    existing_auto_file_ids = {
        cue.file_id for cue in existing_cues if _is_auto_cue(cue)
    }
    force_generated_rows = force or force_analysis
    if force_generated_rows:
        cue_points = [
            cue for cue in existing_cues
            if not (_is_auto_cue(cue) and cue.file_id in selected_file_ids)
        ]
    else:
        cue_points = list(existing_cues)

    raw_dir = os.path.join(config.raw_dir, "cue_analysis")
    os.makedirs(raw_dir, exist_ok=True)

    stats = CueAnalysisStats(
        xml_tracks=len(refs),
        matched=len(all_matches),
        unmatched=max(0, len(refs) - len(all_matches)),
        cue_points_path=os.path.join(store.output_dir, "cue_points_master.csv"),
    )

    progress = ProgressBar(len(matches), label="Cue analysis", enabled=show_progress)
    ucache = cache_mod.get_cache(config.raw_cache_path)
    since_cache_checkpoint = 0
    for index, match in enumerate(matches, start=1):
        frec = match.file_record
        label = frec.file_name or os.path.basename(frec.path_abs)
        if not force_generated_rows and frec.file_id in existing_auto_file_ids:
            stats.skipped_existing += 1
            progress.update(index, label, skipped=stats.skipped_existing)
            continue
        if not frec.path_abs or not os.path.exists(frec.path_abs):
            logger.warning("Cue analysis skipped missing file: %s", frec.path_abs)
            stats.failed += 1
            progress.update(index, f"{label} missing", failed=stats.failed)
            continue

        try:
            cache_filename = frec.file_name or os.path.basename(frec.path_abs)
            cache_duration = _cache_duration_for_file(frec, fallback=match.ref.total_time_sec)
            cached = None if force_analysis else ucache.get_track(
                cache_filename,
                cache_duration,
                CUE_ANALYSIS_CACHE_LAYER,
            )
            cached_analysis = _cue_analysis_from_cache(cached)
            if cached_analysis is not None:
                grid, raw_analysis = cached_analysis
                stats.cached += 1
            else:
                track_audio = load_audio_features(frec.path_abs)
                section_map = analyze_sections(track_audio)
                raw_analysis = extract_raw_analysis(track_audio)
                grid = cue_grid_from_audio(track_audio, section_map, raw_analysis)
                cache_duration = (
                    cache_mod.quick_duration(frec.path_abs)
                    or frec.audio_duration_sec
                    or match.ref.total_time_sec
                    or float(getattr(track_audio, "duration", 0.0) or 0.0)
                    or None
                )
                _put_cue_analysis_cache(
                    ucache,
                    cache_filename,
                    cache_duration,
                    _cue_analysis_cache_payload(grid, raw_analysis),
                    force=force_analysis,
                    mtime=_file_mtime(frec.path_abs),
                )
                since_cache_checkpoint += 1
                stats.analyzed += 1
                if since_cache_checkpoint >= CUE_ANALYSIS_CHECKPOINT_EVERY:
                    ucache.save()
                    logger.info(
                        "Cue analysis checkpoint: saved %d fresh cue-analysis cache entries",
                        since_cache_checkpoint,
                    )
                    since_cache_checkpoint = 0
            payload_ref = _write_analysis_payload(config, raw_dir, match, grid, raw_analysis)
            generated = select_profile_cues(
                frec.track_id,
                frec.file_id,
                grid,
                profile=cue_profile.name,
                loop_bars=cue_profile.loop_bars,
                profile_config=cue_profile,
                analysis_payload_ref=payload_ref,
            )
        except Exception as exc:
            logger.warning("Cue analysis failed for %s: %s", frec.path_abs, exc)
            stats.failed += 1
            progress.update(index, f"{label} failed", failed=stats.failed)
            continue

        cue_points.extend(generated)
        stats.cues_written += len(generated)
        progress.update(index, label, cues=stats.cues_written, cached=stats.cached)

    progress.finish()
    if ucache.dirty:
        ucache.save()
    store.save_cue_points(cue_points)

    logger.info(
        "Cue analysis: %d XML tracks, %d matched, %d analyzed, %d cached, %d cues, %d skipped, %d failed",
        stats.xml_tracks,
        stats.matched,
        stats.analyzed,
        stats.cached,
        stats.cues_written,
        stats.skipped_existing,
        stats.failed,
    )
    return stats


def _is_auto_cue(cue: CuePoint) -> bool:
    return str(cue.source_system or "").startswith("auto_")


def _cache_duration_for_file(frec, *, fallback: float = 0.0) -> float | None:
    return cache_mod.quick_duration(frec.path_abs) or frec.audio_duration_sec or fallback or None


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _put_cue_analysis_cache(
    ucache,
    filename: str,
    duration: float | None,
    payload: dict,
    *,
    force: bool,
    mtime: float,
) -> None:
    if force:
        key = ucache.track_key(filename, duration, CUE_ANALYSIS_CACHE_LAYER)
        if key in ucache._entries:
            del ucache._entries[key]
            ucache._dirty_raw = True
    ucache.put_track(
        filename,
        duration,
        CUE_ANALYSIS_CACHE_LAYER,
        payload,
        mtime=mtime,
    )


def _cue_analysis_cache_payload(grid: CueGrid, raw_analysis: dict | None) -> dict:
    raw_analysis = raw_analysis if isinstance(raw_analysis, dict) else {}
    return {
        "schema": "cue_analysis_v1",
        "grid": {
            "duration_sec": float(grid.duration_sec),
            "beat_times": [float(v) for v in grid.beat_times],
            "bar_times": [float(v) for v in grid.bar_times],
            "bar_energies": [float(v) for v in grid.bar_energies],
            "sections": [asdict(section) for section in grid.sections],
        },
        "raw_analysis": {
            "tempo": raw_analysis.get("tempo", ""),
        },
    }


def _cue_analysis_from_cache(data) -> tuple[CueGrid, dict] | None:
    if not isinstance(data, dict):
        return None
    grid_data = data.get("grid")
    if not isinstance(grid_data, dict):
        return None
    try:
        sections = [
            CueSection(
                label=str(section.get("label", "")),
                start_bar=int(section.get("start_bar", 0) or 0),
                end_bar=int(section.get("end_bar", 0) or 0),
                energy=float(section.get("energy", 0.0) or 0.0),
            )
            for section in grid_data.get("sections", []) or []
            if isinstance(section, dict)
        ]
        grid = CueGrid(
            duration_sec=float(grid_data.get("duration_sec", 0.0) or 0.0),
            beat_times=_float_list(grid_data.get("beat_times")),
            bar_times=_float_list(grid_data.get("bar_times")),
            bar_energies=_float_list(grid_data.get("bar_energies")),
            sections=sections,
        )
    except (TypeError, ValueError):
        return None
    if grid.duration_sec <= 0 and grid.n_bars <= 0:
        return None
    raw_analysis = data.get("raw_analysis")
    return grid, raw_analysis if isinstance(raw_analysis, dict) else {}


def _float_list(values) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    return [float(value) for value in values]


def _write_analysis_payload(config: RegistryConfig, raw_dir: str, match, grid, raw_analysis: dict) -> str:
    payload_path = os.path.join(raw_dir, f"{match.file_id}_cue_analysis.json")
    payload = {
        "rekordbox_track": asdict(match.ref),
        "match_method": match.match_method,
        "track_id": match.track_id,
        "file_id": match.file_id,
        "duration_sec": round(grid.duration_sec, 3),
        "n_bars": grid.n_bars,
        "beat_count": len(grid.beat_times),
        "bar_times": [round(v, 3) for v in grid.bar_times],
        "bar_energies": [round(float(v), 6) for v in grid.bar_energies],
        "sections": [asdict(section) for section in grid.sections],
        "raw_tempo": raw_analysis.get("tempo", ""),
    }
    with open(payload_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return os.path.relpath(payload_path, config.output_dir)
