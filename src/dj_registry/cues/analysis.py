"""CLI-facing cue analysis orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from dj_tagger.analyzers.sections import analyze_sections
from dj_tagger.audio import load_audio_features
from dj_tagger.raw_features import extract_raw_analysis

from ..config import RegistryConfig
from ..models import CuePoint
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .rekordbox import match_rekordbox_tracks, parse_rekordbox_tracks
from .selection import cue_grid_from_audio, select_default_hot_cues

logger = logging.getLogger(__name__)


@dataclass
class CueAnalysisStats:
    xml_tracks: int = 0
    matched: int = 0
    analyzed: int = 0
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
    show_progress: bool = False,
) -> CueAnalysisStats:
    """Generate v1 cue points for registry tracks present in a Rekordbox XML export."""
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
        cue.file_id for cue in existing_cues if cue.source_system == "auto_v1"
    }
    if force:
        cue_points = [
            cue for cue in existing_cues
            if not (cue.source_system == "auto_v1" and cue.file_id in selected_file_ids)
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
    new_cues: list[CuePoint] = []
    for index, match in enumerate(matches, start=1):
        frec = match.file_record
        label = frec.file_name or os.path.basename(frec.path_abs)
        if not force and frec.file_id in existing_auto_file_ids:
            stats.skipped_existing += 1
            progress.update(index, label, skipped=stats.skipped_existing)
            continue
        if not frec.path_abs or not os.path.exists(frec.path_abs):
            logger.warning("Cue analysis skipped missing file: %s", frec.path_abs)
            stats.failed += 1
            progress.update(index, f"{label} missing", failed=stats.failed)
            continue

        try:
            track_audio = load_audio_features(frec.path_abs)
            section_map = analyze_sections(track_audio)
            raw_analysis = extract_raw_analysis(track_audio)
            grid = cue_grid_from_audio(track_audio, section_map, raw_analysis)
            payload_ref = _write_analysis_payload(config, raw_dir, match, grid, raw_analysis)
            generated = select_default_hot_cues(
                frec.track_id,
                frec.file_id,
                grid,
                analysis_payload_ref=payload_ref,
            )
        except Exception as exc:
            logger.warning("Cue analysis failed for %s: %s", frec.path_abs, exc)
            stats.failed += 1
            progress.update(index, f"{label} failed", failed=stats.failed)
            continue

        new_cues.extend(generated)
        stats.analyzed += 1
        stats.cues_written += len(generated)
        progress.update(index, label, cues=stats.cues_written)

    progress.finish()
    cue_points.extend(new_cues)
    store.save_cue_points(cue_points)

    logger.info(
        "Cue analysis: %d XML tracks, %d matched, %d analyzed, %d cues, %d skipped, %d failed",
        stats.xml_tracks,
        stats.matched,
        stats.analyzed,
        stats.cues_written,
        stats.skipped_existing,
        stats.failed,
    )
    return stats


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
