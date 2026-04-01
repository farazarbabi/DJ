"""Analysis pipeline: orchestrates all analyzers for a single track."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .audio import load_audio_features
from .analyzers.energy import EnergyResult, analyze_energy
from .analyzers.key import KeyResult, analyze_key
from .analyzers.sections import SectionMap, analyze_sections
from .analyzers.structure import StructureResult, analyze_structure
from .analyzers.vibe import VibeResult, analyze_vibe
from .analyzers.vocal import VocalResult, analyze_vocal
from .formats import format_tag
from .metadata import read_existing_tag, write_tag

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    dry_run: bool = True
    overwrite: bool = False
    use_essentia: bool = False
    verbose: bool = False
    max_duration: float | None = None


def analyze_track(path: str, config: AnalysisConfig) -> dict:
    """Run the full analysis pipeline on a single track.

    Returns a dict suitable for CSV/JSON export, including vibe_scores,
    confidences, and section info for downstream grouping.
    """
    # Check for existing tag
    if not config.overwrite:
        existing = read_existing_tag(path)
        if existing:
            logger.info("Skipping %s (already tagged: %s)", path, existing)
            return {
                "file": path,
                "tag": existing,
                "skipped": True,
            }

    # Load audio and precompute shared features
    track_audio = load_audio_features(path, max_duration=config.max_duration)

    # Run each analyzer independently; catch failures per-analyzer
    energy_result = _safe_analyze("energy", analyze_energy, track_audio)
    key_result = _safe_analyze("key", analyze_key, track_audio, config.use_essentia)
    structure_result = _safe_analyze("structure", analyze_structure, track_audio)
    vibe_result = _safe_analyze("vibe", analyze_vibe, track_audio)
    vocal_result = _safe_analyze("vocal", analyze_vocal, track_audio)
    section_map = _safe_analyze("sections", analyze_sections, track_audio)

    # Build tag string
    bpm_rounded = round(track_audio.tempo) if track_audio.tempo > 0 else None
    tag = format_tag(
        energy=energy_result.level if energy_result else None,
        camelot=key_result.camelot if key_result else None,
        bpm=bpm_rounded,
        structure=structure_result.formatted if structure_result else None,
        vibe=vibe_result.label if vibe_result else None,
        has_vocals=vocal_result.has_vocals if vocal_result else None,
    )

    # Write tag
    write_tag(path, tag, dry_run=config.dry_run)

    # Collect confidences
    confidences = {
        "energy": energy_result.confidence if energy_result else 0.0,
        "key": key_result.confidence if key_result else 0.0,
        "structure": structure_result.confidence if structure_result else 0.0,
        "vibe": vibe_result.confidence if vibe_result else 0.0,
        "vocal": vocal_result.confidence if vocal_result else 0.0,
    }

    result = {
        "file": path,
        "tag": tag,
        "bpm": round(track_audio.tempo, 1),
        "energy": energy_result.level if energy_result else None,
        "key": key_result.key_name if key_result else None,
        "camelot": key_result.camelot if key_result else None,
        "structure": structure_result.formatted if structure_result else None,
        "vibe": vibe_result.label if vibe_result else None,
        "vocal": ("V" if vocal_result.has_vocals else "NV") if vocal_result else None,
        "key_confidence": round(key_result.confidence, 3) if key_result else None,
        "vocal_ratio": round(vocal_result.vocal_ratio, 3) if vocal_result else None,
        # New: continuous vibe scores and confidences for grouper
        "vibe_scores": {k: round(v, 4) for k, v in vibe_result.scores.items()} if vibe_result else {},
        "confidences": {k: round(v, 3) for k, v in confidences.items()},
        # Section info
        "sections": [
            {"label": s.label, "start": s.start_bar, "end": s.end_bar, "energy": round(s.energy, 3)}
            for s in (section_map.sections if section_map else [])
        ],
    }

    if config.verbose:
        if energy_result:
            result["energy_details"] = {
                k: round(v, 4) for k, v in energy_result.details.items()
            }
        if structure_result:
            result["intro_bars"] = structure_result.intro_bars
            result["flow_type"] = structure_result.flow_type

    return result


def _safe_analyze(name: str, fn, *args):
    """Call an analyzer and return None on failure instead of crashing."""
    try:
        return fn(*args)
    except Exception:
        logger.warning("Analyzer '%s' failed", name, exc_info=True)
        return None
