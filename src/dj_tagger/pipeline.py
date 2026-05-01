"""Analysis pipeline: orchestrates all analyzers for a single track."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .audio import load_audio_features
from .formats import format_tag
from .metadata import read_existing_tag, write_tag
from .raw_features import compute_tagger_artifacts

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    dry_run: bool = True
    overwrite: bool = False
    use_essentia: bool = False
    verbose: bool = False
    max_duration: float | None = None
    audio_features: dict[str, float] | None = None


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

    artifacts = compute_tagger_artifacts(
        track_audio,
        audio_features=config.audio_features,
        use_essentia=config.use_essentia,
    )
    canonical = artifacts["tagger_result"]

    # Build tag string
    bpm_rounded = round(canonical["bpm"]) if canonical.get("bpm") else None
    tag = format_tag(
        energy=canonical.get("energy"),
        camelot=canonical.get("camelot"),
        bpm=bpm_rounded,
        structure=canonical.get("structure"),
        vibe=canonical.get("vibe"),
        has_vocals=canonical.get("has_vocals"),
        vocal_profile=canonical.get("vocal_profile", canonical.get("vocal")),
    )

    # Write tag
    write_tag(path, tag, dry_run=config.dry_run)

    result = {
        "file": path,
        "tag": tag,
        "bpm": canonical.get("bpm"),
        "energy": canonical.get("energy"),
        "key": canonical.get("key"),
        "camelot": canonical.get("camelot"),
        "structure": canonical.get("structure"),
        "vibe": canonical.get("vibe"),
        "mood": canonical.get("mood", canonical.get("vibe")),
        "vocal": canonical.get("vocal"),
        "vocal_profile": canonical.get("vocal_profile", canonical.get("vocal")),
        "key_confidence": canonical.get("key_confidence"),
        "vocal_ratio": canonical.get("vocal_ratio"),
        "vocal_scores": canonical.get("vocal_scores", {}),
        "vibe_scores": canonical.get("vibe_scores", {}),
        "mood_scores": canonical.get("mood_scores", canonical.get("vibe_scores", {})),
        "confidences": canonical.get("confidences", {}),
        "sections": canonical.get("sections", []),
    }

    if config.verbose:
        result["intro_bars"] = canonical.get("intro_bars")
        result["flow_type"] = canonical.get("flow_type")

    return result


def _safe_analyze(name: str, fn, *args):
    """Call an analyzer and return None on failure instead of crashing."""
    try:
        return fn(*args)
    except Exception:
        logger.warning("Analyzer '%s' failed", name, exc_info=True)
        return None
