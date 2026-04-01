"""Track structure analysis: intro bar length + flow type."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from ..audio import TrackAudio
from ..constants import (
    FLOW_JUMP_DROP,
    FLOW_JUMP_HYPNOTIC,
    FLOW_PLATEAU_MIN,
    FLOW_TREND_BUILDER,
    FLOW_TREND_GROOVE,
    FLOW_VARIANCE_HYPNOTIC,
    FLOW_VARIANCE_LINEAR,
    INTRO_ENERGY_RATIO,
    INTRO_SUSTAIN_BARS,
    STANDARD_INTRO_BARS,
)

logger = logging.getLogger(__name__)

BEATS_PER_BAR = 4
N_SEGMENTS = 8


@dataclass
class StructureResult:
    intro_bars: int
    flow_type: str
    formatted: str
    confidence: float = 1.0


def _bar_energies(onset_env: NDArray, beat_frames: NDArray) -> NDArray:
    """Compute mean onset energy per bar (4 beats)."""
    beat_energies = []
    for i in range(len(beat_frames) - 1):
        seg = onset_env[beat_frames[i] : beat_frames[i + 1]]
        beat_energies.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)

    n_bars = len(beat_energies) // BEATS_PER_BAR
    if n_bars == 0:
        return np.array([])

    bar_e = []
    for i in range(n_bars):
        chunk = beat_energies[i * BEATS_PER_BAR : (i + 1) * BEATS_PER_BAR]
        bar_e.append(float(np.mean(chunk)))
    return np.array(bar_e)


def _detect_intro(bar_energies: NDArray) -> tuple[int, float]:
    """Find intro end bar and snap to nearest standard length.

    Returns (intro_bars, confidence).
    """
    if len(bar_energies) < INTRO_SUSTAIN_BARS:
        return STANDARD_INTRO_BARS[0], 0.3

    median_energy = float(np.median(bar_energies))
    threshold = INTRO_ENERGY_RATIO * median_energy

    intro_end = 0
    for i in range(len(bar_energies) - INTRO_SUSTAIN_BARS + 1):
        window = bar_energies[i : i + INTRO_SUSTAIN_BARS]
        if np.all(window > threshold):
            intro_end = i
            break

    snapped = min(STANDARD_INTRO_BARS, key=lambda x: abs(x - intro_end))
    # Confidence: how close raw intro was to snapped value
    snap_error = abs(intro_end - snapped)
    confidence = max(0.2, 1.0 - snap_error / 16.0)

    return snapped, confidence


def _count_plateaus(seg_norm: NDArray, tolerance: float = 0.08) -> int:
    """Count distinct plateau regions (consecutive segments within tolerance)."""
    if len(seg_norm) < 2:
        return 1
    plateaus = 1
    in_plateau = True
    for i in range(1, len(seg_norm)):
        if abs(seg_norm[i] - seg_norm[i - 1]) <= tolerance:
            if not in_plateau:
                plateaus += 1
                in_plateau = True
        else:
            in_plateau = False
    return plateaus


def _detect_flow_type(bar_energies: NDArray) -> str:
    """Classify the track's energy flow into G/H/D/B/L."""
    if len(bar_energies) < N_SEGMENTS:
        return "H"

    seg_len = len(bar_energies) // N_SEGMENTS
    seg_energies = np.array([
        float(np.mean(bar_energies[i * seg_len : (i + 1) * seg_len]))
        for i in range(N_SEGMENTS)
    ])

    seg_max = float(np.max(seg_energies))
    seg_norm = seg_energies / seg_max if seg_max > 0 else seg_energies

    variance = float(np.var(seg_norm))
    energy_diff = np.diff(seg_norm)
    max_jump = float(np.max(np.abs(energy_diff))) if len(energy_diff) > 0 else 0.0
    trend = float(np.polyfit(np.arange(len(seg_norm)), seg_norm, 1)[0])
    n_plateaus = _count_plateaus(seg_norm)

    logger.debug(
        "Flow metrics: var=%.4f jump=%.3f trend=%.3f plateaus=%d",
        variance, max_jump, trend, n_plateaus,
    )

    if variance < FLOW_VARIANCE_LINEAR:
        return "L"
    if variance < FLOW_VARIANCE_HYPNOTIC and max_jump < FLOW_JUMP_HYPNOTIC:
        return "H"
    if max_jump > FLOW_JUMP_DROP:
        return "D"
    if trend > FLOW_TREND_BUILDER and n_plateaus >= FLOW_PLATEAU_MIN:
        return "B"
    if trend > FLOW_TREND_GROOVE:
        return "G"
    return "H"


def analyze_structure(track_audio: TrackAudio) -> StructureResult:
    """Detect intro bar count and flow type."""
    onset_env = librosa.onset.onset_strength(y=track_audio.y, sr=track_audio.sr)
    bar_e = _bar_energies(onset_env, track_audio.beat_frames)

    if len(bar_e) == 0:
        logger.warning("No bars detected for %s, defaulting to 16H", track_audio.path)
        return StructureResult(intro_bars=16, flow_type="H", formatted="16H", confidence=0.2)

    intro_bars, intro_conf = _detect_intro(bar_e)
    flow_type = _detect_flow_type(bar_e)
    formatted = f"{intro_bars}{flow_type}"

    logger.debug("Structure: %s (intro=%d bars, flow=%s, conf=%.2f)", formatted, intro_bars, flow_type, intro_conf)
    return StructureResult(intro_bars=intro_bars, flow_type=flow_type, formatted=formatted, confidence=intro_conf)
