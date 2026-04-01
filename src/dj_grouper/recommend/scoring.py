"""Directional recommendation scoring for DJ usability."""

from __future__ import annotations

from ..config import GrouperConfig
from ..features.builder import TrackFeatures
from ..grouping.distance import blended_distance
from .transition import structure_compatibility

# Camelot helpers
_CAMELOT_POSITIONS: dict[str, int] = {}
for i in range(1, 13):
    _CAMELOT_POSITIONS[f"{i}A"] = i - 1
    _CAMELOT_POSITIONS[f"{i}B"] = i - 1 + 12


def _camelot_distance(key_a: str | None, key_b: str | None) -> float:
    """Distance on Camelot wheel. [0, 1]."""
    if not key_a or not key_b:
        return 0.0
    if key_a not in _CAMELOT_POSITIONS or key_b not in _CAMELOT_POSITIONS:
        return 0.5
    if key_a[-1] == key_b[-1]:
        num_a, num_b = int(key_a[:-1]), int(key_b[:-1])
        diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
        return diff / 6.0
    num_a, num_b = int(key_a[:-1]), int(key_b[:-1])
    if num_a == num_b:
        return 0.1
    return 0.7


def _bpm_penalty(a: TrackFeatures, b: TrackFeatures, config: GrouperConfig) -> float:
    """Soft BPM penalty. Returns 0 when close, scales up, hard cutoff at 8%.

    Replaces the old hard 4% filter. Allows good recommendations through when
    groove/structure are compatible, while still penalizing large gaps.
    """
    bpm_a = a.info.bpm or 128
    bpm_b = b.info.bpm or 128
    if bpm_a == 0 or bpm_b == 0:
        return 0.0

    pct_diff = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)

    # Hard cutoff: reject if > 8%
    if pct_diff > config.bpm_hard_cutoff_pct:
        return float("inf")

    # Soft penalty: starts at 4%, scales quadratically
    if pct_diff <= config.bpm_soft_penalty_pct:
        return 0.0

    excess = (pct_diff - config.bpm_soft_penalty_pct) / (config.bpm_hard_cutoff_pct - config.bpm_soft_penalty_pct)
    return config.bpm_penalty_weight * excess ** 2


def _key_penalty(a: TrackFeatures, b: TrackFeatures, config: GrouperConfig) -> float:
    """Key penalty, weighted by vibe and vocal presence."""
    d = _camelot_distance(a.info.key, b.info.key)
    vibe_a = a.info.vibe or "HYPN"
    kw = config.key_weight_by_vibe.get(vibe_a, 0.2)
    if a.info.vocal == "V":
        kw = min(1.0, kw + config.key_weight_vocal_boost)
    return d * kw * 0.15


def _struct_bonus(a: TrackFeatures, b: TrackFeatures) -> float:
    return structure_compatibility(
        a.info.flow_type, a.info.intro_bars,
        b.info.flow_type, b.info.intro_bars,
    )


# ─── DJ usability features (directional: source -> destination) ─────────────

def _intro_usability(dest: TrackFeatures) -> float:
    """How clean/usable is the destination's intro for mixing in?

    Longer intros with groove/hypnotic flow are more layerable.
    """
    bars = dest.info.intro_bars or 32
    flow = dest.info.flow_type or "H"

    bar_score = {16: 0.3, 32: 0.7, 64: 1.0}.get(bars, 0.5)
    flow_bonus = {"H": 0.15, "G": 0.10, "L": 0.15, "B": 0.0, "D": -0.10}.get(flow, 0.0)

    return bar_score + flow_bonus


def _bass_conflict_risk(source: TrackFeatures, dest: TrackFeatures) -> float:
    """Risk of bass clashing when layering. Higher = worse.

    Two tracks with very different low-end profiles clash more.
    """
    low_a = source.raw_dsp.get("low_freq_ratio", 0.0)
    low_b = dest.raw_dsp.get("low_freq_ratio", 0.0)
    # Both very bass-heavy = conflict risk
    if low_a > 0.3 and low_b > 0.3:
        return 0.1
    return 0.0


def _groove_compatibility(source: TrackFeatures, dest: TrackFeatures) -> float:
    """How well the grooves match. Uses section DSP if available."""
    # Compare groove-section onset density and beat strength
    s_groove = source.section_dsp.get("groove", source.raw_dsp)
    d_groove = dest.section_dsp.get("groove", dest.raw_dsp)

    onset_a = s_groove.get("onset_density", 0.0)
    onset_b = d_groove.get("onset_density", 0.0)
    beat_a = s_groove.get("beat_strength", 0.0)
    beat_b = d_groove.get("beat_strength", 0.0)

    # Similar groove density = good compatibility
    onset_sim = 1.0 - min(1.0, abs(onset_a - onset_b) / (max(onset_a, onset_b) + 1e-8))
    beat_sim = 1.0 - min(1.0, abs(beat_a - beat_b) / (max(beat_a, beat_b) + 1e-8))

    return (onset_sim + beat_sim) / 2.0 * 0.15


def _energy_direction_bonus(source: TrackFeatures, dest: TrackFeatures) -> float:
    """Small bonus/penalty for energy direction. Smooth transitions preferred."""
    e_a = source.info.energy or 3
    e_b = dest.info.energy or 3
    diff = e_b - e_a
    if diff == 0:
        return 0.05  # same energy = smooth
    if abs(diff) == 1:
        return 0.02  # one step = manageable
    return -0.05 * (abs(diff) - 1)  # jumps of 2+ = risky


def _breakdown_risk(dest: TrackFeatures) -> float:
    """Penalty if destination has unexpected breakdowns that interrupt the flow."""
    if dest.info.flow_type == "B":
        return -0.08
    return 0.0


# ─── Main scoring function ──────────────────────────────────────────────────

def recommend_score(
    source: TrackFeatures,
    dest: TrackFeatures,
    config: GrouperConfig,
) -> float:
    """Directional recommendation score: source -> dest.

    Combines similarity with DJ usability. Higher = better recommendation.
    Score is asymmetric: recommend_score(A, B) != recommend_score(B, A).
    """
    # BPM: soft penalty with hard cutoff
    bpm_pen = _bpm_penalty(source, dest, config)
    if bpm_pen == float("inf"):
        return -float("inf")

    # Base similarity (symmetric)
    similarity = 1.0 - blended_distance(source, dest, config)

    # Penalties and bonuses
    key_pen = _key_penalty(source, dest, config)
    struct_bon = _struct_bonus(source, dest)

    # DJ usability features (directional)
    intro_use = _intro_usability(dest) * 0.10
    bass_risk = _bass_conflict_risk(source, dest)
    groove_compat = _groove_compatibility(source, dest)
    energy_dir = _energy_direction_bonus(source, dest)
    breakdown_risk = _breakdown_risk(dest)

    score = (
        similarity
        - key_pen
        - bpm_pen
        - bass_risk
        + struct_bon
        + intro_use
        + groove_compat
        + energy_dir
        + breakdown_risk
    )
    return score
