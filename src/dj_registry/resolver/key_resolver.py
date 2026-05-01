"""Canonical key resolver — weighted multi-source scoring."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from ..config import RegistryConfig
from ..models import LogicalTrack, SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .explanation import (
    classify_resolution,
    format_evidence_summary,
    CONFLICT_REVIEW,
    LOW_CONFIDENCE_REVIEW,
    NO_ANALYSIS,
    NO_EVIDENCE,
    MANUAL_OVERRIDE,
    REVIEW_CONFLICT,
    REVIEW_LOW_CONFIDENCE,
    REVIEW_NO_ANALYSIS,
)

logger = logging.getLogger(__name__)


@dataclass
class KeyCandidate:
    """A candidate key with its aggregated score."""
    camelot: str
    standard: str
    score: float
    sources: list[str]  # source_system names that agree on this key
    weighted_base: float = 0.0  # base weighted score (for confidence calc)


def resolve_track_key(
    track: LogicalTrack,
    observations: list[SourceObservation],
    config: RegistryConfig,
) -> None:
    """Resolve the canonical key for a single track. Mutates the track in place."""
    key_obs = [o for o in observations if o.key_camelot]

    if not key_obs:
        track.canonical_key_standard = ""
        track.canonical_key_camelot = ""
        track.canonical_key_confidence = 0.0
        track.canonical_key_source = ""
        track.canonical_key_resolution_reason = NO_EVIDENCE
        track.key_evidence_summary = format_evidence_summary({}, None)
        track.needs_manual_review = True
        track.review_reason = REVIEW_NO_ANALYSIS
        return

    # Check for manual override
    manual_obs = [o for o in key_obs if o.source_system == "manual"]
    if manual_obs:
        mo = manual_obs[-1]  # Latest manual observation
        track.canonical_key_standard = mo.key_standard
        track.canonical_key_camelot = mo.key_camelot
        track.canonical_key_confidence = 1.0
        track.canonical_key_source = "manual"
        track.canonical_key_resolution_reason = MANUAL_OVERRIDE
        source_keys = {o.source_system: o.key_camelot for o in key_obs}
        track.key_evidence_summary = format_evidence_summary(source_keys, mo.key_camelot)
        track.needs_manual_review = False
        track.review_reason = ""
        track.last_resolved_at = now_iso()
        return

    # Group observations by Camelot key
    candidates: dict[str, KeyCandidate] = {}
    for o in key_obs:
        cam = o.key_camelot.upper()
        if cam not in candidates:
            candidates[cam] = KeyCandidate(
                camelot=cam,
                standard=o.key_standard,
                score=0.0,
                sources=[],
            )
        candidates[cam].sources.append(o.source_system)

    weights = config.source_weights

    # Score each candidate (for ranking — determines which key wins)
    # Confidence is computed separately as weighted vote share
    total_weighted = 0.0
    for cam, cand in candidates.items():
        base = 0.0
        for o in key_obs:
            if o.key_camelot.upper() == cam:
                w = weights.get(o.source_system, 0.5)
                base += w * o.key_confidence
        total_weighted += base

        # Ranking bonuses (don't affect confidence, only ordering)
        n_sources = len(cand.sources)
        agreement = config.agreement_boost * max(0, n_sources - 1)
        has_analysis = any(s.startswith("analysis_") for s in cand.sources)
        has_external = any(s in ("rekordbox", "songstats") for s in cand.sources)
        cross_boost = config.cross_type_boost if (has_analysis and has_external) else 0.0

        cand.score = base + agreement + cross_boost
        cand.weighted_base = base  # store for confidence calc

    # Confidence = weighted vote share of winning key (0.0 to 1.0)
    # All sources agree → 1.0, any disagreement → < 1.0
    def _confidence(cand: KeyCandidate) -> float:
        if total_weighted <= 0:
            return 0.0
        return round(cand.weighted_base / total_weighted, 3)

    # Tie-breaking priority: when scores are equal, prefer these sources
    _TIE_BREAK_PRIORITY = {"tag": 0, "rekordbox": 1, "songstats": 2, "analysis_librosa": 3, "analysis_essentia": 4}

    def _best_source_rank(sources: list[str]) -> int:
        return min(_TIE_BREAK_PRIORITY.get(s, 99) for s in sources)

    # Sort by: rounded score descending (avoids float noise), then tie-break priority
    ranked = sorted(
        candidates.values(),
        key=lambda c: (-round(c.score, 4), _best_source_rank(c.sources)),
    )
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    # Build evidence summary
    source_keys = {o.source_system: o.key_camelot for o in key_obs}
    has_conflict = len(ranked) > 1
    top_confidence = _confidence(top)

    # Auto-resolve logic:
    # 1. Majority wins (more than half of sources agree)
    # 2. Score threshold + margin met
    # 3. Scores are effectively equal but top has better source priority (tie-break)
    has_majority = len(top.sources) > len(key_obs) / 2
    score_sufficient = top.score >= config.confidence_threshold and (
        second is None or (top.score - second.score) >= config.margin_threshold
    )
    tie_broken = (
        second is not None
        and top.score >= config.confidence_threshold
        and round(top.score, 4) >= round(second.score, 4)
        and _best_source_rank(top.sources) < _best_source_rank(second.sources)
    )

    if has_majority or score_sufficient or tie_broken:
        track.canonical_key_standard = top.standard
        track.canonical_key_camelot = top.camelot
        track.canonical_key_confidence = top_confidence
        track.canonical_key_source = "+".join(sorted(set(top.sources)))
        track.canonical_key_resolution_reason = classify_resolution(top.sources, has_conflict)
        track.key_evidence_summary = format_evidence_summary(source_keys, top.camelot)
        track.needs_manual_review = False
        track.review_reason = ""
        track.last_resolved_at = now_iso()
    else:
        # Fallback: use the best-scoring candidate so downstream tools always
        # have a key. The track is flagged for manual review so the conflict is
        # visible, but an empty key is worse than a best-guess key.
        track.canonical_key_standard = top.standard
        track.canonical_key_camelot = top.camelot
        track.canonical_key_confidence = top_confidence
        track.canonical_key_source = "+".join(sorted(set(top.sources)))

        if not any(s.startswith("analysis_") for s in source_keys):
            track.canonical_key_resolution_reason = NO_ANALYSIS
            track.review_reason = REVIEW_NO_ANALYSIS
        elif has_conflict:
            track.canonical_key_resolution_reason = CONFLICT_REVIEW
            track.review_reason = REVIEW_CONFLICT
        else:
            track.canonical_key_resolution_reason = LOW_CONFIDENCE_REVIEW
            track.review_reason = REVIEW_LOW_CONFIDENCE

        track.key_evidence_summary = format_evidence_summary(source_keys, top.camelot)
        track.needs_manual_review = True
        track.last_resolved_at = now_iso()


def resolve_all_keys(
    config: RegistryConfig,
    store: CsvStore,
    *,
    force: bool = False,
    show_progress: bool = False,
) -> tuple[int, int]:
    """Resolve canonical keys for all tracks.

    Returns (resolved_count, review_count).
    """
    tracks = store.load_tracks()
    observations = store.load_observations()

    # Group observations by track
    obs_by_track: dict[str, list[SourceObservation]] = defaultdict(list)
    for o in observations:
        if o.track_id:
            obs_by_track[o.track_id].append(o)

    resolved = 0
    review = 0
    progress = ProgressBar(len(tracks), label="Resolve keys", enabled=show_progress)

    for index, track in enumerate(tracks, start=1):
        if not force and track.canonical_key_camelot and not track.needs_manual_review:
            # Already resolved and no new evidence needed
            resolved += 1
            progress.update(index, track.title_canonical, resolved=resolved, review=review)
            continue

        track_obs = obs_by_track.get(track.track_id, [])
        resolve_track_key(track, track_obs, config)

        if track.canonical_key_camelot:
            resolved += 1
        else:
            review += 1
        progress.update(index, track.title_canonical, resolved=resolved, review=review)

    progress.finish()
    store.save_tracks(tracks)
    logger.info("Resolve: %d keys resolved, %d need review", resolved, review)
    return resolved, review
