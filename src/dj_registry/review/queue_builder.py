"""Generate review_queue.csv from unresolved tracks."""

from __future__ import annotations

import logging
from collections import defaultdict

from ..config import RegistryConfig
from ..models import ReviewItem, SourceObservation, now_iso
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def build_review_queue(
    config: RegistryConfig,
    store: CsvStore,
) -> int:
    """Generate or refresh review_queue.csv for unresolved tracks.

    Returns number of review items created.
    """
    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()

    file_by_id = {f.file_id: f for f in files}

    # Group observations with key data by track
    obs_by_track: dict[str, list[SourceObservation]] = defaultdict(list)
    for o in observations:
        if o.track_id and o.key_camelot:
            obs_by_track[o.track_id].append(o)

    items: list[ReviewItem] = []

    for track in tracks:
        if not track.needs_manual_review:
            continue

        key_obs = obs_by_track.get(track.track_id, [])

        # Score candidates (same logic as resolver but simplified for display)
        candidates: dict[str, tuple[float, str, list[str]]] = {}
        for o in key_obs:
            cam = o.key_camelot.upper()
            if cam not in candidates:
                candidates[cam] = (0.0, o.key_standard, [])
            weight = config.source_weights.get(o.source_system, 0.5)
            old_score, std, sources = candidates[cam]
            sources.append(o.source_system)
            candidates[cam] = (old_score + weight * o.key_confidence, std, sources)

        ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)

        # Get primary file info
        primary_file = file_by_id.get(track.primary_file_id)
        primary_path = primary_file.path_abs if primary_file else ""

        # Determine priority
        if track.review_reason == "source_conflict":
            priority = "high"
        elif track.review_reason == "no_analysis":
            priority = "low"
        else:
            priority = "medium"

        item = ReviewItem(
            review_id=f"REV-{track.track_id}",
            track_id=track.track_id,
            priority=priority,
            reason_code=track.review_reason,
            reason_detail=track.canonical_key_resolution_reason,
            primary_file_id=track.primary_file_id,
            primary_file_path=primary_path,
            key_evidence_summary=track.key_evidence_summary,
            created_at=now_iso(),
        )

        # Fill candidate slots (up to 3)
        for i, (cam, (score, std, sources)) in enumerate(ranked[:3]):
            idx = i + 1
            setattr(item, f"candidate_key_standard_{idx}", std)
            setattr(item, f"candidate_key_camelot_{idx}", cam)
            setattr(item, f"candidate_source_{idx}", "+".join(sources))
            setattr(item, f"candidate_score_{idx}", round(score, 3))

        # Pre-fill suggestion with top candidate
        if ranked:
            cam, (score, std, sources) = ranked[0]
            item.suggested_key_standard = std
            item.suggested_key_camelot = cam

        items.append(item)

    # Sort: high priority first, then by track_id
    priority_order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda x: (priority_order.get(x.priority, 1), x.track_id))

    store.save_review_queue(items)
    logger.debug("Review queue: %d items", len(items))
    return len(items)
