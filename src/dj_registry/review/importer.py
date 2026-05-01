"""Import manual review decisions from edited review_queue.csv."""

from __future__ import annotations

import logging

from ..key_utils import parse_any_key
from ..models import SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def import_reviews(store: CsvStore, *, show_progress: bool = False) -> tuple[int, int, int]:
    """Import review decisions from review_queue.csv.

    Returns (accepted, overridden, skipped).
    """
    queue = store.load_review_queue()
    tracks = store.load_tracks()
    track_by_id = {t.track_id: t for t in tracks}

    new_obs: list[SourceObservation] = []
    accepted = 0
    overridden = 0
    skipped = 0

    progress = ProgressBar(len(queue), label="Import reviews", enabled=show_progress)

    for index, item in enumerate(queue, start=1):
        decision = item.reviewer_decision.strip().lower()
        if not decision or decision == "skip":
            skipped += 1
            progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
            continue

        track = track_by_id.get(item.track_id)
        if not track:
            logger.warning("Review item %s: track %s not found", item.review_id, item.track_id)
            skipped += 1
            progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
            continue

        if decision == "accept":
            # Use the suggested key
            key_std = item.suggested_key_standard
            key_cam = item.suggested_key_camelot
            if not key_cam:
                logger.warning("Review item %s: no suggested key to accept", item.review_id)
                skipped += 1
                progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
                continue
            accepted += 1
        elif decision == "override":
            # Use reviewer's key
            raw = item.reviewer_key_camelot or item.reviewer_key_standard
            if not raw:
                logger.warning("Review item %s: override but no reviewer key provided", item.review_id)
                skipped += 1
                progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
                continue
            parsed = parse_any_key(raw)
            if not parsed:
                logger.warning("Review item %s: invalid reviewer key '%s'", item.review_id, raw)
                skipped += 1
                progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
                continue
            key_std, key_cam = parsed
            overridden += 1
        else:
            logger.warning("Review item %s: unknown decision '%s'", item.review_id, decision)
            skipped += 1
            progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)
            continue

        # Create manual observation
        new_obs.append(SourceObservation(
            observation_id=f"OBS-manual-{item.track_id}",
            track_id=item.track_id,
            source_system="manual",
            key_standard=key_std,
            key_camelot=key_cam,
            key_confidence=1.0,
            observed_at=now_iso(),
            notes=item.reviewer_note or f"decision={decision}",
        ))

        # Update track directly
        track.canonical_key_standard = key_std
        track.canonical_key_camelot = key_cam
        track.canonical_key_confidence = 1.0
        track.canonical_key_source = "manual"
        track.canonical_key_resolution_reason = "manual_override"
        track.needs_manual_review = False
        track.review_reason = ""
        track.last_resolved_at = now_iso()
        progress.update(index, accepted=accepted, overridden=overridden, skipped=skipped)

    if new_obs:
        store.add_observations(new_obs)
    store.save_tracks(tracks)
    progress.finish()

    logger.info("Review import: %d accepted, %d overridden, %d skipped", accepted, overridden, skipped)
    return accepted, overridden, skipped
