"""Full pipeline orchestrator: scan -> link -> ingest -> analyze -> resolve -> sync."""

from __future__ import annotations

import logging
import uuid

from ..adapters.file_scanner import scan_files
from ..adapters.local_analysis import run_analysis
from ..adapters.rekordbox_xml import ingest_rekordbox
from ..adapters.songstats import ingest_songstats
from ..adapters.spotify_isrc import enrich_isrcs
from ..config import RegistryConfig
from ..identity.matcher import link_files_to_tracks
from ..resolver.bpm_resolver import resolve_all_bpms
from ..resolver.key_resolver import resolve_all_keys
from ..review.queue_builder import build_review_queue
from ..store.csv_store import CsvStore
from ..store.obs_cache import ObsCache
from ..sync.export import generate_reports
from ..sync.tag_writer import sync_tags
from ..taxonomy.classifier import classify_all_taxonomies

logger = logging.getLogger(__name__)


def _enrich_observations(store: CsvStore) -> None:
    """Fill artist/title on all observations from tracks_master for easy review."""
    tracks = store.load_tracks()
    track_lookup = {t.track_id: t for t in tracks}
    obs = store.load_observations()
    changed = False
    for o in obs:
        t = track_lookup.get(o.track_id)
        if t and (o.artist != t.artist_canonical or o.title != t.title_canonical):
            o.artist = t.artist_canonical
            o.title = t.title_canonical
            changed = True
    if changed:
        store.save_observations(obs)


def _classify_dj_taxonomy_for_tags(store: CsvStore, *, show_progress: bool = False) -> int:
    """Populate DJ category fields before COMMENT tag sync when models exist."""
    try:
        from ..taxonomy.dj_model import classify_all_dj_taxonomies

        model_dir = _dj_taxonomy_model_dir_for_tags(store)
        if not model_dir:
            logger.info("DJ taxonomy: skipped (no model found)")
            return 0
        return classify_all_dj_taxonomies(
            store,
            model_dir=model_dir,
            primary_model="internal",
            show_progress=show_progress,
        )
    except FileNotFoundError as exc:
        logger.info("DJ taxonomy: skipped (%s)", exc)
        return 0


def _dj_taxonomy_model_dir_for_tags(store: CsvStore) -> str | None:
    """Prefer the active registry model, then the project-level trained model."""
    from pathlib import Path

    active = Path(store.output_dir) / "dj_taxonomy_model"
    if (active / "internal").exists() or (active / "external").exists():
        return str(active)
    fallback = Path("outputs") / "registry" / "dj_taxonomy_model"
    if (fallback / "internal").exists() or (fallback / "external").exists():
        return str(fallback)
    return None


def run_full_pipeline(
    config: RegistryConfig,
    *,
    dry_run: bool = True,
    write_tags: bool = False,
    include_rekordbox: bool = False,
    include_songstats: bool = False,
    songstats_limit: int | None = None,
    no_essentia: bool = False,
    analysis_workers: int | None = None,
    show_progress: bool = False,
) -> dict:
    """Run the full registry pipeline.

    pkl cache is the source of truth for collected data.
    CSV observations are rebuilt fresh every run from cache + new data.

    Returns a summary dict with counts.
    """
    run_id = uuid.uuid4().hex[:8]
    store = CsvStore(config.output_dir)
    obs_cache = ObsCache()
    store.snapshot(run_id)
    summary: dict = {"run_id": run_id}

    if analysis_workers is not None:
        config.analysis_workers = analysis_workers

    # Clear stale observations — they'll be rebuilt from cache
    store.save_observations([])

    files = scan_files(config, store, obs_cache=obs_cache, show_progress=show_progress)
    summary["files_scanned"] = len(files)

    link_files_to_tracks(config, store, show_progress=show_progress)
    tracks = store.load_tracks()
    summary["tracks_total"] = len(tracks)

    if include_rekordbox and config.rekordbox_xml_path:
        summary["rekordbox_matched"] = ingest_rekordbox(
            config,
            store,
            obs_cache=obs_cache,
            show_progress=show_progress,
        )
    else:
        summary["rekordbox_matched"] = 0

    if include_songstats:
        summary["isrcs_enriched"] = enrich_isrcs(store, show_progress=show_progress)
        ss_stats = ingest_songstats(
            config, store, obs_cache=obs_cache, limit=songstats_limit, show_progress=show_progress
        )
        summary["songstats_fetched"] = ss_stats["total"]
        summary["songstats_cached"] = ss_stats["cached"]
    else:
        summary["isrcs_enriched"] = 0
        summary["songstats_fetched"] = 0

    # Flush Songstats (and any other ISRC-keyed data) to the universal cache
    # before analysis so the tagger can look up audio features during vibe scoring.
    obs_cache.save()

    analysis_stats = run_analysis(config, store, no_essentia=no_essentia, show_progress=show_progress)
    summary["tracks_analyzed"] = analysis_stats["total"]
    summary["tracks_analyzed_cached"] = analysis_stats["cached"]

    _enrich_observations(store)

    resolved, review = resolve_all_keys(config, store, force=True, show_progress=show_progress)
    summary["keys_resolved"] = resolved
    summary["keys_need_review"] = review

    bpm_resolved, bpm_missing = resolve_all_bpms(config, store, force=True, show_progress=show_progress)
    summary["bpm_resolved"] = bpm_resolved
    summary["bpm_missing"] = bpm_missing
    summary["review_items"] = build_review_queue(config, store, show_progress=show_progress)
    summary["dj_taxonomy_classified"] = _classify_dj_taxonomy_for_tags(store, show_progress=show_progress)

    if write_tags and not dry_run:
        written, _, errors = sync_tags(store, dry_run=False, show_progress=show_progress)
        summary["tags_written"] = written
        summary["tags_errors"] = errors
    else:
        sync_tags(store, dry_run=True, show_progress=show_progress)
        summary["tags_written"] = 0
        summary["tags_errors"] = 0

    summary["taxonomy_classified"] = classify_all_taxonomies(store, show_progress=show_progress)

    generate_reports(store, config.reports_dir, show_progress=show_progress)
    return summary
