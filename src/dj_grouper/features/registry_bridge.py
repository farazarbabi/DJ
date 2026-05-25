"""Load registry data and match to grouper tracks by filename."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RegistryEnrichment:
    """Enrichment data from the registry for a single track."""
    # Songstats audio features (0-1 scale, None if unavailable)
    danceability: float | None = None
    valence: float | None = None
    # Genre
    primary_genre: str = ""
    all_genres: list[str] = field(default_factory=list)
    # Canonical resolved values
    canonical_key: str | None = None
    canonical_bpm: int | None = None
    # Flags
    has_songstats: bool = False


class RegistryBridge:
    """Bridge between dj-registry output and dj-grouper feature pipeline."""

    def __init__(self, registry_dir: str = "./outputs/registry") -> None:
        self._registry_dir = registry_dir

    def load(self) -> dict[str, RegistryEnrichment]:
        """Load registry_overview.csv and return enrichments keyed by filename.

        Returns an empty dict if the file doesn't exist or can't be parsed.
        """
        overview_path = Path(self._registry_dir) / "registry_overview.csv"
        if not overview_path.exists():
            logger.info("Registry overview not found at %s", overview_path)
            return {}

        enrichments: dict[str, RegistryEnrichment] = {}
        try:
            current_canonical = _load_tracks_master_canonical(Path(self._registry_dir))
            with open(overview_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("file_name", "").strip()
                    if not filename:
                        continue
                    enr = _parse_row(row)
                    _apply_current_canonical(enr, row.get("track_id", ""), current_canonical)
                    enrichments[filename] = enr
        except Exception as e:
            logger.warning("Failed to load registry overview: %s", e)
            return {}

        logger.info(
            "Registry bridge loaded %d tracks (%d with Songstats)",
            len(enrichments),
            sum(1 for e in enrichments.values() if e.has_songstats),
        )
        return enrichments


def match_enrichments(
    track_paths: list[str],
    enrichments: dict[str, RegistryEnrichment],
) -> dict[str, RegistryEnrichment]:
    """Match enrichments to track paths by filename.

    Returns dict keyed by full track path (same keys as the grouper uses).
    """
    matched: dict[str, RegistryEnrichment] = {}
    for path in track_paths:
        filename = Path(path).name
        enr = enrichments.get(filename)
        if enr is not None:
            matched[path] = enr
    n_matched = len(matched)
    n_total = len(track_paths)
    if n_total > 0:
        logger.info("Registry matched %d/%d tracks (%.0f%%)", n_matched, n_total, n_matched / n_total * 100)
    return matched


def _load_tracks_master_canonical(registry_dir: Path) -> dict[str, tuple[str, int | None]]:
    """Load fresher canonical key/BPM values keyed by track_id, if available."""
    tracks_path = registry_dir / "tracks_master.csv"
    if not tracks_path.exists():
        return {}

    canonical: dict[str, tuple[str, int | None]] = {}
    with open(tracks_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row.get("track_id", "").strip()
            if not track_id:
                continue
            key = row.get("canonical_key_camelot", "").strip()
            bpm = _parse_int(row.get("canonical_bpm"))
            if key or bpm:
                canonical[track_id] = (key, bpm)
    return canonical


def _apply_current_canonical(
    enrichment: RegistryEnrichment,
    track_id: str,
    canonical: dict[str, tuple[str, int | None]],
) -> None:
    """Overlay tracks_master canonical values when overview values are stale/blank."""
    key, bpm = canonical.get(track_id.strip(), ("", None))
    if key and key != "??":
        enrichment.canonical_key = key
    if bpm and bpm > 0:
        enrichment.canonical_bpm = bpm


def apply_registry_upgrades(
    infos: list,
    enrichments: dict[str, RegistryEnrichment],
) -> int:
    """Upgrade TrackInfo key/bpm from registry canonical values.

    Modifies infos in-place. Returns number of upgrades applied.
    """
    n_upgrades = 0
    for info in infos:
        enr = enrichments.get(info.path)
        if enr is None:
            continue
        if enr.canonical_key:
            info.key = enr.canonical_key
            n_upgrades += 1
        if enr.canonical_bpm and enr.canonical_bpm > 0:
            info.bpm = enr.canonical_bpm
            n_upgrades += 1
    if n_upgrades:
        logger.info("Registry upgraded %d key/bpm values", n_upgrades)
    return n_upgrades


def _parse_row(row: dict[str, str]) -> RegistryEnrichment:
    """Parse a single registry_overview.csv row into RegistryEnrichment."""
    enr = RegistryEnrichment()

    # Songstats audio features. The export writes these under the ss_*
    # prefix (see dj_registry/sync/export.py: ss_danceability, ss_valence,
    # etc.). Older versions of this reader looked for the unprefixed names
    # and silently reported "0 with Songstats" for every track.
    enr.danceability = _parse_float(row.get("ss_danceability") or row.get("danceability"))
    enr.valence = _parse_float(row.get("ss_valence") or row.get("valence"))
    enr.has_songstats = enr.danceability is not None

    # Genre (prefer Songstats, fall back to Rekordbox)
    ss_genre = row.get("songstats_genre", "").strip()
    rb_genre = row.get("rekordbox_genre", "").strip()
    enr.primary_genre = ss_genre or rb_genre

    genres_all_str = row.get("songstats_genres_all", "").strip()
    if genres_all_str:
        enr.all_genres = [g.strip() for g in genres_all_str.split(";") if g.strip()]
    elif enr.primary_genre:
        enr.all_genres = [enr.primary_genre]

    # Canonical key/bpm
    canonical_key = row.get("canonical_key", "").strip()
    if canonical_key and canonical_key != "??":
        enr.canonical_key = canonical_key

    canonical_bpm = row.get("canonical_bpm", "").strip()
    if canonical_bpm:
        enr.canonical_bpm = _parse_int(canonical_bpm)

    return enr


def _parse_float(val: str | None) -> float | None:
    if not val or not val.strip():
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_int(val: str | None) -> int | None:
    if not val or not val.strip():
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
