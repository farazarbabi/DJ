"""Library scanning and tag reading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from dj_tagger.cli import find_audio_files
from dj_tagger.formats import parse_tag
from dj_tagger.metadata import read_existing_tag

logger = logging.getLogger(__name__)

VIBE_LABELS = ["HYPN", "DRK", "RAW", "DEEP", "TRIB", "MEL", "ACID", "ATM"]


@dataclass
class TrackInfo:
    """Parsed metadata for a single track."""
    path: str
    energy: int | None = None
    key: str | None = None
    bpm: int | None = None
    structure: str | None = None
    intro_bars: int | None = None
    flow_type: str | None = None
    vibe: str | None = None
    vocal: str | None = None  # "V" or "NV"
    group_id: str | None = None
    # Continuous vibe scores (from pipeline analysis)
    vibe_scores: dict[str, float] = field(default_factory=dict)
    # Per-analyzer confidence (0=uncertain, 1=confident)
    confidences: dict[str, float] = field(default_factory=dict)
    # Section info from analysis
    sections: list[dict] = field(default_factory=list)
    # Track role (inferred by grouper)
    role: str | None = None


def scan_library(paths: list[str], recursive: bool = True) -> list[TrackInfo]:
    """Scan paths for audio files and read existing dj_tagger tags."""
    files = find_audio_files(paths, recursive)
    tracks: list[TrackInfo] = []

    for fpath in files:
        info = TrackInfo(path=str(fpath))
        tag_str = read_existing_tag(str(fpath))
        if tag_str:
            parsed = parse_tag(tag_str)
            if parsed:
                info.energy = int(parsed["energy"]) if parsed.get("energy", "?") != "?" else None
                info.key = parsed.get("key")
                if "bpm" in parsed and parsed["bpm"] != "???":
                    info.bpm = int(parsed["bpm"])
                info.structure = parsed.get("structure")
                if info.structure and len(info.structure) >= 3:
                    info.intro_bars = int(info.structure[:-1])
                    info.flow_type = info.structure[-1]
                info.vibe = parsed.get("vibe")
                info.vocal = parsed.get("vocal")
                info.group_id = parsed.get("group_id")
        tracks.append(info)

    logger.info("Scanned %d tracks (%d with tags)", len(tracks), sum(1 for t in tracks if t.energy is not None))
    return tracks
