"""Heuristic 6-dimensional genre embedding for electronic music subgenres."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 6 axes: [electronic-ness, tempo_character, melodic-ness, darkness, deepness, organicness]
# Each value in [0, 1].
#
# electronic-ness:  0 = acoustic/rock, 1 = fully electronic
# tempo_character:  0 = downtempo/chill, 1 = peak-time/driving
# melodic-ness:     0 = raw/minimal, 1 = melodic/progressive
# darkness:         0 = light/uplifting, 1 = dark/industrial
# deepness:         0 = surface/pop, 1 = deep/underground
# organicness:      0 = synthetic, 1 = organic/acoustic elements

GENRE_DIM = 6
NEUTRAL_VECTOR = [0.5] * GENRE_DIM

GENRE_VECTORS: dict[str, list[float]] = {
    # Techno family
    "Techno":                [1.0, 0.8, 0.2, 0.7, 0.6, 0.1],
    "Melodic Techno":        [1.0, 0.7, 0.8, 0.4, 0.6, 0.2],
    "Melodic House & Techno":[1.0, 0.6, 0.8, 0.3, 0.5, 0.3],
    "Dark Techno":           [1.0, 0.8, 0.1, 0.9, 0.7, 0.1],
    "Hard Techno":           [1.0, 0.9, 0.1, 0.8, 0.5, 0.0],
    "Industrial Techno":     [1.0, 0.9, 0.1, 1.0, 0.7, 0.0],
    "Minimal Techno":        [1.0, 0.6, 0.2, 0.5, 0.7, 0.1],
    "Acid Techno":           [1.0, 0.8, 0.3, 0.7, 0.6, 0.1],
    "Peak Time Techno":      [1.0, 1.0, 0.2, 0.7, 0.5, 0.1],
    "Dub Techno":            [1.0, 0.5, 0.4, 0.4, 0.8, 0.2],
    # House family
    "House":                 [0.9, 0.5, 0.5, 0.3, 0.4, 0.3],
    "Deep House":            [0.9, 0.4, 0.6, 0.3, 0.8, 0.4],
    "Tech House":            [1.0, 0.6, 0.3, 0.4, 0.5, 0.2],
    "Minimal Tech House":    [1.0, 0.6, 0.2, 0.4, 0.6, 0.1],
    "Progressive House":     [0.9, 0.6, 0.7, 0.3, 0.5, 0.3],
    "Afro House":            [0.8, 0.5, 0.5, 0.2, 0.5, 0.7],
    "Organic House":         [0.7, 0.4, 0.6, 0.2, 0.6, 0.8],
    "Indie Dance":           [0.7, 0.5, 0.6, 0.2, 0.4, 0.5],
    "Electronica":           [0.8, 0.4, 0.6, 0.3, 0.5, 0.4],
    # Trance family
    "Trance":                [1.0, 0.8, 0.8, 0.3, 0.4, 0.1],
    "Progressive Trance":    [1.0, 0.7, 0.7, 0.3, 0.5, 0.1],
    "Psytrance":             [1.0, 0.9, 0.5, 0.5, 0.5, 0.1],
    # Ambient / Downtempo
    "Ambient":               [0.6, 0.1, 0.8, 0.2, 0.7, 0.6],
    "Downtempo":             [0.7, 0.2, 0.6, 0.3, 0.6, 0.5],
    "Chillout":              [0.7, 0.2, 0.7, 0.1, 0.4, 0.5],
    # Bass / Breaks
    "Drum & Bass":           [0.9, 0.9, 0.3, 0.5, 0.4, 0.1],
    "Breakbeat":             [0.8, 0.7, 0.3, 0.4, 0.4, 0.2],
    "UK Garage":             [0.8, 0.6, 0.4, 0.2, 0.4, 0.3],
    # Other electronic
    "Disco":                 [0.6, 0.5, 0.6, 0.1, 0.3, 0.5],
    "Nu Disco":              [0.7, 0.5, 0.6, 0.1, 0.3, 0.4],
    "EBM":                   [1.0, 0.7, 0.2, 0.8, 0.6, 0.0],
}

# Build a case-insensitive lookup
_LOOKUP: dict[str, list[float]] = {k.lower(): v for k, v in GENRE_VECTORS.items()}


def encode_genres(genres: list[str]) -> NDArray[np.floating]:
    """Encode a list of genre strings into a 6-dimensional vector.

    Multi-genre tracks get the average of their genre vectors.
    Unknown genres are skipped. If all genres are unknown, returns the neutral vector.
    """
    if not genres:
        return np.array(NEUTRAL_VECTOR, dtype=np.float32)

    vectors: list[list[float]] = []
    for genre in genres:
        key = genre.strip().lower()
        vec = _LOOKUP.get(key)
        if vec is not None:
            vectors.append(vec)
        else:
            # Try partial matching for common prefixes
            vec = _fuzzy_match(key)
            if vec is not None:
                vectors.append(vec)

    if not vectors:
        return np.array(NEUTRAL_VECTOR, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def _fuzzy_match(genre_lower: str) -> list[float] | None:
    """Try to match genre by finding the best substring overlap."""
    # Check if any known genre is a substring of the input or vice versa
    for known, vec in _LOOKUP.items():
        if known in genre_lower or genre_lower in known:
            return vec
    return None
