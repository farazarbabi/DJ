"""Central location for all tunable thresholds and lookup tables."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Supported audio formats
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".flac", ".aiff", ".aif", ".wav", ".m4a"}
)

# ---------------------------------------------------------------------------
# Audio loading defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22050

# ---------------------------------------------------------------------------
# Note names (pitch-class index 0 = C)
# ---------------------------------------------------------------------------
NOTE_NAMES: list[str] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
]

# ---------------------------------------------------------------------------
# Camelot wheel mapping  (pitch_class, mode) -> Camelot code
# ---------------------------------------------------------------------------
KEY_TO_CAMELOT: dict[tuple[int, str], str] = {
    (0, "major"): "8B",   (0, "minor"): "5A",    # C
    (1, "major"): "3B",   (1, "minor"): "12A",   # C#
    (2, "major"): "10B",  (2, "minor"): "7A",    # D
    (3, "major"): "5B",   (3, "minor"): "2A",    # Eb
    (4, "major"): "12B",  (4, "minor"): "9A",    # E
    (5, "major"): "7B",   (5, "minor"): "4A",    # F
    (6, "major"): "2B",   (6, "minor"): "11A",   # F#
    (7, "major"): "9B",   (7, "minor"): "6A",    # G
    (8, "major"): "4B",   (8, "minor"): "1A",    # Ab
    (9, "major"): "11B",  (9, "minor"): "8A",    # A
    (10, "major"): "6B",  (10, "minor"): "3A",   # Bb
    (11, "major"): "1B",  (11, "minor"): "10A",  # B
}

# Reverse map for Essentia string output -> Camelot
ESSENTIA_KEY_TO_CAMELOT: dict[tuple[str, str], str] = {
    ("C", "major"): "8B",   ("C", "minor"): "5A",
    ("C#", "major"): "3B",  ("C#", "minor"): "12A",
    ("Db", "major"): "3B",  ("Db", "minor"): "12A",
    ("D", "major"): "10B",  ("D", "minor"): "7A",
    ("D#", "major"): "5B",  ("D#", "minor"): "2A",
    ("Eb", "major"): "5B",  ("Eb", "minor"): "2A",
    ("E", "major"): "12B",  ("E", "minor"): "9A",
    ("F", "major"): "7B",   ("F", "minor"): "4A",
    ("F#", "major"): "2B",  ("F#", "minor"): "11A",
    ("Gb", "major"): "2B",  ("Gb", "minor"): "11A",
    ("G", "major"): "9B",   ("G", "minor"): "6A",
    ("G#", "major"): "4B",  ("G#", "minor"): "1A",
    ("Ab", "major"): "4B",  ("Ab", "minor"): "1A",
    ("A", "major"): "11B",  ("A", "minor"): "8A",
    ("A#", "major"): "6B",  ("A#", "minor"): "3A",
    ("Bb", "major"): "6B",  ("Bb", "minor"): "3A",
    ("B", "major"): "1B",   ("B", "minor"): "10A",
}

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------
MAJOR_PROFILE: list[float] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
]
MINOR_PROFILE: list[float] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
]

# ---------------------------------------------------------------------------
# Energy thresholds
# ---------------------------------------------------------------------------
ENERGY_WEIGHTS: dict[str, float] = {
    "rms": 0.30,
    "centroid": 0.15,
    "flux": 0.20,
    "onset": 0.20,
    "low_freq": 0.15,
}

# (min, range) for normalization: normalized = clip((value - min) / range, 0, 1)
ENERGY_NORM: dict[str, tuple[float, float]] = {
    "rms":      (0.02, 0.18),
    "centroid": (1000.0, 4000.0),
    "flux":     (0.5, 4.5),
    "onset":    (1.0, 7.0),
    "low_freq": (0.1, 0.5),
}

# Composite score boundaries -> energy level
ENERGY_THRESHOLDS: list[float] = [0.20, 0.40, 0.60, 0.80]

# ---------------------------------------------------------------------------
# Structure thresholds
# ---------------------------------------------------------------------------
STANDARD_INTRO_BARS: list[int] = [16, 32, 64]
INTRO_ENERGY_RATIO: float = 0.80        # fraction of median bar energy
INTRO_SUSTAIN_BARS: int = 8             # consecutive bars above threshold

FLOW_VARIANCE_LINEAR: float = 0.005
FLOW_VARIANCE_HYPNOTIC: float = 0.02
FLOW_JUMP_HYPNOTIC: float = 0.15
FLOW_JUMP_DROP: float = 0.40
FLOW_TREND_BUILDER: float = 0.05
FLOW_TREND_GROOVE: float = 0.03
FLOW_PLATEAU_MIN: int = 2

# ---------------------------------------------------------------------------
# Vocal detection thresholds
# ---------------------------------------------------------------------------
VOCAL_FREQ_LOW: float = 300.0
VOCAL_FREQ_HIGH: float = 3000.0
VOCAL_ENERGY_RATIO: float = 0.15
VOCAL_FLATNESS_MAX: float = 0.4
VOCAL_FRAME_THRESHOLD: float = 0.08

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
TAG_IDENTIFIER: str = "DJTAGGER"
