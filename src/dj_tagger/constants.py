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
# Key profiles (Temperley 2007 — better major/minor and fifth separation
# than Krumhansl-Schmuckler, especially on produced/electronic music)
# ---------------------------------------------------------------------------
MAJOR_PROFILE: list[float] = [
    5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0,
]
MINOR_PROFILE: list[float] = [
    5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0,
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
# Tightened for mastered electronic music (techno/house/downtempo)
ENERGY_NORM: dict[str, tuple[float, float]] = {
    "rms":      (0.04, 0.08),      # mastered electronic: ~0.04-0.12
    "centroid": (1500.0, 2500.0),   # electronic: ~1500-4000
    "flux":     (0.5, 2.5),         # electronic: ~0.5-3.0
    "onset":    (1.5, 4.0),         # electronic: ~1.5-5.5
    "low_freq": (0.15, 0.30),       # bass-heavy music: ~0.15-0.45
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
