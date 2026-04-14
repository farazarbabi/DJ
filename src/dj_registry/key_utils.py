"""Key normalization and conversion utilities.

Bidirectional mapping between standard key names and Camelot codes.
"""

from __future__ import annotations

import re

# Canonical note names (index = pitch class 0-11).
# Matches dj_tagger.constants.NOTE_NAMES.
NOTE_NAMES: list[str] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
]

# Enharmonic normalization: map non-canonical spellings to canonical.
_ENHARMONIC: dict[str, str] = {
    "Db": "C#",
    "D#": "Eb",
    "Gb": "F#",
    "G#": "Ab",  # major context — but Ab vs G# depends on mode
    "A#": "Bb",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}

# (pitch_class, mode) -> Camelot code.
# Matches dj_tagger.constants.KEY_TO_CAMELOT.
_PC_MODE_TO_CAMELOT: dict[tuple[int, str], str] = {
    (0, "major"): "8B",   (0, "minor"): "5A",
    (1, "major"): "3B",   (1, "minor"): "12A",
    (2, "major"): "10B",  (2, "minor"): "7A",
    (3, "major"): "5B",   (3, "minor"): "2A",
    (4, "major"): "12B",  (4, "minor"): "9A",
    (5, "major"): "7B",   (5, "minor"): "4A",
    (6, "major"): "2B",   (6, "minor"): "11A",
    (7, "major"): "9B",   (7, "minor"): "6A",
    (8, "major"): "4B",   (8, "minor"): "1A",
    (9, "major"): "11B",  (9, "minor"): "8A",
    (10, "major"): "6B",  (10, "minor"): "3A",
    (11, "major"): "1B",  (11, "minor"): "10A",
}

# Reverse: Camelot code -> (pitch_class, mode).
_CAMELOT_TO_PC_MODE: dict[str, tuple[int, str]] = {
    v: k for k, v in _PC_MODE_TO_CAMELOT.items()
}

# Standard key string -> (pitch_class, mode).
# Built from NOTE_NAMES for both major and minor.
_STANDARD_TO_PC_MODE: dict[str, tuple[int, str]] = {}
for _pc, _note in enumerate(NOTE_NAMES):
    _STANDARD_TO_PC_MODE[f"{_note} major"] = (_pc, "major")
    _STANDARD_TO_PC_MODE[f"{_note} minor"] = (_pc, "minor")

# Special case: G# minor (pitch class 8, but NOTE_NAMES[8] is "Ab")
# In minor context, G# is the conventional spelling.
_STANDARD_TO_PC_MODE["G# minor"] = (8, "minor")

# Pitch class -> standard note for minor keys.
# Minor keys conventionally use sharps where major uses flats.
_PC_TO_NOTE_MINOR: dict[int, str] = {
    0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "Bb", 11: "B",
}

# Pitch class -> standard note for major keys.
_PC_TO_NOTE_MAJOR: dict[int, str] = {
    0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

# Regex patterns
_RE_CAMELOT = re.compile(r"^(1[0-2]|[1-9])[AB]$")
_RE_STANDARD = re.compile(
    r"^([A-G][b#]?)\s*(major|minor|maj|min|m)$", re.IGNORECASE
)

# Short mode -> canonical mode
_MODE_ALIASES: dict[str, str] = {
    "major": "major",
    "minor": "minor",
    "maj": "major",
    "min": "minor",
    "m": "minor",
}

# Common DJ software key notations (Open Key, Rekordbox shorthand)
_DJ_KEY_ALIASES: dict[str, str] = {
    # Rekordbox sometimes uses sharp/flat + m for minor
    "Cm": "C minor", "C#m": "C# minor", "Dm": "D minor",
    "Ebm": "Eb minor", "Em": "E minor", "Fm": "F minor",
    "F#m": "F# minor", "Gm": "G minor", "G#m": "G# minor",
    "Abm": "G# minor", "Am": "A minor", "Bbm": "Bb minor",
    "Bm": "B minor",
    # Dbm etc.
    "Dbm": "C# minor", "D#m": "Eb minor", "Gbm": "F# minor",
    "A#m": "Bb minor",
}


def is_camelot(key: str) -> bool:
    """Check if a string is a valid Camelot code (e.g., '9A', '12B')."""
    return bool(_RE_CAMELOT.match(key.strip()))


def normalize_standard_key(raw: str) -> str | None:
    """Normalize a standard key string to canonical form.

    Accepts: 'G# minor', 'Abm', 'Ab major', 'am', 'Em', 'f# min', etc.
    Returns: 'G# minor', 'Ab major', 'A minor', 'E minor', 'F# minor', etc.
    Returns None if the input is not a recognizable key.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Try DJ software shorthand first (e.g., "Cm", "G#m")
    if raw in _DJ_KEY_ALIASES:
        return _DJ_KEY_ALIASES[raw]

    m = _RE_STANDARD.match(raw)
    if not m:
        return None

    note_raw, mode_raw = m.group(1), m.group(2).lower()
    mode = _MODE_ALIASES.get(mode_raw)
    if mode is None:
        return None

    # Normalize enharmonic spelling
    note = note_raw[0].upper() + note_raw[1:] if len(note_raw) > 1 else note_raw.upper()
    if note in _ENHARMONIC:
        note = _ENHARMONIC[note]

    # For minor keys, use G# instead of Ab
    if mode == "minor" and note == "Ab":
        note = "G#"
    # For major keys, use Ab instead of G#
    if mode == "major" and note == "G#":
        note = "Ab"

    # Validate pitch class exists
    lookup = f"{note} {mode}"
    if lookup not in _STANDARD_TO_PC_MODE:
        return None

    return lookup


def standard_to_camelot(standard_key: str) -> str | None:
    """Convert standard key (e.g., 'G# minor') to Camelot (e.g., '1A').

    Returns None if the key is not valid.
    """
    normalized = normalize_standard_key(standard_key)
    if normalized is None:
        return None
    pc_mode = _STANDARD_TO_PC_MODE.get(normalized)
    if pc_mode is None:
        return None
    return _PC_MODE_TO_CAMELOT.get(pc_mode)


def camelot_to_standard(camelot: str) -> str | None:
    """Convert Camelot code (e.g., '1A') to standard key (e.g., 'G# minor').

    Returns None if the code is not valid.
    """
    camelot = camelot.strip().upper()
    if not is_camelot(camelot):
        return None
    pc_mode = _CAMELOT_TO_PC_MODE.get(camelot)
    if pc_mode is None:
        return None
    pc, mode = pc_mode
    if mode == "minor":
        note = _PC_TO_NOTE_MINOR[pc]
    else:
        note = _PC_TO_NOTE_MAJOR[pc]
    return f"{note} {mode}"


def parse_any_key(raw: str) -> tuple[str, str] | None:
    """Parse any key representation and return (standard, camelot).

    Accepts Camelot codes, standard key names, DJ shorthand, etc.
    Returns None if the input is not a recognizable key.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Try Camelot first
    if is_camelot(raw):
        standard = camelot_to_standard(raw)
        if standard:
            return (standard, raw.upper())

    # Try standard key
    camelot = standard_to_camelot(raw)
    if camelot:
        standard = normalize_standard_key(raw)
        if standard:
            return (standard, camelot)

    return None
