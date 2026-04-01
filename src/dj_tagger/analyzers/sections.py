"""Section detection: identify intro, groove, peak, and breakdown regions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np
from numpy.typing import NDArray

from ..audio import TrackAudio

logger = logging.getLogger(__name__)

BEATS_PER_BAR = 4


@dataclass
class Section:
    """A detected section with start/end in bars and mean energy."""
    start_bar: int
    end_bar: int
    energy: float
    label: str  # "intro", "groove", "peak", "breakdown", "outro"


@dataclass
class SectionMap:
    """All detected sections for a track."""
    sections: list[Section] = field(default_factory=list)
    bar_energies: NDArray | None = None
    n_bars: int = 0

    @property
    def intro(self) -> Section | None:
        return next((s for s in self.sections if s.label == "intro"), None)

    @property
    def groove(self) -> Section | None:
        return next((s for s in self.sections if s.label == "groove"), None)

    @property
    def peak(self) -> Section | None:
        return next((s for s in self.sections if s.label == "peak"), None)

    @property
    def breakdown(self) -> Section | None:
        return next((s for s in self.sections if s.label == "breakdown"), None)


def analyze_sections(track_audio: TrackAudio) -> SectionMap:
    """Detect sections by analyzing bar-level energy envelope."""
    onset_env = librosa.onset.onset_strength(y=track_audio.y, sr=track_audio.sr)
    beat_frames = track_audio.beat_frames

    # Compute per-bar energy
    beat_energies = []
    for i in range(len(beat_frames) - 1):
        seg = onset_env[beat_frames[i]:beat_frames[i + 1]]
        beat_energies.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)

    n_bars = len(beat_energies) // BEATS_PER_BAR
    if n_bars < 4:
        return SectionMap(n_bars=n_bars)

    bar_energies = np.array([
        np.mean(beat_energies[i * BEATS_PER_BAR:(i + 1) * BEATS_PER_BAR])
        for i in range(n_bars)
    ])

    # Normalize bar energies to [0, 1]
    e_max = float(np.max(bar_energies))
    if e_max > 0:
        bar_norm = bar_energies / e_max
    else:
        bar_norm = bar_energies

    median_e = float(np.median(bar_norm))
    sections: list[Section] = []

    # Detect intro: bars from start until energy sustains above 60% of median for 4+ bars
    intro_end = 0
    threshold = 0.6 * median_e
    for i in range(n_bars - 3):
        if np.all(bar_norm[i:i + 4] > threshold):
            intro_end = i
            break

    if intro_end > 0:
        sections.append(Section(
            start_bar=0, end_bar=intro_end,
            energy=float(np.mean(bar_norm[:intro_end])),
            label="intro",
        ))

    # Classify remaining bars into groove/peak/breakdown
    # Use thresholds relative to the track's energy distribution
    high_threshold = float(np.percentile(bar_norm[intro_end:], 75))
    low_threshold = float(np.percentile(bar_norm[intro_end:], 25))

    # Scan through bars in 8-bar phrases
    phrase_len = 8
    i = intro_end
    while i + phrase_len <= n_bars:
        phrase_e = float(np.mean(bar_norm[i:i + phrase_len]))

        if phrase_e >= high_threshold:
            label = "peak"
        elif phrase_e <= low_threshold:
            label = "breakdown"
        else:
            label = "groove"

        # Extend if next phrase has same label
        end = i + phrase_len
        while end + phrase_len <= n_bars:
            next_e = float(np.mean(bar_norm[end:end + phrase_len]))
            if next_e >= high_threshold and label == "peak":
                end += phrase_len
            elif next_e <= low_threshold and label == "breakdown":
                end += phrase_len
            elif low_threshold < next_e < high_threshold and label == "groove":
                end += phrase_len
            else:
                break

        sections.append(Section(
            start_bar=i, end_bar=end,
            energy=float(np.mean(bar_norm[i:end])),
            label=label,
        ))
        i = end

    # Handle remaining bars
    if i < n_bars:
        sections.append(Section(
            start_bar=i, end_bar=n_bars,
            energy=float(np.mean(bar_norm[i:n_bars])),
            label="outro",
        ))

    # If no groove was found, label the largest non-intro section as groove
    if not any(s.label == "groove" for s in sections):
        non_intro = [s for s in sections if s.label not in ("intro", "outro")]
        if non_intro:
            longest = max(non_intro, key=lambda s: s.end_bar - s.start_bar)
            longest.label = "groove"

    sm = SectionMap(sections=sections, bar_energies=bar_energies, n_bars=n_bars)
    logger.debug(
        "Sections: %s",
        [(s.label, s.start_bar, s.end_bar, f"{s.energy:.2f}") for s in sections],
    )
    return sm
