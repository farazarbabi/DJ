"""Deterministic v1 cue selection heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..models import CuePoint, now_iso
from .slots import hot_cue_num

BEATS_PER_BAR = 4
PHRASE_BARS = 8
LOW_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class CueSection:
    """Small section representation independent of tagger internals."""

    label: str
    start_bar: int
    end_bar: int
    energy: float = 0.0


@dataclass
class CueGrid:
    """Beat, bar, energy, and section data used by the v1 selector."""

    duration_sec: float
    beat_times: list[float] = field(default_factory=list)
    bar_times: list[float] = field(default_factory=list)
    bar_energies: list[float] = field(default_factory=list)
    sections: list[CueSection] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return max(len(self.bar_times), len(self.bar_energies))

    def time_for_bar(self, bar_index: int) -> float:
        if self.n_bars <= 0:
            return 0.0
        clamped = max(0, min(int(bar_index), self.n_bars - 1))
        if clamped < len(self.bar_times):
            return max(0.0, float(self.bar_times[clamped]))
        if len(self.bar_times) >= 2:
            interval = self.bar_times[-1] - self.bar_times[-2]
            return max(0.0, float(self.bar_times[-1] + interval * (clamped - len(self.bar_times) + 1)))
        if self.duration_sec > 0 and self.n_bars > 1:
            return max(0.0, float(self.duration_sec * clamped / self.n_bars))
        return 0.0


ROLE_PROFILES = (
    ("mix_in", "A", "MIX IN", (255, 55, 111)),
    ("drop_1", "B", "DROP 1", (69, 172, 255)),
    ("mix_out", "C", "MIX OUT", (125, 193, 75)),
)


def cue_grid_from_audio(track_audio: Any, section_map: Any | None = None, raw_analysis: dict | None = None) -> CueGrid:
    """Build a CueGrid from existing tagger audio/section artifacts."""
    import librosa

    beat_times = [
        float(t)
        for t in librosa.frames_to_time(track_audio.beat_frames, sr=track_audio.sr)
    ]
    if len(beat_times) < BEATS_PER_BAR and float(getattr(track_audio, "tempo", 0.0) or 0.0) > 0:
        beat_times = _synthetic_beat_times(float(track_audio.duration), float(track_audio.tempo))

    bar_times = [
        beat_times[i]
        for i in range(0, len(beat_times) - (len(beat_times) % BEATS_PER_BAR), BEATS_PER_BAR)
    ]

    bar_energies = _bar_energies_from(section_map, raw_analysis)
    if not bar_times and bar_energies and float(getattr(track_audio, "tempo", 0.0) or 0.0) > 0:
        seconds_per_bar = 60.0 / float(track_audio.tempo) * BEATS_PER_BAR
        bar_times = [i * seconds_per_bar for i in range(len(bar_energies))]

    sections = _sections_from(section_map)
    return CueGrid(
        duration_sec=float(getattr(track_audio, "duration", 0.0) or 0.0),
        beat_times=beat_times,
        bar_times=bar_times,
        bar_energies=bar_energies,
        sections=sections,
    )


def select_default_hot_cues(
    track_id: str,
    file_id: str,
    grid: CueGrid,
    *,
    analysis_payload_ref: str = "",
) -> list[CuePoint]:
    """Generate the v1 default A-C hot cues for one track."""
    if grid.n_bars <= 0:
        return _fallback_duration_cues(track_id, file_id, grid.duration_sec, analysis_payload_ref)

    mix_bar, mix_conf, mix_reason = _choose_mix_in(grid)
    drop_bar, drop_conf, drop_reason = _choose_drop_1(grid, mix_bar)
    out_bar, out_conf, out_reason = _choose_mix_out(grid, drop_bar)

    sequence_notes: dict[str, str] = {}
    if drop_bar <= mix_bar and grid.n_bars > PHRASE_BARS:
        drop_bar = min(grid.n_bars - 1, _snap_phrase(mix_bar + PHRASE_BARS, grid.n_bars, mode="ceil"))
        drop_conf = min(drop_conf, 0.5)
        sequence_notes["drop_1"] = "sequence_adjusted_after_mix_in"
    if out_bar <= drop_bar and grid.n_bars > PHRASE_BARS * 2:
        out_bar = min(grid.n_bars - 1, _snap_phrase(drop_bar + PHRASE_BARS, grid.n_bars, mode="ceil"))
        out_conf = min(out_conf, 0.5)
        sequence_notes["mix_out"] = "sequence_adjusted_after_drop_1"

    choices = {
        "mix_in": (mix_bar, mix_conf, mix_reason),
        "drop_1": (drop_bar, drop_conf, drop_reason),
        "mix_out": (out_bar, out_conf, out_reason),
    }

    now = now_iso()
    cues: list[CuePoint] = []
    for role, slot, name, color in ROLE_PROFILES:
        bar, confidence, reason = choices[role]
        if role in sequence_notes:
            reason = f"{reason};{sequence_notes[role]}"
        confidence = round(float(max(0.0, min(confidence, 1.0))), 3)
        manual_review = confidence < LOW_CONFIDENCE_THRESHOLD or "fallback" in reason or "adjusted" in reason
        cues.append(CuePoint(
            cue_id=f"CUE-{file_id}-{slot}",
            track_id=track_id,
            file_id=file_id,
            source_system="auto_v1",
            cue_kind="hot",
            cue_role=role,
            cue_slot=slot,
            cue_name=name,
            cue_time_sec=round(grid.time_for_bar(bar), 3),
            cue_bar_index=int(bar),
            cue_beat_index=int(bar * BEATS_PER_BAR),
            rekordbox_num=hot_cue_num(slot),
            rekordbox_type="0",
            red=color[0],
            green=color[1],
            blue=color[2],
            score=confidence,
            confidence=confidence,
            selection_reason=reason,
            analysis_payload_ref=analysis_payload_ref,
            manual_review_required=manual_review,
            created_at=now,
            updated_at=now,
        ))
    return cues


def _synthetic_beat_times(duration_sec: float, tempo: float) -> list[float]:
    if duration_sec <= 0 or tempo <= 0:
        return []
    seconds_per_beat = 60.0 / tempo
    total_beats = int(duration_sec / seconds_per_beat)
    return [i * seconds_per_beat for i in range(total_beats)]


def _bar_energies_from(section_map: Any | None, raw_analysis: dict | None) -> list[float]:
    energies = getattr(section_map, "bar_energies", None)
    if energies is not None:
        try:
            return [float(v) for v in list(energies)]
        except TypeError:
            pass
    if raw_analysis and isinstance(raw_analysis.get("bar_energies"), list):
        return [float(v) for v in raw_analysis["bar_energies"]]
    return []


def _sections_from(section_map: Any | None) -> list[CueSection]:
    sections = []
    for section in getattr(section_map, "sections", []) or []:
        sections.append(CueSection(
            label=str(getattr(section, "label", "")),
            start_bar=int(getattr(section, "start_bar", 0) or 0),
            end_bar=int(getattr(section, "end_bar", 0) or 0),
            energy=float(getattr(section, "energy", 0.0) or 0.0),
        ))
    return sections


def _choose_mix_in(grid: CueGrid) -> tuple[int, float, str]:
    intro = _first_section(grid, "intro")
    if intro and intro.end_bar > 0:
        return _snap_phrase(intro.end_bar, grid.n_bars, mode="nearest"), 0.82, "section_intro_end"

    groove = _first_section(grid, "groove")
    if groove:
        return _snap_phrase(groove.start_bar, grid.n_bars, mode="nearest"), 0.76, "section_groove_start"

    energies = _normalized_energies(grid.bar_energies)
    if energies:
        threshold = max(0.05, median(energies) * 0.6)
        for index in range(max(1, len(energies) - 3)):
            if all(value > threshold for value in energies[index:index + 4]):
                return _snap_phrase(index, grid.n_bars, mode="nearest"), 0.58, "energy_sustain_fallback"

    return 0, 0.45, "start_fallback"


def _choose_drop_1(grid: CueGrid, mix_bar: int) -> tuple[int, float, str]:
    peak = _first_section(grid, "peak")
    if peak and peak.start_bar > mix_bar:
        return _snap_phrase(peak.start_bar, grid.n_bars, mode="nearest"), 0.8, "section_peak_start"

    energies = _normalized_energies(grid.bar_energies)
    phrase_bars = [bar for bar in range(PHRASE_BARS * 2, grid.n_bars, PHRASE_BARS) if bar > mix_bar]
    best_bar = 0
    best_delta = -1.0
    for bar in phrase_bars:
        previous = _mean(energies[max(0, bar - PHRASE_BARS):bar])
        current = _mean(energies[bar:min(len(energies), bar + PHRASE_BARS)])
        delta = current - previous
        if delta > best_delta:
            best_bar = bar
            best_delta = delta
    if best_bar:
        confidence = 0.66 if best_delta > 0.05 else 0.55
        return _snap_phrase(best_bar, grid.n_bars, mode="nearest"), confidence, "phrase_energy_rise"

    fallback = min(grid.n_bars - 1, max(mix_bar + PHRASE_BARS, grid.n_bars // 3))
    return _snap_phrase(fallback, grid.n_bars, mode="nearest"), 0.45, "drop_fallback"


def _choose_mix_out(grid: CueGrid, drop_bar: int) -> tuple[int, float, str]:
    outro = _first_section(grid, "outro")
    if outro and outro.start_bar > drop_bar:
        return _snap_phrase(outro.start_bar, grid.n_bars, mode="floor"), 0.78, "section_outro_start"

    runway = PHRASE_BARS * 2 if grid.n_bars >= PHRASE_BARS * 4 else PHRASE_BARS
    fallback = max(drop_bar + PHRASE_BARS, grid.n_bars - runway)
    fallback = min(grid.n_bars - 1, fallback)
    return _snap_phrase(fallback, grid.n_bars, mode="floor"), 0.56, "outro_runway_fallback"


def _first_section(grid: CueGrid, label: str) -> CueSection | None:
    return next((s for s in grid.sections if s.label == label), None)


def _snap_phrase(bar_index: int, n_bars: int, *, mode: str) -> int:
    if n_bars <= 0:
        return 0
    bar = max(0, min(int(bar_index), n_bars - 1))
    if mode == "floor":
        snapped = (bar // PHRASE_BARS) * PHRASE_BARS
    elif mode == "ceil":
        snapped = ((bar + PHRASE_BARS - 1) // PHRASE_BARS) * PHRASE_BARS
    else:
        snapped = round(bar / PHRASE_BARS) * PHRASE_BARS
    return max(0, min(snapped, n_bars - 1))


def _normalized_energies(values: list[float]) -> list[float]:
    if not values:
        return []
    vmax = max(values)
    if vmax <= 0:
        return [0.0 for _ in values]
    return [float(v) / vmax for v in values]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _fallback_duration_cues(
    track_id: str,
    file_id: str,
    duration_sec: float,
    analysis_payload_ref: str,
) -> list[CuePoint]:
    now = now_iso()
    duration = max(0.0, float(duration_sec or 0.0))
    times = (0.0, duration * 0.33, duration * 0.75)
    cues: list[CuePoint] = []
    for (role, slot, name, color), cue_time in zip(ROLE_PROFILES, times):
        cues.append(CuePoint(
            cue_id=f"CUE-{file_id}-{slot}",
            track_id=track_id,
            file_id=file_id,
            source_system="auto_v1",
            cue_kind="hot",
            cue_role=role,
            cue_slot=slot,
            cue_name=name,
            cue_time_sec=round(cue_time, 3),
            rekordbox_num=hot_cue_num(slot),
            rekordbox_type="0",
            red=color[0],
            green=color[1],
            blue=color[2],
            score=0.25,
            confidence=0.25,
            selection_reason="no_bar_grid_fallback",
            analysis_payload_ref=analysis_payload_ref,
            manual_review_required=True,
            created_at=now,
            updated_at=now,
        ))
    return cues
