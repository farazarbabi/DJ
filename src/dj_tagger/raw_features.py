"""Shared raw feature extraction and canonical tagger derivation."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from dj_grouper.features.dsp import extract_dsp_features, extract_section_dsp

from .analyzers.key import analyze_key
from .analyzers.sections import analyze_sections
from .audio import TrackAudio
from .derive import derive_all
from .formats import format_tag

logger = logging.getLogger(__name__)


def extract_raw_analysis(track_audio: TrackAudio) -> dict:
    """Extract raw intermediate features needed to re-derive tagger outputs."""
    raw: dict[str, float | list[float] | int] = {}

    onset_env = librosa.onset.onset_strength(y=track_audio.y, sr=track_audio.sr)
    beat_frames = track_audio.beat_frames
    beat_energies = []
    for i in range(len(beat_frames) - 1):
        seg = onset_env[beat_frames[i]: beat_frames[i + 1]]
        beat_energies.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)

    n_bars = len(beat_energies) // 4
    bar_energies = []
    for i in range(n_bars):
        chunk = beat_energies[i * 4: (i + 1) * 4]
        bar_energies.append(float(np.mean(chunk)))

    raw["bar_energies"] = bar_energies
    raw["n_bars"] = n_bars
    raw["tempo"] = float(track_audio.tempo)

    # Vocal raw features used by derive_vocal.
    S = np.abs(librosa.stft(track_audio.y_harmonic, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=track_audio.sr, n_fft=2048)
    vocal_mask = (freqs >= 300) & (freqs <= 3400)
    vocal_energy = np.mean(S[vocal_mask, :] ** 2, axis=0)
    total_energy = np.mean(S ** 2, axis=0)
    vocal_ratio_per_frame = vocal_energy / (total_energy + 1e-8)

    vocal_S = S[vocal_mask, :]
    log_mean = np.mean(np.log(vocal_S + 1e-8), axis=0)
    flatness_per_frame = np.exp(log_mean) / (np.mean(vocal_S, axis=0) + 1e-8)
    hi_mask = freqs > 5000
    hi_energy = np.mean(S[hi_mask, :] ** 2, axis=0) if np.any(hi_mask) else np.ones_like(vocal_energy)
    vocal_vs_hi = vocal_energy / (hi_energy + 1e-8)
    vocal_frames = (vocal_ratio_per_frame > 0.12) & (flatness_per_frame < 0.5) & (vocal_vs_hi > 1.5)
    raw["vocal_ratio"] = float(np.mean(vocal_frames))
    if np.any(vocal_frames):
        diffs = np.diff(vocal_frames.astype(int))
        n_runs = max(1, np.sum(diffs == 1))
        avg_run_len = np.sum(vocal_frames) / n_runs
        raw["vocal_temporal_bonus"] = float(min(1.0, avg_run_len / 20.0))
    else:
        raw["vocal_temporal_bonus"] = 0.0

    onsets = librosa.onset.onset_detect(y=track_audio.y, sr=track_audio.sr)
    duration = len(track_audio.y) / track_audio.sr
    raw["onset_rate"] = len(onsets) / duration if duration > 0 else 0.0

    return raw


def compute_tagger_artifacts(
    track_audio: TrackAudio,
    *,
    audio_features: dict[str, float] | None = None,
    dsp: dict[str, float] | None = None,
    use_essentia: bool = False,
) -> dict:
    """Compute canonical tagger outputs plus reusable raw artifacts."""
    if dsp is None:
        dsp = extract_dsp_features(track_audio)
    raw_analysis = extract_raw_analysis(track_audio)
    section_map = analyze_sections(track_audio)
    section_dsp = extract_section_dsp(track_audio, section_map)
    derived = derive_all(dsp, raw_analysis, audio_features=audio_features)

    key_result = None
    try:
        key_result = analyze_key(track_audio, use_essentia=use_essentia)
    except Exception:
        logger.warning("Key analysis failed", exc_info=True)

    bpm = round(float(raw_analysis.get("tempo", track_audio.tempo)), 1)
    tag = format_tag(
        energy=derived.get("energy"),
        camelot=key_result.camelot if key_result else None,
        bpm=round(bpm) if bpm > 0 else None,
        vibe=derived.get("vibe"),
        has_vocals=derived.get("has_vocals"),
        vocal_profile=derived.get("vocal_profile"),
    )

    confidences = {k: round(v, 3) for k, v in derived["confidences"].items()}
    confidences["key"] = round(key_result.confidence, 3) if key_result else 0.0

    tagger_result = {
        "tag": tag,
        "bpm": bpm,
        "energy": derived["energy"],
        "key": key_result.key_name if key_result else None,
        "camelot": key_result.camelot if key_result else None,
        "structure": derived["structure"],
        "vibe": derived["vibe"],
        "mood": derived["mood"],
        "vocal": derived["vocal"],
        "vocal_profile": derived["vocal_profile"],
        "key_confidence": round(key_result.confidence, 3) if key_result else None,
        "vocal_ratio": round(float(derived["vocal_ratio"]), 3),
        "vocal_scores": {k: round(v, 4) for k, v in derived["vocal_scores"].items()},
        "vibe_scores": {k: round(v, 4) for k, v in derived["vibe_scores"].items()},
        "mood_scores": {k: round(v, 4) for k, v in derived["mood_scores"].items()},
        "confidences": confidences,
        "intro_bars": derived["intro_bars"],
        "flow_type": derived["flow_type"],
        "has_vocals": derived["has_vocals"],
        "sections": [
            {"label": s.label, "start": s.start_bar, "end": s.end_bar, "energy": round(s.energy, 3)}
            for s in section_map.sections
        ],
    }

    return {
        "tagger_result": tagger_result,
        "dsp": dsp,
        "raw_analysis": raw_analysis,
        "section_dsp": section_dsp,
    }
