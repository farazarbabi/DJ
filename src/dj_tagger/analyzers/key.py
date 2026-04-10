"""Musical key detection with Camelot notation output."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np

from ..audio import TrackAudio
from ..constants import (
    KEY_GAP_DIVISOR,
    KEY_MINOR_BIAS,
    KEY_PROFILES,
    KEY_SAME_ROOT_MINOR_PREF,
    KEY_TO_CAMELOT,
    NOTE_NAMES,
)

logger = logging.getLogger(__name__)

# Number of equal-length segments for voting-based key detection
N_SEGMENTS = 8

# Minimum segment length in seconds (skip very short segments)
MIN_SEGMENT_SECONDS = 2.0


@dataclass
class KeyResult:
    camelot: str
    key_name: str
    confidence: float


def analyze_key(track_audio: TrackAudio, use_essentia: bool = False) -> KeyResult:
    """Detect the musical key and return Camelot notation."""
    if use_essentia:
        try:
            return _detect_key_essentia(track_audio)
        except ImportError:
            logger.warning("Essentia not installed, falling back to librosa")
        except Exception:
            logger.warning("Essentia key detection failed, falling back to librosa", exc_info=True)

    return _detect_key_librosa(track_audio)


def _correlate_key(chroma_avg: np.ndarray, major: np.ndarray, minor: np.ndarray) -> list[tuple[float, int, str]]:
    """Correlate a chroma vector against all 24 key profiles.

    Returns sorted list of (correlation, pitch_class, mode), best first.
    """
    scores: list[tuple[float, int, str]] = []

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)
        corr_maj = float(np.corrcoef(rotated, major)[0, 1])
        corr_min = float(np.corrcoef(rotated, minor)[0, 1])
        scores.append((corr_maj, shift, "major"))
        scores.append((corr_min, shift, "minor"))

    scores.sort(key=lambda x: -x[0])
    return scores


def _detect_key_single_profile(
    segment_chromas: list[np.ndarray],
    major: np.ndarray,
    minor: np.ndarray,
) -> tuple[str, float]:
    """Run key detection with one profile on pre-computed segment chromas.

    Uses gap*abs confidence, sum-squared aggregation, minor bias,
    and same-root minor preference.
    """
    votes: dict[tuple[int, str], float] = {}

    for chroma_avg in segment_chromas:
        scores = _correlate_key(chroma_avg, major, minor)
        best_corr, best_key, best_mode = scores[0]
        second_corr = scores[1][0]

        # Confidence: gap scaled by absolute correlation strength
        gap = best_corr - second_corr
        conf = max(0.0, min(1.0, gap / KEY_GAP_DIVISOR)) * max(0.0, best_corr)

        if best_mode == "minor":
            conf += KEY_MINOR_BIAS

        # Sum-squared aggregation
        vote_key = (best_key, best_mode)
        votes[vote_key] = votes.get(vote_key, 0.0) + conf ** 2

    if not votes:
        return "??", 0.0

    # Same-root minor preference: when top-2 share the same root pitch
    # but differ in mode, boost the minor candidate. Electronic music
    # is predominantly minor-key; this resolves mode ambiguity.
    sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
    if len(sorted_votes) >= 2:
        (k1, m1), w1 = sorted_votes[0]
        (k2, m2), w2 = sorted_votes[1]
        if k1 == k2 and m1 != m2:
            minor_key = (k1, "minor")
            votes[minor_key] = votes.get(minor_key, 0.0) + KEY_SAME_ROOT_MINOR_PREF * w1

    winner = max(votes, key=votes.get)
    total_weight = sum(votes.values())
    vote_conf = votes[winner] / total_weight if total_weight > 0 else 0.0

    camelot = KEY_TO_CAMELOT[(winner[0], winner[1])]
    return camelot, vote_conf


def _detect_key_librosa(track_audio: TrackAudio) -> KeyResult:
    """Key detection via multi-profile ensemble with segment voting.

    Splits the track into segments, skips first and last (intro/outro),
    runs 4 key profiles independently, and takes majority vote.
    Each profile uses confidence-weighted voting with same-root minor
    preference to resolve mode ambiguity.
    """
    y = track_audio.y
    sr = track_audio.sr
    n_samples = len(y)
    min_segment_samples = int(MIN_SEGMENT_SECONDS * sr)

    # Determine segment count based on track length
    n_seg = N_SEGMENTS
    segment_len = n_samples // n_seg
    while segment_len < min_segment_samples and n_seg > 1:
        n_seg -= 1
        segment_len = n_samples // n_seg

    if n_seg <= 1:
        return _detect_key_whole_track(y, sr)

    # Compute chroma per segment
    all_chromas: list[np.ndarray] = []
    for seg_idx in range(n_seg):
        start = seg_idx * segment_len
        end = start + segment_len if seg_idx < n_seg - 1 else n_samples
        segment = y[start:end]

        if len(segment) < min_segment_samples:
            continue

        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, n_chroma=12)
        chroma_avg = np.mean(chroma, axis=1)
        chroma_max = np.max(chroma_avg)
        if chroma_max > 0:
            chroma_avg = chroma_avg / chroma_max
        all_chromas.append(chroma_avg)

    if not all_chromas:
        return _detect_key_whole_track(y, sr)

    # Skip first and last segments (intro/outro have weak key signal)
    if len(all_chromas) > 3:
        use_chromas = all_chromas[1:-1]
    elif len(all_chromas) > 2:
        use_chromas = all_chromas[:-1]
    else:
        use_chromas = all_chromas

    # Multi-profile ensemble: each profile votes independently
    profile_results: list[str] = []
    profile_confs: list[float] = []

    for prof in KEY_PROFILES.values():
        major = np.array(prof["major"])
        minor = np.array(prof["minor"])
        camelot, conf = _detect_key_single_profile(use_chromas, major, minor)
        profile_results.append(camelot)
        profile_confs.append(conf)

    # Majority vote across profiles
    counter = Counter(profile_results)
    best_camelot = counter.most_common(1)[0][0]

    # Confidence: fraction of profiles that agree
    agreement = counter[best_camelot] / len(profile_results)

    # Reverse-lookup key name from Camelot
    key_name = _camelot_to_key_name(best_camelot)

    logger.debug(
        "Key: %s (%s) agreement=%.0f%% profiles=%s",
        key_name, best_camelot, agreement * 100,
        ",".join(profile_results),
    )
    return KeyResult(camelot=best_camelot, key_name=key_name, confidence=agreement)


def _camelot_to_key_name(camelot: str) -> str:
    """Convert Camelot code back to key name (e.g. '6A' -> 'Gm')."""
    for (pc, mode), cam in KEY_TO_CAMELOT.items():
        if cam == camelot:
            suffix = "m" if mode == "minor" else ""
            return f"{NOTE_NAMES[pc]}{suffix}"
    return "??"


def _detect_key_whole_track(y: np.ndarray, sr: int) -> KeyResult:
    """Fallback: whole-track chroma averaging for very short tracks."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_max = np.max(chroma_avg)
    if chroma_max > 0:
        chroma_avg = chroma_avg / chroma_max

    # Use all profiles even for whole-track
    profile_results = []
    for prof in KEY_PROFILES.values():
        major = np.array(prof["major"])
        minor = np.array(prof["minor"])
        camelot, _ = _detect_key_single_profile([chroma_avg], major, minor)
        profile_results.append(camelot)

    counter = Counter(profile_results)
    best_camelot = counter.most_common(1)[0][0]
    agreement = counter[best_camelot] / len(profile_results)
    key_name = _camelot_to_key_name(best_camelot)

    logger.debug("Key (whole-track): %s (%s) agreement=%.0f%%", key_name, best_camelot, agreement * 100)
    return KeyResult(camelot=best_camelot, key_name=key_name, confidence=agreement)


def _detect_key_essentia(track_audio: TrackAudio) -> KeyResult:
    """Key detection via Essentia KeyExtractor (EDMA profile)."""
    from essentia.standard import KeyExtractor
    from ..constants import ESSENTIA_KEY_TO_CAMELOT

    extractor = KeyExtractor(profileType="edma")
    key, scale, confidence = extractor(track_audio.y.astype(np.float32))

    camelot = ESSENTIA_KEY_TO_CAMELOT.get((key, scale), "??")
    suffix = "m" if scale == "minor" else ""
    key_name = f"{key}{suffix}"

    logger.debug("Key (essentia): %s (%s) confidence=%.3f", key_name, camelot, confidence)
    return KeyResult(camelot=camelot, key_name=key_name, confidence=float(confidence))
