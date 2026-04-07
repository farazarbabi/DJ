"""Musical key detection with Camelot notation output."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from ..audio import TrackAudio
from ..constants import KEY_TO_CAMELOT, MAJOR_PROFILE, MINOR_PROFILE, NOTE_NAMES

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


def _correlate_key(chroma_avg: np.ndarray) -> tuple[int, str, float, float]:
    """Correlate a chroma vector against all 24 key profiles.

    Returns (best_key, best_mode, best_corr, second_best_corr).
    """
    major = np.array(MAJOR_PROFILE)
    minor = np.array(MINOR_PROFILE)

    scores: list[tuple[float, int, str]] = []

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)
        corr_maj = float(np.corrcoef(rotated, major)[0, 1])
        corr_min = float(np.corrcoef(rotated, minor)[0, 1])
        scores.append((corr_maj, shift, "major"))
        scores.append((corr_min, shift, "minor"))

    scores.sort(key=lambda x: -x[0])
    best_corr, best_key, best_mode = scores[0]
    second_corr = scores[1][0]

    return best_key, best_mode, best_corr, second_corr


def _gap_confidence(best: float, second: float) -> float:
    """Confidence based on the gap between top two candidates.

    Large gap = clear winner = high confidence.
    Tiny gap = ambiguous = low confidence.
    """
    gap = best - second
    # Scale so that a gap of 0.15+ maps to ~1.0 confidence,
    # and a gap of 0 maps to 0.
    conf = min(1.0, gap / 0.15)
    return max(0.0, conf)


def _detect_key_librosa(track_audio: TrackAudio) -> KeyResult:
    """Key detection via segment voting with confidence weighting.

    Splits the track into segments, detects key per segment, and
    votes weighted by each segment's confidence. This handles intros,
    breakdowns, and key changes much better than whole-track averaging.

    Uses the full signal (not harmonic-only) so bass content contributes
    to key detection — critical for electronic music where bass defines
    the root note.
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

    # Detect key per segment
    votes: dict[tuple[int, str], float] = {}

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

        key, mode, best_corr, second_corr = _correlate_key(chroma_avg)
        conf = _gap_confidence(best_corr, second_corr)

        # Weight vote by segment confidence
        vote_key = (key, mode)
        votes[vote_key] = votes.get(vote_key, 0.0) + conf

    if not votes:
        return _detect_key_whole_track(y, sr)

    # Pick the key with the highest weighted votes
    winner = max(votes, key=votes.get)
    best_key, best_mode = winner
    total_weight = sum(votes.values())
    winner_weight = votes[winner]

    # Overall confidence: what fraction of weighted votes went to the winner
    vote_confidence = winner_weight / total_weight if total_weight > 0 else 0.0

    camelot = KEY_TO_CAMELOT[(best_key, best_mode)]
    suffix = "m" if best_mode == "minor" else ""
    key_name = f"{NOTE_NAMES[best_key]}{suffix}"

    logger.debug(
        "Key: %s (%s) confidence=%.3f (voted %d/%d segments)",
        key_name, camelot, vote_confidence, n_seg, N_SEGMENTS,
    )
    return KeyResult(camelot=camelot, key_name=key_name, confidence=vote_confidence)


def _detect_key_whole_track(y: np.ndarray, sr: int) -> KeyResult:
    """Fallback: whole-track chroma averaging for very short tracks."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_max = np.max(chroma_avg)
    if chroma_max > 0:
        chroma_avg = chroma_avg / chroma_max

    best_key, best_mode, best_corr, second_corr = _correlate_key(chroma_avg)
    confidence = _gap_confidence(best_corr, second_corr)

    camelot = KEY_TO_CAMELOT[(best_key, best_mode)]
    suffix = "m" if best_mode == "minor" else ""
    key_name = f"{NOTE_NAMES[best_key]}{suffix}"

    logger.debug("Key (whole-track): %s (%s) confidence=%.3f", key_name, camelot, confidence)
    return KeyResult(camelot=camelot, key_name=key_name, confidence=confidence)


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
