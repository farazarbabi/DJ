"""Extract raw audio features for all tracks and cache them.

Run once (slow), then use score_from_cache.py for fast iteration.
Caches: raw RMS, centroid, flux, onset_rate, low_ratio, chroma stats,
        perc_ratio, flatness, bandwidth, peakiness, vocal stats, etc.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import librosa
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dj_tagger.audio import load_audio_features
from dj_tagger.constants import SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = "outputs/raw_features.pkl"


def extract_raw_features(track_audio) -> dict:
    """Extract all raw features that analyzers need, before normalization/scoring."""
    y, sr = track_audio.y, track_audio.sr
    y_h = track_audio.y_harmonic
    y_p = track_audio.y_percussive

    S = np.abs(librosa.stft(y))
    S_power = S ** 2

    # --- Energy features (raw, pre-normalization) ---
    rms_raw = float(np.mean(librosa.feature.rms(y=y)[0]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_var = float(np.var(centroid)) / (centroid_mean ** 2 + 1e-8)

    diff = np.diff(S, axis=1)
    flux_raw = float(np.mean(np.sqrt(np.mean(diff ** 2, axis=0))))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration_sec = len(y) / sr
    onset_rate = len(onsets) / duration_sec if duration_sec > 0 else 0.0
    onset_density = float(np.mean(onset_env))
    onset_var = float(np.var(onset_env)) / (float(np.mean(onset_env)) ** 2 + 1e-8)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    S_full = np.abs(librosa.stft(y, n_fft=2048))
    low_mask = freqs < 150
    low_energy = float(np.mean(S_full[low_mask, :] ** 2))
    total_energy = float(np.mean(S_full ** 2))
    low_ratio = low_energy / (total_energy + 1e-8)

    # --- Vibe features ---
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma_strength = float(np.mean(np.max(chroma, axis=0)))
    chroma_var = float(np.mean(np.var(chroma, axis=1)))

    spectral_stability = 1.0 - min(1.0, centroid_var * 10)

    harmonic_energy = float(np.mean(y_h ** 2))
    percussive_energy = float(np.mean(y_p ** 2))
    perc_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-8)

    spec_avg = np.mean(S, axis=1)
    peakiness = float(np.max(spec_avg)) / (float(np.mean(spec_avg)) + 1e-8)

    # --- Vocal features ---
    S_h = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=512))
    vocal_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    vocal_energy_frames = np.mean(S_h[vocal_mask, :] ** 2, axis=0)
    total_energy_frames = np.mean(S_h ** 2, axis=0)
    vocal_ratio_per_frame = vocal_energy_frames / (total_energy_frames + 1e-8)

    vocal_S = S_h[vocal_mask, :]
    log_mean = np.mean(np.log(vocal_S + 1e-8), axis=0)
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(vocal_S, axis=0)
    flatness_per_frame = geo_mean / (arith_mean + 1e-8)

    # HF reference for vocal detection
    hi_mask = freqs > 5000
    hi_energy_frames = np.mean(S_h[hi_mask, :] ** 2, axis=0) if np.any(hi_mask) else np.ones_like(vocal_energy_frames)
    vocal_vs_hi = vocal_energy_frames / (hi_energy_frames + 1e-8)

    # Sub-bass reference (original approach)
    noise_mask = freqs < 300.0
    noise_energy_frames = np.mean(S_h[noise_mask, :] ** 2, axis=0) if np.any(noise_mask) else np.ones_like(vocal_energy_frames)
    vocal_vs_sub = vocal_energy_frames / (noise_energy_frames + 1e-8)

    # Vocal frame statistics at various thresholds (for tuning)
    vocal_stats = {}
    for energy_thresh in [0.08, 0.10, 0.12, 0.15, 0.20]:
        for flat_thresh in [0.3, 0.4, 0.5]:
            for hi_thresh in [1.5, 2.0, 3.0, 5.0]:
                frames = (
                    (vocal_ratio_per_frame > energy_thresh)
                    & (flatness_per_frame < flat_thresh)
                    & (vocal_vs_hi > hi_thresh)
                )
                fraction = float(np.mean(frames))
                # Temporal consistency
                if np.any(frames):
                    diffs = np.diff(frames.astype(int))
                    n_runs = max(1, np.sum(diffs == 1))
                    avg_run_len = np.sum(frames) / n_runs
                    temporal_bonus = min(1.0, avg_run_len / 20.0)
                else:
                    temporal_bonus = 0.0
                score = fraction * (0.6 + 0.4 * temporal_bonus)
                key = f"e{energy_thresh}_f{flat_thresh}_h{hi_thresh}"
                vocal_stats[key] = round(score, 4)

    # --- Key features: per-segment chroma + correlations ---
    from dj_tagger.constants import MAJOR_PROFILE, MINOR_PROFILE

    y_full = track_audio.y
    n_samples = len(y_full)
    n_seg = 8
    min_seg_samples = int(2.0 * sr)
    seg_len = n_samples // n_seg
    while seg_len < min_seg_samples and n_seg > 1:
        n_seg -= 1
        seg_len = n_samples // n_seg

    major = np.array(MAJOR_PROFILE)
    minor = np.array(MINOR_PROFILE)

    segment_chromas = []  # list of normalized chroma vectors (12,)
    segment_correlations = []  # list of {(pitch_class, mode): corr}

    for seg_idx in range(n_seg):
        start = seg_idx * seg_len
        end = start + seg_len if seg_idx < n_seg - 1 else n_samples
        segment = y_full[start:end]
        if len(segment) < min_seg_samples:
            continue

        # Full signal chroma (bass contributes to key)
        seg_chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, n_chroma=12)
        seg_avg = np.mean(seg_chroma, axis=1)
        seg_max = np.max(seg_avg)
        if seg_max > 0:
            seg_avg = seg_avg / seg_max
        segment_chromas.append(seg_avg.tolist())

        # Correlate with all 24 keys
        corrs = {}
        for shift in range(12):
            rotated = np.roll(seg_avg, -shift)
            corr_maj = float(np.corrcoef(rotated, major)[0, 1])
            corr_min = float(np.corrcoef(rotated, minor)[0, 1])
            corrs[f"{shift}_major"] = round(corr_maj, 4)
            corrs[f"{shift}_minor"] = round(corr_min, 4)
        segment_correlations.append(corrs)

    # Also compute harmonic-only chroma for comparison
    h_chroma = librosa.feature.chroma_cqt(y=track_audio.y_harmonic, sr=sr, n_chroma=12)
    h_avg = np.mean(h_chroma, axis=1)
    h_max = np.max(h_avg)
    if h_max > 0:
        h_avg = h_avg / h_max

    h_corrs = {}
    for shift in range(12):
        rotated = np.roll(h_avg, -shift)
        h_corrs[f"{shift}_major"] = round(float(np.corrcoef(rotated, major)[0, 1]), 4)
        h_corrs[f"{shift}_minor"] = round(float(np.corrcoef(rotated, minor)[0, 1]), 4)

    return {
        # Energy raw features
        "rms_raw": rms_raw,
        "centroid_mean": centroid_mean,
        "flux_raw": flux_raw,
        "onset_rate": onset_rate,
        "low_ratio": low_ratio,
        # Vibe raw features
        "flatness": flatness,
        "bandwidth": bandwidth,
        "chroma_strength": chroma_strength,
        "chroma_var": chroma_var,
        "spectral_stability": spectral_stability,
        "onset_density": onset_density,
        "onset_var": onset_var,
        "harmonic_energy": harmonic_energy,
        "percussive_energy": percussive_energy,
        "perc_ratio": perc_ratio,
        "centroid_var": centroid_var,
        "peakiness": peakiness,
        "low_ratio_vibe": low_ratio,
        # Vocal pre-computed stats at various thresholds
        "vocal_stats": vocal_stats,
        # Key: per-segment chroma + correlations
        "segment_chromas": segment_chromas,
        "segment_correlations": segment_correlations,
        "harmonic_chroma": h_avg.tolist(),
        "harmonic_correlations": h_corrs,
        # Tempo
        "bpm": track_audio.tempo,
    }


def main():
    files_dir = Path("files")
    from dj_tagger.constants import SUPPORTED_EXTENSIONS

    audio_files = sorted(
        f for f in files_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Found %d audio files", len(audio_files))

    # Load existing cache if present
    features = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            features = pickle.load(f)
        logger.info("Loaded %d cached features", len(features))

    for i, fpath in enumerate(audio_files):
        fname = fpath.name
        if fname in features and "segment_chromas" in features[fname]:
            logger.info("[%d/%d] %s -- cached", i + 1, len(audio_files), fname)
            continue

        logger.info("[%d/%d] %s -- extracting...", i + 1, len(audio_files), fname)
        t0 = time.perf_counter()
        try:
            track_audio = load_audio_features(str(fpath))
            raw = extract_raw_features(track_audio)
            features[fname] = raw
            elapsed = time.perf_counter() - t0
            logger.info("  -> done in %.1fs (rms=%.4f, centroid=%.0f, bpm=%.0f)",
                        elapsed, raw["rms_raw"], raw["centroid_mean"], raw["bpm"])

            # Save incrementally every 5 files
            if (i + 1) % 5 == 0:
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(features, f)
                logger.info("  [saved %d features]", len(features))
        except Exception as e:
            logger.error("  FAILED: %s", e)

    # Final save
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(features, f)
    logger.info("Saved %d features to %s", len(features), CACHE_PATH)


if __name__ == "__main__":
    main()
