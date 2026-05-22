"""Demucs vocal-stem analyzer.

Separates the vocal stem from short slices of a track using htdemucs and
reports compact scalar metrics for the cache:

- vocal_stem_rms_db        RMS of the vocal stem in dBFS (mono mix of stem)
- vocal_stem_mix_ratio_db  stem RMS minus full-mix RMS, in dB
- vocal_stem_activity_frac fraction of 0.5s windows where stem energy
                           exceeds (noise_floor + activity_margin) dB
- vocal_stem_envelope_var  variance of stem RMS envelope across slices
                           (FVOC vs VOC discriminator: hook-shaped vs sustained)

Slices: five 30s windows at 10/30/50/70/90% of the track. Extended mixes
concentrate vocals in specific verses; three slices at 20/50/75 frequently
missed them. Aggregates are reported as median across slices.

The model is loaded lazily and cached process-wide. Failure is non-fatal: the
caller receives None and should fall back to the spectral heuristic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..audio import TrackAudio

logger = logging.getLogger(__name__)

_DEMUCS_MODEL = None
_DEMUCS_MODEL_NAME = "htdemucs"

# Slice geometry — five 30s windows for broader coverage. Three slices at
# 20/50/75 frequently missed vocal moments in extended mixes where vocals
# are concentrated in specific verses and the rest is instrumental.
SLICE_SECONDS = 30.0
SLICE_POSITIONS = (0.10, 0.30, 0.50, 0.70, 0.90)

# Activity-fraction parameters.
NOISE_FLOOR_DB = -55.0
ACTIVITY_MARGIN_DB = 6.0
ACTIVITY_WINDOW_SECONDS = 0.5


@dataclass
class VocalStemResult:
    vocal_stem_rms_db: float
    vocal_stem_mix_ratio_db: float
    vocal_stem_activity_frac: float
    vocal_stem_envelope_var: float
    n_slices: int


def is_demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        _DEMUCS_MODEL = get_model(_DEMUCS_MODEL_NAME)
        _DEMUCS_MODEL.eval()
        logger.info("Demucs model loaded: %s", _DEMUCS_MODEL_NAME)
    return _DEMUCS_MODEL


def _to_db(x: float) -> float:
    return 20.0 * float(np.log10(max(float(x), 1e-9)))


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2) + 1e-12))


def _activity_fraction(stem_mono: np.ndarray, sr: int) -> float:
    win = int(ACTIVITY_WINDOW_SECONDS * sr)
    if win <= 0 or stem_mono.size < win:
        return 0.0
    n_win = stem_mono.size // win
    energies_db = np.array([
        _to_db(_rms(stem_mono[i * win:(i + 1) * win]))
        for i in range(n_win)
    ])
    return float(np.mean(energies_db > (NOISE_FLOOR_DB + ACTIVITY_MARGIN_DB)))


def _load_stereo_for_demucs(track_audio: TrackAudio, target_sr: int) -> np.ndarray:
    """Resample TrackAudio.y to demucs target SR and stack to stereo (2, N)."""
    import librosa
    y = track_audio.y
    sr = track_audio.sr
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.ndim == 2 and y.shape[0] != 2:
        y = y.T if y.shape[1] == 2 else np.stack([y[0], y[0]], axis=0)
    return y.astype(np.float32)


def _slice_ranges(n_samples: int, sr: int) -> list[tuple[int, int]]:
    win = int(SLICE_SECONDS * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    out = []
    for frac in SLICE_POSITIONS:
        center = int(frac * n_samples)
        start = max(0, min(n_samples - win, center - win // 2))
        out.append((start, start + win))
    return out


def _analyze_slice(model, mix_slice: np.ndarray, sr: int) -> dict | None:
    import torch
    from demucs.apply import apply_model

    tensor = torch.from_numpy(mix_slice).unsqueeze(0)  # (1, 2, N)
    with torch.no_grad():
        stems = apply_model(model, tensor, split=True, overlap=0.25, progress=False)
    sources = list(model.sources)
    if "vocals" not in sources:
        return None
    vocals_idx = sources.index("vocals")
    vocal_stem = stems[0, vocals_idx].cpu().numpy()  # (2, N)
    vocal_mono = vocal_stem.mean(axis=0)
    mix_mono = mix_slice.mean(axis=0)

    rms_vocal = _rms(vocal_mono)
    rms_mix = _rms(mix_mono)
    return {
        "vocal_stem_rms_db": _to_db(rms_vocal),
        "vocal_stem_mix_ratio_db": _to_db(rms_vocal) - _to_db(rms_mix),
        "vocal_stem_activity_frac": _activity_fraction(vocal_mono, sr),
    }


MIN_DURATION_SECONDS = 30.0


def analyze_vocal_stem(track_audio: TrackAudio) -> VocalStemResult | None:
    """Run htdemucs on 3 slices of the track and report stem metrics.

    Returns None if demucs is unavailable, the audio is too short to be a
    real track, or separation fails. Callers fall back to spectral
    heuristics in that case.
    """
    if not is_demucs_available():
        return None
    if getattr(track_audio, "duration", 0.0) < MIN_DURATION_SECONDS:
        # Synthetic fixtures and clips this short cannot give a reliable
        # vocal/instrumental verdict from a separation model.
        return None

    try:
        model = _get_model()
    except Exception:
        logger.warning("Failed to load demucs model", exc_info=True)
        return None

    target_sr = model.samplerate
    try:
        mix_stereo = _load_stereo_for_demucs(track_audio, target_sr)
    except Exception:
        logger.warning("Failed to prepare audio for demucs", exc_info=True)
        return None

    slice_metrics: list[dict] = []
    for (a, b) in _slice_ranges(mix_stereo.shape[1], target_sr):
        try:
            m = _analyze_slice(model, mix_stereo[:, a:b], target_sr)
            if m is not None:
                slice_metrics.append(m)
        except Exception:
            logger.warning("Demucs slice failed", exc_info=True)
            continue

    if not slice_metrics:
        return None

    rms_list = [s["vocal_stem_rms_db"] for s in slice_metrics]
    ratio_list = [s["vocal_stem_mix_ratio_db"] for s in slice_metrics]
    act_list = [s["vocal_stem_activity_frac"] for s in slice_metrics]

    # Envelope variance across slices — high variance suggests hook-shaped
    # (FVOC: only in chorus) vs sustained background (VOC: throughout).
    envelope_var = float(np.var(rms_list))

    return VocalStemResult(
        vocal_stem_rms_db=float(np.median(rms_list)),
        vocal_stem_mix_ratio_db=float(np.median(ratio_list)),
        vocal_stem_activity_frac=float(np.median(act_list)),
        vocal_stem_envelope_var=envelope_var,
        n_slices=len(slice_metrics),
    )


def result_to_dict(r: VocalStemResult | None) -> dict[str, float | int]:
    """Serialize VocalStemResult to a flat dict for the cache."""
    if r is None:
        return {}
    return {
        "vocal_stem_rms_db": round(r.vocal_stem_rms_db, 2),
        "vocal_stem_mix_ratio_db": round(r.vocal_stem_mix_ratio_db, 2),
        "vocal_stem_activity_frac": round(r.vocal_stem_activity_frac, 3),
        "vocal_stem_envelope_var": round(r.vocal_stem_envelope_var, 3),
        "vocal_stem_n_slices": r.n_slices,
    }
