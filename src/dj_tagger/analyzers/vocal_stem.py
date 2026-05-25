"""Demucs full-stem analyzer.

Separates a track into its four htdemucs stems (drums / bass / other / vocals)
and reports per-stem scalar metrics for the cache. Demucs is the expensive
step (~90% of the per-track time); once the stems are in memory, computing
the metrics is essentially free, so we extract everything that meaningfully
discriminates between subgenres — not just vocals.

Per stem (drums / bass / other / vocals), per slice (5 slices at 10/30/50/70/90%
of the track), we compute:
  - rms_db        — stem RMS in dBFS (mono mix of stem)
  - mix_ratio_db  — stem RMS minus full-mix RMS, in dB
  - activity_frac — fraction of 0.5s windows where stem energy exceeds the
                    noise floor
  - centroid_hz   — spectral centroid: where the energy sits in frequency
  - flatness      — spectral flatness: tonal vs noisy
  - onset_rate    — onsets per second (rhythmic density)
  - zcr           — zero-crossing rate (percussion vs tonal discriminator)

Then aggregate to medians across slices, plus per-stem envelope variance
(across-slice RMS variance) and stem dominance ratios (each stem's RMS as
a fraction of the total stem energy).

The model is loaded lazily and cached process-wide. Failure is non-fatal: the
caller receives None and should fall back to the spectral heuristic.

Backward compatibility: the cache dict still includes the original
``vocal_stem_*`` keys (rms_db, mix_ratio_db, activity_frac, envelope_var,
n_slices) so the tagger's vocal-profile derivation in
``src/dj_tagger/derive.py`` reads the same names. The new per-stem keys
(``stem_<name>_<metric>``, ``dominance_<name>``) are additive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

# htdemucs source order. Validated at runtime against model.sources so the
# code stays robust if a future Demucs checkpoint reorders or renames stems.
STEM_NAMES = ("drums", "bass", "other", "vocals")
# Per-stem metrics we extract per slice. Order is for documentation only;
# all metrics flow through dicts.
STEM_METRICS = ("rms_db", "mix_ratio_db", "activity_frac",
                "centroid_hz", "flatness", "onset_rate", "zcr")


@dataclass
class VocalStemResult:
    """Aggregated per-track stem metrics. Keeps the legacy four `vocal_stem_*`
    fields populated for backward compat with the tagger's vocal-profile
    derivation, plus a generic ``stems`` block with per-stem metrics for the
    new subgenre-classifier feature wiring."""

    # Legacy vocal-only metrics (mirror of stems["vocals"]["*"] + envelope var)
    vocal_stem_rms_db: float
    vocal_stem_mix_ratio_db: float
    vocal_stem_activity_frac: float
    vocal_stem_envelope_var: float
    n_slices: int
    # Per-stem aggregated metrics: stems[name][metric] -> median across slices.
    stems: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-stem envelope variance of RMS across slices.
    envelope_var: dict[str, float] = field(default_factory=dict)
    # Per-stem dominance: stem RMS (linear) / sum of all stem RMS (linear).
    dominance: dict[str, float] = field(default_factory=dict)


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


def _stem_metrics(stem_mono: np.ndarray, mix_mono_rms: float, sr: int) -> dict[str, float]:
    """Compute per-stem scalar metrics from a mono stem signal.

    All metrics fit on a single track-level dict, so this stays cheap to
    aggregate later. Spectral features use librosa; failures fall back to
    NaN-safe sentinels so a single bad stem can't tank the whole track.
    """
    rms = _rms(stem_mono)
    metrics: dict[str, float] = {
        "rms_db": _to_db(rms),
        "mix_ratio_db": _to_db(rms) - _to_db(mix_mono_rms),
        "activity_frac": _activity_fraction(stem_mono, sr),
        # Linear RMS — needed downstream to compute stem dominance ratios.
        # Aggregator strips it from the returned dict so it doesn't leak into
        # the model's feature space.
        "_rms_linear": float(rms),
    }

    try:
        import librosa
    except ImportError:
        return metrics

    audio = stem_mono.astype(np.float32, copy=False)
    if audio.size < 2048:
        return metrics

    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        metrics["centroid_hz"] = float(np.nanmean(centroid))
    except Exception:
        pass
    try:
        flatness = librosa.feature.spectral_flatness(y=audio)
        metrics["flatness"] = float(np.nanmean(flatness))
    except Exception:
        pass
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        metrics["zcr"] = float(np.nanmean(zcr))
    except Exception:
        pass
    try:
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="time")
        duration = audio.size / sr if sr > 0 else 0.0
        metrics["onset_rate"] = float(len(onsets) / duration) if duration > 0 else 0.0
    except Exception:
        pass

    return metrics


def _analyze_slice(model, mix_slice: np.ndarray, sr: int) -> dict | None:
    """Run htdemucs on one slice; return per-stem metrics for each source.

    Returns ``{stem_name: {metric: value}}`` or ``None`` if the stems
    couldn't be produced.
    """
    import torch
    from demucs.apply import apply_model

    tensor = torch.from_numpy(mix_slice).unsqueeze(0)  # (1, 2, N)
    with torch.no_grad():
        stems = apply_model(model, tensor, split=True, overlap=0.25, progress=False)

    sources = list(model.sources)
    mix_mono = mix_slice.mean(axis=0)
    mix_rms = _rms(mix_mono)

    out: dict[str, dict[str, float]] = {}
    for stem_name in STEM_NAMES:
        if stem_name not in sources:
            continue
        idx = sources.index(stem_name)
        stem_audio = stems[0, idx].cpu().numpy()  # (2, N)
        stem_mono = stem_audio.mean(axis=0)
        out[stem_name] = _stem_metrics(stem_mono, mix_rms, sr)
    if not out:
        return None
    return out


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

    # slice_metrics[i] = {stem_name: {metric: value}} for slice i.
    slice_metrics: list[dict[str, dict[str, float]]] = []
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

    # Aggregate per-stem-per-metric using median across slices.
    stems_agg: dict[str, dict[str, float]] = {}
    envelope_var: dict[str, float] = {}
    for stem_name in STEM_NAMES:
        per_slice = [s.get(stem_name) for s in slice_metrics if s.get(stem_name)]
        if not per_slice:
            continue
        metric_keys = {k for d in per_slice for k in d.keys() if not k.startswith("_")}
        agg: dict[str, float] = {}
        for key in metric_keys:
            values = [d[key] for d in per_slice if key in d and np.isfinite(d[key])]
            if values:
                agg[key] = float(np.median(values))
        if agg:
            stems_agg[stem_name] = agg
            # Envelope variance is computed on the linear RMS series so it's
            # comparable across stems (dB envelope variance is scale-dependent).
            rms_lin = [d.get("_rms_linear") for d in per_slice if "_rms_linear" in d]
            envelope_var[stem_name] = float(np.var(rms_lin)) if rms_lin else 0.0

    # Stem dominance: each stem's median linear RMS / sum of all stems' medians.
    # Linear (not dB) because dB sums don't add — we want energy proportions.
    median_rms_lin: dict[str, float] = {}
    for stem_name, agg in stems_agg.items():
        per_slice = [s[stem_name].get("_rms_linear") for s in slice_metrics
                     if stem_name in s and "_rms_linear" in s[stem_name]]
        if per_slice:
            median_rms_lin[stem_name] = float(np.median(per_slice))
    total_lin = sum(median_rms_lin.values()) or 1.0
    dominance = {name: rms / total_lin for name, rms in median_rms_lin.items()}

    # Legacy vocals fields — mirror what the old code produced. The tagger's
    # derive.py reads these specific keys to score vocal profile.
    vocals = stems_agg.get("vocals", {})
    vocal_rms_db = vocals.get("rms_db", -90.0)
    vocal_mix_ratio_db = vocals.get("mix_ratio_db", -90.0)
    vocal_activity = vocals.get("activity_frac", 0.0)
    vocal_envelope_var = envelope_var.get("vocals", 0.0)

    return VocalStemResult(
        vocal_stem_rms_db=vocal_rms_db,
        vocal_stem_mix_ratio_db=vocal_mix_ratio_db,
        vocal_stem_activity_frac=vocal_activity,
        vocal_stem_envelope_var=vocal_envelope_var,
        n_slices=len(slice_metrics),
        stems=stems_agg,
        envelope_var=envelope_var,
        dominance=dominance,
    )


def result_to_dict(r: VocalStemResult | None) -> dict[str, float | int]:
    """Serialize VocalStemResult to a flat dict for the cache.

    Output schema (all float unless noted):
      Legacy vocals-only (kept for tagger backward compat):
        vocal_stem_rms_db, vocal_stem_mix_ratio_db, vocal_stem_activity_frac,
        vocal_stem_envelope_var, vocal_stem_n_slices (int)
      Per-stem aggregated:
        stem_<name>_<metric>      e.g. stem_drums_centroid_hz
        stem_<name>_envelope_var  cross-slice variance of linear RMS
      Stem energy proportions:
        dominance_<name>          fraction of total stem RMS (sums to ~1)
    """
    if r is None:
        return {}
    out: dict[str, float | int] = {
        "vocal_stem_rms_db": round(r.vocal_stem_rms_db, 2),
        "vocal_stem_mix_ratio_db": round(r.vocal_stem_mix_ratio_db, 2),
        "vocal_stem_activity_frac": round(r.vocal_stem_activity_frac, 3),
        # Envelope variance is now linear-RMS-based (was dB^2 before, but no
        # downstream consumer applied a threshold to it) — bump precision so
        # the typical 1e-4 magnitude doesn't round to zero.
        "vocal_stem_envelope_var": round(r.vocal_stem_envelope_var, 5),
        "vocal_stem_n_slices": r.n_slices,
    }
    for stem_name, agg in r.stems.items():
        for metric, value in agg.items():
            digits = 2 if metric in {"rms_db", "mix_ratio_db", "centroid_hz"} else 4
            out[f"stem_{stem_name}_{metric}"] = round(float(value), digits)
        if stem_name in r.envelope_var:
            out[f"stem_{stem_name}_envelope_var"] = round(float(r.envelope_var[stem_name]), 5)
    for stem_name, value in r.dominance.items():
        out[f"dominance_{stem_name}"] = round(float(value), 3)
    return out
