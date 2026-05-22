"""Spike: validate Demucs vocal-stem ratios separate INST from VOC.

Reads a labeled mini-set (file basenames + expected_label), runs htdemucs on
3 x 30s slices (early/middle/late) per track, and reports:

    vocal_stem_rms_db        - RMS of the vocal stem in dBFS
    vocal_stem_mix_ratio_db  - vocal stem RMS - full-mix RMS in dB
    vocal_stem_activity_frac - fraction of 0.5s windows where vocal stem RMS
                               exceeds (noise_floor + 6 dB)

Aggregates per track as the median across slices. Prints per-class summaries
so we can decide on threshold bands.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = REPO_ROOT / "Music"
REGISTRY_CSV = REPO_ROOT / "outputs" / "registry" / "registry_overview.csv"
OUTPUT_CSV = REPO_ROOT / "outputs" / "vocal_spike_results.csv"
OUTPUT_JSON = REPO_ROOT / "outputs" / "vocal_spike_results.json"

SLICE_SECONDS = 30.0
SLICE_POSITIONS = (0.20, 0.50, 0.75)  # fraction of track for early/middle/late
NOISE_FLOOR_DB = -55.0
ACTIVITY_MARGIN_DB = 6.0
WINDOW_SECONDS = 0.5


def _to_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2) + 1e-12))


def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load stereo float32 at target_sr. Returns (samples [2, N], sr)."""
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    return y.astype(np.float32), sr


def _slice_indices(n_samples: int, sr: int) -> list[tuple[int, int]]:
    win = int(SLICE_SECONDS * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    out = []
    for frac in SLICE_POSITIONS:
        center = int(frac * n_samples)
        start = max(0, min(n_samples - win, center - win // 2))
        out.append((start, start + win))
    return out


def _activity_fraction(stem_mono: np.ndarray, sr: int) -> float:
    win = int(WINDOW_SECONDS * sr)
    if win <= 0 or stem_mono.size == 0:
        return 0.0
    n_win = stem_mono.size // win
    if n_win == 0:
        return 0.0
    energies = []
    for i in range(n_win):
        seg = stem_mono[i * win : (i + 1) * win]
        energies.append(_to_db(_rms(seg)))
    threshold = NOISE_FLOOR_DB + ACTIVITY_MARGIN_DB
    return float(np.mean(np.array(energies) > threshold))


def _analyze_slice(model, mix_slice: np.ndarray, sr: int) -> dict:
    """Run demucs on a slice and return per-slice metrics."""
    from demucs.apply import apply_model

    # apply_model expects shape (batch, channels, samples)
    tensor = torch.from_numpy(mix_slice).unsqueeze(0)  # (1, 2, N)
    with torch.no_grad():
        stems = apply_model(model, tensor, split=True, overlap=0.25, progress=False)
    # stems shape: (batch, n_sources, channels, samples)
    sources = list(model.sources)
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
        "mix_rms_db": _to_db(rms_mix),
    }


def _aggregate(slices: list[dict]) -> dict:
    if not slices:
        return {}
    keys = slices[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in slices]
        out[k + "_median"] = float(median(vals))
        out[k + "_max"] = float(max(vals))
        out[k + "_min"] = float(min(vals))
    return out


def load_labeled_set() -> list[tuple[str, str]]:
    """Pick a labeled mini-set from the registry overview.

    INST: ss_instrumentalness >= 0.85 and currently tagged VOC (failure cases).
    VOC:  ss_instrumentalness <= 0.10.
    """
    if not REGISTRY_CSV.exists():
        print(f"Registry not found: {REGISTRY_CSV}", file=sys.stderr)
        return []

    def _to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    inst, voc = [], []
    with REGISTRY_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            si = _to_float(r.get("ss_instrumentalness", ""))
            fn = r.get("file_name", "")
            if not fn or si is None:
                continue
            if not (MUSIC_DIR / fn).exists():
                continue
            if si >= 0.85 and r.get("tagger_vocal", "") == "VOC":
                inst.append((si, fn))
            elif si <= 0.10:
                voc.append((si, fn))

    inst.sort(reverse=True)
    voc.sort()
    pairs = [(fn, "INST") for _, fn in inst[:15]] + [(fn, "VOC") for _, fn in voc[:11]]
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tracks (0=all).")
    ap.add_argument("--model", default="htdemucs")
    args = ap.parse_args()

    pairs = load_labeled_set()
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("No labeled tracks found. Make sure registry overview exists.", file=sys.stderr)
        return 1

    print(f"Labeled set: {len(pairs)} tracks", flush=True)
    n_inst = sum(1 for _, lbl in pairs if lbl == "INST")
    n_voc = sum(1 for _, lbl in pairs if lbl == "VOC")
    print(f"  INST: {n_inst}   VOC: {n_voc}", flush=True)

    from demucs.pretrained import get_model

    print(f"Loading {args.model}...", flush=True)
    model = get_model(args.model)
    model.eval()
    target_sr = model.samplerate

    results: list[dict] = []
    t0 = time.time()
    for idx, (fn, label) in enumerate(pairs, 1):
        path = MUSIC_DIR / fn
        try:
            samples, sr = _load_audio(path, target_sr)
        except Exception as e:
            print(f"  [{idx}/{len(pairs)}] LOAD FAIL  {fn}: {e}", flush=True)
            continue

        slice_ranges = _slice_indices(samples.shape[1], sr)
        slice_metrics = []
        ts = time.time()
        for (a, b) in slice_ranges:
            slice_audio = samples[:, a:b]
            try:
                m = _analyze_slice(model, slice_audio, sr)
            except Exception as e:
                print(f"  slice fail @{a/sr:.0f}s: {e}", flush=True)
                continue
            slice_metrics.append(m)
        agg = _aggregate(slice_metrics)
        rec = {"file_name": fn, "expected_label": label, **agg, "elapsed_s": round(time.time() - ts, 2)}
        results.append(rec)
        print(
            f"  [{idx}/{len(pairs)}] {label}  "
            f"ratio_db={agg.get('vocal_stem_mix_ratio_db_median', float('nan')):6.2f}  "
            f"act={agg.get('vocal_stem_activity_frac_median', float('nan')):.2f}  "
            f"rms_db={agg.get('vocal_stem_rms_db_median', float('nan')):6.2f}  "
            f"{rec['elapsed_s']}s  {fn[:60]}",
            flush=True,
        )

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/max(len(pairs),1):.1f}s/track avg)")

    # Persist
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in results for k in r.keys()})
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_JSON}")

    # Class summary
    def _stats(values: list[float]) -> str:
        if not values:
            return "n=0"
        a = np.array(values)
        return f"n={len(a)}  median={np.median(a):.2f}  mean={a.mean():.2f}  min={a.min():.2f}  max={a.max():.2f}  p25={np.percentile(a,25):.2f}  p75={np.percentile(a,75):.2f}"

    for metric in ("vocal_stem_mix_ratio_db_median", "vocal_stem_activity_frac_median", "vocal_stem_rms_db_median"):
        print(f"\n=== {metric} ===")
        for cls in ("INST", "VOC"):
            vals = [r[metric] for r in results if r["expected_label"] == cls and metric in r]
            print(f"  {cls}: {_stats(vals)}")

    # Separability check on mix-ratio
    inst_vals = sorted(r["vocal_stem_mix_ratio_db_median"] for r in results if r["expected_label"] == "INST" and "vocal_stem_mix_ratio_db_median" in r)
    voc_vals = sorted(r["vocal_stem_mix_ratio_db_median"] for r in results if r["expected_label"] == "VOC" and "vocal_stem_mix_ratio_db_median" in r)
    if inst_vals and voc_vals:
        inst_p90 = float(np.percentile(inst_vals, 90))
        voc_p10 = float(np.percentile(voc_vals, 10))
        gap = voc_p10 - inst_p90
        print(f"\nSeparability (mix_ratio_db):  INST p90={inst_p90:.2f}   VOC p10={voc_p10:.2f}   gap={gap:.2f} dB")
        if gap > 2.0:
            print("  -> CLEAN separation; threshold band feasible.")
        elif gap > 0.0:
            print("  -> Marginal separation; need combined feature (ratio + activity_frac).")
        else:
            print("  -> OVERLAP; ratio alone insufficient. Investigate failure modes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
