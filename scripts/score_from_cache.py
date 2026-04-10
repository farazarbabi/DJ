"""Fast scoring from cached raw features. No audio loading needed.

Usage: python scripts/score_from_cache.py [label]
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path


CACHE_PATH = "outputs/raw_features.pkl"
GT_PATH = "files/ground_truth.csv"


# ─── Tunable parameters ──────────────────────────────────────────────

# Energy normalization: (min, range) -- normalized = clip((val - min) / range, 0, 1)
# Calibrated from actual raw feature ranges across 40-track dataset:
#   rms: 0.167 - 0.345      centroid: 1165 - 2645
#   flux: 1.64 - 4.37       onset_rate: 1.46 - 7.27
#   low_ratio: 34 - 67
ENERGY_NORM = {
    "rms":      (0.15, 0.20),       # 0.15->0, 0.35->1.0
    "centroid": (1100.0, 1800.0),   # 1100->0, 2900->1.0
    "flux":     (1.5, 3.0),         # 1.5->0, 4.5->1.0
    "onset":    (1.0, 6.5),         # 1.0->0, 7.5->1.0
    "low_freq": (30.0, 40.0),       # 30->0, 70->1.0
    "bpm":      (85.0, 55.0),       # 85->0, 140->1.0
}

ENERGY_WEIGHTS = {
    "rms": 0.25,
    "centroid": 0.10,
    "flux": 0.25,
    "onset": 0.15,
    "low_freq": 0.05,
    "bpm": 0.20,
}

ENERGY_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]

# Vocal: which pre-computed threshold combo to use
# Key format matches Python f-string of floats: e.g. 0.10 -> "0.1"
VOCAL_KEY = "e0.12_f0.5_h1.5"
VOCAL_THRESHOLD = 0.20


def clip01(v):
    return max(0.0, min(1.0, v))


# ─── Dataset-relative normalization for vibe ───────────────────────────

def compute_dataset_norms(features: dict) -> dict:
    """Compute percentile-based normalization for vibe features across the dataset."""
    import numpy as np

    feat_keys = [
        "rms_raw", "centroid_mean", "centroid_var", "flatness", "bandwidth",
        "chroma_strength", "chroma_var", "spectral_stability",
        "onset_density", "onset_var", "perc_ratio", "low_ratio_vibe", "peakiness",
    ]
    norms = {}
    for k in feat_keys:
        vals = [f[k] for f in features.values()]
        p2 = float(np.percentile(vals, 2))
        p98 = float(np.percentile(vals, 98))
        norms[k] = (p2, max(p98 - p2, 1e-8))  # (min, range)
    return norms


def norm_feat(val: float, norms: dict, key: str) -> float:
    """Normalize a feature to [0, 1] using dataset percentile range."""
    lo, rng = norms[key]
    return clip01((val - lo) / rng)


def score_energy(f: dict) -> tuple[int, float, dict]:
    """Score energy from raw features. Returns (level, composite, details)."""
    raw_map = {
        "rms": "rms_raw",
        "centroid": "centroid_mean",
        "flux": "flux_raw",
        "onset": "onset_rate",
        "low_freq": "low_ratio",
        "bpm": "bpm",
    }
    norms = {}
    for key, (lo, rng) in ENERGY_NORM.items():
        norms[key] = clip01((f[raw_map[key]] - lo) / rng)

    composite = sum(ENERGY_WEIGHTS[k] * norms[k] for k in ENERGY_WEIGHTS)

    level = 1
    for i, t in enumerate(ENERGY_THRESHOLDS):
        if composite >= t:
            level = i + 2

    return level, composite, norms


def score_vibe(f: dict, dnorms: dict) -> tuple[str, dict]:
    """Score vibe using evidence-based override approach.

    MEL is the baseline (79% of tracks). Other vibes only win when
    there's strong, specific evidence. Based on feature analysis:
    - DRK: high onset_var + high flatness (rhythmic aggression + noise)
    - TRB: high perc_ratio (percussion-dominant)
    - RAW: low low_ratio (not bass-heavy, non-electronic character)
    """
    # Raw feature values (not normalized — use actual thresholds)
    onset_var = f["onset_var"]
    perc_ratio = f["perc_ratio"]
    flatness = f["flatness"]
    low_ratio = f["low_ratio_vibe"]
    chroma_var = f["chroma_var"]
    centroid_mean = f["centroid_mean"]
    flux_raw = f.get("flux_raw", 3.0)
    onset_rate = f.get("onset_rate", 5.0)
    rms = f["rms_raw"]

    scores = {}

    # MEL: baseline score — chroma_var is the melodic signal
    mel_chroma = min(1.0, chroma_var / 0.065)
    mel_lowflat = clip01((0.025 - flatness) / 0.015)   # clean (low flatness) = more melodic
    scores["MEL"] = 0.38 + 0.25 * mel_chroma + 0.12 * mel_lowflat

    # DRK: aggressive rhythmic character + noisiness + bass-heavy
    drk_onset = clip01((onset_var - 1.5) / 2.0)        # 1.5->0, 3.5->1
    drk_flat = clip01((flatness - 0.010) / 0.030)       # 0.010->0, 0.04->1
    drk_low = clip01((low_ratio - 45) / 25)              # 45->0, 70->1
    drk_flux = clip01((flux_raw - 2.0) / 2.5)           # 2.0->0, 4.5->1
    scores["DRK"] = (
        0.25 * drk_onset
        + 0.30 * drk_flat
        + 0.25 * drk_low
        + 0.20 * drk_flux
    )

    # TRB: very high percussion ratio — must be clearly percussive, not noisy
    trib_perc = clip01((perc_ratio - 0.40) / 0.15)      # 0.40->0, 0.55->1 (tight)
    trib_clean = clip01((0.03 - flatness) / 0.02)       # low flatness = clean percussion
    scores["TRB"] = 0.70 * trib_perc + 0.30 * trib_clean

    # RAW: non-electronic character (very low bass ratio + sparse)
    raw_nobass = clip01((42 - low_ratio) / 12)           # 42->0, 30->1
    raw_sparse = clip01((1.5 - f["onset_density"]) / 0.8)
    raw_lowvar = clip01((1.5 - onset_var) / 1.0)        # low rhythmic complexity
    scores["RAW"] = 0.45 * raw_nobass + 0.30 * raw_sparse + 0.25 * raw_lowvar

    # HYPN: high stability, low onset variance
    stability = f["spectral_stability"]
    scores["HYPN"] = 0.50 * stability + 0.50 * clip01((2.0 - onset_var) / 1.5)

    # DEEP: low centroid + low loudness + bass-heavy
    scores["DEEP"] = (
        0.35 * clip01((2000 - centroid_mean) / 1000)
        + 0.35 * clip01((0.22 - rms) / 0.10)
        + 0.30 * clip01((low_ratio - 50) / 20)
    )

    # ACID: extreme spectral movement (filter sweeps) — very rare
    centroid_var = f["centroid_var"]
    scores["ACID"] = clip01((centroid_var - 0.75) / 0.40)

    # ATM: sparse + quiet
    onset_density = f["onset_density"]
    scores["ATM"] = (
        0.50 * clip01((1.5 - onset_density) / 1.0)
        + 0.50 * clip01((0.20 - rms) / 0.08)
    )

    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    best = max(scores, key=scores.get)
    return best, scores


def score_vocal(f: dict) -> tuple[str, float]:
    """Score vocal from cached stats. Returns (V/NV, score)."""
    stats = f.get("vocal_stats", {})
    score = stats.get(VOCAL_KEY, 0.0)
    has_vocals = score > VOCAL_THRESHOLD
    return "V" if has_vocals else "NV", score


# Camelot mapping: (pitch_class, mode) -> Camelot code
_KEY_TO_CAMELOT = {
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

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _gap_confidence(best: float, second: float) -> float:
    gap = best - second
    return max(0.0, min(1.0, gap / 0.15))


def score_key(f: dict) -> tuple[str, str, float]:
    """Score key from cached segment chromas using EDMA profiles.

    Skips last segment (outro). Returns (camelot, key_name, confidence).
    """
    import numpy as np

    seg_chromas = f.get("segment_chromas", [])
    if not seg_chromas:
        return "??", "??", 0.0

    # EDMA profiles (matches constants.py)
    major = np.array([6.0, 1.0, 3.5, 1.0, 5.0, 3.0, 1.0, 5.5, 1.0, 2.5, 1.0, 3.0])
    minor = np.array([6.0, 1.0, 3.0, 5.0, 1.0, 3.0, 1.0, 5.5, 3.5, 1.0, 2.0, 3.0])

    # Skip last segment (outro has weak key signal)
    use_chromas = seg_chromas[:-1] if len(seg_chromas) > 2 else seg_chromas

    votes: dict[tuple[int, str], float] = {}
    for seg_chroma in use_chromas:
        chroma_avg = np.array(seg_chroma)
        best_corr = -1.0
        best_key = 0
        best_mode = "major"
        second_corr = -1.0
        for shift in range(12):
            rotated = np.roll(chroma_avg, -shift)
            for mode, prof in [("major", major), ("minor", minor)]:
                c = float(np.corrcoef(rotated, prof)[0, 1])
                if c > best_corr:
                    second_corr = best_corr
                    best_corr = c
                    best_key = shift
                    best_mode = mode
                elif c > second_corr:
                    second_corr = c
        conf = _gap_confidence(best_corr, second_corr)
        vote_key = (best_key, best_mode)
        votes[vote_key] = votes.get(vote_key, 0.0) + conf

    if not votes:
        return "??", "??", 0.0

    winner = max(votes, key=votes.get)
    best_key, best_mode = winner
    total_weight = sum(votes.values())
    vote_conf = votes[winner] / total_weight if total_weight > 0 else 0.0

    camelot = _KEY_TO_CAMELOT[(best_key, best_mode)]
    suffix = "m" if best_mode == "minor" else ""
    key_name = f"{NOTE_NAMES[best_key]}{suffix}"
    return camelot, key_name, vote_conf


def camelot_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    try:
        an, al = int(a[:-1]), a[-1]
        bn, bl = int(b[:-1]), b[-1]
    except (ValueError, IndexError):
        return 99
    if al == bl:
        return min(abs(an - bn), 12 - abs(an - bn))
    return min(abs(an - bn), 12 - abs(an - bn)) + 1


def load_ground_truth() -> dict[str, dict]:
    gt = {}
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt[row["file_name"].strip()] = {
                "energy": row["Energy"].strip(),
                "vibe": row["Vibe"].strip(),
                "key": row["Key_rekordbox"].strip(),
                "vocal": row["Vocal"].strip(),
            }
    return gt


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "current"

    if not os.path.exists(CACHE_PATH):
        print(f"No cached features at {CACHE_PATH}. Run extract_features.py first.")
        return

    with open(CACHE_PATH, "rb") as f:
        features = pickle.load(f)

    gt = load_ground_truth()
    matched = sorted(set(features.keys()) & set(gt.keys()))

    if not matched:
        print("No matching files between cache and ground truth!")
        return

    # Compute dataset-level normalization for vibe
    dnorms = compute_dataset_norms(features)

    n = len(matched)
    e_ok = e_off1 = v_ok = voc_ok = k_ok = k_compat = 0
    errors = []

    # Check if key data is available
    has_key_data = any("segment_chromas" in features[fn] for fn in matched)

    print(f"\n{'='*70}")
    print(f"  Scoring: {label} | {n} tracks")
    print(f"{'='*70}")

    for fname in matched:
        f = features[fname]
        g = gt[fname]

        elevel, ecomp, enorms = score_energy(f)
        vlabel, vscores = score_vibe(f, dnorms)
        voclabel, vocscore = score_vocal(f)

        pred_e = f"E{elevel}"
        gt_e = g["energy"]
        gt_v = g["vibe"]
        gt_voc = g["vocal"]
        gt_k = g["key"]

        e_match = pred_e == gt_e
        e_close = abs(elevel - int(gt_e[1])) <= 1
        v_match = vlabel == gt_v
        voc_match = voclabel == gt_voc

        if e_match: e_ok += 1
        if e_close: e_off1 += 1
        if v_match: v_ok += 1
        if voc_match: voc_ok += 1

        errs = []
        if not e_match: errs.append(f"E:{gt_e}->{pred_e}({ecomp:.2f})")
        if not v_match:
            top2 = sorted(vscores.items(), key=lambda x: -x[1])[:2]
            errs.append(f"V:{gt_v}->{vlabel}({top2[0][1]:.2f},{top2[1][0]}={top2[1][1]:.2f})")
        if not voc_match: errs.append(f"Voc:{gt_voc}->{voclabel}({vocscore:.3f})")

        # Key scoring
        if has_key_data and "segment_chromas" in f:
            pred_k, key_name, key_conf = score_key(f)
            k_dist = camelot_distance(pred_k, gt_k)
            if k_dist == 0: k_ok += 1
            if k_dist <= 1: k_compat += 1
            if k_dist > 0:
                compat_str = " (compat)" if k_dist == 1 else ""
                errs.append(f"K:{gt_k}->{pred_k}{compat_str}")

        if errs:
            errors.append(f"  {fname[:55]:<55s} {' | '.join(errs)}")

    print(f"  Energy exact:    {e_ok:2d}/{n} ({100*e_ok/n:.0f}%)")
    print(f"  Energy +/-1:     {e_off1:2d}/{n} ({100*e_off1/n:.0f}%)")
    print(f"  Vibe exact:      {v_ok:2d}/{n} ({100*v_ok/n:.0f}%)")
    print(f"  Vocal exact:     {voc_ok:2d}/{n} ({100*voc_ok/n:.0f}%)")
    if has_key_data:
        print(f"  Key exact:       {k_ok:2d}/{n} ({100*k_ok/n:.0f}%)")
        print(f"  Key compatible:  {k_compat:2d}/{n} ({100*k_compat/n:.0f}%)")
    print(f"\n  Overall:         {(e_ok + v_ok + voc_ok + k_ok)}/{4*n} ({100*(e_ok+v_ok+voc_ok+k_ok)/(4*n):.0f}%)")

    if errors:
        print(f"\n--- Mismatches ---")
        for e in errors:
            print(e)

    # Dump raw feature ranges for calibration
    print(f"\n--- Raw feature ranges (for energy calibration) ---")
    for key in ["rms_raw", "centroid_mean", "flux_raw", "onset_rate", "low_ratio"]:
        vals = [features[fn][key] for fn in matched]
        print(f"  {key:15s}: min={min(vals):.4f}  max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}")

    # Vocal stats ranges
    print(f"\n--- Vocal score distribution (key={VOCAL_KEY}, threshold={VOCAL_THRESHOLD}) ---")
    for fname in matched:
        f = features[fname]
        g = gt[fname]
        stats = f.get("vocal_stats", {})
        score = stats.get(VOCAL_KEY, 0.0)
        pred = "V" if score > VOCAL_THRESHOLD else "NV"
        marker = " OK" if pred == g["vocal"] else " XX"
        print(f"  {fname[:50]:<50s} {score:.3f} -> {pred} (gt={g['vocal']}){marker}")


if __name__ == "__main__":
    main()
