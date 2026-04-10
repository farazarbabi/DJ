"""Grid search for optimal key detection parameters.

Pre-computes all correlations per (profile, track, segment), then
sweeps parameter combos with fast lookups. Runs in seconds.
"""

from __future__ import annotations

import csv
import itertools
import pickle
import sys

import numpy as np

CACHE_PATH = "outputs/raw_features.pkl"
GT_PATH = "files/ground_truth.csv"

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KEY_TO_CAMELOT = {
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

PROFILES = {
    "edma": {
        "major": [6.0, 1.0, 3.5, 1.0, 5.0, 3.0, 1.0, 5.5, 1.0, 2.5, 1.0, 3.0],
        "minor": [6.0, 1.0, 3.0, 5.0, 1.0, 3.0, 1.0, 5.5, 3.5, 1.0, 2.0, 3.0],
    },
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    },
    "simple": {
        "major": [5.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 4.5, 1.0, 2.0, 1.0, 2.0],
        "minor": [5.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 4.5, 2.0, 1.0, 2.0, 2.0],
    },
    "edma_sharp": {
        "major": [7.0, 0.5, 3.0, 0.5, 4.5, 2.5, 0.5, 6.0, 0.5, 2.0, 0.5, 2.5],
        "minor": [7.0, 0.5, 2.5, 5.0, 0.5, 2.5, 0.5, 6.0, 3.0, 0.5, 1.5, 2.5],
    },
    "edma_bass": {
        "major": [8.0, 0.5, 3.0, 0.5, 4.5, 2.5, 0.5, 5.0, 0.5, 2.0, 0.5, 2.5],
        "minor": [8.0, 0.5, 2.5, 4.5, 0.5, 2.5, 0.5, 5.0, 3.0, 0.5, 1.5, 2.5],
    },
}


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


def correlate_chroma(chroma: np.ndarray, major: np.ndarray, minor: np.ndarray):
    """Correlate chroma against all 24 keys. Returns sorted list of (corr, shift, mode)."""
    scores = []
    for shift in range(12):
        rotated = np.roll(chroma, -shift)
        cmaj = float(np.corrcoef(rotated, major)[0, 1])
        cmin = float(np.corrcoef(rotated, minor)[0, 1])
        scores.append((cmaj, shift, "major"))
        scores.append((cmin, shift, "minor"))
    scores.sort(key=lambda x: -x[0])
    return scores


def precompute_correlations(features, matched, profiles):
    """Pre-compute all (profile, track, segment) correlations.

    Returns: dict[profile_name][fname] = list of segment results,
    where each segment result = (best_corr, second_corr, best_key, best_mode)
    for both full-signal and harmonic chromas.
    """
    precomp = {}

    for pname, prof in profiles.items():
        major = np.array(prof["major"])
        minor = np.array(prof["minor"])
        precomp[pname] = {}

        for fname in matched:
            f = features[fname]
            seg_chromas = f.get("segment_chromas", [])
            h_chroma = f.get("harmonic_chroma", None)

            # Full signal per-segment
            full_segments = []
            for sc in seg_chromas:
                scores = correlate_chroma(np.array(sc), major, minor)
                best = scores[0]
                second = scores[1]
                full_segments.append({
                    "best_corr": best[0], "best_key": best[1], "best_mode": best[2],
                    "second_corr": second[0],
                })

            # Harmonic whole-track
            h_result = None
            if h_chroma is not None:
                scores = correlate_chroma(np.array(h_chroma), major, minor)
                best = scores[0]
                second = scores[1]
                h_result = {
                    "best_corr": best[0], "best_key": best[1], "best_mode": best[2],
                    "second_corr": second[0],
                }

            # Blended chromas
            blend_segments = {}
            if h_chroma is not None:
                h_arr = np.array(h_chroma)
                for bw in [30, 50, 70]:
                    w = bw / 100.0
                    segs = []
                    for sc in seg_chromas:
                        blended = w * np.array(sc) + (1 - w) * h_arr
                        bmax = np.max(blended)
                        if bmax > 0:
                            blended = blended / bmax
                        scores = correlate_chroma(blended, major, minor)
                        best = scores[0]
                        second = scores[1]
                        segs.append({
                            "best_corr": best[0], "best_key": best[1], "best_mode": best[2],
                            "second_corr": second[0],
                        })
                    blend_segments[bw] = segs

            precomp[pname][fname] = {
                "full": full_segments,
                "harmonic": h_result,
                "blends": blend_segments,
            }

    return precomp


def score_key_fast(
    precomp_track: dict,
    signal: str,
    skip: str,
    conf_mode: str,
    agg: str,
    minor_bias: float,
    gap_divisor: float,
) -> tuple[str, float]:
    """Score key from pre-computed correlations."""

    if signal == "harmonic":
        h = precomp_track["harmonic"]
        if h is None:
            return "??", 0.0
        segments = [h]
    elif signal.startswith("blend_"):
        bw = int(signal.split("_")[1])
        segments = precomp_track["blends"].get(bw, precomp_track["full"])
    else:
        segments = precomp_track["full"]

    if not segments:
        return "??", 0.0

    # Skip segments
    n = len(segments)
    if signal != "harmonic":
        if skip == "last" and n > 2:
            segments = segments[:-1]
        elif skip == "first_last" and n > 3:
            segments = segments[1:-1]

    # Vote
    votes: dict[tuple[int, str], float] = {}
    for seg in segments:
        best_corr = seg["best_corr"]
        second_corr = seg["second_corr"]
        best_key = seg["best_key"]
        best_mode = seg["best_mode"]

        gap = best_corr - second_corr
        if conf_mode == "gap":
            conf = max(0.0, min(1.0, gap / gap_divisor))
        elif conf_mode == "gap_abs":
            conf = max(0.0, min(1.0, gap / gap_divisor)) * max(0.0, best_corr)
        elif conf_mode == "gap_sq":
            conf = max(0.0, min(1.0, gap / gap_divisor)) ** 2
        elif conf_mode == "uniform":
            conf = 1.0
        else:
            conf = max(0.0, min(1.0, gap / gap_divisor))

        if best_mode == "minor":
            conf += minor_bias

        vk = (best_key, best_mode)
        if agg == "sum":
            votes[vk] = votes.get(vk, 0.0) + conf
        elif agg == "sum_sq":
            votes[vk] = votes.get(vk, 0.0) + conf ** 2

    if not votes:
        return "??", 0.0

    winner = max(votes, key=votes.get)
    total_weight = sum(votes.values())
    vote_conf = votes[winner] / total_weight if total_weight > 0 else 0.0
    camelot = KEY_TO_CAMELOT[(winner[0], winner[1])]
    return camelot, vote_conf


def load_ground_truth() -> dict[str, dict]:
    gt = {}
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt[row["file_name"].strip()] = {"key": row["Key_rekordbox"].strip()}
    return gt


def main():
    with open(CACHE_PATH, "rb") as f:
        features = pickle.load(f)

    gt = load_ground_truth()
    matched = sorted(set(features.keys()) & set(gt.keys()))
    matched = [fn for fn in matched if "segment_chromas" in features[fn]]
    n = len(matched)
    print(f"Tracks with key data: {n}")

    # Pre-compute all correlations
    print("Pre-computing correlations...")
    precomp = precompute_correlations(features, matched, PROFILES)
    print("Done. Starting grid search...")

    # Parameter grid
    profile_names = list(PROFILES.keys())
    signals = ["full", "harmonic", "blend_30", "blend_50", "blend_70"]
    skips = ["none", "last", "first_last"]
    conf_modes = ["gap", "gap_abs", "gap_sq", "uniform"]
    aggs = ["sum", "sum_sq"]
    minor_biases = [0.0, 0.05, 0.10]
    gap_divisors = [0.10, 0.15, 0.20]

    combos = list(itertools.product(
        profile_names, signals, skips, conf_modes, aggs, minor_biases, gap_divisors
    ))
    print(f"Testing {len(combos)} parameter combinations...")

    results = []

    for combo in combos:
        prof, sig, skip, conf, agg, mbias, gdiv = combo
        exact = 0
        neighbour = 0
        non_neighbour = 0

        for fname in matched:
            gt_key = gt[fname]["key"]
            pred_key, _ = score_key_fast(
                precomp[prof][fname], sig, skip, conf, agg, mbias, gdiv
            )
            dist = camelot_distance(pred_key, gt_key)
            if dist == 0:
                exact += 1
            if dist <= 1:
                neighbour += 1
            if dist > 1:
                non_neighbour += 1

        results.append((non_neighbour, -exact, -neighbour, combo, exact, neighbour))

    # Sort: min non-neighbour, then max exact, then max neighbour
    results.sort()

    print(f"\n{'='*100}")
    print(f"  TOP 30 CONFIGURATIONS (sorted: min non-neighbour, then max exact)")
    print(f"{'='*100}")
    hdr = f"  {'NN':>3s} {'Exact':>5s} {'Neigh':>5s}  {'Profile':<12s} {'Signal':<10s} {'Skip':<12s} {'Conf':<10s} {'Agg':<7s} {'Mbias':>5s} {'Gdiv':>5s}"
    print(hdr)
    print(f"  {'---':>3s} {'-----':>5s} {'-----':>5s}  {'-'*11:<12s} {'-'*9:<10s} {'-'*11:<12s} {'-'*9:<10s} {'-'*6:<7s} {'-----':>5s} {'----':>5s}")

    printed = 0
    for nn, neg_ex, neg_nb, combo, exact, neighbour in results:
        if printed >= 30:
            break
        prof, sig, skip, conf, agg, mbias, gdiv = combo
        print(f"  {nn:3d} {exact:5d} {neighbour:5d}  {prof:<12s} {sig:<10s} {skip:<12s} {conf:<10s} {agg:<7s} {mbias:5.2f} {gdiv:5.2f}")
        printed += 1

    # Best config details
    best = results[0]
    nn, _, _, combo, exact, neighbour = best
    prof, sig, skip, conf, agg, mbias, gdiv = combo
    print(f"\n{'='*100}")
    print(f"  BEST: non-neighbour={nn}, exact={exact}/{n} ({100*exact/n:.0f}%), "
          f"neighbour={neighbour}/{n} ({100*neighbour/n:.0f}%)")
    print(f"  Config: profile={prof}, signal={sig}, skip={skip}, "
          f"conf={conf}, agg={agg}, minor_bias={mbias}, gap_div={gdiv}")
    print(f"{'='*100}")

    # Per-track for the best
    print(f"\n--- Per-track results (best config) ---")
    for fname in matched:
        gt_key = gt[fname]["key"]
        pred_key, key_conf = score_key_fast(
            precomp[prof][fname], sig, skip, conf, agg, mbias, gdiv
        )
        dist = camelot_distance(pred_key, gt_key)
        marker = " OK" if dist == 0 else f" +{dist}" if dist <= 1 else f" XX({dist})"
        short = fname[:55]
        print(f"  {short:<55s} GT={gt_key:>3s}  Pred={pred_key:>3s}  conf={key_conf:.3f}{marker}")

    # Also show the current baseline for comparison
    print(f"\n--- Current baseline (edma, full, skip-last, gap, sum, 0.0, 0.15) ---")
    for fname in matched:
        gt_key = gt[fname]["key"]
        pred_key, key_conf = score_key_fast(
            precomp["edma"][fname], "full", "last", "gap", "sum", 0.0, 0.15
        )
        dist = camelot_distance(pred_key, gt_key)
        marker = " OK" if dist == 0 else f" +{dist}" if dist <= 1 else f" XX({dist})"
        short = fname[:55]
        if dist > 0:
            print(f"  {short:<55s} GT={gt_key:>3s}  Pred={pred_key:>3s}  conf={key_conf:.3f}{marker}")


if __name__ == "__main__":
    main()
