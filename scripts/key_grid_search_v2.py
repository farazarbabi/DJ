"""Key grid search v2: advanced strategies.

Adds:
- Profile ensemble (majority vote across profiles)
- Top-2 voting (second-best key gets fractional vote)
- Merged segments (average adjacent pairs for 4-segment analysis)
- Harmonic+full ensemble
"""

from __future__ import annotations

import csv
import itertools
import pickle
import sys
from collections import Counter

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
    """Returns sorted list of (corr, shift, mode) for all 24 keys."""
    scores = []
    for shift in range(12):
        rotated = np.roll(chroma, -shift)
        cmaj = float(np.corrcoef(rotated, major)[0, 1])
        cmin = float(np.corrcoef(rotated, minor)[0, 1])
        scores.append((cmaj, shift, "major"))
        scores.append((cmin, shift, "minor"))
    scores.sort(key=lambda x: -x[0])
    return scores


def precompute_all(features, matched, profiles):
    """Pre-compute correlations for all profiles, tracks, segments, signals."""
    precomp = {}

    for pname, prof in profiles.items():
        major = np.array(prof["major"])
        minor = np.array(prof["minor"])
        precomp[pname] = {}

        for fname in matched:
            f = features[fname]
            seg_chromas = f.get("segment_chromas", [])
            h_chroma = f.get("harmonic_chroma", None)

            # Full signal: top-2 per segment
            full_segs = []
            for sc in seg_chromas:
                scores = correlate_chroma(np.array(sc), major, minor)
                full_segs.append({
                    "top1": (scores[0][0], scores[0][1], scores[0][2]),
                    "top2": (scores[1][0], scores[1][1], scores[1][2]),
                })

            # Merged segments (average adjacent pairs)
            merged_segs = []
            for i in range(0, len(seg_chromas) - 1, 2):
                avg = (np.array(seg_chromas[i]) + np.array(seg_chromas[i + 1])) / 2.0
                avg_max = np.max(avg)
                if avg_max > 0:
                    avg = avg / avg_max
                scores = correlate_chroma(avg, major, minor)
                merged_segs.append({
                    "top1": (scores[0][0], scores[0][1], scores[0][2]),
                    "top2": (scores[1][0], scores[1][1], scores[1][2]),
                })

            # Harmonic whole-track
            h_result = None
            if h_chroma is not None:
                scores = correlate_chroma(np.array(h_chroma), major, minor)
                h_result = {
                    "top1": (scores[0][0], scores[0][1], scores[0][2]),
                    "top2": (scores[1][0], scores[1][1], scores[1][2]),
                }

            # Blended 50/50
            blend_segs = []
            if h_chroma is not None:
                h_arr = np.array(h_chroma)
                for sc in seg_chromas:
                    blended = 0.5 * np.array(sc) + 0.5 * h_arr
                    bmax = np.max(blended)
                    if bmax > 0:
                        blended = blended / bmax
                    scores = correlate_chroma(blended, major, minor)
                    blend_segs.append({
                        "top1": (scores[0][0], scores[0][1], scores[0][2]),
                        "top2": (scores[1][0], scores[1][1], scores[1][2]),
                    })

            precomp[pname][fname] = {
                "full": full_segs,
                "merged": merged_segs,
                "harmonic": h_result,
                "blend50": blend_segs,
            }

    return precomp


def vote_segments(segments, skip, conf_mode, agg, minor_bias, gdiv, top2_weight):
    """Run voting on a list of segment results. Returns (camelot, confidence)."""
    if not segments:
        return "??", 0.0

    n = len(segments)
    if skip == "last" and n > 2:
        segments = segments[:-1]
    elif skip == "first_last" and n > 3:
        segments = segments[1:-1]

    votes = {}
    for seg in segments:
        best_corr, best_key, best_mode = seg["top1"]
        second_corr = seg["top2"][0]

        gap = best_corr - second_corr
        if conf_mode == "gap":
            conf = max(0.0, min(1.0, gap / gdiv))
        elif conf_mode == "gap_abs":
            conf = max(0.0, min(1.0, gap / gdiv)) * max(0.0, best_corr)
        elif conf_mode == "gap_sq":
            conf = max(0.0, min(1.0, gap / gdiv)) ** 2
        else:
            conf = 1.0

        if best_mode == "minor":
            conf += minor_bias

        vk = (best_key, best_mode)
        if agg == "sum":
            votes[vk] = votes.get(vk, 0.0) + conf
        else:
            votes[vk] = votes.get(vk, 0.0) + conf ** 2

        # Top-2 voting: second best key gets fractional vote
        if top2_weight > 0:
            _, k2, m2 = seg["top2"]
            conf2 = conf * top2_weight
            vk2 = (k2, m2)
            if agg == "sum":
                votes[vk2] = votes.get(vk2, 0.0) + conf2
            else:
                votes[vk2] = votes.get(vk2, 0.0) + conf2 ** 2

    if not votes:
        return "??", 0.0

    winner = max(votes, key=votes.get)
    total = sum(votes.values())
    vote_conf = votes[winner] / total if total > 0 else 0.0
    camelot = KEY_TO_CAMELOT[(winner[0], winner[1])]
    return camelot, vote_conf


def load_ground_truth():
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
    print(f"Tracks: {n}")

    print("Pre-computing correlations...")
    precomp = precompute_all(features, matched, PROFILES)
    print("Done.")

    # ─── Strategy 1: Single-profile grid search (refined from v1) ────
    print("\n=== Strategy 1: Single profile (refined) ===")
    profile_names = list(PROFILES.keys())
    signals = ["full", "merged", "blend50"]
    skips = ["none", "last", "first_last"]
    conf_modes = ["gap", "gap_abs"]
    aggs = ["sum", "sum_sq"]
    minor_biases = [0.0, 0.05, 0.10]
    gap_divs = [0.10, 0.15, 0.20]
    top2_weights = [0.0, 0.25, 0.50]

    combos = list(itertools.product(
        profile_names, signals, skips, conf_modes, aggs, minor_biases, gap_divs, top2_weights
    ))
    print(f"Testing {len(combos)} single-profile combos...")

    results = []
    for combo in combos:
        prof, sig, skip, conf, agg, mbias, gdiv, t2w = combo
        exact = non_nb = 0
        for fname in matched:
            segs = precomp[prof][fname].get(sig, precomp[prof][fname]["full"])
            if sig == "harmonic":
                continue  # skip harmonic-only for segment voting
            if isinstance(segs, dict) and "top1" in segs:
                segs = [segs]
            pred, _ = vote_segments(segs, skip, conf, agg, mbias, gdiv, t2w)
            dist = camelot_distance(pred, gt[fname]["key"])
            if dist == 0: exact += 1
            if dist > 1: non_nb += 1
        results.append((non_nb, -exact, combo, exact))

    results.sort()
    print(f"\n  Top 15 single-profile configs:")
    print(f"  {'NN':>3s} {'Exact':>5s}  {'Profile':<12s} {'Signal':<8s} {'Skip':<12s} {'Conf':<8s} {'Agg':<6s} {'Mb':>4s} {'Gd':>4s} {'T2':>4s}")
    for i, (nn, neg_ex, combo, exact) in enumerate(results[:15]):
        prof, sig, skip, conf, agg, mbias, gdiv, t2w = combo
        print(f"  {nn:3d} {exact:5d}  {prof:<12s} {sig:<8s} {skip:<12s} {conf:<8s} {agg:<6s} {mbias:4.2f} {gdiv:4.2f} {t2w:4.2f}")

    # ─── Strategy 2: Profile ensemble (majority vote) ────────────────
    print("\n=== Strategy 2: Profile ensemble ===")

    # For each track, run N profiles and take majority vote
    ensemble_groups = [
        ("edma+simple", ["edma", "simple"]),
        ("edma+krum", ["edma", "krumhansl"]),
        ("edma+temp", ["edma", "temperley"]),
        ("all3", ["edma", "krumhansl", "temperley"]),
        ("all4", ["edma", "krumhansl", "temperley", "simple"]),
        ("edma+sharp+bass", ["edma", "edma_sharp", "edma_bass"]),
        ("all6", list(PROFILES.keys())),
    ]

    # Use best single-profile settings from strategy 1
    best_single = results[0][2]
    _, best_sig, best_skip, best_conf, best_agg, best_mbias, best_gdiv, best_t2w = best_single

    ens_results = []
    for ens_name, prof_list in ensemble_groups:
        for sig in ["full", "merged", "blend50"]:
            for skip in ["last", "first_last"]:
                for t2w in [0.0, 0.25]:
                    exact = non_nb = 0
                    for fname in matched:
                        votes_per_profile = []
                        for prof in prof_list:
                            segs = precomp[prof][fname].get(sig, precomp[prof][fname]["full"])
                            if isinstance(segs, dict) and "top1" in segs:
                                segs = [segs]
                            pred, conf_val = vote_segments(
                                segs, skip, best_conf, best_agg, best_mbias, best_gdiv, t2w
                            )
                            votes_per_profile.append(pred)

                        # Majority vote
                        counter = Counter(votes_per_profile)
                        pred = counter.most_common(1)[0][0]
                        dist = camelot_distance(pred, gt[fname]["key"])
                        if dist == 0: exact += 1
                        if dist > 1: non_nb += 1

                    ens_results.append((non_nb, -exact, ens_name, sig, skip, t2w, exact))

    ens_results.sort()
    print(f"\n  Top 15 ensemble configs:")
    print(f"  {'NN':>3s} {'Exact':>5s}  {'Ensemble':<18s} {'Signal':<8s} {'Skip':<12s} {'T2':>4s}")
    for i, (nn, neg_ex, ens_name, sig, skip, t2w, exact) in enumerate(ens_results[:15]):
        print(f"  {nn:3d} {exact:5d}  {ens_name:<18s} {sig:<8s} {skip:<12s} {t2w:4.2f}")

    # ─── Strategy 3: Harmonic + Full signal ensemble ─────────────────
    print("\n=== Strategy 3: Full + Harmonic ensemble ===")

    fh_results = []
    for prof_name in profile_names:
        for skip in ["last", "first_last"]:
            for t2w in [0.0, 0.25]:
                exact = non_nb = 0
                for fname in matched:
                    # Full signal vote
                    full_segs = precomp[prof_name][fname]["full"]
                    pred_full, conf_full = vote_segments(
                        full_segs, skip, best_conf, best_agg, best_mbias, best_gdiv, t2w
                    )

                    # Harmonic vote
                    h = precomp[prof_name][fname]["harmonic"]
                    if h is not None:
                        h_key = KEY_TO_CAMELOT[(h["top1"][1], h["top1"][2])]
                    else:
                        h_key = pred_full

                    # If full and harmonic agree, use that; else use full
                    # (could also try: if they disagree, check which has higher confidence)
                    if pred_full == h_key:
                        pred = pred_full
                    else:
                        # Full gets priority but harmonic breaks ties if full is low confidence
                        pred = pred_full

                    dist = camelot_distance(pred, gt[fname]["key"])
                    if dist == 0: exact += 1
                    if dist > 1: non_nb += 1
                fh_results.append((non_nb, -exact, prof_name, skip, t2w, exact))

    # Not super useful since we just default to full... Let's try weighted merge instead
    print("  (Full+harmonic simple ensemble: same as full-only, skipping)")

    # ─── Strategy 4: Confidence-weighted full+harmonic fusion ────────
    print("\n=== Strategy 4: Full + Harmonic weighted fusion ===")

    fuse_results = []
    for prof_name in profile_names:
        for skip in ["last", "first_last"]:
            for h_weight in [0.5, 1.0, 2.0]:  # weight of harmonic vote relative to each segment
                for t2w in [0.0, 0.25]:
                    exact = non_nb = 0
                    for fname in matched:
                        full_segs = precomp[prof_name][fname]["full"]
                        h = precomp[prof_name][fname]["harmonic"]

                        # Copy segments and optionally skip
                        segs = list(full_segs)
                        ns = len(segs)
                        if skip == "last" and ns > 2:
                            segs = segs[:-1]
                        elif skip == "first_last" and ns > 3:
                            segs = segs[1:-1]

                        # Build votes from segments
                        votes = {}
                        for seg in segs:
                            bc, bk, bm = seg["top1"]
                            sc = seg["top2"][0]
                            gap = bc - sc
                            conf = max(0.0, min(1.0, gap / best_gdiv)) * max(0.0, bc)
                            if bm == "minor": conf += best_mbias
                            vk = (bk, bm)
                            votes[vk] = votes.get(vk, 0.0) + conf
                            if t2w > 0:
                                _, k2, m2 = seg["top2"]
                                votes[(k2, m2)] = votes.get((k2, m2), 0.0) + conf * t2w

                        # Add harmonic vote
                        if h is not None:
                            hc, hk, hm = h["top1"]
                            hsc = h["top2"][0]
                            hgap = hc - hsc
                            hconf = max(0.0, min(1.0, hgap / best_gdiv)) * max(0.0, hc) * h_weight
                            if hm == "minor": hconf += best_mbias * h_weight
                            votes[(hk, hm)] = votes.get((hk, hm), 0.0) + hconf
                            if t2w > 0:
                                _, hk2, hm2 = h["top2"]
                                votes[(hk2, hm2)] = votes.get((hk2, hm2), 0.0) + hconf * t2w

                        if not votes:
                            pred = "??"
                        else:
                            winner = max(votes, key=votes.get)
                            pred = KEY_TO_CAMELOT[(winner[0], winner[1])]

                        dist = camelot_distance(pred, gt[fname]["key"])
                        if dist == 0: exact += 1
                        if dist > 1: non_nb += 1

                    fuse_results.append((non_nb, -exact, prof_name, skip, h_weight, t2w, exact))

    fuse_results.sort()
    print(f"\n  Top 15 fusion configs:")
    print(f"  {'NN':>3s} {'Exact':>5s}  {'Profile':<12s} {'Skip':<12s} {'Hw':>4s} {'T2':>4s}")
    for i, (nn, neg_ex, prof, skip, hw, t2w, exact) in enumerate(fuse_results[:15]):
        print(f"  {nn:3d} {exact:5d}  {prof:<12s} {skip:<12s} {hw:4.1f} {t2w:4.2f}")

    # ─── Show overall best across all strategies ────────────────────
    print(f"\n{'='*90}")
    all_best = []

    # Best single
    b = results[0]
    all_best.append((b[0], b[3], "single", str(b[2])))

    # Best ensemble
    if ens_results:
        b = ens_results[0]
        all_best.append((b[0], b[6], "ensemble", f"{b[2]} {b[3]} {b[4]} t2={b[5]}"))

    # Best fusion
    if fuse_results:
        b = fuse_results[0]
        all_best.append((b[0], b[6], "fusion", f"{b[2]} {b[3]} hw={b[4]} t2={b[5]}"))

    all_best.sort(key=lambda x: (x[0], -x[1]))
    print(f"  OVERALL BEST:")
    for nn, exact, strategy, config in all_best:
        nb = n - nn - (n - exact - (nn))
        print(f"    {strategy:10s}: NN={nn}, Exact={exact}/{n} ({100*exact/n:.0f}%), Config: {config}")

    # ─── Per-track details for the absolute best ────────────────────
    # Re-run the single best
    best = results[0]
    _, _, combo, _ = best
    prof, sig, skip, conf, agg, mbias, gdiv, t2w = combo
    print(f"\n--- Per-track (best single: {prof}, {sig}, skip={skip}, t2={t2w}) ---")
    for fname in matched:
        segs = precomp[prof][fname].get(sig, precomp[prof][fname]["full"])
        if isinstance(segs, dict) and "top1" in segs:
            segs = [segs]
        pred, kconf = vote_segments(segs, skip, conf, agg, mbias, gdiv, t2w)
        gt_key = gt[fname]["key"]
        dist = camelot_distance(pred, gt_key)
        marker = " OK" if dist == 0 else f" +{dist}" if dist <= 1 else f" XX({dist})"
        short = fname[:55]
        if dist > 0:
            print(f"  {short:<55s} GT={gt_key:>3s}  Pred={pred:>3s}  conf={kconf:.3f}{marker}")


if __name__ == "__main__":
    main()
