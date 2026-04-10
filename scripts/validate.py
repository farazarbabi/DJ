"""Compare tagger output against ground truth and report accuracy."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Camelot wheel: map camelot codes to numbers for distance calculation
CAMELOT_NUMBER = {}
for code in [f"{n}{l}" for n in range(1, 13) for l in "AB"]:
    CAMELOT_NUMBER[code] = (int(code[:-1]), code[-1])


def camelot_distance(pred: str, truth: str) -> int:
    """Distance on the Camelot wheel (0 = exact, 1 = compatible, etc.)."""
    if pred == truth:
        return 0
    pn, pl = CAMELOT_NUMBER.get(pred, (None, None))
    tn, tl = CAMELOT_NUMBER.get(truth, (None, None))
    if pn is None or tn is None:
        return 99  # unknown
    # Same letter: circular distance on 1-12
    if pl == tl:
        return min(abs(pn - tn), 12 - abs(pn - tn))
    # Different letter (major/minor): check if same number (relative key)
    num_dist = min(abs(pn - tn), 12 - abs(pn - tn))
    return num_dist + 1  # penalty for mode mismatch


def key_compatible(pred: str, truth: str) -> bool:
    """Keys are compatible if distance <= 1 on Camelot wheel."""
    return camelot_distance(pred, truth) <= 1


def load_ground_truth(path: str) -> dict[str, dict]:
    """Load ground truth CSV keyed by filename."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file_name"].strip()
            gt[fname] = {
                "energy": row.get("Energy", "").strip(),
                "vibe": row.get("Vibe", "").strip(),
                "key": row.get("Key_rekordbox", "").strip(),
                "vocal": row.get("Vocal", "").strip(),
            }
    return gt


def load_predictions(path: str) -> dict[str, dict]:
    """Load tagger CSV output keyed by filename."""
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = Path(row["file"]).name
            preds[fname] = {
                "energy": f"E{row.get('energy', '?')}",
                "vibe": row.get("vibe", "??"),
                "key": row.get("camelot", "??"),
                "vocal": row.get("vocal", "??"),
            }
    return preds


def compare(gt: dict, preds: dict, label: str = "baseline") -> dict:
    """Compare predictions to ground truth, return accuracy stats."""
    matched_files = sorted(set(gt.keys()) & set(preds.keys()))
    total = len(matched_files)
    if total == 0:
        print("No matching files found!")
        return {}

    stats = {
        "energy_exact": 0, "energy_off1": 0,
        "vibe_exact": 0,
        "key_exact": 0, "key_compatible": 0,
        "vocal_exact": 0,
    }
    details = []

    for fname in matched_files:
        g = gt[fname]
        p = preds[fname]

        e_match = g["energy"] == p["energy"]
        e_off1 = abs(int(g["energy"][1]) - int(p["energy"][1])) <= 1 if g["energy"][1].isdigit() and p["energy"][1].isdigit() else False
        v_match = g["vibe"] == p["vibe"]
        k_exact = g["key"] == p["key"]
        k_compat = key_compatible(p["key"], g["key"])
        voc_match = g["vocal"] == p["vocal"]

        if e_match:
            stats["energy_exact"] += 1
        if e_off1:
            stats["energy_off1"] += 1
        if v_match:
            stats["vibe_exact"] += 1
        if k_exact:
            stats["key_exact"] += 1
        if k_compat:
            stats["key_compatible"] += 1
        if voc_match:
            stats["vocal_exact"] += 1

        details.append({
            "file": fname,
            "gt_energy": g["energy"], "pred_energy": p["energy"], "energy_ok": e_match,
            "gt_vibe": g["vibe"], "pred_vibe": p["vibe"], "vibe_ok": v_match,
            "gt_key": g["key"], "pred_key": p["key"], "key_ok": k_exact, "key_compat": k_compat,
            "gt_vocal": g["vocal"], "pred_vocal": p["vocal"], "vocal_ok": voc_match,
        })

    print(f"\n{'='*60}")
    print(f"  Validation Report: {label}")
    print(f"  Tracks matched: {total}/{len(gt)}")
    print(f"{'='*60}")
    print(f"  Energy exact:      {stats['energy_exact']:2d}/{total} ({100*stats['energy_exact']/total:.0f}%)")
    print(f"  Energy +/-1:       {stats['energy_off1']:2d}/{total} ({100*stats['energy_off1']/total:.0f}%)")
    print(f"  Vibe exact:        {stats['vibe_exact']:2d}/{total} ({100*stats['vibe_exact']/total:.0f}%)")
    print(f"  Key exact:         {stats['key_exact']:2d}/{total} ({100*stats['key_exact']/total:.0f}%)")
    print(f"  Key compatible:    {stats['key_compatible']:2d}/{total} ({100*stats['key_compatible']/total:.0f}%)")
    print(f"  Vocal exact:       {stats['vocal_exact']:2d}/{total} ({100*stats['vocal_exact']/total:.0f}%)")

    # Print errors
    print(f"\n--- Mismatches ---")
    for d in details:
        errors = []
        if not d["energy_ok"]:
            errors.append(f"E:{d['gt_energy']}->{d['pred_energy']}")
        if not d["vibe_ok"]:
            errors.append(f"V:{d['gt_vibe']}->{d['pred_vibe']}")
        if not d["key_ok"]:
            compat = " (compat)" if d["key_compat"] else ""
            errors.append(f"K:{d['gt_key']}->{d['pred_key']}{compat}")
        if not d["vocal_ok"]:
            errors.append(f"Voc:{d['gt_vocal']}->{d['pred_vocal']}")
        if errors:
            short_name = d["file"][:50]
            print(f"  {short_name:<52s} {', '.join(errors)}")

    return stats


if __name__ == "__main__":
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "files/ground_truth.csv"
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/baseline.csv"
    label = sys.argv[3] if len(sys.argv) > 3 else "baseline"

    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)
    compare(gt, preds, label)
