"""Filter a ground-truth labels CSV down to tracks with complete cache coverage.

Use this when you want to evaluate the DJ subgenre model on the subset of
tracks that have ALL dense audio features populated (DSP + CLAP + full
Demucs stems), so the holdout metric isn't diluted by tracks with sparse
features. Typical workflow:

    python scripts/populate_audio_embeddings.py --files "D:\\Music"
    python scripts/filter_labels_by_features.py \\
        --labels "D:\\Music\\outputs\\dj_taxonomy_ground_truth.csv" \\
        --files "D:\\Music" \\
        --out   "D:\\Music\\outputs\\labels_full_features.csv"
    dj-registry dj-taxonomy evaluate "D:\\Music" \\
        --labels  "D:\\Music\\outputs\\labels_full_features.csv" \\
        --model-dir "D:\\Music\\outputs\\registry\\dj_taxonomy_model"

By default a row is kept iff all of these are cached:
  - ``dsp`` layer
  - ``clap`` layer
  - ``vocal_stem`` layer WITH the new per-stem schema (presence of
    ``stem_drums_rms_db`` flags the post-expansion analyzer; legacy
    vocals-only entries are NOT counted as complete).

Use ``--require`` to relax: e.g. ``--require dsp clap`` skips the stems
check, which is useful while Demucs is still running.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import os  # noqa: E402
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

LAYER_PROBES = {
    # layer name -> predicate(cache_value) -> bool. Returning True means
    # "this row counts as having the layer populated".
    "dsp": lambda v: isinstance(v, dict) and "rms_mean" in v,
    "clap": lambda v: v is not None,  # ndarray is truthy-by-presence
    "vocal_stem": lambda v: isinstance(v, dict) and "vocal_stem_rms_db" in v,
    "stems_full": lambda v: isinstance(v, dict) and "stem_drums_rms_db" in v,
}


def _index_files(files_root: Path) -> dict[str, Path]:
    """Map lowercase filename -> absolute path for the music root."""
    audio_exts = {".mp3", ".aiff", ".aif", ".wav", ".flac", ".m4a"}
    return {
        p.name.lower(): p
        for p in files_root.rglob("*")
        if p.is_file() and p.suffix.lower() in audio_exts
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="Path to the ground-truth CSV.")
    ap.add_argument("--files", required=True, help="Music library root (for filename-to-path mapping).")
    ap.add_argument("--out", required=True, help="Where to write the filtered CSV.")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"),
                    help="Directory containing raw_cache.pkl (default: ./cache).")
    ap.add_argument(
        "--require",
        nargs="+",
        default=["dsp", "clap", "stems_full"],
        choices=list(LAYER_PROBES.keys()),
        help="Cache layers a row must have to be kept. Default is the full set "
             "(DSP + CLAP + full Demucs stems). Use 'vocal_stem' instead of "
             "'stems_full' if you want to count legacy-schema entries too.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from dj_tagger.universal_cache import get_cache, quick_duration

    cache_dir = Path(args.cache_dir)
    ucache = get_cache(str(cache_dir / "raw_cache.pkl"))

    files_index = _index_files(Path(args.files))
    print(f"Indexed {len(files_index)} audio files under {args.files}")

    labels_path = Path(args.labels)
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"Loaded {len(rows)} label rows from {labels_path}")

    # Per-layer counts so the user sees what's missing where.
    layer_present: Counter[str] = Counter()
    layer_absent: Counter[str] = Counter()
    missing_file = 0
    kept_rows: list[dict] = []

    for row in rows:
        file_name = (row.get("file_name") or row.get("filename") or "").strip()
        if not file_name:
            missing_file += 1
            continue
        path = files_index.get(file_name.lower())
        if path is None:
            missing_file += 1
            continue
        duration = quick_duration(str(path))
        row_layers: dict[str, bool] = {}
        for layer in LAYER_PROBES:
            cache_layer = "vocal_stem" if layer == "stems_full" else layer
            data = ucache.get_track(path.name, duration, cache_layer)
            present = LAYER_PROBES[layer](data)
            row_layers[layer] = present
            if present:
                layer_present[layer] += 1
            else:
                layer_absent[layer] += 1
        if all(row_layers[layer] for layer in args.require):
            kept_rows.append(row)

    print("\nCoverage in labels CSV:")
    for layer in LAYER_PROBES:
        total = layer_present[layer] + layer_absent[layer]
        pct = (layer_present[layer] * 100) // total if total else 0
        marker = "  required" if layer in args.require else ""
        print(f"  {layer:11s} {layer_present[layer]:4d}/{total:<4d}  {pct:>3}%{marker}")
    if missing_file:
        print(f"  rows skipped (no matching audio file): {missing_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)
    print(
        f"\nWrote {len(kept_rows)} of {len(rows)} rows to {out_path} "
        f"(filter: require all of {args.require})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
