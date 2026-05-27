"""Phase-1 ablation: train XGB once per feature-block variant, compare metrics.

Diagnoses which dense feature blocks (CLAP, vocal_stem, DSP, popularity) help
vs. hurt the DJ-functional subgenre classifier on the current label set, and
whether `class_weight="balanced"` is hurting in the long-tail regime.

Each variant trains from scratch on the same examples (built once) with the
same seed, then runs the same holdout evaluation. The variant filters keys
out of each example's feature dict in-memory — no production-code changes
and no cache writes.

Outputs:
  - A side-by-side table printed to stdout.
  - JSON at  <output>/dj_taxonomy_model/ablation.json  with all variant
    metrics for the record.

Usage:
    python scripts/ablate_feature_blocks.py ^
        --labels "D:\\Music\\outputs\\labels_full_features.csv" ^
        --output "D:\\Music\\outputs\\registry"
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logger = logging.getLogger(__name__)


def _drop_keys(features: dict, predicate: Callable[[str], bool]) -> dict:
    """Return a new feature dict with keys matching predicate removed."""
    return {k: v for k, v in features.items() if not predicate(k)}


# Key-stripping rules per variant. Each predicate returns True when the key
# should be DROPPED from the feature dict.
_VARIANT_FILTERS: dict[str, Callable[[str], bool]] = {
    "baseline": lambda k: False,
    "no_clap": lambda k: k.startswith("clap:"),
    "no_stems": lambda k: (
        k.startswith("num:stem:")
        or k.startswith("num:dominance:")
        or k.startswith("num:vocal_stem:")
    ),
    "no_dsp": lambda k: k.startswith("num:dsp:"),
    "no_popularity": lambda k: k.startswith("popularity_band:") or k.endswith(":popularity"),
    "no_dense": lambda k: (
        k.startswith("clap:")
        or k.startswith("num:stem:")
        or k.startswith("num:dominance:")
        or k.startswith("num:vocal_stem:")
        or k.startswith("num:dsp:")
        or k.startswith("popularity_band:")
        or k.endswith(":popularity")
    ),
    # no_class_weight: features unchanged, but pass class_balancing="none"
    "no_class_weight": lambda k: False,
}


def _filter_examples(examples: list[dict], predicate: Callable[[str], bool]) -> list[dict]:
    """Deep-copy examples and strip filtered keys from each feature dict."""
    out: list[dict] = []
    for ex in examples:
        new_ex = copy.copy(ex)  # shallow copy of the dict
        new_ex["features"] = _drop_keys(ex["features"], predicate)
        out.append(new_ex)
    return out


def _run_variant(
    name: str,
    examples: list[dict],
    taxonomy,
    *,
    labels_path: str,
    validation_split: float,
    seed: int,
    n_iter: int,
) -> dict:
    """Train one variant and return a small metrics summary."""
    from dj_registry.taxonomy.dj_model_xgb import train_xgb_model

    predicate = _VARIANT_FILTERS[name]
    variant_examples = _filter_examples(examples, predicate)
    class_balancing = "none" if name == "no_class_weight" else "balanced"

    # Use the FIRST example's feature dict to report block sizes (sanity check).
    sample = variant_examples[0]["features"] if variant_examples else {}
    block_counts = {
        "total_keys": len(sample),
        "clap": sum(1 for k in sample if k.startswith("clap:")),
        "stems": sum(
            1 for k in sample
            if k.startswith("num:stem:") or k.startswith("num:dominance:") or k.startswith("num:vocal_stem:")
        ),
        "dsp": sum(1 for k in sample if k.startswith("num:dsp:")),
        "popularity": sum(
            1 for k in sample if k.startswith("popularity_band:") or k.endswith(":popularity")
        ),
    }

    t0 = time.time()
    _model, metrics, warnings = train_xgb_model(
        variant_examples,
        taxonomy,
        labels_path=labels_path,
        validation_split=validation_split,
        seed=seed,
        n_iter=n_iter,
        class_balancing=class_balancing,
    )
    elapsed = time.time() - t0

    return {
        "variant": name,
        "class_balancing": class_balancing,
        "examples": len(variant_examples),
        "block_counts_in_first_example": block_counts,
        "top1_accuracy": metrics.get("top1_accuracy", 0.0),
        "top3_accuracy": metrics.get("top3_accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "average_confidence": metrics.get("average_confidence", 0.0),
        "validation_kind": metrics.get("validation_kind", ""),
        "validation_examples": metrics.get("validation_examples", 0),
        "fit_time_seconds": round(elapsed, 2),
        "warnings": list(warnings or []),
    }


def _print_table(results: list[dict]) -> None:
    """Render a compact one-line-per-variant comparison table."""
    base = next((r for r in results if r["variant"] == "baseline"), None)

    def delta_pp(row: dict, value: float, key: str) -> str:
        if base is None or row is base:
            return ""
        d = (value - base[key]) * 100
        sign = "+" if d >= 0 else ""
        return f" ({sign}{d:.1f}pp)"

    sep = "-" * 88
    print(sep)
    print(
        f"{'variant':18s}{'keys':>6s}  {'top-1':>14s}  {'top-3':>14s}  "
        f"{'macro-F1':>14s}  {'fit':>6s}"
    )
    print(sep)
    for r in results:
        keys = r["block_counts_in_first_example"]["total_keys"]
        print(
            f"{r['variant']:18s}{keys:>6d}  "
            f"{r['top1_accuracy']:>6.1%}{delta_pp(r, r['top1_accuracy'], 'top1_accuracy'):>8s}  "
            f"{r['top3_accuracy']:>6.1%}{delta_pp(r, r['top3_accuracy'], 'top3_accuracy'):>8s}  "
            f"{r['macro_f1']:>6.3f}{delta_pp(r, r['macro_f1'], 'macro_f1'):>8s}  "
            f"{r['fit_time_seconds']:>5.1f}s"
        )
    print(sep)
    if base is None:
        print("(no baseline result — deltas unavailable)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="DJ taxonomy ground-truth labels CSV.")
    ap.add_argument("--output", default="./outputs/registry",
                    help="Registry output dir (default: ./outputs/registry).")
    ap.add_argument("--taxonomy", default=None, help="Optional dj_taxonomy.json path.")
    ap.add_argument("--validation-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=6,
                    help="RandomizedSearchCV iterations for XGB (default 6 — kept small for ablation speed).")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(_VARIANT_FILTERS.keys()),
        choices=list(_VARIANT_FILTERS.keys()),
        help="Subset of variants to run (default: all).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from dj_registry.store.csv_store import CsvStore
    from dj_registry.taxonomy.dj_model import (
        _build_training_examples,
        _load_label_rows,
        _resolve_cache,
    )
    from dj_registry.taxonomy.dj_schema import load_dj_taxonomy

    store = CsvStore(args.output)
    taxonomy = load_dj_taxonomy(args.taxonomy)
    labels = _load_label_rows(args.labels, taxonomy)

    # Build feature examples ONCE — the slow step. Each variant filters
    # in-memory and reuses these.
    ucache = _resolve_cache()
    print(f"Building examples (this is the slow step — runs once)...")
    examples, skipped, dropped = _build_training_examples(
        labels, store, mode="external", show_progress=False, ucache=ucache,
    )
    if not examples:
        print("No examples — labels CSV produced 0 feature-bearing rows.")
        return 1
    print(
        f"Built {len(examples)} examples across "
        f"{len({e['category_id'] for e in examples})} categories. "
        f"Skipped {skipped} rows, dropped {len(dropped)} categories with <3 examples."
    )
    sample = examples[0]["features"]
    print(
        f"First example: {len(sample)} total keys "
        f"({sum(1 for k in sample if k.startswith('clap:'))} CLAP, "
        f"{sum(1 for k in sample if k.startswith('num:stem:') or k.startswith('num:dominance:') or k.startswith('num:vocal_stem:'))} stems, "
        f"{sum(1 for k in sample if k.startswith('num:dsp:'))} DSP)"
    )
    print()

    results: list[dict] = []
    for variant in args.variants:
        print(f"  -> running {variant} ...")
        result = _run_variant(
            variant,
            examples,
            taxonomy,
            labels_path=args.labels,
            validation_split=args.validation_split,
            seed=args.seed,
            n_iter=args.n_iter,
        )
        results.append(result)
        print(
            f"    top-1={result['top1_accuracy']:.1%}  "
            f"top-3={result['top3_accuracy']:.1%}  "
            f"mF1={result['macro_f1']:.3f}  fit={result['fit_time_seconds']}s"
        )

    print()
    print("=" * 70)
    print("Ablation results")
    print("=" * 70)
    _print_table(results)

    out_path = Path(args.output) / "dj_taxonomy_model" / "ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_path": args.labels,
                "seed": args.seed,
                "n_iter": args.n_iter,
                "validation_split": args.validation_split,
                "examples_total": len(examples),
                "categories_total": len({e["category_id"] for e in examples}),
                "results": results,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"\nWrote ablation report to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
