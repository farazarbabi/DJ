"""Pure presentation helpers for LR-vs-XGB model comparison.

No training, no IO with stores. Input: two `training_report.json`-shaped
dicts (or the equivalent in-memory dicts returned by
train_dj_taxonomy_unified). Output: formatted strings + a single JSON
artifact with all metric deltas computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_GATE_THRESHOLDS = (
    ("top-1 ≥ 50%", "top1_accuracy", 0.50, "{:.1%}"),
    ("top-3 ≥ 75%", "top3_accuracy", 0.75, "{:.1%}"),
    ("macro F1 ≥ 0.30", "macro_f1", 0.30, "{:.3f}"),
)


def _get_metric(report: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not report:
        return default
    metrics = report.get("metrics") or {}
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta_pp(lr: float, xgb: float) -> str:
    delta = (xgb - lr) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}pp"


def _delta_abs(lr: float, xgb: float, places: int = 3) -> str:
    delta = xgb - lr
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{places}f}"


def format_comparison_table(
    lr_report: dict[str, Any] | None,
    xgb_report: dict[str, Any] | None,
) -> str:
    """Multi-row table with LR / XGB / Δ columns for the §9.4 metrics + fit time."""
    rows = [
        ("top-1 accuracy", "top1_accuracy", "pct"),
        ("top-3 accuracy", "top3_accuracy", "pct"),
        ("macro F1", "macro_f1", "abs"),
        ("weighted F1", "weighted_f1", "abs"),
        ("avg confidence", "average_confidence", "abs2"),
        ("fit time", "fit_time_seconds", "secs"),
    ]
    lines = [f"{'':18s}{'LR':>12s}{'XGB':>12s}{'Δ':>12s}"]
    for label, key, kind in rows:
        lr_val = _get_metric(lr_report, key)
        xgb_val = _get_metric(xgb_report, key)
        if kind == "pct":
            lines.append(f"{label:18s}{_percent(lr_val):>12s}{_percent(xgb_val):>12s}{_delta_pp(lr_val, xgb_val):>12s}")
        elif kind == "abs":
            lines.append(f"{label:18s}{lr_val:>12.3f}{xgb_val:>12.3f}{_delta_abs(lr_val, xgb_val):>12s}")
        elif kind == "abs2":
            lines.append(f"{label:18s}{lr_val:>12.2f}{xgb_val:>12.2f}{_delta_abs(lr_val, xgb_val, places=2):>12s}")
        elif kind == "secs":
            lines.append(f"{label:18s}{lr_val:>11.1f}s{xgb_val:>11.1f}s{(xgb_val - lr_val):>+11.1f}s")
    return "\n".join(lines)


def format_gate_table(
    lr_report: dict[str, Any] | None,
    xgb_report: dict[str, Any] | None,
) -> str:
    """Pass/fail per §9.4 gate, side-by-side."""
    lines = [f"{'':18s}{'LR':>6s}{'XGB':>6s}"]
    for label, key, threshold, _fmt in _GATE_THRESHOLDS:
        lr_pass = _get_metric(lr_report, key) >= threshold
        xgb_pass = _get_metric(xgb_report, key) >= threshold
        lines.append(f"{label:18s}{'✓' if lr_pass else '✗':>6s}{'✓' if xgb_pass else '✗':>6s}")
    return "\n".join(lines)


def format_distribution_table(
    lr_check: dict[str, Any] | None,
    xgb_check: dict[str, Any] | None,
    *,
    cap: float = 0.20,
) -> str:
    """Largest-bucket distribution side-by-side."""
    lines = [f"{'':24s}{'LR':>30s}{'XGB':>30s}"]
    lr_id = (lr_check or {}).get("largest_bucket_id", "—")
    xgb_id = (xgb_check or {}).get("largest_bucket_id", "—")
    lr_share = (lr_check or {}).get("largest_bucket_share", 0.0)
    xgb_share = (xgb_check or {}).get("largest_bucket_share", 0.0)
    lines.append(
        f"{'largest bucket':24s}{f'{lr_id} ({lr_share:.1%})':>30s}{f'{xgb_id} ({xgb_share:.1%})':>30s}"
    )
    lr_pass = lr_share <= cap
    xgb_pass = xgb_share <= cap
    lines.append(f"{f'≤{cap:.0%} cap':24s}{'✓' if lr_pass else '✗':>30s}{'✓' if xgb_pass else '✗':>30s}")
    return "\n".join(lines)


def format_xgb_hyperparams(xgb_report: dict[str, Any] | None) -> str:
    """Format the best XGB hyperparameters for terminal output."""
    if not xgb_report:
        return "  (no XGB report)"
    best = xgb_report.get("best_params") or (xgb_report.get("metrics") or {}).get("best_params", {})
    if not best:
        return "  (no best params — tuning was skipped)"
    parts = []
    for k in (
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight",
        "gamma", "reg_alpha", "reg_lambda",
    ):
        if k in best:
            val = best[k]
            if isinstance(val, float):
                parts.append(f"{k}={val:.3g}")
            else:
                parts.append(f"{k}={val}")
    if not parts:
        return "  (best_params dict empty)"
    # Group 3-per-line for readability
    out_lines: list[str] = []
    for i in range(0, len(parts), 3):
        out_lines.append("  " + "  ".join(parts[i:i + 3]))
    return "\n".join(out_lines)


def format_full_report(
    lr_report: dict[str, Any] | None,
    xgb_report: dict[str, Any] | None,
    *,
    lr_check: dict[str, Any] | None = None,
    xgb_check: dict[str, Any] | None = None,
    examples_total: int | None = None,
    classes: int | None = None,
    cap: float = 0.20,
) -> str:
    """Compose the full terminal report — metrics, gates, distribution, hyperparams."""
    lines = ["=== DJ Taxonomy Model Comparison ==="]
    if examples_total is not None and classes is not None:
        lines.append(f"Train/test split: 80/20, {examples_total} examples, {classes} categories")
    lines.append("Feature mode: unified (internal + provider + audio + priors)")
    lines.append("")
    lines.append(format_comparison_table(lr_report, xgb_report))
    lines.append("")
    lines.append("§9.4 gate pass/fail")
    lines.append(format_gate_table(lr_report, xgb_report))
    if lr_check or xgb_check:
        lines.append("")
        lines.append(f"Library bucket distribution (largest of {classes or '—'})")
        lines.append(format_distribution_table(lr_check, xgb_check, cap=cap))
    lines.append("")
    lines.append("Best XGB hyperparameters")
    lines.append(format_xgb_hyperparams(xgb_report))
    return "\n".join(lines)


def write_comparison_json(
    out_path: str | Path,
    lr_result: dict[str, Any] | None,
    xgb_result: dict[str, Any] | None,
) -> Path:
    """Dump a JSON artifact containing both reports + computed deltas."""
    deltas: dict[str, float] = {}
    for _label, key, _threshold, _fmt in _GATE_THRESHOLDS:
        deltas[key] = round(_get_metric(xgb_result, key) - _get_metric(lr_result, key), 4)
    deltas["average_confidence"] = round(
        _get_metric(xgb_result, "average_confidence") - _get_metric(lr_result, "average_confidence"), 4
    )
    deltas["fit_time_seconds"] = round(
        _get_metric(xgb_result, "fit_time_seconds") - _get_metric(lr_result, "fit_time_seconds"), 2
    )

    payload = {
        "lr": lr_result,
        "xgb": xgb_result,
        "deltas": deltas,
        "gates_pass_lr": {
            key: _get_metric(lr_result, key) >= threshold
            for _, key, threshold, _ in _GATE_THRESHOLDS
        },
        "gates_pass_xgb": {
            key: _get_metric(xgb_result, key) >= threshold
            for _, key, threshold, _ in _GATE_THRESHOLDS
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return out
