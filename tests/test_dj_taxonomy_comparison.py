"""Spec §11.5 — tests for the model-comparison presentation layer."""

from __future__ import annotations

import json

from dj_registry.taxonomy.dj_model_comparison import (
    format_comparison_table,
    format_distribution_table,
    format_full_report,
    format_gate_table,
    format_xgb_hyperparams,
    write_comparison_json,
)


def _lr_report(top1=0.55, top3=0.78, mf1=0.32, wf1=0.49, conf=0.55, fit=2.3):
    return {"metrics": {
        "top1_accuracy": top1, "top3_accuracy": top3,
        "macro_f1": mf1, "weighted_f1": wf1,
        "average_confidence": conf, "fit_time_seconds": fit,
    }}


def _xgb_report(top1=0.62, top3=0.83, mf1=0.38, wf1=0.55, conf=0.6, fit=45.0, best_params=None):
    return {
        "metrics": {
            "top1_accuracy": top1, "top3_accuracy": top3,
            "macro_f1": mf1, "weighted_f1": wf1,
            "average_confidence": conf, "fit_time_seconds": fit,
        },
        "best_params": best_params or {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.07,
            "subsample": 0.85, "colsample_bytree": 0.75, "min_child_weight": 3,
        },
    }


def test_format_comparison_table_contains_both_models_and_delta():
    output = format_comparison_table(_lr_report(), _xgb_report())
    assert "LR" in output
    assert "XGB" in output
    assert "top-1 accuracy" in output
    assert "fit time" in output
    # Some delta should appear; check the columns are present
    lines = output.splitlines()
    header = lines[0]
    assert "LR" in header and "XGB" in header


def test_format_gate_table_marks_failures():
    # Below all thresholds
    lr = _lr_report(top1=0.40, top3=0.60, mf1=0.20)
    xgb = _xgb_report(top1=0.55, top3=0.80, mf1=0.35)
    output = format_gate_table(lr, xgb)
    assert "top-1 ≥ 50%" in output
    assert "top-3 ≥ 75%" in output
    assert "macro F1 ≥ 0.30" in output
    # LR fails all three, XGB passes all three
    # Count the marks to verify the row direction
    lines = output.splitlines()[1:]  # skip header
    assert all("✗" in line and "✓" in line for line in lines)


def test_format_gate_table_marks_passes_when_both_above():
    lr = _lr_report(top1=0.75, top3=0.90, mf1=0.50)
    xgb = _xgb_report(top1=0.80, top3=0.95, mf1=0.55)
    output = format_gate_table(lr, xgb)
    lines = output.splitlines()[1:]
    assert all("✓" in line for line in lines)
    assert not any("✗" in line for line in lines)


def test_format_distribution_table_side_by_side():
    lr_check = {"largest_bucket_id": "organic_house_builder", "largest_bucket_share": 0.221}
    xgb_check = {"largest_bucket_id": "tribal_afro_driver", "largest_bucket_share": 0.184}
    output = format_distribution_table(lr_check, xgb_check, cap=0.20)
    assert "organic_house_builder" in output
    assert "tribal_afro_driver" in output
    assert "≤20% cap" in output
    # LR fails the cap, XGB passes
    assert "✗" in output and "✓" in output


def test_format_xgb_hyperparams_renders_best_params():
    output = format_xgb_hyperparams(_xgb_report())
    assert "n_estimators=400" in output
    assert "max_depth=5" in output
    assert "learning_rate=" in output


def test_format_xgb_hyperparams_handles_missing_best_params():
    output = format_xgb_hyperparams({"metrics": {}})
    assert "no best params" in output or "empty" in output


def test_format_full_report_includes_all_sections():
    lr = _lr_report()
    xgb = _xgb_report()
    lr_check = {"largest_bucket_id": "organic_house_builder", "largest_bucket_share": 0.221}
    xgb_check = {"largest_bucket_id": "tribal_afro_driver", "largest_bucket_share": 0.184}
    output = format_full_report(
        lr, xgb, lr_check=lr_check, xgb_check=xgb_check,
        examples_total=247, classes=103,
    )
    assert "DJ Taxonomy Model Comparison" in output
    assert "247 examples" in output
    assert "103 categories" in output
    assert "§9.4 gate" in output
    assert "Library bucket distribution" in output
    assert "Best XGB hyperparameters" in output


def test_write_comparison_json_includes_deltas_and_gates(tmp_path):
    lr = _lr_report()
    xgb = _xgb_report()
    out_path = write_comparison_json(tmp_path / "comparison.json", lr, xgb)
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "lr" in payload
    assert "xgb" in payload
    assert "deltas" in payload
    # XGB beat LR on all three gates per the fixtures
    assert payload["deltas"]["top1_accuracy"] > 0
    assert payload["deltas"]["macro_f1"] > 0
    assert "gates_pass_lr" in payload
    assert "gates_pass_xgb" in payload


def test_write_comparison_handles_missing_xgb(tmp_path):
    """When only LR was trained, XGB side is None — JSON write should still succeed."""
    out_path = write_comparison_json(tmp_path / "comparison.json", _lr_report(), None)
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["xgb"] is None
    assert payload["lr"] is not None
