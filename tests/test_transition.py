"""Tests for transition compatibility."""

from dj_grouper.recommend.transition import (
    flow_compatibility, intro_compatibility, structure_compatibility,
)


def test_hypnotic_pair_positive():
    assert flow_compatibility("H", "H") > 0


def test_drop_to_hypnotic_negative():
    assert flow_compatibility("D", "H") < 0


def test_layer_with_groove_positive():
    assert flow_compatibility("L", "G") > 0


def test_same_intro_positive():
    assert intro_compatibility(64, 64) > 0


def test_large_intro_mismatch_negative():
    assert intro_compatibility(16, 64) < 0


def test_structure_combined():
    score = structure_compatibility("H", 64, "H", 64)
    assert score > 0  # both positive components
