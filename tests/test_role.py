"""Tests for track role inference."""

from dj_grouper.scanner import TrackInfo
from dj_grouper.features.role import infer_role


def test_tool_low_energy_layer():
    info = TrackInfo(path="t.aiff", energy=1, flow_type="L", vibe="HYPN")
    assert infer_role(info, {}) == "TOOL"


def test_peak_high_energy():
    info = TrackInfo(path="t.aiff", energy=5, flow_type="D", vibe="RAW")
    assert infer_role(info, {}) == "PEAK"


def test_driver_mid_energy():
    info = TrackInfo(path="t.aiff", energy=3, flow_type="G", vibe="HYPN")
    assert infer_role(info, {}) == "DRIVER"


def test_reset_atm():
    info = TrackInfo(path="t.aiff", energy=2, flow_type="H", vibe="ATM")
    assert infer_role(info, {}) == "RESET"


def test_breakdown_flow():
    info = TrackInfo(path="t.aiff", energy=3, flow_type="B", vibe="MEL")
    assert infer_role(info, {}) == "BREAKDOWN"


def test_bridge_melodic():
    info = TrackInfo(path="t.aiff", energy=3, flow_type="H", vibe="MEL")
    assert infer_role(info, {}) == "BRIDGE"
