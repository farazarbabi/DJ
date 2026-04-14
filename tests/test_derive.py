"""Tests for derive.py — recomputing derived values from cached raw features."""

from dj_tagger.derive import derive_energy, derive_vibe, derive_vocal, derive_structure, derive_all


def _make_dsp(**overrides):
    """Create a typical DSP feature dict."""
    dsp = {
        "rms_mean": 0.25,
        "centroid_mean": 1800.0,
        "centroid_var": 50000.0,
        "flatness_mean": 0.015,
        "onset_density": 2.0,
        "onset_variance": 1.0,
        "perc_harmonic_ratio": 0.3,
        "low_freq_ratio": 50.0,
        "flux_mean": 2.5,
        "chroma_var": 0.04,
        "beat_strength": 2.5,
        "bandwidth_mean": 2000.0,
        "chroma_strength": 0.5,
    }
    dsp.update(overrides)
    return dsp


def _make_raw(**overrides):
    """Create a typical raw_analysis dict."""
    raw = {
        "bar_energies": [0.3, 0.3, 0.5, 0.7, 0.8, 0.8, 0.9, 0.9, 0.8, 0.7, 0.5, 0.3],
        "n_bars": 12,
        "tempo": 126.0,
        "vocal_ratio": 0.05,
        "vocal_temporal_bonus": 0.0,
        "onset_rate": 3.5,
    }
    raw.update(overrides)
    return raw


def test_derive_energy_returns_valid_level():
    result = derive_energy(_make_dsp())
    assert 1 <= result["energy"] <= 5
    assert 0.0 <= result["energy_confidence"] <= 1.0


def test_derive_energy_high_features_give_high_energy():
    dsp = _make_dsp(rms_mean=0.35, beat_strength=4.0, onset_density=3.0, centroid_mean=3000.0)
    result = derive_energy(dsp)
    assert result["energy"] >= 4


def test_derive_energy_low_features_give_low_energy():
    dsp = _make_dsp(rms_mean=0.15, beat_strength=1.5, onset_density=1.0, centroid_mean=1000.0)
    result = derive_energy(dsp)
    assert result["energy"] <= 2


def test_derive_vibe_returns_valid_label():
    result = derive_vibe(_make_dsp())
    assert result["vibe"] in ("HYPN", "DRK", "RAW", "DEEP", "TRIB", "MEL", "ACID", "ATM")
    assert len(result["vibe_scores"]) == 8
    assert all(0.0 <= v <= 1.0 for v in result["vibe_scores"].values())


def test_derive_vibe_dark_features():
    dsp = _make_dsp(onset_variance=4.0, flatness_mean=0.04, low_freq_ratio=70.0, flux_mean=4.5)
    result = derive_vibe(dsp)
    assert result["vibe_scores"]["DRK"] > 0.3


def test_derive_vocal_no_vocals():
    raw = _make_raw(vocal_ratio=0.05, vocal_temporal_bonus=0.0)
    result = derive_vocal(raw)
    assert result["vocal"] == "NV"
    assert result["has_vocals"] is False


def test_derive_vocal_has_vocals():
    raw = _make_raw(vocal_ratio=0.5, vocal_temporal_bonus=0.8)
    result = derive_vocal(raw)
    assert result["vocal"] == "V"
    assert result["has_vocals"] is True


def test_derive_structure_returns_valid():
    raw = _make_raw()
    result = derive_structure(raw)
    assert result["intro_bars"] in (16, 32, 64)
    assert result["flow_type"] in ("G", "H", "D", "B", "L")
    assert result["structure"] == f"{result['intro_bars']}{result['flow_type']}"


def test_derive_structure_empty_bars():
    raw = _make_raw(bar_energies=[], n_bars=0)
    result = derive_structure(raw)
    assert result["structure"] == "16H"  # default fallback


def test_derive_all_returns_complete():
    dsp = _make_dsp()
    raw = _make_raw()
    result = derive_all(dsp, raw)
    assert "energy" in result
    assert "vibe" in result
    assert "vocal" in result
    assert "structure" in result
    assert "vibe_scores" in result
    assert "confidences" in result
    assert "energy" in result["confidences"]
    assert "vibe" in result["confidences"]
    assert "vocal" in result["confidences"]
    assert "structure" in result["confidences"]


def test_derive_all_deterministic():
    """Same inputs should always produce same outputs."""
    dsp = _make_dsp()
    raw = _make_raw()
    r1 = derive_all(dsp, raw)
    r2 = derive_all(dsp, raw)
    assert r1["energy"] == r2["energy"]
    assert r1["vibe"] == r2["vibe"]
    assert r1["vocal"] == r2["vocal"]
    assert r1["structure"] == r2["structure"]
