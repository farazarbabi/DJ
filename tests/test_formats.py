"""Tests for tag formatting and parsing."""

from dj_tagger.formats import format_tag, parse_tag


def test_format_with_bpm():
    assert format_tag(3, "9A", 126, "64H", "HYPN", False) == "E3 | 9A | 126 | 64H | HYPN | NV"


def test_format_with_bpm_and_group():
    result = format_tag(3, "9A", 126, "64H", "HYPN", False, "G017")
    assert result == "E3 | 9A | 126 | 64H | HYPN | NV | G017"


def test_format_vocals():
    assert format_tag(4, "5A", 130, "32D", "RAW", True) == "E4 | 5A | 130 | 32D | RAW | V"


def test_format_missing_fields():
    result = format_tag(None, None)
    assert result == "E? | ?? | ??? | ?? | ?? | ??"


def test_parse_v2_with_bpm():
    result = parse_tag("E3 | 9A | 126 | 64H | HYPN | NV")
    assert result == {
        "energy": "3",
        "key": "9A",
        "bpm": "126",
        "structure": "64H",
        "vibe": "HYPN",
        "vocal": "NV",
    }


def test_parse_v2_with_group():
    result = parse_tag("E3 | 9A | 126 | 64H | HYPN | NV | G017")
    assert result is not None
    assert result["group_id"] == "G017"
    assert result["bpm"] == "126"


def test_parse_v1_legacy():
    """Legacy format without BPM should still parse."""
    result = parse_tag("E3 | 9A | 64H | HYPN | NV")
    assert result is not None
    assert result["energy"] == "3"
    assert result["key"] == "9A"
    assert result["structure"] == "64H"
    assert "bpm" not in result


def test_parse_all_vibes():
    for vibe in ("HYPN", "DRK", "RAW", "DEEP", "TRIB", "MEL", "ACID", "ATM"):
        tag = f"E3 | 9A | 128 | 64H | {vibe} | NV"
        result = parse_tag(tag)
        assert result is not None
        assert result["vibe"] == vibe


def test_parse_invalid():
    assert parse_tag("not a tag") is None
    assert parse_tag("") is None
    assert parse_tag("E6 | 9A | 128 | 64H | HYPN | NV") is None  # E6 invalid


def test_parse_all_flow_types():
    for flow in ("G", "H", "D", "B", "L"):
        tag = f"E3 | 9A | 130 | 32{flow} | HYPN | NV"
        result = parse_tag(tag)
        assert result is not None
        assert result["structure"] == f"32{flow}"
