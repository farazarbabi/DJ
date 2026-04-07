"""Tests for tag formatting and parsing."""

from dj_tagger.formats import format_tag, parse_tag


def test_format_v3():
    assert format_tag(3, "9A", 126, "64H", "HYPN", False) == "9A_E3_HYPN_64H_NV_126"


def test_format_vocals():
    assert format_tag(4, "5A", 130, "32D", "RAW", True) == "5A_E4_RAW_32D_V_130"


def test_format_missing_fields():
    result = format_tag(None, None)
    assert result == "??_E?_??_??_??_???"


def test_parse_v3():
    result = parse_tag("9A_E3_HYPN_64H_NV_126")
    assert result == {
        "key": "9A",
        "energy": "3",
        "vibe": "HYPN",
        "structure": "64H",
        "vocal": "NV",
        "bpm": "126",
    }


def test_parse_v2_legacy():
    """Legacy v2 format should still parse."""
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
    """Legacy v2 with group ID should still parse."""
    result = parse_tag("E3 | 9A | 126 | 64H | HYPN | NV | G017")
    assert result is not None
    assert result["group_id"] == "G017"
    assert result["bpm"] == "126"


def test_parse_v1_legacy():
    """Legacy v1 format without BPM should still parse."""
    result = parse_tag("E3 | 9A | 64H | HYPN | NV")
    assert result is not None
    assert result["energy"] == "3"
    assert result["key"] == "9A"
    assert result["structure"] == "64H"
    assert "bpm" not in result


def test_parse_all_vibes_v3():
    for vibe in ("HYPN", "DRK", "RAW", "DEEP", "TRIB", "MEL", "ACID", "ATM"):
        tag = f"9A_E3_{vibe}_64H_NV_128"
        result = parse_tag(tag)
        assert result is not None
        assert result["vibe"] == vibe


def test_parse_invalid():
    assert parse_tag("not a tag") is None
    assert parse_tag("") is None
    assert parse_tag("E6 | 9A | 128 | 64H | HYPN | NV") is None  # E6 invalid


def test_parse_all_flow_types_v3():
    for flow in ("G", "H", "D", "B", "L"):
        tag = f"9A_E3_HYPN_32{flow}_NV_130"
        result = parse_tag(tag)
        assert result is not None
        assert result["structure"] == f"32{flow}"


def test_format_with_group_id():
    assert format_tag(3, "9A", 126, "64H", "HYPN", False, "G001") == "9A_E3_HYPN_64H_NV_126_G001"


def test_format_without_group_id():
    """Without group_id, tag should not have trailing underscore."""
    assert format_tag(3, "9A", 126, "64H", "HYPN", False) == "9A_E3_HYPN_64H_NV_126"


def test_parse_v3_with_group():
    result = parse_tag("9A_E3_HYPN_64H_NV_126_G017")
    assert result is not None
    assert result["group_id"] == "G017"
    assert result["key"] == "9A"
    assert result["bpm"] == "126"


def test_parse_v3_without_group():
    """Tags without group_id should still parse and not have group_id key."""
    result = parse_tag("9A_E3_HYPN_64H_NV_126")
    assert result is not None
    assert "group_id" not in result
