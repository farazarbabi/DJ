"""Tests for tag formatting and parsing."""

from dj_tagger.formats import format_tag, parse_tag
from dj_tagger.moods import MOOD_LABELS


def test_format_current():
    assert (
        format_tag(3, "9A", 126, vibe="HYPN", has_vocals=False)
        == "9A|E3|HYPN|INST"
    )


def test_format_vocals():
    assert (
        format_tag(4, "5A", 130, vibe="RAW", has_vocals=True)
        == "5A|E4|RAW|VOC"
    )


def test_format_explicit_vocal_profile():
    assert (
        format_tag(4, "5A", 130, vibe="RAW", vocal_profile="featured_vocal")
        == "5A|E4|RAW|FVOC"
    )


def test_format_missing_fields():
    result = format_tag(None, None)
    assert result == "??|E?|??|??"


def test_format_unknown_vocab_fields_use_placeholders():
    result = format_tag(3, "9A", 126, vibe="UNKNOWN", vocal_profile="UNKNOWN")
    assert result == "9A|E3|??|??"


def test_format_ignores_bpm_argument():
    """BPM is accepted for forward-compat but must not appear in the tag."""
    with_bpm = format_tag(3, "9A", 126, vibe="HYPN", has_vocals=False)
    without_bpm = format_tag(3, "9A", None, vibe="HYPN", has_vocals=False)
    assert with_bpm == without_bpm == "9A|E3|HYPN|INST"


def test_parse_current():
    result = parse_tag("9A|E3|HYPN|INST")
    assert result == {
        "key": "9A",
        "energy": "3",
        "vibe": "HYPN",
        "vocal": "INST",
    }


def test_parse_does_not_emit_bpm_key():
    result = parse_tag("9A|E3|HYPN|INST")
    assert result is not None
    assert "bpm" not in result


def test_parse_all_vibes_current():
    for vibe in MOOD_LABELS:
        tag = f"9A|E3|{vibe}|INST"
        result = parse_tag(tag)
        assert result is not None
        assert result["vibe"] == vibe


def test_parse_taxonomy_mood():
    result = parse_tag("9A|E3|WARM|INST")
    assert result is not None
    assert result["vibe"] == "WARM"


def test_parse_invalid():
    assert parse_tag("not a tag") is None
    assert parse_tag("") is None
    assert parse_tag("9A|E6|HYPN|INST") is None
    # Old underscore format should no longer parse.
    assert parse_tag("9A_126_E3_HYPN_INST") is None


def test_format_with_category_code():
    assert (
        format_tag(3, "9A", 126, vibe="HYPN", has_vocals=False, category="DRK.TECH.HOUS.DRV")
        == "9A|E3|HYPN|INST|DRK.TECH.HOUS.DRV"
    )


def test_parse_current_with_category_code():
    result = parse_tag("9A|E3|HYPN|INST|DRK.TECH.HOUS.DRV")
    assert result == {
        "key": "9A",
        "energy": "3",
        "vibe": "HYPN",
        "vocal": "INST",
        "category": "DRK.TECH.HOUS.DRV",
    }


def test_format_with_group_id():
    assert (
        format_tag(3, "9A", 126, vibe="HYPN", has_vocals=False, group_id="G001")
        == "9A|E3|HYPN|INST|G001"
    )


def test_format_without_group_id():
    assert (
        format_tag(3, "9A", 126, vibe="HYPN", has_vocals=False)
        == "9A|E3|HYPN|INST"
    )


def test_parse_current_with_group():
    result = parse_tag("9A|E3|HYPN|INST|G017")
    assert result is not None
    assert result["group_id"] == "G017"
    assert "category" not in result


def test_parse_current_with_category_and_group():
    result = parse_tag("9A|E3|HYPN|INST|DRK.TECH.HOUS.DRV|G017")
    assert result is not None
    assert result["category"] == "DRK.TECH.HOUS.DRV"
    assert result["group_id"] == "G017"
