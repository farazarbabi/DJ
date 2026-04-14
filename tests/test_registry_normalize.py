"""Tests for dj_registry.identity.normalize — text normalization."""

import pytest

from dj_registry.identity.normalize import (
    extract_mix_from_title,
    normalize_artist,
    normalize_mix,
    normalize_text,
    normalize_title,
)


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Hello   World  ") == "hello world"

    def test_unicode(self):
        assert normalize_text("Ame\u0301") == "am\u00e9"  # NFD -> NFC

    def test_empty(self):
        assert normalize_text("") == ""


class TestNormalizeArtist:
    def test_featuring(self):
        assert "feat" in normalize_artist("Artist feat. Singer")
        assert "feat" in normalize_artist("Artist featuring Singer")
        assert "feat" in normalize_artist("Artist ft Singer")
        assert "feat" in normalize_artist("Artist ft. Singer")

    def test_separators(self):
        result = normalize_artist("A & B, C and D vs E")
        assert " & " in result

    def test_case(self):
        assert normalize_artist("ARTIST") == "artist"


class TestNormalizeTitle:
    def test_strip_featuring(self):
        result = normalize_title("Song (feat. Singer)")
        assert "feat" not in result
        assert "singer" not in result

    def test_plain(self):
        assert normalize_title("Simple Title") == "simple title"


class TestNormalizeMix:
    def test_canonical_forms(self):
        assert normalize_mix("Original Mix") == "original mix"
        assert normalize_mix("Original") == "original mix"
        assert normalize_mix("Extended Mix") == "extended mix"
        assert normalize_mix("Extended") == "extended mix"
        assert normalize_mix("Radio Edit") == "radio edit"
        assert normalize_mix("Radio Mix") == "radio edit"
        assert normalize_mix("Dub") == "dub mix"
        assert normalize_mix("VIP Mix") == "vip"

    def test_strip_parens(self):
        assert normalize_mix("(Original Mix)") == "original mix"
        assert normalize_mix("[Extended Mix]") == "extended mix"

    def test_remix_preserved(self):
        result = normalize_mix("Dixon Remix")
        assert "dixon remix" == result

    def test_empty(self):
        assert normalize_mix("") == ""


class TestExtractMixFromTitle:
    def test_extract(self):
        title, mix = extract_mix_from_title("Song Name (Original Mix)")
        assert title == "Song Name"
        assert mix == "original mix"

    def test_remix(self):
        title, mix = extract_mix_from_title("Song (Dixon Remix)")
        assert title == "Song"
        assert "dixon remix" in mix

    def test_extended(self):
        title, mix = extract_mix_from_title("Song Title (Extended Mix)")
        assert title == "Song Title"
        assert mix == "extended mix"

    def test_no_mix(self):
        title, mix = extract_mix_from_title("Simple Song")
        assert title == "Simple Song"
        assert mix == ""

    def test_brackets(self):
        title, mix = extract_mix_from_title("Song [Radio Edit]")
        assert title == "Song"
        assert mix == "radio edit"
