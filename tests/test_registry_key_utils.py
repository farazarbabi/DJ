"""Tests for dj_registry.key_utils — key normalization and conversion."""

import pytest

from dj_registry.key_utils import (
    camelot_to_standard,
    is_camelot,
    normalize_standard_key,
    parse_any_key,
    standard_to_camelot,
)


class TestIsCamelot:
    def test_valid_codes(self):
        for n in range(1, 13):
            assert is_camelot(f"{n}A")
            assert is_camelot(f"{n}B")

    def test_invalid(self):
        assert not is_camelot("0A")
        assert not is_camelot("13A")
        assert not is_camelot("1C")
        assert not is_camelot("Am")
        assert not is_camelot("")


class TestNormalizeStandardKey:
    def test_canonical_forms(self):
        assert normalize_standard_key("C major") == "C major"
        assert normalize_standard_key("G# minor") == "G# minor"
        assert normalize_standard_key("Ab major") == "Ab major"
        assert normalize_standard_key("F# minor") == "F# minor"

    def test_mode_aliases(self):
        assert normalize_standard_key("C maj") == "C major"
        assert normalize_standard_key("A min") == "A minor"
        assert normalize_standard_key("E m") == "E minor"

    def test_case_insensitive(self):
        assert normalize_standard_key("c major") == "C major"
        assert normalize_standard_key("G# MINOR") == "G# minor"

    def test_enharmonic(self):
        assert normalize_standard_key("Db major") == "C# major"
        assert normalize_standard_key("D# minor") == "Eb minor"
        assert normalize_standard_key("Gb minor") == "F# minor"
        assert normalize_standard_key("A# minor") == "Bb minor"

    def test_dj_shorthand(self):
        assert normalize_standard_key("Cm") == "C minor"
        assert normalize_standard_key("G#m") == "G# minor"
        assert normalize_standard_key("Abm") == "G# minor"
        assert normalize_standard_key("Em") == "E minor"

    def test_ab_vs_gsharp(self):
        # Ab for major, G# for minor
        assert normalize_standard_key("Ab major") == "Ab major"
        assert normalize_standard_key("G# minor") == "G# minor"
        # Cross-context normalization
        assert normalize_standard_key("G# major") == "Ab major"
        assert normalize_standard_key("Ab minor") == "G# minor"

    def test_invalid(self):
        assert normalize_standard_key("") is None
        assert normalize_standard_key("X major") is None
        assert normalize_standard_key("not a key") is None


class TestStandardToCamelot:
    def test_all_major_keys(self):
        expected = {
            "C major": "8B", "C# major": "3B", "D major": "10B",
            "Eb major": "5B", "E major": "12B", "F major": "7B",
            "F# major": "2B", "G major": "9B", "Ab major": "4B",
            "A major": "11B", "Bb major": "6B", "B major": "1B",
        }
        for std, cam in expected.items():
            assert standard_to_camelot(std) == cam, f"{std} -> {cam}"

    def test_all_minor_keys(self):
        expected = {
            "C minor": "5A", "C# minor": "12A", "D minor": "7A",
            "Eb minor": "2A", "E minor": "9A", "F minor": "4A",
            "F# minor": "11A", "G minor": "6A", "G# minor": "1A",
            "A minor": "8A", "Bb minor": "3A", "B minor": "10A",
        }
        for std, cam in expected.items():
            assert standard_to_camelot(std) == cam, f"{std} -> {cam}"

    def test_shorthand(self):
        assert standard_to_camelot("Am") == "8A"
        assert standard_to_camelot("Em") == "9A"


class TestCamelotToStandard:
    def test_round_trip(self):
        """Every Camelot code should round-trip through standard and back."""
        for n in range(1, 13):
            for letter in "AB":
                cam = f"{n}{letter}"
                std = camelot_to_standard(cam)
                assert std is not None, f"Failed to convert {cam}"
                cam2 = standard_to_camelot(std)
                assert cam2 == cam, f"Round trip failed: {cam} -> {std} -> {cam2}"

    def test_specific(self):
        assert camelot_to_standard("1A") == "G# minor"
        assert camelot_to_standard("8B") == "C major"
        assert camelot_to_standard("9A") == "E minor"

    def test_invalid(self):
        assert camelot_to_standard("0A") is None
        assert camelot_to_standard("13B") is None
        assert camelot_to_standard("Am") is None


class TestParseAnyKey:
    def test_camelot_input(self):
        result = parse_any_key("9A")
        assert result == ("E minor", "9A")

    def test_standard_input(self):
        result = parse_any_key("E minor")
        assert result == ("E minor", "9A")

    def test_shorthand_input(self):
        result = parse_any_key("Em")
        assert result == ("E minor", "9A")

    def test_invalid(self):
        assert parse_any_key("") is None
        assert parse_any_key("not a key") is None

    def test_whitespace(self):
        result = parse_any_key("  9A  ")
        assert result == ("E minor", "9A")
