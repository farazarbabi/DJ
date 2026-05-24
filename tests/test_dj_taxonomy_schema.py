"""Spec §11.1 — schema validation for dj_taxonomy.json v2.

Confirms every entry has all required fields and the values are well-formed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dj_registry.taxonomy.dj_schema import load_dj_taxonomy

_REQUIRED_FIELDS = (
    "id",
    "label",
    "family",
    "source_genres",
    "moods",
    "grooves",
    "set_roles",
    "bpm_range",
    "energy_range",
    "vocal_profiles",
    "keywords",
)

_ALLOWED_VOCAL_PROFILES = {
    "instrumental",
    "vocal",
    "featured_vocal",
    "spoken",
    "chant",
    "dub",
    "tool",
}

_SNAKE_CASE_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _load_raw() -> dict:
    path = Path("src/dj_registry/taxonomy/dj_taxonomy.json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_loads_via_loader():
    """Loader accepts the v2 file and yields the dataclass-validated taxonomy."""
    taxonomy = load_dj_taxonomy()
    assert taxonomy.version.startswith("dj-taxonomy-v")
    assert len(taxonomy.categories) >= 90


def test_required_fields_present():
    raw = _load_raw()
    for entry in raw["categories"]:
        missing = [f for f in _REQUIRED_FIELDS if f not in entry]
        assert not missing, f"category {entry.get('id')!r} missing fields: {missing}"


def test_ids_are_unique():
    raw = _load_raw()
    ids = [entry["id"] for entry in raw["categories"]]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate IDs: {duplicates}"


def test_ids_are_snake_case():
    raw = _load_raw()
    bad = [entry["id"] for entry in raw["categories"] if not _SNAKE_CASE_RE.match(entry["id"])]
    assert not bad, f"non-snake_case IDs: {bad}"


def test_labels_nonempty():
    raw = _load_raw()
    for entry in raw["categories"]:
        assert entry["label"].strip(), f"category {entry['id']} has empty label"


def test_bpm_range_is_int_pair_ascending():
    raw = _load_raw()
    for entry in raw["categories"]:
        bpm = entry["bpm_range"]
        assert isinstance(bpm, list) and len(bpm) == 2, f"{entry['id']}: bpm_range not pair"
        low, high = bpm
        assert isinstance(low, int) and isinstance(high, int)
        assert 50 <= low <= high <= 220, f"{entry['id']}: bpm out of expected range {bpm}"


def test_energy_range_is_int_pair_in_1_to_5():
    raw = _load_raw()
    for entry in raw["categories"]:
        energy = entry["energy_range"]
        assert isinstance(energy, list) and len(energy) == 2
        low, high = energy
        assert 1 <= low <= high <= 5, f"{entry['id']}: energy out of range {energy}"


def test_vocal_profiles_in_allowed_set():
    raw = _load_raw()
    for entry in raw["categories"]:
        for profile in entry["vocal_profiles"]:
            assert profile in _ALLOWED_VOCAL_PROFILES, (
                f"{entry['id']}: vocal_profile {profile!r} not in allowed set"
            )


def test_string_lists_are_lists_of_strings():
    raw = _load_raw()
    list_fields = ("source_genres", "moods", "grooves", "set_roles", "vocal_profiles", "keywords")
    for entry in raw["categories"]:
        for field in list_fields:
            value = entry[field]
            assert isinstance(value, list), f"{entry['id']}.{field} should be list"
            for item in value:
                assert isinstance(item, str) and item.strip(), (
                    f"{entry['id']}.{field}: invalid item {item!r}"
                )


def test_non_edm_categories_present():
    """The 5 named regression-target tracks need at least one non-EDM home."""
    raw = _load_raw()
    families = {entry["family"] for entry in raw["categories"]}
    # At minimum: an R&B / Soul home for Nina Simone-style tracks.
    has_soul_home = any("R&B" in f or "Soul" in f or "Jazz" in f for f in families)
    assert has_soul_home, (
        f"Taxonomy must include an R&B/Soul/Jazz family for non-EDM tracks; "
        f"families seen: {sorted(families)}"
    )


def test_named_split_categories_present():
    """The §3 split categories must exist as IDs in the new taxonomy."""
    raw = _load_raw()
    ids = {entry["id"] for entry in raw["categories"]}
    expected_new_ids = {
        # Tribal Afro splits (warmup/builder/cinematic_builder were consolidated
        # away in the v2.2 reshuffle; surviving splits are the chant and 3-step)
        "spiritual_afro_chant",
        "afro_3_step",
        # Organic House splits
        "organic_house_warmup",
        "desert_organic_house",
        "balearic_organic_house",
        "organic_downtempo_crossover",
        # Melodic House splits
        "warm_melodic_house",
        "cinematic_melodic_house",
        # Dark Indie Tech-House splits
        "rolling_dark_indie_tech",
        "driving_dark_indie_tech",
        "hypnotic_dark_indie_tech",
        # Hypnotic Indie Dance splits
        "warm_hypnotic_indie",
        "psychedelic_hypnotic_indie",
    }
    missing = expected_new_ids - ids
    assert not missing, f"missing split IDs: {missing}"


def test_modern_additions_present():
    raw = _load_raw()
    ids = {entry["id"] for entry in raw["categories"]}
    expected_new_ids = {
        "afro_tech_driver",
        "afro_house_peak",
        "micro_house",
        "minimal_dub_tool",
        "bass_tech_house",
        "driving_bass_house",
        "hardgroove_techno_driver",
        "cosmic_indie_dance",
        "synth_wave_indie",
        "darkwave_indie_crossover",
        "lo_fi_deep_tech",
        "dub_deep_tech_house",
        "sunset_balearic_house",
        "slow_melodic_house",
        "psy_organic_crossover",
    }
    missing = expected_new_ids - ids
    assert not missing, f"missing modern-addition IDs: {missing}"


def test_provisional_field_optional():
    """Per spec Appendix B — provisional field is optional; loader defaults to False."""
    taxonomy = load_dj_taxonomy()
    for category in taxonomy.categories:
        # Should have the attribute thanks to the schema change
        assert hasattr(category, "provisional")
        # All categories default to False in the shipped JSON
        assert category.provisional is False
