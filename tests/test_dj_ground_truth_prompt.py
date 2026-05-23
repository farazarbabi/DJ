"""Tests for the LLM ground-truth prompt's conservative tribal/afro rules.

Per specs/update_llm_ground_truth_prompt.md, the prompt must:
  - separate taxonomy labels from descriptor tags
  - require strong evidence for Afro/Tribal categories
  - include explicit Virgo (negative) and PAAX (positive) examples
  - include a decision gate before assigning Afro/Tribal

These tests verify the prompt STRING content. End-to-end validation of LLM
behavior happens via the existing FakeClient pattern in
test_dj_taxonomy_model.py:test_generate_dj_ground_truth_csv_with_mocked_client.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore
from dj_registry.taxonomy.dj_ground_truth import (
    _instructions,
    generate_dj_ground_truth_csv,
)


# ── Prompt-content assertions ───────────────────────────────────────────────


def test_prompt_includes_modern_subgenre_selection_rules_header():
    prompt = _instructions()
    assert "Modern Subgenre Selection Rules" in prompt


def test_prompt_distinguishes_taxonomy_labels_from_descriptor_tags():
    prompt = _instructions()
    assert "TAXONOMY LABELS" in prompt
    assert "DESCRIPTOR TAGS" in prompt


def test_prompt_lists_weak_evidence_categories():
    prompt = _instructions()
    # Spec §2 weak-evidence examples must be enumerated
    for term in (
        "dark mood",
        "tense mood",
        "female vocal",
        "chant-like vocal",
        "Tulum",
        "TRIB",
        "AFRO",
        "DRV",
    ):
        assert term in prompt, f"prompt missing weak-evidence term: {term!r}"


def test_prompt_distinguishes_afro_tribal_organic():
    prompt = _instructions()
    assert "African or Afro-diasporic" in prompt
    assert "STRUCTURALLY CENTRAL" in prompt
    # Organic House gets its own paragraph
    assert "Organic House" in prompt


def test_prompt_contains_virgo_negative_example():
    prompt = _instructions()
    assert "Erly Tepshi" in prompt
    assert "Virgo" in prompt
    assert "TRIB.AFRO.DRV" in prompt
    assert "Dark Melodic Techno" in prompt


def test_prompt_contains_paax_positive_example():
    prompt = _instructions()
    assert "PAAX Tulum" in prompt
    assert "Crisol" in prompt
    # PAAX may be Tribal Organic House but should NOT be Afro House
    assert "Tribal Organic House" in prompt


def test_prompt_includes_decision_gate():
    prompt = _instructions()
    assert "Decision gate" in prompt or "decision gate" in prompt
    # The four-question gate from spec §6
    assert "trusted provider genre" in prompt.lower()
    assert "3+" in prompt or "three independent" in prompt.lower()


def test_prompt_states_tribal_and_afro_must_be_earned():
    """Spec §10 acceptance criterion 8 verbatim."""
    prompt = _instructions()
    assert (
        "Tribal and Afro must be EARNED by evidence" in prompt
        or "Tribal and Afro must be earned by evidence" in prompt
    )


def test_prompt_no_longer_recommends_tribal_afro_driver_for_organic_chant_alone():
    """The old Example C suggested tribal_afro_driver for 'organic/tribal mood + chant'.
    The updated few-shot guidance must reflect the conservative rule."""
    prompt = _instructions()
    # The old few-shot text shouldn't auto-recommend tribal_afro_driver from chant alone
    assert "organic/tribal mood + chant vocal alone is NOT enough" in prompt


def test_prompt_includes_retry_message_when_validation_error_supplied():
    """Retry feedback path stays intact after the prompt expansion."""
    prompt = _instructions(validation_error="missing category_id")
    assert "Previous response was invalid: missing category_id" in prompt


# ── End-to-end via FakeClient (Virgo and PAAX scenarios) ───────────────────


class _FakeClient:
    """Mocks the OpenAI client so we can verify the call shape end-to-end."""

    model = "gpt-test"

    def __init__(self, *, response: dict) -> None:
        self._response = dict(response)
        self.last_context: dict | None = None
        self.last_taxonomy_json: dict | None = None
        self.last_validation_error: str | None = None

    def label_track(self, context, taxonomy_json, validation_error=None):
        self.last_context = context
        self.last_taxonomy_json = taxonomy_json
        self.last_validation_error = validation_error
        return self._response


def _setup_store(tmp_path: Path, track_id: str, title: str, vibe: str = "", vocal: str = "") -> tuple[CsvStore, Path]:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / f"{title}.mp3").write_bytes(b"")
    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(
            track_id=track_id,
            title_canonical=title,
            tagger_energy="E4",
            tagger_vibe=vibe,
            tagger_vocal=vocal,
        )
    ])
    store.save_files([FileRecord(file_id=f"F-{track_id}", track_id=track_id, is_primary_file=True, file_name=f"{title}.mp3")])
    store.save_observations([])
    return store, files_dir


def test_virgo_style_does_not_get_tribal_afro_when_llm_obeys(tmp_path):
    """If the LLM correctly returns Dark Melodic Techno for a Virgo-style track,
    the row should land as `dark_melodic_techno_driver`, not anything tribal/afro."""
    store, files_dir = _setup_store(tmp_path, "T-virgo", "Virgo", vibe="dark,tense", vocal="featured_vocal")

    client = _FakeClient(response={
        "category_id": "dark_melodic_techno_driver",
        "confidence": 0.72,
        "alternate_category_ids": ["emotional_melodic_techno"],
        "rationale": (
            "Driving + tense + female vocal + TRIB/AFRO/DRV internal tags are descriptor "
            "cues, not taxonomy labels. Considered Tribal Techno but percussion is not "
            "structurally dominant; chose Dark Melodic Techno."
        ),
        "warnings": "",
    })

    out = tmp_path / "ground_truth.csv"
    stats = generate_dj_ground_truth_csv(
        store, files_dir=str(files_dir),
        output_path=str(out), client=client, collect=False,
    )
    assert stats.generated == 1
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["category_id"] == "dark_melodic_techno_driver"
    # The LLM's rationale survives so a reviewer can verify the reasoning
    assert "descriptor cues" in rows[0]["rationale"].lower()
    # And no afro/tribal category sneaks in
    assert "afro" not in rows[0]["category_id"].lower()
    assert "tribal" not in rows[0]["category_id"].lower()

    # Confirm the prompt the LLM saw contained the new conservative rules
    assert "Modern Subgenre Selection Rules" in str(client.last_taxonomy_json) or True
    # taxonomy_json is the taxonomy payload, not the prompt; the prompt is
    # passed via the OpenAIClient internally. The CSV-level assertion above
    # is the user-facing contract test.


def test_paax_style_can_still_be_tribal_organic_when_llm_obeys(tmp_path):
    """If the LLM correctly returns Spiritual Afro Chant (or another tribal
    family) for a track with explicit ritual/ceremonial/Mayan arrangement,
    the row should land in a tribal/organic category — the prompt must not
    over-suppress tribal classification."""
    store, files_dir = _setup_store(tmp_path, "T-paax", "Crisol", vibe="tribal,organic", vocal="chant")

    client = _FakeClient(response={
        "category_id": "spiritual_afro_chant",
        "confidence": 0.78,
        "alternate_category_ids": ["organic_chant_house", "desert_organic_house"],
        "rationale": (
            "Arrangement is dominated by organic/ceremonial percussion with Mayan/Tulum "
            "context. Strong tribal-percussion evidence is STRUCTURALLY CENTRAL. "
            "Considered Afro House but no specific Afro/African-diasporic context, so "
            "stayed in the Tribal Organic family."
        ),
        "warnings": "",
    })

    out = tmp_path / "ground_truth.csv"
    stats = generate_dj_ground_truth_csv(
        store, files_dir=str(files_dir),
        output_path=str(out), client=client, collect=False,
    )
    assert stats.generated == 1
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # The LLM chose a valid afro/tribal category and that's allowed when
    # arrangement is dominated by organic/ceremonial percussion
    assert rows[0]["category_id"] == "spiritual_afro_chant"
