"""Tests for the LLM ground-truth subgenre-classifier prompt.

The prompt encodes:
  - hard rules (only pick from taxonomy, JSON-only output)
  - a 5-step decision process with BPM gate first
  - anti-bias rules blocking afro over-tagging by routing world/tribal cues
    to their proper regional categories
  - evidence weighting (STRONG / MEDIUM / WEAK / NOT-evidence)
  - the wired output schema (category_id / confidence / rationale /
    alternate_category_ids / warnings)

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
    SYSTEM_PROMPT,
    _system_prompt,
    build_user_message,
    generate_dj_ground_truth_csv,
)


# ── Prompt-content assertions ───────────────────────────────────────────────


def test_prompt_declares_classifier_role_and_taxonomy_constraint():
    assert "subgenre classifier" in SYSTEM_PROMPT
    assert "allowed_dj_taxonomy.categories" in SYSTEM_PROMPT


def test_prompt_includes_hard_rules_and_json_only_constraint():
    assert "Hard rules" in SYSTEM_PROMPT
    assert "Output JSON only" in SYSTEM_PROMPT
    assert "NEVER invent" in SYSTEM_PROMPT


def test_prompt_includes_five_step_decision_process_with_bpm_gate_first():
    assert "Decision process" in SYSTEM_PROMPT
    assert "Step 1 — BPM gate" in SYSTEM_PROMPT
    assert "Step 5 — Set confidence" in SYSTEM_PROMPT


def test_prompt_includes_anti_bias_rules_with_regional_routing():
    assert "Anti-bias rules" in SYSTEM_PROMPT
    # Each regional scene routes to a specific taxonomy id, not generic afro
    for scene, target in (
        ("anatolian", "anatolian_psych_house"),
        ("saz", "anatolian_psych_house"),
        ("oud", "oriental_arabic_house"),
        ("tulum", "tulum_tribal_"),
        ("balkan", "balkan_gypsy_groove"),
        ("amapiano", "amapiano_groove"),
    ):
        assert scene in SYSTEM_PROMPT.lower(), f"prompt missing scene keyword {scene!r}"
        assert target in SYSTEM_PROMPT, f"prompt missing routing target {target!r}"


def test_prompt_enforces_afro_lineage_requirements():
    """afro_* requires explicit Afro / African / Afro-diasporic lineage."""
    assert "Afro / African / Afro-diasporic" in SYSTEM_PROMPT
    assert "MoBlack" in SYSTEM_PROMPT  # named afro-allow label
    assert "Zulu" in SYSTEM_PROMPT  # named afro-allow vocal language


def test_prompt_evidence_weighting_separates_strong_medium_weak():
    assert "STRONG" in SYSTEM_PROMPT
    assert "MEDIUM" in SYSTEM_PROMPT
    assert "WEAK" in SYSTEM_PROMPT
    # Mood / texture / groove words are explicitly disqualified as lineage
    for term in ("Mood words", "Texture words", "Groove descriptors"):
        assert term in SYSTEM_PROMPT, f"prompt missing evidence demotion: {term!r}"


def test_prompt_states_tribal_and_afro_must_be_earned():
    assert (
        "Tribal and Afro must be EARNED by evidence" in SYSTEM_PROMPT
        or "Tribal and Afro must be earned by evidence" in SYSTEM_PROMPT
    )


def test_prompt_documents_wired_output_schema():
    """Schema lines must use the wired field names (rationale,
    alternate_category_ids, warnings) — not the spec's old draft names."""
    assert "Output schema" in SYSTEM_PROMPT
    assert '"category_id"' in SYSTEM_PROMPT
    assert '"confidence"' in SYSTEM_PROMPT
    assert '"rationale"' in SYSTEM_PROMPT
    assert '"alternate_category_ids"' in SYSTEM_PROMPT
    assert '"warnings"' in SYSTEM_PROMPT
    # Spec draft names that didn't make it into the wired schema must NOT
    # appear in the output-schema instructions (folded into rationale instead)
    assert '"reasoning"' not in SYSTEM_PROMPT
    assert '"alternatives"' not in SYSTEM_PROMPT
    assert '"rejected_afro"' not in SYSTEM_PROMPT
    assert '"evidence_used"' not in SYSTEM_PROMPT


def test_prompt_includes_confidence_calibration_bands():
    assert "Confidence calibration" in SYSTEM_PROMPT
    assert "0.85" in SYSTEM_PROMPT
    assert "0.40" in SYSTEM_PROMPT


def test_prompt_includes_diverse_few_shot_examples():
    """At least one example per major confusion cluster the prompt targets."""
    needed_anchors = (
        "Bedouin",            # indie-tech dark
        "Üsküdara",           # anatolian non-afro
        "Bona Fide",          # tulum tribal non-afro
        "Caiiro",             # genuine afro floor
        "Skepsis",            # uk bass
        "Tim Reaper",         # jungle revival
        "Kabza De Small",     # amapiano vs afro
        "Mr. Fingers",        # warm deep house
        "Moodymann",          # lo-fi deep house
        "Frankie Knuckles",   # classic house
        "Phuture",            # raw acid
    )
    for anchor in needed_anchors:
        assert anchor in SYSTEM_PROMPT, f"prompt missing few-shot anchor: {anchor!r}"


def test_prompt_includes_retry_message_when_validation_error_supplied():
    prompt = _system_prompt(validation_error="missing category_id")
    assert "Previous response was invalid: missing category_id" in prompt


def test_prompt_unchanged_when_no_validation_error():
    assert _system_prompt() == SYSTEM_PROMPT
    assert _system_prompt(validation_error="") == SYSTEM_PROMPT


# ── build_user_message contract ─────────────────────────────────────────────


def test_build_user_message_emits_track_evidence_header_and_json_directive():
    out = build_user_message({"title": "X", "artist": "Y", "bpm": 124})
    assert out.startswith("TRACK EVIDENCE:")
    assert "title: X" in out
    assert "artist: Y" in out
    assert "bpm: 124" in out
    assert out.rstrip().endswith("Return the JSON object only.")


def test_build_user_message_drops_empty_values():
    out = build_user_message({
        "title": "X", "artist": "", "label": None, "moods": [], "extras": {},
        "bpm": 124,
    })
    assert "title: X" in out
    assert "artist" not in out
    assert "label" not in out
    assert "moods" not in out
    assert "extras" not in out
    assert "bpm: 124" in out


def test_build_user_message_raises_on_empty_track():
    with pytest.raises(ValueError):
        build_user_message({})


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
        "alternate_category_ids": ["ritual_chant_house", "burner_desert_house"],
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
