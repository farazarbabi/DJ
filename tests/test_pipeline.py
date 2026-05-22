"""Tests for the full analysis pipeline."""

from dj_tagger.pipeline import AnalysisConfig, analyze_track
from dj_tagger.formats import parse_tag
from dj_tagger.moods import MOOD_LABELS
from dj_tagger.vocals import VOCAL_PROFILE_LABELS


def test_full_pipeline(sine_440hz):
    """Full pipeline should produce a valid tag."""
    config = AnalysisConfig(dry_run=True)
    result = analyze_track(sine_440hz, config)
    assert "tag" in result
    assert parse_tag(result["tag"]) is not None
    assert result["bpm"] is not None
    assert result["energy"] in (1, 2, 3, 4, 5)
    assert result["camelot"] is not None
    assert result["vibe"] in MOOD_LABELS
    assert result["vocal"] in VOCAL_PROFILE_LABELS
    assert result["vocal_profile"] == result["vocal"]


def test_pipeline_skip_existing(flac_file):
    """If a file already has a tag and overwrite is False, it should be skipped."""
    from dj_tagger.metadata import write_tag
    existing_tag = "8A|E2|DEEP|INST"
    write_tag(flac_file, existing_tag, dry_run=False)

    config = AnalysisConfig(dry_run=True, overwrite=False)
    result = analyze_track(flac_file, config)
    assert result.get("skipped") is True
    assert result["tag"] == existing_tag


def test_pipeline_overwrite(flac_file):
    """With overwrite=True, it should re-analyze even if tagged."""
    from dj_tagger.metadata import write_tag
    write_tag(flac_file, "1A|E1|ATM|INST", dry_run=False)

    config = AnalysisConfig(dry_run=True, overwrite=True)
    result = analyze_track(flac_file, config)
    assert result.get("skipped") is not True
    assert parse_tag(result["tag"]) is not None
