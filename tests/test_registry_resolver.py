"""Tests for dj_registry.resolver.key_resolver — weighted scoring."""

import pytest

from dj_registry.config import RegistryConfig
from dj_registry.models import LogicalTrack, SourceObservation
from dj_registry.resolver.key_resolver import resolve_track_key


def _make_obs(source: str, camelot: str, standard: str = "", confidence: float = 1.0) -> SourceObservation:
    return SourceObservation(
        observation_id=f"OBS-{source}-{camelot}",
        track_id="T-001",
        source_system=source,
        key_standard=standard or f"{camelot} key",
        key_camelot=camelot,
        key_confidence=confidence,
    )


class TestResolveTrackKey:
    def setup_method(self):
        self.config = RegistryConfig()
        self.track = LogicalTrack(track_id="T-001")

    def test_no_observations(self):
        resolve_track_key(self.track, [], self.config)
        assert self.track.needs_manual_review is True
        assert self.track.canonical_key_camelot == ""

    def test_single_high_confidence_source(self):
        obs = [_make_obs("rekordbox", "9A", "E minor")]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "9A"
        assert self.track.needs_manual_review is False

    def test_all_sources_agree_confidence_1(self):
        """All sources agree -> confidence = 1.0."""
        obs = [
            _make_obs("tag", "8A", "A minor"),
            _make_obs("rekordbox", "8A", "A minor"),
            _make_obs("analysis_librosa", "8A", "A minor", 0.9),
        ]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "8A"
        assert self.track.needs_manual_review is False
        assert self.track.canonical_key_confidence == 1.0

    def test_conflict_sends_to_review(self):
        obs = [
            _make_obs("tag", "1A", "G# minor"),
            _make_obs("analysis_librosa", "2A", "Eb minor", 0.6),
        ]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.needs_manual_review is True
        assert "review" in self.track.key_evidence_summary

    def test_manual_override(self):
        obs = [
            _make_obs("tag", "1A", "G# minor"),
            _make_obs("analysis_librosa", "2A", "Eb minor"),
            _make_obs("manual", "1A", "G# minor"),
        ]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "1A"
        assert self.track.canonical_key_source == "manual"
        assert self.track.needs_manual_review is False

    def test_both_analyses_agree(self):
        obs = [
            _make_obs("analysis_librosa", "5A", "F minor", 0.9),
            _make_obs("analysis_essentia", "5A", "F minor", 0.92),
        ]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "5A"
        assert self.track.needs_manual_review is False

    def test_analyses_disagree_low_confidence(self):
        obs = [
            _make_obs("analysis_librosa", "5A", "F minor", 0.5),
            _make_obs("analysis_essentia", "6A", "G minor", 0.5),
        ]
        resolve_track_key(self.track, obs, self.config)
        # Both low confidence, different keys -> review
        assert self.track.needs_manual_review is True

    def test_evidence_summary_format(self):
        obs = [
            _make_obs("tag", "9A", "E minor"),
            _make_obs("rekordbox", "9A", "E minor"),
        ]
        resolve_track_key(self.track, obs, self.config)
        summary = self.track.key_evidence_summary
        assert "tag=9A" in summary
        assert "rekordbox=9A" in summary
        assert "canonical=9A" in summary

    def test_strong_single_analysis(self):
        """A single analysis with high confidence should auto-resolve."""
        obs = [_make_obs("analysis_librosa", "7A", "D minor", 0.95)]
        resolve_track_key(self.track, obs, self.config)
        # score = 0.80 * 0.95 = 0.76 > 0.70 threshold
        assert self.track.canonical_key_camelot == "7A"
        assert self.track.needs_manual_review is False

    def test_single_tag_resolves_as_majority(self):
        """A single tag observation is 100% majority — should auto-resolve."""
        obs = [_make_obs("tag", "7A", "D minor")]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "7A"
        assert self.track.needs_manual_review is False

    def test_tie_breaks_by_tag_priority(self):
        """2v2 tie: tag+rekordbox vs analysis+songstats — tag side wins."""
        obs = [
            _make_obs("tag", "6A", "G minor"),
            _make_obs("rekordbox", "6A", "G minor"),
            _make_obs("songstats", "5A", "F minor"),
            _make_obs("analysis_librosa", "5A", "F minor", 0.9),
        ]
        resolve_track_key(self.track, obs, self.config)
        assert self.track.canonical_key_camelot == "6A"
        assert self.track.needs_manual_review is False
        assert "tag" in self.track.canonical_key_source
