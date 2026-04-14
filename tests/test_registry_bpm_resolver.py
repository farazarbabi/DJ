"""Tests for dj_registry.resolver.bpm_resolver — majority vote + tag tiebreaker."""

import pytest

from dj_registry.config import RegistryConfig
from dj_registry.models import LogicalTrack, SourceObservation
from dj_registry.resolver.bpm_resolver import resolve_track_bpm


def _make_obs(source: str, bpm: str) -> SourceObservation:
    return SourceObservation(
        observation_id=f"OBS-{source}-{bpm}",
        track_id="T-001",
        source_system=source,
        bpm=bpm,
    )


class TestResolveTrackBpm:
    def setup_method(self):
        self.track = LogicalTrack(track_id="T-001")

    def test_no_observations(self):
        resolve_track_bpm(self.track, [])
        assert self.track.canonical_bpm == ""
        assert self.track.canonical_bpm_confidence == 0.0

    def test_single_source(self):
        obs = [_make_obs("rekordbox", "128")]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "128"
        assert self.track.canonical_bpm_confidence == 1.0

    def test_all_sources_agree(self):
        obs = [
            _make_obs("tag", "126"),
            _make_obs("rekordbox", "126"),
            _make_obs("songstats", "126"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "126"
        assert self.track.canonical_bpm_confidence == 1.0

    def test_majority_wins_over_tag(self):
        """2 sources say 128, tag says 126 — majority wins."""
        obs = [
            _make_obs("tag", "126"),
            _make_obs("rekordbox", "128"),
            _make_obs("songstats", "128"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "128"
        assert self.track.canonical_bpm_confidence == pytest.approx(2 / 3, abs=0.01)

    def test_tie_tag_wins(self):
        """1v1 tie: tag=126 vs rekordbox=128 — tag wins as tiebreaker."""
        obs = [
            _make_obs("tag", "126"),
            _make_obs("rekordbox", "128"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "126"
        assert self.track.canonical_bpm_confidence == 0.5

    def test_tie_without_tag_uses_source_priority(self):
        """1v1 tie: rekordbox=128 vs songstats=126 — rekordbox wins by priority."""
        obs = [
            _make_obs("rekordbox", "128"),
            _make_obs("songstats", "126"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "128"
        assert self.track.canonical_bpm_confidence == 0.5

    def test_float_bpm_rounded(self):
        """BPM values like '127.5' get rounded to '128'."""
        obs = [
            _make_obs("tag", "127.5"),
            _make_obs("rekordbox", "128"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "128"
        assert self.track.canonical_bpm_confidence == 1.0

    def test_empty_bpm_ignored(self):
        obs = [
            SourceObservation(observation_id="OBS-1", track_id="T-001", source_system="tag", bpm=""),
            _make_obs("rekordbox", "130"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "130"
        assert self.track.canonical_bpm_confidence == 1.0

    def test_three_way_tie_tag_wins(self):
        """3-way tie: tag=124, rekordbox=126, songstats=128 — tag wins."""
        obs = [
            _make_obs("tag", "124"),
            _make_obs("rekordbox", "126"),
            _make_obs("songstats", "128"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "124"

    def test_majority_2v1v1(self):
        """2 agree on 128, 1 says 126, 1 says 130 — 128 wins."""
        obs = [
            _make_obs("tag", "126"),
            _make_obs("rekordbox", "128"),
            _make_obs("songstats", "128"),
            _make_obs("analysis_librosa", "130"),
        ]
        resolve_track_bpm(self.track, obs)
        assert self.track.canonical_bpm == "128"
        assert self.track.canonical_bpm_confidence == 0.5
