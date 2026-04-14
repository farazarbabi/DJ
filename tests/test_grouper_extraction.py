"""Tests for dj_grouper extraction service — cache, stats, deleted file cleanup."""

import os
import numpy as np
import soundfile as sf
import pytest

from dj_grouper.scanner import TrackInfo
from dj_grouper.cli import _run_extraction, _ExtractionStats
from dj_tagger.universal_cache import reset_cache


@pytest.fixture(autouse=True)
def _reset_ucache():
    """Reset universal cache singleton between tests."""
    reset_cache()
    yield
    reset_cache()


def _make_wav(path: str, duration_sec: float = 3.0) -> None:
    """Create a minimal WAV file."""
    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, y, sr)


class TestRunExtraction:
    def test_extract_single_track(self, tmp_path):
        """Extracting a single track returns populated cache and stats."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")

        raw_cache, stats = _run_extraction(tracks, cache_path, workers=1)

        assert stats.n_extracted == 1
        assert stats.n_cached == 0
        assert stats.n_failed == 0
        assert wav in raw_cache
        assert "dsp" in dir(raw_cache[wav]) or hasattr(raw_cache[wav], "dsp")

    def test_cache_hit_on_rerun(self, tmp_path):
        """Second extraction of same file uses cache."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")

        # First run
        _, stats1 = _run_extraction(tracks, cache_path, workers=1)
        assert stats1.n_extracted == 1
        assert stats1.n_cached == 0

        # Second run — same file, same mtime
        _, stats2 = _run_extraction(tracks, cache_path, workers=1)
        assert stats2.n_extracted == 0
        assert stats2.n_cached == 1

    def test_force_re_extracts(self, tmp_path):
        """force=True ignores existing cache."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")

        # First run
        _run_extraction(tracks, cache_path, workers=1)

        # Force re-extract
        _, stats = _run_extraction(tracks, cache_path, force=True, workers=1)
        assert stats.n_extracted == 1
        assert stats.n_cached == 0

    def test_deleted_file_removed_from_cache(self, tmp_path):
        """Tracks removed from the library are evicted from cache."""
        wav1 = str(tmp_path / "track1.wav")
        wav2 = str(tmp_path / "track2.wav")
        _make_wav(wav1)
        _make_wav(wav2)
        cache_path = str(tmp_path / "cache.pkl")

        # Extract both
        tracks_both = [TrackInfo(path=wav1), TrackInfo(path=wav2)]
        raw_cache, stats = _run_extraction(tracks_both, cache_path, workers=1)
        assert stats.n_extracted == 2
        assert wav1 in raw_cache
        assert wav2 in raw_cache

        # Re-run with only track1 — track2 should be evicted
        tracks_one = [TrackInfo(path=wav1)]
        raw_cache2, stats2 = _run_extraction(tracks_one, cache_path, workers=1)
        assert stats2.n_cached == 1
        assert stats2.n_removed == 1
        assert wav1 in raw_cache2
        assert wav2 not in raw_cache2

    def test_summary_parts(self):
        """ExtractionStats.summary_parts formats correctly."""
        stats = _ExtractionStats()
        stats.n_cached = 5
        stats.n_extracted = 3
        stats.n_failed = 1
        assert stats.summary_parts() == ["5 cached", "3 extracted", "1 failed"]

    def test_summary_parts_empty(self):
        stats = _ExtractionStats()
        assert stats.summary_parts() == []

    def test_vibe_and_vocal_populated(self, tmp_path):
        """Extraction populates vibe and vocal fields on tracks."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav, duration_sec=5.0)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")

        _run_extraction(tracks, cache_path, workers=1)

        assert tracks[0].vibe is not None
        assert tracks[0].vocal in ("V", "NV")
        assert "vibe" in tracks[0].confidences
        assert "vocal" in tracks[0].confidences
