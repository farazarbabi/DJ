"""Tests for dj_grouper extraction service — cache, stats, deleted file cleanup."""

import os
import numpy as np
import soundfile as sf
import pytest

from dj_grouper.scanner import TrackInfo
from dj_grouper.cli import _run_extraction, _ExtractionStats, _apply_tagger_result
from dj_tagger.tagger_cache import hydrate_tagger_result
from dj_tagger.universal_cache import get_cache, quick_duration, reset_cache


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

    def test_rerun_with_fewer_tracks(self, tmp_path):
        """Re-running with fewer tracks still works (removed tracks stay in raw cache)."""
        wav1 = str(tmp_path / "track1.wav")
        wav2 = str(tmp_path / "track2.wav")
        _make_wav(wav1)
        _make_wav(wav2)
        cache_path = str(tmp_path / "cache.pkl")

        # Extract both
        tracks_both = [TrackInfo(path=wav1), TrackInfo(path=wav2)]
        raw_cache, stats = _run_extraction(tracks_both, cache_path, workers=1)
        assert stats.n_extracted == 2

        # Re-run with only track1 — track1 should be cached, track2 not in result
        tracks_one = [TrackInfo(path=wav1)]
        raw_cache2, stats2 = _run_extraction(tracks_one, cache_path, workers=1)
        assert stats2.n_cached == 1
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
        assert tracks[0].vocal in {"CHANT", "DUB", "FVOC", "INST", "SPK", "TOOL", "VOC"}
        assert "vibe" in tracks[0].confidences
        assert "vocal" in tracks[0].confidences

    def test_cached_tagger_replaces_unknown_key_placeholder(self):
        """A parsed ?? tag must not block a valid cached tagger key."""
        track = TrackInfo(path="track.wav", key="??", energy=3)

        _apply_tagger_result(
            track,
            {
                "camelot": "9A",
                "energy": 4,
                "bpm": 126.0,
                "structure": "16H",
                "vibe": "DRK",
                "vocal": "NV",
                "confidences": {"key": 0.9},
            },
            analyze_untagged=True,
        )

        assert track.key == "9A"
        assert track.vocal == "INST"
        assert track.confidences["key"] == 0.9

    def test_extract_stores_canonical_raw_layers(self, tmp_path):
        """Extraction should persist reusable raw layers plus hydrated tagger metadata."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav, duration_sec=5.0)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")

        _run_extraction(tracks, cache_path, workers=1, analyze_untagged=True)

        ucache = get_cache(str(tmp_path / "raw_cache.pkl"))
        cache_dur = quick_duration(wav)
        assert ucache.get_track("track.wav", cache_dur, "dsp") is not None
        assert ucache.get_track("track.wav", cache_dur, "raw_analysis") is not None
        assert ucache.get_track("track.wav", cache_dur, "section_dsp") is not None

        tagger = ucache.get_track("track.wav", cache_dur, "tagger")
        assert tagger is not None
        assert tagger["_tagger_version"]
        assert tagger["_tagger_raw_sig"]
        assert tagger["_tagger_derived_sig"]
        assert tagger["_tagger_key_sig"]

    def test_extract_preserves_richer_songstats_tagger(self, tmp_path):
        """Grouper should not overwrite a current Songstats-aware tagger entry with DSP-only output."""
        wav = str(tmp_path / "track.wav")
        _make_wav(wav, duration_sec=5.0)
        tracks = [TrackInfo(path=wav)]
        cache_path = str(tmp_path / "cache.pkl")
        ucache = get_cache(str(tmp_path / "raw_cache.pkl"))
        cache_dur = quick_duration(wav)

        richer = hydrate_tagger_result(
            {
                "camelot": "8A",
                "key": "A minor",
                "key_confidence": 0.9,
                "energy": 3,
                "vibe": "MEL",
                "vocal": "NV",
                "structure": "32H",
                "bpm": 128.0,
                "vibe_scores": {"MEL": 0.9},
                "confidences": {"energy": 0.8, "vibe": 0.8, "vocal": 0.9, "structure": 0.7, "key": 0.9},
            },
            {"valence": 0.9, "energy": 0.6},
        )
        ucache.put_track("track.wav", cache_dur, "tagger", richer)
        ucache.save()

        _run_extraction(tracks, cache_path, workers=1)

        tagger = ucache.get_track("track.wav", cache_dur, "tagger")
        assert tagger is not None
        assert tagger["vibe"] == "MEL"
        assert tagger["_tagger_audio_features_sig"]
