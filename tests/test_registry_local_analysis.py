"""Tests for dj_registry.adapters.local_analysis — cache hits, misses, and observation creation."""

import os
import pytest
import numpy as np
import soundfile as sf

from dj_registry.config import RegistryConfig
from dj_registry.models import LogicalTrack, FileRecord
from dj_registry.store.csv_store import CsvStore
from dj_registry.adapters.local_analysis import run_analysis
from dj_tagger.cache import load_cache, put_cached, save_cache, ANALYZER_VERSION


def _make_wav(path: str, duration_sec: float = 2.0) -> None:
    """Create a minimal WAV file."""
    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, y, sr)


class TestRunAnalysis:
    def setup_method(self):
        self.config = RegistryConfig()

    def test_no_candidates(self, tmp_path):
        """No tracks with primary files -> 0 processed."""
        self.config.output_dir = str(tmp_path / "registry")
        store = CsvStore(self.config.output_dir)
        store.save_tracks([LogicalTrack(track_id="T-001")])
        store.save_files([])
        store.save_observations([])

        count = run_analysis(self.config, store, no_essentia=True)
        assert count == 0

    def test_cache_hit_creates_observation(self, tmp_path):
        """A file with a tagger cache entry should create a librosa observation from cache."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path)
        mtime = os.path.getmtime(wav_path)
        duration = 2.0

        # Set up tagger cache with a known result
        cache_path = str(tmp_path / "cache" / "tagger_cache.pkl")
        tagger_cache = {}
        put_cached(tagger_cache, wav_path, mtime, {
            "camelot": "8A",
            "key": "A minor",
            "key_confidence": 0.92,
        }, duration=duration)
        save_cache(tagger_cache, cache_path)

        # Set up registry store
        self.config.output_dir = str(tmp_path / "registry")

        store = CsvStore(self.config.output_dir)
        store.save_tracks([
            LogicalTrack(track_id="T-001", primary_file_id="F-001"),
        ])
        store.save_files([
            FileRecord(
                file_id="F-001", track_id="T-001", path_abs=wav_path,
                is_primary_file=True, audio_duration_sec=duration,
            ),
        ])
        store.save_observations([])

        # Monkey-patch the cache path
        import dj_registry.adapters.local_analysis as la_mod
        orig_cache_path = la_mod.TAGGER_CACHE_PATH
        la_mod.TAGGER_CACHE_PATH = cache_path
        try:
            count = run_analysis(self.config, store, no_essentia=True)
        finally:
            la_mod.TAGGER_CACHE_PATH = orig_cache_path

        assert count == 1

        obs = store.load_observations()
        librosa_obs = [o for o in obs if o.source_system == "analysis_librosa"]
        assert len(librosa_obs) == 1
        assert librosa_obs[0].key_camelot == "8A"
        assert librosa_obs[0].key_standard == "A minor"
        assert librosa_obs[0].key_confidence == pytest.approx(0.92)

    def test_cache_miss_runs_analysis(self, tmp_path):
        """A file not in tagger cache should be analyzed and create observations."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path, duration_sec=5.0)

        # Empty tagger cache
        cache_path = str(tmp_path / "cache" / "tagger_cache.pkl")
        save_cache({}, cache_path)

        self.config.output_dir = str(tmp_path / "registry")

        store = CsvStore(self.config.output_dir)
        store.save_tracks([
            LogicalTrack(track_id="T-001", primary_file_id="F-001"),
        ])
        store.save_files([
            FileRecord(
                file_id="F-001", track_id="T-001", path_abs=wav_path,
                is_primary_file=True, audio_duration_sec=5.0,
            ),
        ])
        store.save_observations([])

        import dj_registry.adapters.local_analysis as la_mod
        orig_cache_path = la_mod.TAGGER_CACHE_PATH
        la_mod.TAGGER_CACHE_PATH = cache_path
        try:
            count = run_analysis(self.config, store, no_essentia=True)
        finally:
            la_mod.TAGGER_CACHE_PATH = orig_cache_path

        assert count == 1

        obs = store.load_observations()
        librosa_obs = [o for o in obs if o.source_system == "analysis_librosa"]
        assert len(librosa_obs) == 1
        assert librosa_obs[0].key_camelot  # some key was detected
        assert librosa_obs[0].key_confidence > 0

        # Verify result was written to tagger cache
        updated_cache = load_cache(cache_path)
        assert len(updated_cache) > 0

    def test_rerun_uses_cache(self, tmp_path):
        """Running analysis twice should use cache on second run."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path, duration_sec=5.0)

        cache_path = str(tmp_path / "cache" / "tagger_cache.pkl")
        save_cache({}, cache_path)

        self.config.output_dir = str(tmp_path / "registry")

        store = CsvStore(self.config.output_dir)
        store.save_tracks([
            LogicalTrack(track_id="T-001", primary_file_id="F-001"),
        ])
        store.save_files([
            FileRecord(
                file_id="F-001", track_id="T-001", path_abs=wav_path,
                is_primary_file=True, audio_duration_sec=5.0,
            ),
        ])
        store.save_observations([])

        import dj_registry.adapters.local_analysis as la_mod
        orig_cache_path = la_mod.TAGGER_CACHE_PATH
        la_mod.TAGGER_CACHE_PATH = cache_path
        try:
            # First run — cache miss, actual analysis
            count1 = run_analysis(self.config, store, no_essentia=True)
            assert count1 == 1

            # Second run — cache hit, no analysis needed
            count2 = run_analysis(self.config, store, no_essentia=True)
            assert count2 == 1
        finally:
            la_mod.TAGGER_CACHE_PATH = orig_cache_path
