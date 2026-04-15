"""Tests for dj_registry.adapters.local_analysis — cache hits, misses, and observation creation."""

import os
import pytest
import numpy as np
import soundfile as sf

from dj_registry.config import RegistryConfig
from dj_registry.models import LogicalTrack, FileRecord
from dj_registry.store.csv_store import CsvStore
from dj_registry.adapters.local_analysis import run_analysis
from dj_tagger.universal_cache import get_cache, reset_cache


def _make_wav(path: str, duration_sec: float = 2.0) -> None:
    """Create a minimal WAV file."""
    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, y, sr)


class TestRunAnalysis:
    def setup_method(self):
        self.config = RegistryConfig()
        reset_cache()

    def teardown_method(self):
        reset_cache()

    def test_no_candidates(self, tmp_path):
        """No tracks with primary files -> 0 processed."""
        self.config.output_dir = str(tmp_path / "registry")
        store = CsvStore(self.config.output_dir)
        store.save_tracks([LogicalTrack(track_id="T-001")])
        store.save_files([])
        store.save_observations([])

        stats = run_analysis(self.config, store, no_essentia=True)
        assert stats["total"] == 0

    def test_cache_hit_creates_observation(self, tmp_path, monkeypatch):
        """A file with a tagger cache entry should create a librosa observation from cache."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path)
        duration = 2.0

        # chdir so relative cache path resolves to test dir
        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)

        # Set up universal cache with a known tagger result
        ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))
        ucache.put_track("track.wav", duration, "tagger", {
            "camelot": "8A",
            "key": "A minor",
            "key_confidence": 0.92,
            "energy": 3,
            "vibe": "DRK",
            "vocal": "NV",
            "structure": "32H",
            "bpm": 128.0,
            "vibe_scores": {"DRK": 0.8, "HYPN": 0.1},
            "confidences": {"energy": 0.9, "vibe": 0.7, "vocal": 0.8},
        })
        ucache.save()

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

        stats = run_analysis(self.config, store, no_essentia=True)

        assert stats["total"] == 1

        obs = store.load_observations()
        librosa_obs = [o for o in obs if o.source_system == "analysis_librosa"]
        assert len(librosa_obs) == 1
        assert librosa_obs[0].key_camelot == "8A"
        assert librosa_obs[0].key_standard == "A minor"
        assert librosa_obs[0].key_confidence == pytest.approx(0.92)

    def test_cache_miss_runs_analysis(self, tmp_path, monkeypatch):
        """A file not in tagger cache should be analyzed and create observations."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path, duration_sec=5.0)

        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)
        ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))

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

        stats = run_analysis(self.config, store, no_essentia=True)

        assert stats["total"] == 1

        obs = store.load_observations()
        librosa_obs = [o for o in obs if o.source_system == "analysis_librosa"]
        assert len(librosa_obs) == 1
        assert librosa_obs[0].key_camelot  # some key was detected
        assert librosa_obs[0].key_confidence > 0

        # Verify result was stored in universal cache
        tagger_entries = [k for k in ucache._entries if k.endswith("|tagger")]
        assert len(tagger_entries) > 0

    def test_rerun_uses_cache(self, tmp_path, monkeypatch):
        """Running analysis twice should use cache on second run."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path, duration_sec=5.0)

        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)
        get_cache(os.path.join("cache", "raw_cache.pkl"))

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

        # First run — cache miss, actual analysis
        stats1 = run_analysis(self.config, store, no_essentia=True)
        assert stats1["total"] == 1
        assert stats1["analyzed"] == 1

        # Second run — cache hit, no analysis needed
        stats2 = run_analysis(self.config, store, no_essentia=True)
        assert stats2["total"] == 1
        assert stats2["cached"] == 1
