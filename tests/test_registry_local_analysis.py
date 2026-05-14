"""Tests for dj_registry.adapters.local_analysis — cache hits, misses, and observation creation."""

import os
import pytest
import numpy as np
import soundfile as sf

from dj_registry.config import RegistryConfig
from dj_registry.models import LogicalTrack, FileRecord
from dj_registry.store.csv_store import CsvStore
from dj_registry.adapters.local_analysis import run_analysis
from dj_grouper.features.dsp import DSP_CURATED_NAMES
from dj_tagger.derive import derive_all
from dj_tagger.tagger_cache import hydrate_tagger_result
from dj_tagger.universal_cache import get_cache, quick_duration, reset_cache


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

        # Set up universal cache with a known tagger result (must include metadata)
        ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))
        tagger_entry = hydrate_tagger_result({
            "camelot": "8A",
            "key": "A minor",
            "key_confidence": 0.92,
            "energy": 3,
            "vibe": "DRK",
            "vocal": "INST",
            "structure": "32H",
            "bpm": 128.0,
            "vibe_scores": {"DRK": 0.8, "HYPN": 0.1},
            "confidences": {"energy": 0.9, "vibe": 0.7, "vocal": 0.8},
        })
        ucache.put_track("track.wav", duration, "tagger", tagger_entry)
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
        cache_dur = quick_duration(wav_path) or 5.0
        assert ucache.get_track("track.wav", cache_dur, "dsp") is not None
        assert ucache.get_track("track.wav", cache_dur, "raw_analysis") is not None
        assert ucache.get_track("track.wav", cache_dur, "section_dsp") is not None

        track = store.load_tracks()[0]
        assert track.tagger_version
        assert track.tagger_raw_signature
        assert track.tagger_derived_signature
        assert track.tagger_key_signature

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

    def test_cache_hit_refreshes_vibe_when_songstats_signature_changes(self, tmp_path, monkeypatch):
        """Current-version tagger entries should be refreshed when Songstats-aware vibe inputs change."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path)
        duration = 2.0

        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)
        cache_dur = quick_duration(wav_path) or duration
        ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))

        dsp = {
            "rms_mean": 0.18,
            "centroid_mean": 1700.0,
            "centroid_var": 42000.0,
            "flatness_mean": 0.012,
            "onset_density": 1.8,
            "onset_variance": 0.8,
            "perc_harmonic_ratio": 0.28,
            "low_freq_ratio": 47.0,
            "flux_mean": 2.1,
            "chroma_var": 0.03,
            "beat_strength": 2.2,
        }
        raw = {
            "bar_energies": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6],
            "n_bars": 8,
            "tempo": 128.0,
            "vocal_ratio": 0.04,
            "vocal_temporal_bonus": 0.0,
            "onset_rate": 3.0,
        }
        audio_features = {
            "valence": 0.92,
            "energy": 0.55,
            "instrumentalness": 0.25,
            "liveness": 0.10,
            "acousticness": 0.03,
        }
        expected = derive_all(dsp, raw, audio_features=audio_features)
        stale_vibe = "DRK" if expected["vibe"] != "DRK" else "MEL"

        ucache.put_track("track.wav", cache_dur, "dsp", dsp)
        ucache.put_track("track.wav", cache_dur, "raw_analysis", raw)
        # Hydrate with current metadata but WITHOUT audio_features — the sig mismatch
        # (stored="" vs expected=non-empty from Songstats) triggers re-derivation.
        stale_entry = hydrate_tagger_result({
            "camelot": "8A",
            "key": "A minor",
            "key_confidence": 0.92,
            "energy": 3,
            "vibe": stale_vibe,
            "vocal": "INST",
            "structure": "32H",
            "bpm": 128.0,
            "vibe_scores": {stale_vibe: 0.9},
            "confidences": {"energy": 0.9, "vibe": 0.7, "vocal": 0.8},
        })
        ucache.put_track("track.wav", cache_dur, "tagger", stale_entry)
        ucache.put("isrc:ISRC123|songstats", audio_features)
        ucache.save()

        self.config.output_dir = str(tmp_path / "registry")
        store = CsvStore(self.config.output_dir)
        store.save_tracks([
            LogicalTrack(track_id="T-001", primary_file_id="F-001", isrc_canonical="ISRC123"),
        ])
        store.save_files([
            FileRecord(
                file_id="F-001",
                track_id="T-001",
                path_abs=wav_path,
                file_name="track.wav",
                is_primary_file=True,
                audio_duration_sec=duration,
            ),
        ])
        store.save_observations([])

        stats = run_analysis(self.config, store, no_essentia=True)

        assert stats["total"] == 1
        assert stats["cached"] == 1
        assert stats["analyzed"] == 0

        track = store.load_tracks()[0]
        assert track.tagger_vibe == expected["vibe"]

        obs = store.load_observations()
        assert len(obs) == 1
        assert obs[0].key_camelot == "8A"

        refreshed = ucache.get_track("track.wav", cache_dur, "tagger")
        assert refreshed is not None
        assert refreshed["vibe"] == expected["vibe"]
        assert refreshed["_tagger_audio_features_sig"]

    def test_identity_keyed_raw_layers_are_rederived_without_audio(self, tmp_path, monkeypatch):
        """Registry analysis should reuse raw layers despite stale version stamps."""
        wav_path = str(tmp_path / "track.wav")
        _make_wav(wav_path)
        duration = 2.0

        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)
        cache_dur = quick_duration(wav_path) or duration
        ucache = get_cache(os.path.join("cache", "raw_cache.pkl"))

        dsp = {name: 0.2 for name in DSP_CURATED_NAMES}
        raw = {
            "bar_energies": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6],
            "n_bars": 8,
            "tempo": 126.0,
            "vocal_ratio": 0.04,
            "vocal_temporal_bonus": 0.0,
            "onset_rate": 3.0,
        }
        ucache.put(ucache.track_key("track.wav", cache_dur, "dsp"), dsp, version="old-raw-version")
        ucache.put(ucache.track_key("track.wav", cache_dur, "raw_analysis"), raw, version="old-raw-version")

        stale_tagger = hydrate_tagger_result({
            "camelot": "9A",
            "key": "A minor",
            "key_confidence": 0.91,
            "energy": 2,
            "vibe": "DRK",
            "vocal": "INST",
            "structure": "16H",
            "bpm": 126.0,
            "confidences": {"key": 0.91, "vibe": 0.7, "vocal": 0.8},
        })
        stale_tagger["_tagger_raw_sig"] = "old-raw-signature"
        ucache.put_track("track.wav", cache_dur, "tagger", stale_tagger)
        ucache.save()

        def fail_analysis(*args, **kwargs):
            raise AssertionError("audio analysis should not run")

        monkeypatch.setattr("dj_registry.adapters.local_analysis._analyze_full", fail_analysis)

        self.config.output_dir = str(tmp_path / "registry")
        store = CsvStore(self.config.output_dir)
        store.save_tracks([
            LogicalTrack(track_id="T-001", primary_file_id="F-001"),
        ])
        store.save_files([
            FileRecord(
                file_id="F-001",
                track_id="T-001",
                path_abs=wav_path,
                file_name="track.wav",
                is_primary_file=True,
                audio_duration_sec=duration,
            ),
        ])
        store.save_observations([])

        stats = run_analysis(self.config, store, no_essentia=True)

        assert stats["total"] == 1
        assert stats["cached"] == 1
        assert stats["analyzed"] == 0
        assert ucache.get_track("track.wav", cache_dur, "dsp") == dsp
        assert ucache.get_track("track.wav", cache_dur, "raw_analysis") == raw

        refreshed = ucache.get_track("track.wav", cache_dur, "tagger")
        expected = derive_all(dsp, raw)
        assert refreshed is not None
        assert refreshed["vibe"] == expected["vibe"]
        assert refreshed["_tagger_raw_sig"] != "old-raw-signature"
