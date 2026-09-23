"""Cache location defaults and cache-directory merging."""

import os

from dj_tagger import DEFAULT_LIBRARY_DIR
from dj_tagger.universal_cache import (
    DERIVED_VERSIONS,
    CacheEntry,
    _read_entries,
    _write_entries,
    cache_dir_for,
    merge_cache_dirs,
)


def test_cache_dir_for_lives_beside_the_library():
    assert cache_dir_for(r"D:\Music") == os.path.join(r"D:\Music", "cache")
    assert cache_dir_for("files") == "cache"
    assert cache_dir_for("./files") == "cache"


def test_registry_config_cache_root_follows_library_root(tmp_path):
    from dj_registry.config import RegistryConfig

    assert RegistryConfig().cache_root == "cache"
    cfg = RegistryConfig(library_roots=[str(tmp_path)])
    assert cfg.raw_cache_path == str(tmp_path / "cache" / "raw_cache.pkl")
    assert RegistryConfig(library_roots=[str(tmp_path)], cache_dir="X").cache_root == "X"


def test_cli_defaults_point_at_the_music_library():
    from dj_grouper.cli import _DEFAULTS
    from dj_tagger.cli import _build_parser as tagger_parser
    from dj_tools.cli import _build_parser as dj_parser

    p = dj_parser()
    assert p.parse_args(["run"]).paths == [DEFAULT_LIBRARY_DIR]
    assert p.parse_args(["vibe-audit"]).path == DEFAULT_LIBRARY_DIR
    assert p.parse_args(["export-rekordbox"]).paths == [DEFAULT_LIBRARY_DIR]
    assert p.parse_args(["fetch-missing"]).library == DEFAULT_LIBRARY_DIR
    merge = p.parse_args(["merge-cache", "old_cache"])
    assert merge.source == "old_cache"
    assert merge.into == cache_dir_for(DEFAULT_LIBRARY_DIR)
    assert _DEFAULTS.input_dir == DEFAULT_LIBRARY_DIR
    assert tagger_parser().parse_args([]).paths == [DEFAULT_LIBRARY_DIR]


def _entry(data, mtime=1.0, version="1"):
    return CacheEntry(version=version, mtime=mtime, data=data)


def _data(path):
    return {k: e.data for k, e in _read_entries(str(path)).items()}


def test_merge_cache_dirs_unions_and_resolves_conflicts(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    current = DERIVED_VERSIONS["tagger"]
    _write_entries(str(src / "raw_cache.pkl"), {
        "a.aiff|300.0|dsp": _entry("src-a", mtime=5.0),  # newer than dst -> replaces
        "b.aiff|300.0|dsp": _entry("src-b", mtime=1.0),  # older -> dst kept
        "c.aiff|300.0|dsp": _entry("src-c", mtime=1.0),  # tie -> dst kept
        "isrc:X|spotify": _entry("src-x", mtime=0.0),    # only in src -> added
    })
    _write_entries(str(dst / "raw_cache.pkl"), {
        "a.aiff|300.0|dsp": _entry("dst-a", mtime=2.0),
        "b.aiff|300.0|dsp": _entry("dst-b", mtime=2.0),
        "c.aiff|300.0|dsp": _entry("dst-c", mtime=1.0),
        "d.aiff|300.0|dsp": _entry("dst-d"),
    })
    _write_entries(str(src / "derived_cache.pkl"), {
        "a.aiff|300.0|tagger": _entry("src-a", mtime=9.0, version="stale"),  # dst current -> kept
        "b.aiff|300.0|tagger": _entry("src-b", mtime=0.0, version=current),  # src current -> replaces
    })
    _write_entries(str(dst / "derived_cache.pkl"), {
        "a.aiff|300.0|tagger": _entry("dst-a", mtime=1.0, version=current),
        "b.aiff|300.0|tagger": _entry("dst-b", mtime=5.0, version="stale"),
    })

    stats = merge_cache_dirs(str(src), str(dst))

    assert _data(dst / "raw_cache.pkl") == {
        "a.aiff|300.0|dsp": "src-a", "b.aiff|300.0|dsp": "dst-b", "c.aiff|300.0|dsp": "dst-c",
        "d.aiff|300.0|dsp": "dst-d", "isrc:X|spotify": "src-x",
    }
    assert _data(dst / "derived_cache.pkl") == {
        "a.aiff|300.0|tagger": "dst-a", "b.aiff|300.0|tagger": "src-b",
    }
    assert stats == {
        "raw_cache.pkl": {"added": 1, "replaced": 1, "total": 5},
        "derived_cache.pkl": {"added": 0, "replaced": 1, "total": 2},
    }
    assert _data(dst / "raw_cache.pkl.bak")["a.aiff|300.0|dsp"] == "dst-a"
    assert (dst / "derived_cache.pkl.bak").exists()
    assert _data(src / "raw_cache.pkl")["a.aiff|300.0|dsp"] == "src-a"  # source untouched


def test_merge_cache_dirs_into_empty_destination(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_entries(str(src / "raw_cache.pkl"), {"a.aiff|300.0|dsp": _entry("a")})

    stats = merge_cache_dirs(str(src), str(dst))

    assert stats["raw_cache.pkl"] == {"added": 1, "replaced": 0, "total": 1}
    assert stats["derived_cache.pkl"] == {"added": 0, "replaced": 0, "total": 0}
    assert (dst / "raw_cache.pkl").exists() and not (dst / "raw_cache.pkl.bak").exists()
    assert not (dst / "derived_cache.pkl").exists()
