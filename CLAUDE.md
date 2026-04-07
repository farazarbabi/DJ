# CLAUDE.md

Project-specific guidance for AI-assisted development.

## Project overview

DJ music library toolkit: `dj-tagger` (per-track audio analysis) + `dj-grouper` (grouping and recommendations). Python 3.10+, Windows-first, Rekordbox-compatible.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests (80 tests, ~12s)
pytest tests/ -v

# Tag tracks
dj-tagger files -r --write-tags

# Group and recommend (single command)
dj-grouper run -r --write-tags

# Force re-extract (after changing analyzers or adding tracks)
dj-grouper run -r --force-extract --write-tags
```

## Architecture

```
src/dj_tagger/     per-track audio analysis (librosa + mutagen)
src/dj_grouper/    library-level grouping (scipy clustering + custom scoring)
```

dj-grouper depends on dj-tagger. Both are CLI entry points from one pyproject.toml.

## Key design decisions

- **Input**: `./files/` — **Output**: `./outputs/` — nothing generated in project root
- **Tagger cache**: `outputs/tagger_cache.pkl` — per-track analysis results keyed by filename+mtime. Skips audio loading and all analyzers for unchanged files. Flags: `--no-cache` (force re-analysis), `--clear-cache` (delete and re-analyze). Version-gated by `ANALYZER_VERSION` in `cache.py`.
- **Grouper cache**: `outputs/features_cache.pkl` — incremental per-file DSP extraction keyed by path+mtime. Only audio extraction is cached; clustering and recommendations always recompute.
- **Tag format**: `KEY_ENERGY_VIBE_STRUCT_VOC_BPM` (e.g. `9A_E3_HYPN_64H_NV_126`) — backward-compatible parser handles legacy v1 (pipe-separated, no BPM) and v2 (pipe-separated with BPM)
- **Metadata**: written to COMMENT field via mutagen — both generic (for DJ software) and tagged (desc="DJTAGGER" for self-detection)
- **Group folders**: hard-linked files on NTFS (zero extra space), fall back to copy
- **BPM detection**: reads native TBPM from file metadata (Rekordbox/DJ software) before falling back to librosa beat tracking
- **Clustering**: agglomerative with average linkage, soft vocal partitioning (only split when vocal confidence > 0.5), post-clustering BPM validation (force-split groups with >6% BPM spread), post-clustering energy validation (force-split groups with >3 energy levels spread), target group size (4, 12)
- **Recommendations**: directional scoring (A→B ≠ B→A), DJ usability features, soft BPM penalty (4-8%), no recommendation modes — single similarity ranking
- **Vibe**: internally continuous scores (8 floats), exported as argmax label. Used as continuous in distance computation.
- **CLAP**: on by default when installed (opt-out with `--no-clap`). The old `--use-clap` flag is removed.
- **DSP scaling**: percentile-rank normalized to [0,1] across the library (not z-score). Distance: L2-normalized Euclidean (not cosine). Contrast stretching maps 2nd-98th percentile to [0,1] after blending.
- **Confidence**: every analyzer reports confidence (0-1). Low-confidence tags are softened toward neutral in feature encoding.

## Calibration-sensitive code

These files contain tunable thresholds that directly affect output quality. Changes require re-tagging and re-extracting.

- `src/dj_tagger/constants.py` — energy normalization ranges, structure detection, vocal detection thresholds
- `src/dj_tagger/analyzers/vibe.py` — vibe scoring formulas (inline weights, not in constants)
- `src/dj_grouper/config.py` — layer weights (0.45/0.55 without CLAP, 0.25/0.30/0.45 with CLAP), tag sub-weights (energy d^1.5), BPM filtering, clustering params, group size targets
- `src/dj_grouper/grouping/distance.py` — tag distance sub-weights (inline), contrast stretching (2nd-98th percentile), DSP L2-normalized Euclidean

## Common calibration issues

- **All tracks same energy**: normalization ranges in constants.py are too wide for the music genre. Tighten (min, range) values.
- **Vibe all MEL**: chroma_strength is high for any produced music. MEL needs stricter melodic movement (chroma_var) threshold.
- **HYPN missing trance**: HYPN must not penalize loud tracks. Trance is hypnotic AND loud.
- **Bad groupings by BPM**: check bpm_group_max_spread_pct, BPM tag weight in distance.py, and BPM normalization range in builder.py.
- **Groups too large/homogeneous**: lower target_group_size, switch to complete linkage, or increase tag layer weight.
- **Groups too fragmented**: raise target_group_size, lower BPM spread threshold, increase DSP weight.

## Testing

- Tests use synthetic audio (numpy-generated WAVs via soundfile) — no real music files needed
- `conftest.py` provides fixtures: sine waves, noise, silence, chords, etc.
- Tag format tests cover current v3 (underscore-separated) and legacy v1/v2 (pipe-separated) parsing
- Distance/scoring tests use `_make_track()` helper that builds TrackFeatures from parameters
- TrackFeatures DSP vector is 21 dimensions (10 curated spectral/rhythmic + 5 MFCCs + 6 tonnetz)

## File conventions

- Type hints throughout
- Structured logging (logger per module)
- Dataclasses for results and config
- Constants centralized where possible, inline where formula-specific
- No emojis in code or output
- Git: runtime data (outputs/, files/, *.pkl, *.csv, *.m3u8) in .gitignore

## When changing analyzers

1. Update the analyzer code
2. Bump `ANALYZER_VERSION` in `src/dj_tagger/cache.py` (invalidates tagger cache)
3. Run `pytest tests/ -v` to verify
4. Re-tag: `dj-tagger files -r --write-tags --overwrite`
5. Re-extract: `dj-grouper run -r --force-extract --dry-run` to preview
6. Check grouping quality before writing: look at group folder names and sizes

## When changing distance/scoring

1. Update distance.py or scoring.py
2. Run tests
3. No need to re-extract — just re-run: `dj-grouper run -r --dry-run`
4. Cache is reused; only clustering/recommendations recompute
