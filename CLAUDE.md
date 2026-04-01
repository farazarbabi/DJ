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
- **Cache**: `outputs/features_cache.pkl` — incremental per-file keyed by path+mtime. Only audio extraction is cached; clustering and recommendations always recompute.
- **Tag format**: `E# | KEY | BPM | STRUCT | VIBE | VOC [| GID]` — backward-compatible parser handles v1 (no BPM) and v2
- **Metadata**: written to COMMENT field via mutagen — both generic (for DJ software) and tagged (desc="DJTAGGER" for self-detection)
- **Group folders**: hard-linked files on NTFS (zero extra space), fall back to copy
- **Clustering**: agglomerative with average linkage, soft vocal partitioning (only split when vocal confidence > 0.5), post-clustering BPM validation (force-split groups with >6% BPM spread)
- **Recommendations**: directional scoring (A→B ≠ B→A), DJ usability features, soft BPM penalty (4-8%), no recommendation modes — single similarity ranking
- **Vibe**: internally continuous scores (8 floats), exported as argmax label. Used as continuous in distance computation.
- **Confidence**: every analyzer reports confidence (0-1). Low-confidence tags are softened toward neutral in feature encoding.

## Calibration-sensitive code

These files contain tunable thresholds that directly affect output quality. Changes require re-tagging and re-extracting.

- `src/dj_tagger/constants.py` — energy normalization ranges, structure detection, vocal detection thresholds
- `src/dj_tagger/analyzers/vibe.py` — vibe scoring formulas (inline weights, not in constants)
- `src/dj_grouper/config.py` — layer weights, tag sub-weights, BPM filtering, clustering params, group size targets
- `src/dj_grouper/grouping/distance.py` — tag distance sub-weights (inline)

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
- Tag format tests cover v1 legacy and v2 with BPM/GID
- Distance/scoring tests use `_make_track()` helper that builds TrackFeatures from parameters
- TrackFeatures DSP vector is 10 dimensions (curated features, not raw 45)

## File conventions

- Type hints throughout
- Structured logging (logger per module)
- Dataclasses for results and config
- Constants centralized where possible, inline where formula-specific
- No emojis in code or output
- Git: runtime data (outputs/, files/, *.pkl, *.csv, *.m3u8) in .gitignore

## When changing analyzers

1. Update the analyzer code
2. Run `pytest tests/ -v` to verify
3. Re-tag: `dj-tagger files -r --write-tags --overwrite`
4. Re-extract: `dj-grouper run -r --force-extract --dry-run` to preview
5. Check grouping quality before writing: look at group folder names and sizes

## When changing distance/scoring

1. Update distance.py or scoring.py
2. Run tests
3. No need to re-extract — just re-run: `dj-grouper run -r --dry-run`
4. Cache is reused; only clustering/recommendations recompute
