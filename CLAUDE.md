# CLAUDE.md

Project-specific guidance for AI-assisted development.

## Project overview

DJ music library toolkit with three modules: `dj-tagger` (per-track audio analysis), `dj-grouper` (grouping and recommendations), and `dj-registry` (metadata collection, enrichment, and canonical key resolution). Python 3.10+, Windows-first, Rekordbox-compatible.

## Commands

```bash
# Install
pip install -e ".[dev,registry]"

# Run tests (145 tests)
pytest tests/ -v

# Tag tracks (defaults to ./files)
dj-tagger --write-tags
dj-tagger "E:\Music" --write-tags

# Group and recommend (defaults to ./files)
dj-grouper --write-tags
dj-grouper "E:\Music" --write-tags

# Force re-extract (after changing analyzers or adding tracks)
dj-grouper --force-extract --write-tags

# Registry: collect metadata from all sources and resolve canonical key
dj-registry run ./files --rekordbox-xml ./files/export.xml --songstats --no-essentia -w 4

# Registry: write resolved keys to file tags
dj-registry run ./files --rekordbox-xml ./files/export.xml --songstats --no-essentia -w 4 --write-tags

# Registry: individual steps
dj-registry scan ./files
dj-registry link
dj-registry ingest-rekordbox --xml ./files/export.xml
dj-registry enrich-isrcs
dj-registry ingest-songstats
dj-registry analyze --no-essentia -w 4
dj-registry resolve --force
dj-registry review-queue
dj-registry import-reviews
dj-registry sync-tags --write
dj-registry export
```

## Architecture

```
src/dj_tagger/     per-track audio analysis (librosa + mutagen)
src/dj_grouper/    library-level grouping (scipy clustering + custom scoring)
src/dj_registry/   metadata collection, enrichment, canonical key resolution
```

dj-grouper depends on dj-tagger. dj-registry depends on dj-tagger (reuses key analysis, metadata I/O, file scanning). All three are CLI entry points from one pyproject.toml.

## Cache architecture

All caches live in `./cache/` (project root, survives `rm -rf outputs/`):

- **`cache/tagger_cache.pkl`** — shared between dj-tagger and dj-registry. Per-track analysis results + ISRCs. Keyed by `filename|duration` (path-independent). Version-gated by `ANALYZER_VERSION` in `cache.py`. Falls back to filename-only key for backward compat.
- **`cache/registry_cache.pkl`** — all source observations (tags, Rekordbox, Songstats). Keyed by `filename|duration|source` for file-based data, `isrc:XXX|source` for API-based data. Survives registry resets.
- **`cache/features_cache.pkl`** — dj-grouper DSP features. Keyed by path+mtime.
- **`cache/clap_cache.pkl`** — dj-grouper CLAP embeddings.

## Key design decisions

- **Input**: `./files/` — **Output**: `./outputs/` — **Cache**: `./cache/`
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

## Registry design decisions

- **Single source of truth**: canonical registry, not file tags or any single external source
- **Sources**: file tags, Rekordbox XML, Songstats API (via ISRC), Spotify (for ISRC enrichment), local librosa key analysis
- **Key resolution**: weighted vote share confidence (all agree = 1.0, any disagreement < 1.0). Tie-breaking priority: tag > rekordbox > songstats > analysis. Majority always wins.
- **Source weights** (for ranking): manual=1.00, analysis_essentia=0.85, analysis_librosa=0.80, rekordbox=0.75, tag=0.55, songstats=0.40
- **ISRC enrichment**: Spotify search by artist+title with duration matching. Handles artist-in-title (MP3s), multi-artist, featuring syntax.
- **Tag write-back**: writes canonical key to both TKEY field (for DJ software) and key portion of dj-tagger COMMENT tag
- **CSV backend**: atomic writes via temp file + os.replace(). Observations rebuilt fresh from pkl cache every run.
- **Output**: `outputs/registry/registry_overview.csv` — one row per track, all sources as columns, canonical key, confidence, review status

## Registry output files

- `outputs/registry/registry_overview.csv` — main review file (one row per track, all data)
- `outputs/registry/tracks_master.csv` — canonical track registry
- `outputs/registry/files_master.csv` — file inventory
- `outputs/registry/source_observations.csv` — all observations (wide format, one row per source per track)
- `outputs/registry/review_queue.csv` — unresolved tracks for manual review
- `outputs/registry/raw/` — raw API payloads (Songstats JSON, Rekordbox XML, analysis results, tag snapshots)

## Calibration-sensitive code

These files contain tunable thresholds that directly affect output quality. Changes require re-tagging and re-extracting.

- `src/dj_tagger/constants.py` — energy normalization ranges, structure detection, vocal detection thresholds
- `src/dj_tagger/analyzers/vibe.py` — vibe scoring formulas (inline weights, not in constants)
- `src/dj_grouper/config.py` — layer weights (0.45/0.55 without CLAP, 0.25/0.30/0.45 with CLAP), tag sub-weights (energy d^1.5), BPM filtering, clustering params, group size targets
- `src/dj_grouper/grouping/distance.py` — tag distance sub-weights (inline), contrast stretching (2nd-98th percentile), DSP L2-normalized Euclidean
- `src/dj_registry/config.py` — source weights, confidence threshold (0.70), margin threshold (0.20), agreement/cross-type boosts
- `src/dj_registry/resolver/key_resolver.py` — tie-breaking priority, majority/score/tie-break resolution logic

## Common calibration issues

- **All tracks same energy**: normalization ranges in constants.py are too wide for the music genre. Tighten (min, range) values.
- **Vibe all MEL**: chroma_strength is high for any produced music. MEL needs stricter melodic movement (chroma_var) threshold.
- **HYPN missing trance**: HYPN must not penalize loud tracks. Trance is hypnotic AND loud.
- **Bad groupings by BPM**: check bpm_group_max_spread_pct, BPM tag weight in distance.py, and BPM normalization range in builder.py.
- **Groups too large/homogeneous**: lower target_group_size, switch to complete linkage, or increase tag layer weight.
- **Groups too fragmented**: raise target_group_size, lower BPM spread threshold, increase DSP weight.
- **Registry key conflicts**: adjust source_weights in RegistryConfig. Check registry_overview.csv to compare sources.

## Testing

- 145 tests total: 85 tagger/grouper + 60 registry
- Tests use synthetic audio (numpy-generated WAVs via soundfile) — no real music files needed
- `conftest.py` provides fixtures: sine waves, noise, silence, chords, etc.
- Tag format tests cover current v3 (underscore-separated) and legacy v1/v2 (pipe-separated) parsing
- Distance/scoring tests use `_make_track()` helper that builds TrackFeatures from parameters
- TrackFeatures DSP vector is 21 dimensions (10 curated spectral/rhythmic + 5 MFCCs + 6 tonnetz)
- Registry tests cover key normalization, CSV round-trips, text normalization, resolver scoring

## File conventions

- Type hints throughout
- Structured logging (logger per module)
- Dataclasses for results and config
- Constants centralized where possible, inline where formula-specific
- No emojis in code or output
- Git: runtime data (outputs/, files/, cache/, *.pkl, *.csv, *.m3u8) in .gitignore

## When changing analyzers

1. Update the analyzer code
2. Bump `ANALYZER_VERSION` in `src/dj_tagger/cache.py` (invalidates tagger cache)
3. Run `pytest tests/ -v` to verify
4. Re-tag: `dj-tagger --write-tags --overwrite`
5. Re-extract: `dj-grouper --force-extract --dry-run` to preview
6. Check grouping quality before writing: look at group folder names and sizes

## When changing distance/scoring

1. Update distance.py or scoring.py
2. Run tests
3. No need to re-extract — just re-run: `dj-grouper --dry-run`
4. Cache is reused; only clustering/recommendations recompute

## When changing registry resolution

1. Update resolver/key_resolver.py or config.py source_weights
2. Run `pytest tests/test_registry_resolver.py -v`
3. Re-run: `dj-registry resolve --force` (no re-scanning or API calls needed)
4. Check `outputs/registry/registry_overview.csv` for results
