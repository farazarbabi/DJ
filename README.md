# DJ Tools

Local Python toolkit for DJ library analysis, registry management, and grouping.

Current command surface:

- `dj`: unified pipeline
- `dj-tagger`: per-track analysis and tag writing
- `dj-registry`: metadata registry and canonical key/BPM resolution
- `dj-grouper`: grouping and recommendations

The project is Windows-first and built around a local Rekordbox workflow.

## Current Tag Format

The current COMMENT tag format is:

```text
KEY_ENERGY_VIBE_STRUCTURE_VOCAL_BPM[_GID]
```

Examples:

```text
9A_E3_HYPN_64H_NV_126
8A_E4_DRK_32D_V_130_G017
```

Legacy pipe-separated tags are still parsed, but newly written tags use the underscore format.

## Installation

```bash
pip install -e ".[dev,registry]"
```

Optional extras:

```bash
pip install -e ".[essentia]"   # optional key-analysis backend
pip install -e ".[clap]"       # optional CLAP embeddings for grouping
```

Python `>=3.10`.

## Directory Layout

```text
DJ/
  files/                    default library input
  cache/
    raw_cache.pkl           shared raw cache
    derived_cache.pkl       shared derived cache
  outputs/
    registry/
      registry_overview.csv
      tracks_master.csv
      files_master.csv
      source_observations.csv
      review_queue.csv
      source_payload_index.csv
      raw/
    groups.csv
    recommendations.csv
    feedback.csv
    Grouped/
    playlists/
  settings.toml             tunable derived-scoring parameters
  src/
  tests/
```

## Quick Start

Run the full local pipeline:

```bash
dj run
```

Useful variants:

```bash
dj run "E:\\Music" -w 4
dj run --no-grouping
dj run --no-tags
dj run --no-songstats
dj run --force-extract
dj vibe-audit
```

Lower-level commands:

```bash
dj-tagger --write-tags
dj-tagger --write-tags --no-registry

dj-registry run --write-tags --songstats
dj-registry analyze -w 4
dj-registry export

dj-grouper --dry-run
dj-grouper --force-extract
```

## Components

### `dj`

The unified pipeline orchestrates:

1. registry scan and file linking
2. Rekordbox and Songstats ingest
3. full tagger analysis
4. canonical key/BPM resolution
5. tag writing
6. grouping and recommendation generation

### `dj-tagger`

Analyzes a track and produces:

- `energy`
- `key` / `camelot`
- `bpm`
- `structure`
- `vibe`
- `vocal`
- `vibe_scores`
- `confidences`

The canonical compute path is:

```text
audio -> TrackAudio -> compute_tagger_artifacts()
      -> raw layers (dsp, raw_analysis, section_dsp)
      -> derive_all() for tunable fields
      -> key analysis
      -> hydrated tagger result
```

### `dj-registry`

The registry is the source of truth for:

- canonical key
- canonical BPM
- stored tagger outputs on `LogicalTrack`
- exported review and audit reports

`registry_overview.csv` is the main review file. It includes source columns, `tagger_*` columns, and provenance fields such as:

- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

### `dj-grouper`

The grouper now reads the same shared cache used by tagger and registry. It does not maintain a separate feature-analysis cache anymore.

Current grouping pipeline:

1. load tagger results from shared cache when current
2. fall back to raw cache and re-derive when only derived logic changed
3. extract missing DSP/raw artifacts through the canonical tagger pipeline
4. build tag, DSP, optional CLAP, and optional registry-enrichment feature layers
5. cluster with either:
   - `constrained` (default)
   - `agglomerative`
6. write `groups.csv`, `recommendations.csv`, playlists, and optional grouped folders

## Cache Model

There are two shared cache files under `./cache/`.

### `cache/raw_cache.pkl`

Stores reusable raw artifacts and external data:

- `dsp`
- `section_dsp`
- `raw_analysis`
- `clap`
- `tag`
- `rekordbox`
- `songstats`
- `spotify`
- analysis observations

Important: raw layers are no longer treated as permanently valid. The tagger computes a raw extractor signature from the relevant source files, and versioned raw layers miss automatically if that logic changes.

### `cache/derived_cache.pkl`

Stores derived tagger results:

- `tagger`

Each tagger record carries metadata describing the build used to compute it:

- tagger version
- raw signature
- derived signature
- key signature
- Songstats audio-feature signature

## Safe Experimentation

If you change `settings.toml` or derived scoring logic, the derived signature changes automatically and tagger results are re-derived from cached raw layers on the next run.

If you change raw extraction logic, the raw signature changes automatically and those raw layers are recomputed instead of being silently reused.

You do not need to manually bump a cache version string anymore.

The main exception is when you add a brand-new Python module that affects tagger computation. In that case, add the new file to the relevant signature list in `src/dj_tagger/settings.py`:

- `_RAW_VERSION_FILES`
- `_DERIVED_VERSION_FILES`
- `_KEY_VERSION_FILES`

For a focused workflow reference, see [docs/tagger_cache_and_experimentation.md](docs/tagger_cache_and_experimentation.md).

## Current Grouping Constraints

The current implementation is not yet the redesigned grouping system discussed separately. As built today:

- key constraints are tiered by Camelot distance
- same key and direct neighbors are fully allowed
- Camelot distance `2` is treated as a relaxed fallback in constrained clustering
- BPM spread beyond `bpm_group_max_spread_pct` is a hard cannot-link
- agglomerative mode also applies post-clustering key, BPM, and energy validation splits

See [docs/dj_grouping_recommendation_system_spec.md](docs/dj_grouping_recommendation_system_spec.md) for the current as-built grouping behavior.

## Useful Outputs

- `outputs/registry/registry_overview.csv`: main audit and review sheet
- `outputs/groups.csv`: grouped tracks
- `outputs/recommendations.csv`: directional recommendations
- `outputs/Grouped/`: optional grouped folders
- `outputs/playlists/`: generated playlists

## Testing

```bash
pytest -q
```

Current suite size: `245` tests.

## Documentation

- [docs/technical_spec.md](docs/technical_spec.md): current as-built technical architecture
- [docs/dj_grouping_recommendation_system_spec.md](docs/dj_grouping_recommendation_system_spec.md): current grouping and recommendation behavior
- [docs/tagger_cache_and_experimentation.md](docs/tagger_cache_and_experimentation.md): cache-safe tuning workflow
- [specs/track_registery.md](specs/track_registery.md): current registry reference
