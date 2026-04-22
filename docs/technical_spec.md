# Technical Specification

This document describes the current as-built system.

## Package Layout

```text
src/dj_tools/      unified CLI (`dj`)
src/dj_tagger/     per-track audio analysis and tag formatting/writing
src/dj_registry/   registry, enrichment, canonical resolution, exports
src/dj_grouper/    grouping and recommendation generation
```

All four entry points are installed from one `pyproject.toml`.

## Top-Level Data Flow

```text
audio files
  -> dj_registry scan/link/ingest
  -> dj_tagger canonical analysis
  -> registry resolution and export
  -> dj_grouper feature build / grouping / recommendation
```

The registry is the source of truth for canonical key/BPM and for the stored `tagger_*` values used in reporting.

## Canonical Tagger Pipeline

The current tagger path is intentionally shared across `dj-tagger`, `dj-registry`, and `dj-grouper`.

### Core Steps

```text
load_audio_features()
  -> compute_tagger_artifacts()
     -> extract_dsp_features()
     -> extract_raw_analysis()
     -> analyze_sections()
     -> extract_section_dsp()
     -> derive_all()
     -> analyze_key()
  -> hydrate_tagger_result()
```

### Raw Layers

Produced by `src/dj_tagger/raw_features.py`:

- `dsp`
- `raw_analysis`
- `section_dsp`

These are the reusable inputs for cheap re-derivation after scoring changes.

### Derived Fields

Produced by `src/dj_tagger/derive.py`:

- `energy`
- `bpm`
- `vibe`
- `vocal`
- `structure`
- `vibe_scores`
- `confidences`

Key analysis is computed separately and merged into the final hydrated tagger result.

## Cache Architecture

Shared cache lives in `./cache/`.

### Files

```text
cache/raw_cache.pkl
cache/derived_cache.pkl
```

### `raw_cache.pkl`

Contains raw or external layers:

- `dsp`
- `section_dsp`
- `raw_analysis`
- `clap`
- `tag`
- `rekordbox`
- `songstats`
- `spotify`
- `analysis_librosa`
- `analysis_essentia`

`dsp`, `section_dsp`, and `raw_analysis` are versioned raw layers. They are invalidated by the raw extractor signature, not reused forever.

### `derived_cache.pkl`

Contains:

- `tagger`

The `tagger` layer is version-gated by the current tagger signature.

## Automatic Signatures

`src/dj_tagger/settings.py` computes signatures from both settings content and relevant source files.

Current signature types:

- `raw_version()`
- `derived_version()`
- `key_version()`
- `tagger_version()`

Each hydrated tagger result stores:

- `_tagger_version`
- `_tagger_raw_sig`
- `_tagger_derived_sig`
- `_tagger_key_sig`
- `_tagger_audio_features_sig`

This is the main protection against stale cache reuse during tuning.

## Invalidation Rules

### If settings or derived scoring logic changes

- derived signature changes
- existing raw layers remain usable
- tagger results are re-derived from cached raw data

### If raw extraction logic changes

- raw signature changes
- cached `dsp`, `raw_analysis`, and `section_dsp` miss
- fresh extraction is required

### If key logic changes

- key signature changes
- tagger records are not accepted as current until key is recomputed

### If Songstats audio-feature inputs change

- `_tagger_audio_features_sig` changes
- tagger entries are refreshed instead of silently reused

## Signature Coverage

The signature lists live in `src/dj_tagger/settings.py`:

- `_RAW_VERSION_FILES`
- `_DERIVED_VERSION_FILES`
- `_KEY_VERSION_FILES`

If a new Python file becomes part of tagger computation, it must be added to the appropriate list so cache invalidation sees it.

## Registry Architecture

The registry writes CSV-backed state under `outputs/registry/`.

Primary files:

- `tracks_master.csv`
- `files_master.csv`
- `source_observations.csv`
- `review_queue.csv`
- `source_payload_index.csv`
- `registry_overview.csv`

### `LogicalTrack`

Current track-level stored tagger fields:

- `tagger_energy`
- `tagger_vibe`
- `tagger_vocal`
- `tagger_structure`
- `tagger_bpm`
- `tagger_vibe_scores`
- `tagger_confidences`
- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

### Analysis Path

`src/dj_registry/adapters/local_analysis.py`:

1. looks for a current hydrated tagger entry
2. if needed, re-derives from cached raw layers
3. if needed, runs full canonical analysis
4. writes raw layers and hydrated tagger results back to shared cache
5. updates `LogicalTrack`
6. emits `analysis_librosa` observations

This is the same truth later exported by `registry_overview.csv`.

## Unified CLI

`dj run` orchestrates:

1. scan files
2. link files to logical tracks
3. ingest Rekordbox
4. enrich ISRCs and ingest Songstats
5. run local analysis
6. resolve canonical key and BPM
7. sync tags
8. export registry reports
9. run grouping unless skipped

`dj vibe-audit` compares registry-stored vibe labels against fresh `derive_vibe()` output from current DSP plus current Songstats audio features.

## Grouper Architecture

The grouper now uses the same canonical cache pipeline as tagger and registry.

### Extraction

`src/dj_grouper/cli.py`:

- loads current tagger entries when available
- re-derives from raw cache when only derived logic changed
- falls back to canonical artifact extraction when needed
- preserves richer Songstats-aware tagger entries instead of replacing them with DSP-only output

### Feature Layers

- tag vector from parsed tagger fields
- DSP vector
- optional CLAP embedding
- optional registry enrichment

### Grouping Modes

- `constrained` is the default
- `agglomerative` remains available

### Current Constraint Behavior

- key distance `<= 1`: fully allowed
- key distance `== 2`: relaxed cannot-link in constrained mode
- key distance `> 2`: hard cannot-link
- BPM spread above `bpm_group_max_spread_pct`: hard cannot-link
- agglomerative mode also runs post-clustering key/BPM/energy validation splits

## Tag Writing

The registry sync path builds the COMMENT tag from:

- canonical key
- canonical BPM when available, otherwise tagger BPM
- tagger energy
- tagger vibe
- tagger structure
- tagger vocal

It also writes canonical key to the dedicated key field for supported formats.

## Observability

Main review and audit surfaces:

- `outputs/registry/registry_overview.csv`
- `dj vibe-audit`
- hydrated tagger metadata in cache

Recommended verification loop after tuning:

1. run `dj run --no-grouping --no-tags`
2. inspect `registry_overview.csv`
3. run `dj vibe-audit`
4. confirm the exported signature columns are consistent across the affected rows

## Current Limitations

- registry file scanning defaults to `.mp3`, `.aiff`, `.aif` via `RegistryConfig`
- grouping still allows Camelot distance `2` as a relaxed fallback in constrained mode
- cache keys are still filename-plus-duration scoped, not full-path scoped
- some docs or file names still reflect earlier terminology even though runtime behavior has changed
