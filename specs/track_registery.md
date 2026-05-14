# Track Registry Reference

This file replaces the earlier aspirational registry spec with the current as-built reference.

## Purpose

`dj_registry` maintains the project registry under `outputs/registry/`.

It is the source of truth for:

- canonical key
- canonical BPM
- stored tagger outputs per logical track
- review and audit exports

## Current Pipeline

The registry pipeline is:

```text
scan
  -> link
  -> ingest-rekordbox
  -> enrich-isrcs
  -> ingest-songstats
  -> analyze
  -> resolve
  -> review-queue
  -> sync-tags
  -> export
```

The unified `dj run` command orchestrates the same flow and then optionally runs grouping.

## Commands

Available `dj-registry` commands:

- `scan`
- `link`
- `ingest-rekordbox`
- `enrich-isrcs`
- `ingest-songstats`
- `analyze`
- `resolve`
- `review-queue`
- `import-reviews`
- `sync-tags`
- `export`
- `run`

## Current Source Systems

Registry observations currently come from:

- `tag`
- `rekordbox`
- `songstats`
- `analysis_librosa`
- `analysis_essentia`
- `manual`

## Current Data Files

Written under `outputs/registry/`:

- `tracks_master.csv`
- `files_master.csv`
- `source_observations.csv`
- `review_queue.csv`
- `source_payload_index.csv`
- `registry_overview.csv`

Raw payload area:

```text
outputs/registry/raw/
  analysis/
  file_tag_snapshots/
  rekordbox_xml/
  songstats/
```

## `tracks_master.csv`

One row per logical track.

Important current field groups:

### Canonical identity and resolution

- `artist_canonical`
- `title_canonical`
- `mix_canonical`
- `album_canonical`
- `label_canonical`
- `release_date_canonical`
- `duration_sec_canonical`
- `isrc_canonical`
- `canonical_key_standard`
- `canonical_key_camelot`
- `canonical_key_confidence`
- `canonical_key_source`
- `canonical_key_resolution_reason`
- `canonical_bpm`
- `canonical_bpm_confidence`

### Stored tagger outputs

- `tagger_energy`
- `tagger_vibe`
- `tagger_vocal`
- `tagger_structure`
- `tagger_bpm`
- `tagger_vibe_scores`
- `tagger_vocal_scores`
- `tagger_confidences`

`tagger_vibe` stores taxonomy mood codes loaded from
`src/dj_registry/taxonomy/dj_taxonomy.json`; `tagger_vibe_scores` stores the
full mood-score map.

`tagger_vocal` now stores a taxonomy vocal-profile code loaded from the same
dictionary's `vocal_profiles` values. Current codes are `INST`, `VOC`, `FVOC`,
`SPK`, `CHANT`, `DUB`, and `TOOL`. `tagger_vocal_scores` stores the full
profile-score map.

The full mood and vocal-profile code dictionary is documented in `README.md`.

### Tagger provenance

- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

These provenance fields are the main mechanism for verifying that the exported track values were produced by one consistent tagger build.

## `files_master.csv`

One row per physical file.

Tracks filesystem metadata, embedded tags, and read/write status.

Current registry scanning defaults to the extensions configured in `RegistryConfig`, which today are:

- `.mp3`
- `.aiff`
- `.aif`

## `source_observations.csv`

One row per observation in the current wide observation schema.

It stores source-specific fields such as:

- key
- BPM
- genre
- label
- release date
- tagger-derived fields on analysis observations

The registry rebuilds stale analysis observations from cache on each run instead of treating the observation file as the primary analysis store.

## `registry_overview.csv`

This is the main human review export.

Current exported groups include:

- canonical identity
- key values from tag, Rekordbox, Songstats, and local analysis
- canonical key and BPM
- Songstats audio features
- `tagger_*` values
- tagger provenance/signature columns

The overview is what you should compare when tuning derived analysis logic.

## Current Analysis Behavior

`dj-registry analyze` now runs the full canonical tagger pipeline, not just key analysis.

For each track it:

1. looks for a current hydrated tagger cache record
2. re-derives from cached raw layers when possible
3. reuses raw-layer payloads by filename + duration identity even when signatures changed
4. otherwise runs fresh canonical analysis
5. writes raw layers and hydrated tagger output back to shared cache
6. writes `tagger_*` fields onto `LogicalTrack`
7. emits `analysis_librosa` observations

## Current Cache Interaction

The registry shares the same cache with tagger and grouper:

- `cache/raw_cache.pkl`
- `cache/derived_cache.pkl`

Relevant raw layers:

- `dsp`
- `raw_analysis`
- `section_dsp`
- `songstats`
- `songstats_lookup`
- `spotify`
- `rekordbox`
- `tag`

Relevant derived layer:

- `tagger`

The registry trusts a cached derived tagger result only when:

- tagger version matches
- derived signature matches
- Songstats audio-feature signature matches the current inputs

Raw/data-collection cache entries are different: they are identity-only. Track
data uses full filename + rounded duration + layer; API data uses ISRC + layer.
If that entry exists, downstream tag/category/grouping/signature changes do not
recollect DSP, raw analysis, embedded tags, Spotify/Songstats lookups, or local
librosa/Essentia observations. Fresh collection happens only for missing
identities or explicit force/clear workflows.

## Resolution

Canonical resolution currently covers:

- key
- BPM

Key resolution uses weighted multi-source evidence.

BPM resolution uses available source evidence with registry rules.

Tagger-derived fields are not voted across sources; they are stored directly from the canonical local analysis path.

## Tag Sync

`dj-registry sync-tags` writes only the COMMENT tag:

- COMMENT tag built from: canonical key, canonical BPM, stored tagger energy/mood/vocal, and compact internal DJ taxonomy category label code
- Current shape: `KEY_BPM_ENERGY_VIBE_VOCAL[_CATEGORY][_GID]`
- `CATEGORY` is derived from the selected category label, not written as `category_id`

The TKEY/InitialKey field is intentionally left untouched by default so the
original embedded key (e.g. from Rekordbox analysis or manual tagging) is
preserved as a validation signal for multi-source key resolution. Passing
`--write-key-tag` writes the canonical Camelot key there for supported formats.

By default `sync-tags` is dry-run unless `--write` is passed.

## Review Queue

`review_queue.csv` contains unresolved tracks that need manual intervention after automatic resolution.

This is for canonical resolution issues, not for tagger score calibration.

## Current Known Limits

- registry scanning currently defaults to `.mp3`, `.aiff`, `.aif`
- cache keys are still filename-plus-duration scoped
- the CSV store is authoritative for registry data, but some raw payload artifacts remain auxiliary debug material rather than a normalized datastore
- the file name `track_registery.md` is historical even though the implementation is now the current reference

## Recommended Calibration Workflow

If the goal is tuning analysis fields such as mood/vibe, energy, vocal, or structure:

1. edit `settings.toml` or the relevant scorer code
2. run `dj run --no-grouping --no-tags` or `dj-registry analyze` then `dj-registry export`
3. inspect `outputs/registry/registry_overview.csv`
4. use `dj vibe-audit` for vibe-specific drift checks
5. confirm the tagger signature columns changed where expected

For more detail, see [../docs/tagger_cache_and_experimentation.md](../docs/tagger_cache_and_experimentation.md).
