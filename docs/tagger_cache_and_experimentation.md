# Tagger Cache and Experimentation Workflow

This document is for tuning `energy`, `vibe`/mood, `vocal`, `structure`, `bpm`, or key logic without accidentally reading stale cached results.

`vibe` is the historical field name. Its values are now taxonomy mood codes from
`src/dj_registry/taxonomy/dj_taxonomy.json`, loaded through
`src/dj_tagger/moods.py`.

`vocal` is likewise a taxonomy vocal-profile code loaded from the same
dictionary through `src/dj_tagger/vocals.py`.

## Goal

You should be able to:

- change `settings.toml`
- change derived scorer logic
- keep raw data collection stable unless you explicitly force recollection
- rerun the pipeline
- trust that exported derived `tagger_*` values reflect current derived code and settings

That is the current design of the tagger cache system.

## Current Cache Model

Shared cache files (under `<library>/cache/` for a real library root, `./cache/`
for the default `./files` root):

```text
cache/raw_cache.pkl
cache/derived_cache.pkl
```

### Raw Layers

Relevant raw layers:

- `dsp`
- `raw_analysis`
- `section_dsp`

These are reusable inputs for re-deriving tagger outputs without decoding audio again.
`dsp` is the stable audio-extraction boundary. Raw/data-collection entries are
keyed by full filename + rounded duration + layer. API entries are keyed by ISRC
+ layer. Tag formatting, category labels, grouping code, derived scorer changes,
and signature/provenance changes do not force DSP, librosa, tag, Spotify, or
Songstats recollection when a matching identity is already cached.

### Derived Layer

Relevant derived layer:

- `tagger`

This holds the full hydrated tagger result used by registry and grouper.

## What Refreshes Automatically

### 1. `settings.toml` changes

If you change values in the derived scoring sections, the derived signature changes automatically.

Effect:

- existing raw layers stay usable
- tagger entries are re-derived from raw cache
- no manual version bump is needed

### 2. Derived code changes

If you change code used by derived scoring, the derived signature changes automatically.

Current coverage is driven by `_DERIVED_VERSION_FILES` in `src/dj_tagger/settings.py`.

Effect:

- same as a settings change
- raw layers reused
- derived tagger output refreshed

### 3. Raw feature extraction changes

If you change code used to produce `dsp`, `raw_analysis`, or `section_dsp`, the
raw provenance signature changes on newly hydrated tagger records.

Current coverage is driven by `_DSP_VERSION_FILES`, `_SECTION_DSP_VERSION_FILES`,
and `_RAW_ANALYSIS_VERSION_FILES` in `src/dj_tagger/settings.py`.

Effect:

- existing raw/data-collection entries remain cache hits by filename + duration
- fresh extraction runs only for missing identities or explicit force/clear
- derived tagger output is rebuilt or restamped from cached raw payloads

### 4. Key-analysis changes

If key-analysis code or key settings change, the key signature changes automatically.

Current coverage is driven by `_KEY_VERSION_FILES` in `src/dj_tagger/settings.py`.

Effect:

- derived tagger entries are refreshed when needed
- collected key/librosa facts are not recollected solely because the signature changed

### 5. Songstats audio-feature changes

If the Songstats-derived audio features that influence vibe scoring change, the tagger audio-feature signature changes.

Effect:

- stale tagger entries are refreshed instead of silently reused

## What Is Stored on Each Tagger Result

Hydrated tagger records carry:

- `_tagger_version`
- `_tagger_raw_sig`
- `_tagger_derived_sig`
- `_tagger_key_sig`
- `_tagger_audio_features_sig`

The registry exports the same information onto each logical track as:

- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

That makes `registry_overview.csv` the easiest place to verify a whole run was produced by one consistent tagger build.

## Safe Workflows

### A. Tuning thresholds in `settings.toml`

Example: adjust vibe, vocal, energy, or structure settings.

Recommended run:

```bash
dj run --no-grouping --no-tags
```

Why:

- registry analysis refreshes derived tagger outputs automatically
- no file writes
- faster review loop than full grouping

Then inspect:

- `outputs/registry/registry_overview.csv`
- `dj vibe-audit` for vibe-specific checks

### B. Editing derived scorer code

Example files:

- `src/dj_tagger/derive.py`
- `src/dj_tagger/vibe_scoring.py`

Recommended run:

```bash
dj run --no-grouping --no-tags
```

Expected behavior:

- raw cache reused
- derived tagger output recomputed

### C. Editing raw extraction code

Example files:

- `src/dj_tagger/raw_features.py`
- `src/dj_tagger/audio.py`
- `src/dj_grouper/features/dsp.py`

Recommended run:

```bash
dj run --no-grouping --no-tags
```

Expected behavior:

- existing raw feature layers remain cache hits by filename + duration
- fresh extraction runs only for missing identities or when you force/clear cache
- derived tagger output rebuilt/restamped from cached raw layers

### D. Grouping-only experiments

If you only change grouping logic or scoring:

```bash
dj-grouper --dry-run
```

or

```bash
dj run --no-tags
```

You do not need to clear tagger cache for grouping-only changes unless you also changed tagger-derived inputs.

## Verification Checklist

After a tuning run:

1. open `outputs/registry/registry_overview.csv`
2. confirm the relevant `tagger_*` values changed where expected
3. confirm the signature columns are populated
4. check that the signature values are consistent across the affected run
5. for vibe work, run `dj vibe-audit`

Useful signals from `dj vibe-audit`:

- current / stale / missing tagger cache counts
- canonical label distribution from the registry
- drift count between registry vibe and fresh `derive_vibe()` output

## When You Still Need Manual Attention

### New Python files that affect tagger computation

If you create a brand-new module and the tagger starts depending on it, add that
file to one of the signature lists in `src/dj_tagger/settings.py` so derived
outputs and provenance metadata can refresh correctly:

- `_DSP_VERSION_FILES`
- `_SECTION_DSP_VERSION_FILES`
- `_RAW_ANALYSIS_VERSION_FILES`
- `_DERIVED_VERSION_FILES`
- `_KEY_VERSION_FILES`

`_RAW_VERSION_FILES` is kept as aggregate tagger raw provenance metadata. Raw
cache hits still use filename + duration identity, not these signatures.

Examples:

- new helper for raw vocal metrics or vocal-profile scoring
- new shared scorer module for energy or structure
- new key-profile or key-postprocessing helper

If you do not add the new file, changing it later may not refresh derived tagger
metadata.

### External data shape changes

If you change how Songstats data is mapped into tagger inputs, verify that the resulting `_tagger_audio_features_sig` changes as expected.

### Optional Manual Resets

Manual cache clearing is no longer the normal workflow, but it still exists for debugging.

Examples:

```bash
dj-tagger --clear-cache
dj-grouper --force-extract
```

Use those when:

- you suspect cache corruption
- you want a completely cold run for benchmarking
- you intentionally want to recollect raw/data-collection layers

## Practical Recommendation

For most parameter and logic tuning work, use this loop:

```bash
dj run --no-grouping --no-tags
dj vibe-audit
```

Then review:

- `outputs/registry/registry_overview.csv`
- the tagger signature columns
- the drift output from `dj vibe-audit`

That is the current safest way to iterate without worrying about stale derived values.
