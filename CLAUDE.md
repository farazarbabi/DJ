# CLAUDE.md

Project-specific guidance for AI-assisted development.

## Project Overview

DJ music library toolkit with four CLIs:

- `dj`: unified pipeline
- `dj-tagger`: per-track analysis and tag writing
- `dj-registry`: metadata registry, enrichment, canonical resolution
- `dj-grouper`: grouping and recommendations

The registry is the source of truth for canonical key/BPM and stored `tagger_*` outputs. The unified `dj run` command is the preferred end-to-end entry point.

## Common Commands

```bash
# Install
pip install -e ".[dev,registry]"

# Full test suite
pytest -q

# Unified pipeline
dj run
dj run "E:\\Music" -w 4
dj run --no-grouping
dj run --no-tags
dj run --no-songstats
dj run --force-extract
dj vibe-audit

# Lower-level tools
dj-tagger --write-tags
dj-tagger --write-tags --no-registry

dj-registry analyze -w 4
dj-registry resolve --force
dj-registry export

dj-grouper --dry-run
dj-grouper --force-extract
```

## Current Tag Format

Current COMMENT tag format:

```text
KEY_ENERGY_VIBE_STRUCTURE_VOCAL_BPM[_GID]
```

Example:

```text
9A_E3_HYPN_64H_NV_126
```

Legacy pipe-separated tags are still parsed, but new writes use the underscore format.

## Cache Architecture

Shared cache lives in `./cache/`.

Files:

- `cache/raw_cache.pkl`
- `cache/derived_cache.pkl`

Relevant raw layers:

- `dsp`
- `section_dsp`
- `raw_analysis`
- `clap`
- source payload layers such as `tag`, `songstats`, `rekordbox`

Relevant derived layer:

- `tagger`

Important current behavior:

- raw layers `dsp`, `section_dsp`, and `raw_analysis` are versioned
- tagger results are hydrated with provenance metadata
- registry and grouper reuse the same canonical tagger/raw cache
- grouper preserves richer Songstats-aware tagger entries instead of downgrading them

## Automatic Cache Invalidation

Do not rely on manual version bumps.

`src/dj_tagger/settings.py` computes:

- `raw_version()`
- `derived_version()`
- `key_version()`
- `tagger_version()`

These signatures hash both:

- relevant sections of `settings.toml`
- relevant source files listed in:
  - `_RAW_VERSION_FILES`
  - `_DERIVED_VERSION_FILES`
  - `_KEY_VERSION_FILES`

Hydrated tagger records carry:

- `_tagger_version`
- `_tagger_raw_sig`
- `_tagger_derived_sig`
- `_tagger_key_sig`
- `_tagger_audio_features_sig`

If you add a brand-new Python file that affects tagger computation, add it to the appropriate signature list. Otherwise changing that file later will not invalidate cache.

## Safe Tuning Workflow

For tagger tuning:

```bash
dj run --no-grouping --no-tags
dj vibe-audit
```

Then inspect:

- `outputs/registry/registry_overview.csv`
- tagger signature columns in the overview
- drift output from `dj vibe-audit`

Expected behavior:

- settings or derived scorer changes re-derive from raw cache
- raw extraction changes invalidate raw feature layers automatically
- Songstats input changes refresh affected tagger entries automatically

Manual cache clearing is now mainly for debugging:

```bash
dj-tagger --clear-cache
dj-grouper --force-extract
```

## Key Design Notes

- tagger canonical pipeline runs through `compute_tagger_artifacts()` in `src/dj_tagger/raw_features.py`
- registry analysis stores full `tagger_*` values on `LogicalTrack`
- `registry_overview.csv` exports tagger values plus provenance columns
- `dj vibe-audit` audits the same registry-backed vibe truth written by `dj run`
- current grouping defaults to constrained clustering, not the future redesign

## Current Grouping Behavior

As built today:

- default algorithm: `constrained`
- fallback algorithm: `agglomerative`
- same key and direct Camelot neighbors are allowed
- Camelot distance `2` is currently a relaxed fallback in constrained clustering
- BPM spread above `bpm_group_max_spread_pct` is a hard grouping constraint
- agglomerative mode applies post-clustering key/BPM/energy validation

Do not document the proposed redesign as if it already exists.

## Calibration-Sensitive Files

### Tagger and cache behavior

- `settings.toml`
- `src/dj_tagger/settings.py`
- `src/dj_tagger/raw_features.py`
- `src/dj_tagger/derive.py`
- `src/dj_tagger/vibe_scoring.py`
- `src/dj_tagger/tagger_cache.py`
- `src/dj_tagger/universal_cache.py`

### Grouping and recommendation behavior

- `src/dj_grouper/config.py`
- `src/dj_grouper/grouping/constraints.py`
- `src/dj_grouper/grouping/distance.py`
- `src/dj_grouper/recommend/scoring.py`

### Registry resolution behavior

- `src/dj_registry/config.py`
- `src/dj_registry/resolver/key_resolver.py`
- `src/dj_registry/adapters/local_analysis.py`
- `src/dj_registry/sync/export.py`

## When Changing Tagger Logic

1. update the code or `settings.toml`
2. if a new helper file is introduced, add it to the relevant signature list
3. run `pytest -q`
4. run `dj run --no-grouping --no-tags`
5. inspect `outputs/registry/registry_overview.csv`
6. run `dj vibe-audit` if vibe-related

Do not add instructions that tell contributors to bump a manual cache version.

## Testing Notes

- current suite size: `245` tests
- tests use synthetic audio fixtures
- registry, tagger, grouper, and cache behaviors all have direct coverage

## Documentation References

- `README.md`
- `docs/technical_spec.md`
- `docs/dj_grouping_recommendation_system_spec.md`
- `docs/tagger_cache_and_experimentation.md`
- `specs/track_registery.md`
