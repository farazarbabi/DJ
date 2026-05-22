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
It also stores the current 3-level genre taxonomy assignment.

## Canonical Tagger Pipeline

The current tagger path is intentionally shared across `dj-tagger`, `dj-registry`, and `dj-grouper`.

### Core Steps

```text
load_audio_features()
  -> compute_tagger_artifacts()
     -> extract_dsp_features() unless identity-keyed dsp was supplied from cache
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
`dsp` is treated as the stable audio-extraction boundary: tag formatting,
category labels, grouping, and derived scoring changes do not force DSP
re-extraction.

### Derived Fields

Produced by `src/dj_tagger/derive.py`:

- `energy`
- `bpm`
- `vibe`
- `mood`
- `vocal`
- `vocal_profile`
- `vocal_scores`
- `structure`
- `vibe_scores`
- `mood_scores`
- `confidences`

Key analysis is computed separately and merged into the final hydrated tagger result.

`vibe` is the stored tag field for taxonomy mood codes loaded from
`src/dj_registry/taxonomy/dj_taxonomy.json`.
`src/dj_tagger/moods.py` is the shared vocabulary source for tagger, grouper,
registry taxonomy features, and tag parsing.

`vocal` is the stored tag field for the taxonomy vocal profile.
`src/dj_tagger/vocals.py` loads `vocal_profiles` from
`src/dj_registry/taxonomy/dj_taxonomy.json`, currently producing `INST`, `VOC`,
`FVOC`, `SPK`, `CHANT`, `DUB`, and `TOOL`.

The human-facing code dictionary is maintained in `README.md`. The runtime
source of truth is:

- moods: `src/dj_tagger/moods.py`
- vocal profiles: `src/dj_tagger/vocals.py`
- taxonomy input: `src/dj_registry/taxonomy/dj_taxonomy.json`

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
- `songstats_lookup`
- `spotify`
- `analysis_librosa`
- `analysis_essentia`

Raw/data-collection layers are identity-keyed only. Track-scoped entries use the
full filename + rounded duration + layer. API entries use ISRC + layer. Once an
entry exists for that identity, downstream changes do not recollect it. This
includes DSP, section DSP, raw analysis, embedded tags, Spotify lookups,
Songstats observations and not-found lookups, Rekordbox imports, and registry
librosa/Essentia observations.

Raw signatures remain provenance metadata on hydrated tagger records. They are
not cache-hit gates for raw/data-collection layers.

### `derived_cache.pkl`

Contains:

- `tagger`

The `tagger` layer is version-gated by the current tagger signature.

## Automatic Signatures

`src/dj_tagger/settings.py` computes signatures from both settings content and relevant source files.

Current signature types:

- `raw_version()`
- `dsp_version()`
- `section_dsp_version()`
- `raw_analysis_version()`
- `derived_version()`
- `key_version()`
- `tagger_version()`

Each hydrated tagger result stores:

- `_tagger_version`
- `_tagger_raw_sig`
- `_tagger_derived_sig`
- `_tagger_key_sig`
- `_tagger_audio_features_sig`

Derived signatures protect derived tagger outputs during tuning. The aggregate
`_tagger_raw_sig` records the raw-code provenance used when the tagger result was
hydrated, but raw/data-collection entries are still reused by identity.

## Refresh Rules

### If settings or derived scoring logic changes

- derived signature changes
- existing raw layers remain usable
- tagger results are re-derived from cached raw data

### If raw extraction logic changes

- existing raw/data-collection entries remain cache hits by identity
- fresh extraction happens only for missing identities or explicit force/clear
- derived tagger output can be rehydrated/restamped from the cached raw payloads

### If key logic changes

- key signature changes
- tagger records are not accepted as current until key is recomputed

### If Songstats audio-feature inputs change

- `_tagger_audio_features_sig` changes
- tagger entries are refreshed instead of silently reused

## Signature Coverage

The signature lists live in `src/dj_tagger/settings.py`:

- `_RAW_VERSION_FILES`
- `_DSP_VERSION_FILES`
- `_SECTION_DSP_VERSION_FILES`
- `_RAW_ANALYSIS_VERSION_FILES`
- `_DERIVED_VERSION_FILES`
- `_KEY_VERSION_FILES`

If a new Python file becomes part of tagger computation, it must be added to the
appropriate list so derived/provenance signatures see it. `_RAW_VERSION_FILES`
is aggregate raw provenance metadata; it does not invalidate raw/data-collection
cache entries.

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
- `tagger_vocal_scores`
- `tagger_confidences`
- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

`registry_overview.csv` also exports `tagger_mood` and `tagger_mood_scores`
as aliases of `tagger_vibe` and `tagger_vibe_scores`.

`tagger_structure` remains stored and exported for analysis, taxonomy features,
grouping, and recommendations. It is not written into the current COMMENT tag.

Current track-level genre taxonomy fields:

- `genre_family`
- `genre`
- `subgenre`
- `genre_confidence`
- `genre_confidence_level`
- `genre_alternatives`
- `genre_evidence`
- `genre_warnings`
- `genre_taxonomy_version`

Legacy `taxonomy_*` columns are still populated for downstream compatibility.

### Analysis Path

`src/dj_registry/adapters/local_analysis.py`:

1. looks for a current hydrated tagger entry
2. if needed, re-derives from cached raw layers
3. reuses raw-layer payloads by filename + duration identity even when signatures changed
4. if needed, runs full canonical analysis
5. writes raw layers and hydrated tagger results back to shared cache
6. updates `LogicalTrack`
7. emits `analysis_librosa` observations

This is the same truth later exported by `registry_overview.csv`.

## Genre Taxonomy Architecture

The taxonomy classifier lives under `src/dj_registry/taxonomy/`.

Primary modules:

- `classifier.py`: deterministic taxonomy scorer and registry persistence
- `features.py`: unified feature extraction for learned classification
- `ground_truth.py`: GPT/Azure OpenAI label generation and API connection test
- `model.py`: scikit-learn training, evaluation, artifact loading, and prediction
- `dj_schema.py`: flat `dj_taxonomy.json` category loader and metadata expansion
- `dj_ground_truth.py`: GPT/Azure OpenAI labels constrained to `dj_taxonomy.json`
- `dj_model.py`: dual internal/external DJ category models and comparison reports

The taxonomy reference is `music_genre_taxonomy_3_level.json`. Every output path
must validate against this file.

### Evidence Strategy

Provider genre fields are not trusted as truth. Rekordbox, Spotify, Songstats,
embedded genre, and `genres_all` are model features alongside:

- tagger energy/mood/vocal/structure/BPM
- canonical BPM
- Songstats audio features
- filename/path, title, mix, label, artist, comments
- rule-derived cues such as rolling, dark, tribal, breaks, warehouse, acid, and dub

When a trained model exists, `classify_all_taxonomies()` loads it from
`outputs/registry/taxonomy_model/model.pkl` and uses the model prediction as the
taxonomy path. Deterministic evidence remains in `genre_evidence` for audit.

Use `dj-registry taxonomy classify --no-model` to force deterministic-only
classification.

### Flat DJ-Functional Taxonomy

`dj-registry dj-taxonomy` is separate from the 3-level genre workflow. It predicts
one `category_id` from `src/dj_registry/taxonomy/dj_taxonomy.json` and expands all
category metadata from that JSON rather than letting a model invent metadata.

Ground truth generation runs registry/tagger collection first unless
`--no-collect` is used. It can include configured Rekordbox and Songstats evidence,
then asks GPT/Azure OpenAI for a strict JSON label bounded to the allowed category
IDs. Provider genres are treated as hints only.

Training writes two artifacts:

- `outputs/registry/dj_taxonomy_model/internal/model.pkl`
- `outputs/registry/dj_taxonomy_model/external/model.pkl`

The internal model uses only file/embedded tags, local tagger/librosa-derived
values, and identity text. The external model uses the same internal features plus
Rekordbox, Songstats, Spotify/source observations, provider genres, labels, and
provider audio features.

Both models train on the same GPT/Azure OpenAI seeded labels so their metrics are
directly comparable. The LLM labeler receives metadata only, not raw audio, and
must return one allowed `category_id`; category metadata is expanded from
`dj_taxonomy.json`.

The current supervised baseline uses `DictVectorizer` plus balanced
`LogisticRegression`. Reports include top-1/top-3 accuracy, macro and weighted
F1, confidence buckets, per-category accuracy, model agreement, and external
improved/worsened counts.

Classification always runs both available models. Standalone
`dj-registry dj-taxonomy classify` keeps the external model as the default primary
when available, while the tag-writing workflow uses the internal model as primary.
Both paths write the primary `dj_taxonomy_*` metadata plus internal/external
prediction columns so confidence and accuracy can be compared per track.

### Ground-Truth Labels

`dj-registry taxonomy generate-ground-truth` creates `files/taxonomy_ground_truth.csv`
from metadata-only GPT/Azure OpenAI calls. It does not upload raw audio.

The command is fail-fast by default:

- first API, parsing, or validation error stops the run
- the error message includes the failing track
- exit code is `1`
- no new partial CSV is written for that failed run

Use `--keep-going` only to intentionally continue after failures and write
`gpt5_error` rows. Existing error rows are retried on later runs and are never
silently reused as valid labels.

Validate connectivity before long runs:

```bash
dj-registry taxonomy test-api
```

### Learned Model Artifacts

`dj-registry taxonomy train-model --labels files/taxonomy_ground_truth.csv`
writes:

- `outputs/registry/taxonomy_model/model.pkl`
- `outputs/registry/taxonomy_model/training_report.json`
- `outputs/registry/taxonomy_model/training_audit.csv`

The artifact records taxonomy hash, feature schema version, label CSV hash, and
training timestamp. It refuses to load if the taxonomy or feature schema has
changed.

## Progress Reporting

Registry batch operations use dependency-free stderr progress bars through
`src/dj_registry/progress.py`.

Progress is enabled by default for CLI runs and disabled by default for direct
library calls.

Covered workflows:

- scan files
- link files
- Rekordbox ingest
- Spotify ISRC enrichment
- Songstats ingest
- local analysis cache scan and audio analysis
- key and BPM resolution
- review queue generation
- review decision import
- tag sync
- registry overview export
- taxonomy classification
- taxonomy ground-truth generation
- taxonomy model training and evaluation

Disable progress bars with `--no-progress`.

## Unified CLI

`dj run` orchestrates:

1. scan files
2. link files to logical tracks
3. ingest Rekordbox
4. enrich ISRCs and ingest Songstats
5. run local analysis
6. resolve canonical key and BPM
7. classify the flat DJ taxonomy with the internal model as the primary category
8. sync tags
9. export registry reports
10. run grouping unless skipped

The unified CLI passes progress settings into registry batch steps. `dj-grouper`
and `dj-tagger` also have their own existing progress output for analysis and
grouping-specific work.

`dj vibe-audit` compares registry-stored mood/vibe labels against fresh `derive_vibe()` output from current DSP plus current Songstats audio features.

## Grouper Architecture

The grouper now uses the same canonical cache pipeline as tagger and registry.

### Extraction

`src/dj_grouper/cli.py`:

- loads current tagger entries when available
- re-derives from raw cache when only derived logic changed
- reuses cached DSP by filename + duration across downstream tag/category/grouping/signature changes
- uses a lightweight DSP-only worker when tagger analysis is cached but DSP is missing
- falls back to canonical artifact extraction when analysis/raw-analysis is missing or force is requested
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
- tagger mood/vibe
- tagger vocal
- compact category code from the internal DJ taxonomy category label

The current COMMENT shape is `KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]`.
`CATEGORY` is the no-space code from the category label. Each label word becomes
3-4 uppercase characters separated by dots, for example
`Dark Tech-House Driver -> DRK.TECH.HOUS.DRV`.

BPM is not currently encoded in the tag; the formatter still accepts a `bpm`
argument so it can be reintroduced without changing call sites. Canonical BPM
remains in the registry.

By default it leaves the dedicated TKEY/InitialKey field untouched so the
original embedded key remains available as a resolver signal. Passing
`--write-key-tag` also writes the canonical Camelot key there for supported
formats.

## Observability

Main review and audit surfaces:

- `outputs/registry/registry_overview.csv`
- `dj vibe-audit`
- hydrated tagger metadata in cache
- `files/taxonomy_ground_truth.csv`
- `outputs/registry/taxonomy_model/training_report.json`
- taxonomy evidence and warnings in `registry_overview.csv`

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
