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
  -> optional Spotify playlist gap fill (`dj fetch-missing` / `dj run --fetch-missing`)
  -> dj_registry scan/link/ingest
  -> dj_tagger canonical analysis and shared cache refresh
  -> registry key/BPM resolution, taxonomy classification, and exports
  -> categorical playlist generation
  -> optional Rekordbox cue generation / copied XML export / cue reports
  -> dj_grouper feature build / grouping / recommendation
  -> final COMMENT tag sync with group IDs when grouping ran
```

The registry is the source of truth for canonical key/BPM and for the stored `tagger_*` values used in reporting.
It also stores the current 3-level genre taxonomy assignment and the flat DJ
taxonomy category used for COMMENT category codes.

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
- `cue_analysis`
- `analysis_librosa`
- `analysis_essentia`

Raw/data-collection layers are identity-keyed only. Track-scoped entries use the
full filename + rounded duration + layer. API entries use ISRC + layer. Once an
entry exists for that identity, downstream changes do not recollect it. This
includes DSP, section DSP, raw analysis, embedded tags, Spotify lookups,
Songstats observations and not-found lookups, Rekordbox imports, and registry
librosa/Essentia observations. Cue analysis grids use the same identity-keyed
raw-cache rule.

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
- `dj_model.py`: flat DJ category model training, evaluation, artifact loading, and reports

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

Training writes one XGBoost artifact:

- `outputs/registry/dj_taxonomy_model/xgb/model.pkl`

The model uses file/embedded tags, local tagger/librosa-derived values, identity
text, Rekordbox, Songstats, Spotify/source observations, provider genres,
provider labels, and provider audio features. The LLM labeler receives metadata
only, not raw audio, and must return one allowed active `category_id`; category
metadata is expanded from `dj_taxonomy.json`.

The current supervised baseline uses `DictVectorizer` plus `XGBClassifier`.
Reports include held-out top-1/top-3 accuracy, macro and weighted F1, confidence
buckets, per-category accuracy, label alias counts, and dropped under-supported
categories.

`dj_taxonomy.json` may define `deprecated_category_aliases` for merged or renamed
categories. Deprecated IDs are accepted from older labels/caches and normalized
to active IDs during ground-truth reuse and model training, but they are not
shown to GPT as allowed prompt categories.

Classification writes the primary `dj_taxonomy_*` metadata plus the existing
internal/external prediction columns. Those compatibility columns now contain the
same XGB prediction.

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
- cue analysis, cue XML export, cue quality reporting, and cue XML validation

Disable progress bars with `--no-progress`.

## Unified CLI

`dj run` orchestrates:

1. optionally fetch missing Spotify playlist tracks into the target library
2. scan files and link them to logical tracks
3. ingest Rekordbox when an XML is configured or auto-detected
4. enrich ISRCs and ingest Songstats unless skipped or unavailable
5. run local tagger analysis through the shared cache
6. resolve canonical key and BPM, then build the review queue
7. classify the flat DJ taxonomy with the available XGB model for COMMENT category labels
8. classify the 3-level genre taxonomy and export registry reports
9. write categorical playlists under `outputs/playlists/by_key_coarse/`, `by_bpm_coarse/`, and `by_subgenre_coarse/` by default
10. optionally generate Rekordbox cue rows, copied XML exports, cue reports, and XML validation
11. run grouping and recommendations unless skipped
12. sync COMMENT tags immediately when grouping is skipped, or after grouping when `G###` group IDs can be added

The unified CLI passes progress settings into registry batch steps. `dj-grouper`
and `dj-tagger` also have their own existing progress output for analysis and
grouping-specific work.

`dj vibe-audit` compares registry-stored mood/vibe labels against fresh `derive_vibe()` output from current DSP plus current Songstats audio features.

## Cue Point Workflow

Cue generation is part of the registry layer and is exposed through both:

- `dj run ... --cues`
- `dj-registry cues ...`

The unified `dj run` integration is the preferred path for normal use because
it runs cue generation from the same registry state produced by scan, Rekordbox
ingest, local analysis, and taxonomy/report phases.

### Safety Model

The cue system is XML-first and preserve-first:

- it reads a Rekordbox XML export;
- it writes generated rows to `cue_points_master.csv`;
- it caches cue-grid analysis in `cache/raw_cache.pkl` under `cue_analysis`;
- it can write a copied XML export through `--cue-export-xml`;
- it does not write to the live Rekordbox database;
- it does not write to Pioneer USB/PDB databases;
- existing Rekordbox markers are preserved by default.

The default export policy is `preserve`. Conflicting markers are skipped and
reported instead of overwritten. `replace-generated` only replaces XML markers
that match this tool's generated name/type/color shape. `review-only` avoids XML
marker insertion. `--cue-export-dry-run` previews export statuses without
mutating cue rows or XML markers.

Cue generation has two persistence layers:

- `cache/raw_cache.pkl` stores `cue_analysis`, the expensive audio-derived cue
  grid used by all profiles.
- `cue_points_master.csv` stores generated cue rows: hot cues, memory cues,
  loops, colors, names, confidence, review flags, and export statuses.

Cue analysis cache entries use the shared track cache signature:
`filename|duration|cue_analysis`. Fresh cue-grid analyses are saved to
`cache/raw_cache.pkl` every 10 successfully analyzed tracks and once again at
the end of the run. Generated cue rows are profile-specific and are re-created
from the cached cue grid, so profile changes, loop-length changes, cue
name/color changes, and review-policy changes do not force audio decoding.

`--cue-force` replaces generated `auto_*` cue rows from the cached cue grid.
`--cue-force-analysis` bypasses cached cue grids, decodes matched files again,
rewrites `cue_analysis` entries, and also replaces generated `auto_*` cue rows
for those matched files.

### Cue Profiles

Built-in profiles:

| Profile | Output | Source system |
| --- | --- | --- |
| `v1` | Hot cues `MIX IN`, `DROP 1`, `MIX OUT` | `auto_v1` |
| `v2` | V1 hot cues, memory cues, and intro/outro loops | `auto_v2` |
| `v3-default` | Same cue roles as V2 with stricter confidence/review behavior and phrase-aligned section cues | `auto_v3` |

Supported cue roles:

```text
mix_in, drop_1, mix_out,
intro_start, breakdown, peak, outro_start,
intro_loop, outro_loop
```

`configs/cue_profiles/v3-default.json` is the checked-in profile-file example.
Custom profiles can be passed with `--cue-profile-file`; supported fields are
`enabled_roles`, `loop_bars`, `min_confidence`, `source_system`, and
`phrase_align_sections`.

### Unified Run Options

`dj run` cue options:

| Option | Behavior |
| --- | --- |
| `--cues` | Run cue analysis for tracks matched to the Rekordbox XML export. |
| `--cue-profile v1|v2|v3-default` | Select a built-in cue profile. |
| `--cue-profile-file PATH` | Load a custom JSON/YAML profile. |
| `--cue-loop-bars N` | Override loop length in bars. |
| `--cue-force` | Remove existing generated `auto_*` cue rows for matched files before writing new generated rows. |
| `--cue-force-analysis` | Recompute and rewrite cached `cue_analysis` grids from audio, and replace generated `auto_*` cue rows for matched files. |
| `--cue-limit N` | Limit cue analysis to N matched XML tracks. |
| `--cue-quality-report` | Write `reports/cue_quality_report.csv`. |
| `--cue-validate-xml` | Validate known Rekordbox marker shapes in the input XML before cue work. |
| `--cue-export-xml PATH` | Write generated cues into a copied XML export. |
| `--cue-export-policy preserve|replace-generated|replace-empty-slot|review-only` | Select export conflict behavior. |
| `--cue-export-dry-run` | Preview export statuses without mutating cue rows or XML markers. |

Cue XML validation currently recognizes:

- hot cues: `POSITION_MARK Type="0" Num="0..7" Start="..."`
- memory cues: `POSITION_MARK Type="0" Num="-1" Start="..."`
- loops: `POSITION_MARK Type="4" Num="-1" Start="..." End="..."`

Local user XML exports available during implementation contained hot-cue marker
examples only. Memory and loop output is syntactically tested and must be
validated by importing a copied XML into Rekordbox before bulk library use.

### Cue Outputs

Cue-related outputs under the registry directory:

- `cue_points_master.csv`
- `reports/cue_quality_report.csv`
- `reports/cue_rekordbox_export_report.csv`
- `raw/cue_analysis/*.json`

The JSON audit payloads are written for both fresh and cached cue-grid analyses.
The quality report flags cues that need review, including low confidence,
fallback placement, short loops, possible grid offset, cue-order adjustment, and
export conflicts.

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
- compact category code from the selected flat DJ taxonomy category label

The current COMMENT shape is `KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]`.
`CATEGORY` is the no-space code from the category label. Each label word becomes
3-4 uppercase characters separated by dots, for example
`Dark Tech-House Driver -> DRK.TECH.HOUS.DRV`.

`dj run` delays COMMENT tag writing until after grouping when grouping is
enabled, so the final tag can include the `G###` group ID. If grouping is
skipped, tags are synced after registry resolution and taxonomy classification.

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
- `outputs/dj_taxonomy_ground_truth.csv`
- `outputs/registry/taxonomy_model/training_report.json`
- `outputs/registry/dj_taxonomy_model/xgb/training_report.json`
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
- the registry reference filename `track_registery.md` keeps its historical spelling
