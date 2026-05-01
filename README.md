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
9A_E3_HYPN_64H_INST_126
8A_E4_DRK_32D_FVOC_130_G017
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
dj-registry taxonomy test-api
dj-registry taxonomy generate-ground-truth --files ./files --out files/taxonomy_ground_truth.csv
dj-registry taxonomy train-model --labels files/taxonomy_ground_truth.csv
dj-registry taxonomy classify
dj-registry dj-taxonomy test-api
dj-registry dj-taxonomy generate-ground-truth --files ./files
dj-registry dj-taxonomy train-models --labels outputs/registry/dj_taxonomy_ground_truth.csv
dj-registry dj-taxonomy evaluate --labels outputs/registry/dj_taxonomy_ground_truth.csv --model-dir outputs/registry/dj_taxonomy_model
dj-registry dj-taxonomy classify

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
- `vibe` / `mood`
- `vocal`
- `vocal_profile`
- `vocal_scores`
- `vibe_scores` / `mood_scores`
- `confidences`

`vibe` is kept as the compatibility field used in tags and historical CSVs.
Semantically it is now a taxonomy `mood` code loaded from
`src/dj_registry/taxonomy/dj_taxonomy.json` moods. `vocal` is now a taxonomy
`vocal_profile` code loaded from the same taxonomy `vocal_profiles` values.
Legacy `HYP`, `V`, and `NV` tags are still parsed and normalized to `HYPN`,
`VOC`, and `INST`.

#### Mood Codes

| Code | Mood | Meaning for DJ use |
| --- | --- | --- |
| `ACID` | acidic | Acid-line, 303-like, squelchy or psychedelic pressure. |
| `ATM` | atmospheric | Spacious pads, ambience, texture, or float without a strong song hook. |
| `CIN` | cinematic | Dramatic, soundtrack-like, wide or narrative tension. |
| `DEEP` | deep | Late-night, submerged, dubby, restrained, or low-intensity depth. |
| `DRK` | dark | Nocturnal, shadowy, industrial, gothic, or low-valence mood. |
| `EMO` | emotional | Melancholic, romantic, expressive, or sentiment-forward. |
| `EUP` | euphoric | Uplifting, triumphant, trance-leaning, or hands-up release. |
| `GRIT` | gritty | Rough, distorted, noisy, overdriven, or abrasive texture. |
| `HYPN` | hypnotic | Loop-driven, rolling, meditative, repetitive, or trance-inducing. |
| `MEL` | melodic | Harmony-forward, lead-melody driven, or musically lyrical. |
| `MIN` | minimal | Sparse, reduced, micro, stripped-back, or low-density arrangement. |
| `ORG` | organic | Earthy, desert, ethnic, middle-eastern, wood/percussion oriented. |
| `PLAY` | playful | Funky, bouncy, cheeky, bright, or light-footed. |
| `PSY` | psychedelic | Trippy, psy, mental, acidic, or perception-bending. |
| `RAW` | raw | Unpolished, hard-edged, warehouse, industrial, or rough machine feel. |
| `SOUL` | soulful | Soul, gospel, warm vocal feeling, or emotionally human house feel. |
| `SUB` | subby | Bass-heavy, low-end focused, sub-pressure, or weight-driven. |
| `SUN` | sunlit | Sunset, balearic, outdoor, warm-day, or golden-hour feel. |
| `TENS` | tense | Suspenseful, anxious, pressure-building, or unresolved. |
| `TRIB` | tribal | Percussive, ritual, chant-adjacent, shamanic, or drum-circle energy. |
| `WARM` | warm | Rounded, soft, inviting, soulful, or smooth-toned. |
| `WHSE` | warehouse | Rave-room, concrete, peak industrial, dark-club or big-room rawness. |

#### Vocal Profile Codes

| Code | Profile | Meaning for DJ use |
| --- | --- | --- |
| `CHANT` | chant | Ritual, tribal, mantra-like, call-and-response, or chanted vocal content. |
| `DUB` | dub | Dub mix, reduced vocal, echo-heavy version, or vocal treated as texture. |
| `FVOC` | featured vocal | Featured singer, clear vocal hook, topline, or vocal-led chorus moment. |
| `INST` | instrumental | No meaningful vocal content; voice is absent or not a mix-planning factor. |
| `SPK` | spoken | Spoken word, speech sample, voiceover, MC phrase, or talk-like vocal. |
| `TOOL` | tool | DJ tool, percussive/loop track, functional layer, usually non-vocal. |
| `VOC` | vocal | General vocal-led or lyric-bearing track without a stronger specialized profile. |

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
- 3-level genre taxonomy fields
- exported review and audit reports

`registry_overview.csv` is the main review file. It includes source columns, `tagger_*` columns, and provenance fields such as:

- `tagger_version`
- `tagger_raw_signature`
- `tagger_derived_signature`
- `tagger_key_signature`
- `tagger_audio_features_signature`

Registry batch commands show stderr progress bars by default. Use
`--no-progress` on `dj-registry` or `dj run` commands when scripting or when a
clean log stream is preferred.

The taxonomy workflow adds:

- GPT/Azure OpenAI seeded labels in `files/taxonomy_ground_truth.csv`
- a local learned model in `outputs/registry/taxonomy_model/`
- registry columns `genre_family`, `genre`, `subgenre`, confidence,
  alternatives, evidence, and warnings

Provider genres from Rekordbox, Spotify, Songstats, and embedded tags are used
as model features, not as truth. See
[docs/taxonomy_ground_truth_and_model.md](docs/taxonomy_ground_truth_and_model.md).

The separate `dj-registry dj-taxonomy` workflow trains the flat DJ-functional
category model from `src/dj_registry/taxonomy/dj_taxonomy.json`. It always trains
and reports two models against the same GPT-seeded labels:

- `internal`: file tags, tagger/librosa-derived values, filename/title/mix text
- `external`: all internal features plus Rekordbox, Songstats, Spotify/source observations

Classification always records both model predictions and confidences in
`dj_taxonomy_internal_*` and `dj_taxonomy_external_*` columns. The primary
`dj_taxonomy_*` metadata is expanded from the selected category ID, so moods,
grooves, set roles, BPM/energy ranges, vocal profiles, source genres, and
keywords remain bounded by `dj_taxonomy.json`.

Methodology summary:

- Ground truth is metadata-only GPT/Azure OpenAI labeling, constrained to allowed
  `category_id` values from `dj_taxonomy.json`.
- The ground-truth command first runs registry/tagger collection unless
  `--no-collect` is used.
- The internal model uses local file/tagger/librosa evidence only.
- The external model uses the same local evidence plus Rekordbox, Songstats,
  Spotify/source observations, provider genres, provider labels, and provider
  audio features.
- Both models use a sparse `DictVectorizer` plus balanced logistic regression,
  then report top-1 accuracy, top-3 accuracy, F1, confidence buckets, agreement,
  and external-improved/worsened counts.

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
- `files/taxonomy_ground_truth.csv`: GPT/manual seed labels for taxonomy training
- `outputs/registry/taxonomy_model/`: trained taxonomy model artifacts
- `outputs/groups.csv`: grouped tracks
- `outputs/recommendations.csv`: directional recommendations
- `outputs/Grouped/`: optional grouped folders
- `outputs/playlists/`: generated playlists

## Testing

```bash
pytest -q
```

Current suite size: `291` tests.

## Documentation

- [docs/technical_spec.md](docs/technical_spec.md): current as-built technical architecture
- [docs/taxonomy_ground_truth_and_model.md](docs/taxonomy_ground_truth_and_model.md): genre taxonomy labels, GPT-5 seeding, learned model, and progress/failure behavior
- [docs/dj_grouping_recommendation_system_spec.md](docs/dj_grouping_recommendation_system_spec.md): current grouping and recommendation behavior
- [docs/tagger_cache_and_experimentation.md](docs/tagger_cache_and_experimentation.md): cache-safe tuning workflow
- [specs/track_registery.md](specs/track_registery.md): current registry reference
