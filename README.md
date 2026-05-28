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
KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]
```

Examples:

```text
9A|E3|HYPN|INST|DRK.TECH.HOUS.DRV
7A|E2|SOUL|FVOC|ORG.HOUS.BUIL|G015
```

BPM is not currently encoded in the tag (the formatter accepts a `bpm`
argument for forward compatibility but does not emit it). Canonical BPM lives
in the registry.

`CATEGORY` is a compact code derived from the DJ taxonomy category label: each
label word becomes 3-4 uppercase characters separated by dots, for example
`Dark Tech-House Driver -> DRK.TECH.HOUS.DRV`.

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
dj-registry dj-taxonomy train-models --labels outputs/dj_taxonomy_ground_truth.csv
dj-registry dj-taxonomy evaluate --labels outputs/dj_taxonomy_ground_truth.csv --model-dir outputs/registry/dj_taxonomy_model
dj-registry dj-taxonomy classify

dj-grouper --dry-run
dj-grouper --force-extract
```

## Components

### `dj`

The unified pipeline orchestrates:

1. registry scan and file linking
2. Rekordbox and Songstats ingest
3. local tagger analysis and cache refresh
4. canonical key/BPM resolution
5. internal DJ taxonomy category prediction
6. tag writing
7. grouping and recommendation generation

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

`vibe` is the tag field for taxonomy `mood` codes loaded from
`src/dj_registry/taxonomy/dj_taxonomy.json` moods. `vocal` is the tag field for
taxonomy `vocal_profile` codes loaded from the same taxonomy `vocal_profiles`
values.

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

`structure` remains an internal analysis field for registry exports, taxonomy
features, grouping, and recommendations. It is not written into the COMMENT tag.

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
category model from `src/dj_registry/taxonomy/dj_taxonomy.json`. It trains one
XGBoost model using file/tagger/librosa evidence plus Rekordbox, Songstats, and
Spotify/source observations when available.

Classification records the same XGB prediction in both
`dj_taxonomy_internal_*` and `dj_taxonomy_external_*` compatibility columns.
The selected category ID expands to bounded moods, grooves, set roles,
BPM/energy ranges, vocal profiles, source genres, and keywords from
`dj_taxonomy.json`. COMMENT tags use a compact code derived from the selected
category label; they do not write the internal `category_id`.

Methodology summary:

- Ground truth is metadata-only GPT/Azure OpenAI labeling, constrained to allowed
  `category_id` values from `dj_taxonomy.json`.
- The ground-truth command first runs registry/tagger collection unless
  `--no-collect` is used.
- Deprecated category IDs from older labels or caches are normalized through
  `deprecated_category_aliases` before reuse or training; new GPT prompts only
  expose active categories.
- The model uses a sparse `DictVectorizer` plus XGBoost, then reports held-out
  top-1 accuracy, top-3 accuracy, F1, confidence buckets, per-category support,
  alias counts, and dropped under-supported categories.

### `dj-grouper`

The grouper now reads the same shared cache used by tagger and registry. It does not maintain a separate feature-analysis cache anymore.

Current grouping pipeline:

1. load tagger results from shared cache when current
2. fall back to raw cache and re-derive when only derived logic changed
3. reuse cached DSP/raw layers by filename + duration identity, even if downstream signatures changed
4. extract DSP/section-DSP only when the raw cache identity is missing or force is requested
5. run the canonical tagger pipeline only when analysis/raw-analysis is absent from the raw cache
6. build tag, DSP, optional CLAP, and optional registry-enrichment feature layers
7. cluster with either:
   - `constrained` (default)
   - `agglomerative`
8. write `groups.csv`, `recommendations.csv`, playlists, and optional grouped folders

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
- `songstats_lookup`
- `spotify`
- analysis observations

Important: raw/data-collection cache keys are identity-only:

- track-scoped data: full filename + rounded duration + layer
- API data: ISRC + layer

If a raw/data-collection entry exists for that identity, downstream changes do
not recollect it. That includes tag formatting, category labels, grouping,
derived scoring, signature metadata, and taxonomy changes. DSP, section DSP,
raw analysis, embedded tags, Spotify lookups, Songstats observations, Songstats
not-found lookups, Rekordbox imports, and registry analysis observations all use
this rule. Raw versions/signatures are audit metadata only; forced recollection
requires an explicit force/clear workflow.

### `cache/derived_cache.pkl`

Stores derived tagger results:

- `tagger`

Each tagger record carries metadata describing the build used to compute it:

- tagger version
- aggregate raw signature
- derived signature
- key signature
- Songstats audio-feature signature

## Safe Experimentation

If you change `settings.toml` or derived scoring logic, the derived signature changes automatically and tagger results are re-derived from cached raw layers on the next run.

If you change raw extraction logic and want to recollect DSP/librosa/API/tag data,
use an explicit force/clear workflow. Cache identity is still filename +
duration for track data, or ISRC for API data; signature changes alone do not
invalidate raw/data-collection entries.

You do not need to manually bump a cache version string anymore.

The main exception is when you add a brand-new Python module that affects tagger
or raw-layer computation. In that case, add the new file to the relevant
signature list in `src/dj_tagger/settings.py`:

- `_DSP_VERSION_FILES`
- `_SECTION_DSP_VERSION_FILES`
- `_RAW_ANALYSIS_VERSION_FILES`
- `_DERIVED_VERSION_FILES`
- `_KEY_VERSION_FILES`

`_RAW_VERSION_FILES` is kept as aggregate tagger provenance metadata. Derived
outputs may be restamped or re-derived, but the raw/data-collection cache itself
is not invalidated by these signatures.

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

Current suite size: `321` tests.

## Documentation

- [docs/technical_spec.md](docs/technical_spec.md): current as-built technical architecture
- [docs/taxonomy_ground_truth_and_model.md](docs/taxonomy_ground_truth_and_model.md): genre taxonomy labels, GPT-5 seeding, learned model, and progress/failure behavior
- [docs/dj_grouping_recommendation_system_spec.md](docs/dj_grouping_recommendation_system_spec.md): current grouping and recommendation behavior
- [docs/tagger_cache_and_experimentation.md](docs/tagger_cache_and_experimentation.md): cache-safe tuning workflow
- [specs/track_registery.md](specs/track_registery.md): current registry reference
