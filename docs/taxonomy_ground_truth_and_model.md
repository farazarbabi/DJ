# Taxonomy Ground Truth and Learned Genre Model

This document describes the current as-built 3-level genre taxonomy workflow.

## Goal

The registry assigns each track to:

```text
family -> genre -> subgenre
```

The allowed values are constrained by `music_genre_taxonomy_3_level.json`.

Provider genre fields from Rekordbox, Spotify, Songstats, and embedded tags are
not treated as truth. They are normal evidence features alongside tagger output,
BPM, audio features, filename/path text, title/mix text, artist, label, comments,
and rule-derived cues.

## Main Commands

Validate API credentials before generating labels:

```bash
dj-registry taxonomy test-api
```

Generate GPT-seeded labels for files in `./files`:

```bash
dj-registry taxonomy generate-ground-truth --files ./files --out files/taxonomy_ground_truth.csv
```

Train the local model:

```bash
dj-registry taxonomy train-model --labels files/taxonomy_ground_truth.csv
```

Classify registry tracks:

```bash
dj-registry taxonomy classify
```

Evaluate a trained model against a labels CSV:

```bash
dj-registry taxonomy evaluate --labels files/taxonomy_ground_truth.csv --model-dir outputs/registry/taxonomy_model
```

## Ground-Truth Generation

`generate-ground-truth` uses OpenAI or Azure OpenAI through environment
variables loaded from `.env`.

Supported OpenAI variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` optional, defaults to `gpt-5`

Supported Azure OpenAI variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

The command sends metadata only. It does not upload raw audio.

Inputs sent to the model include:

- allowed taxonomy JSON
- file name/path and embedded tags
- canonical registry identity
- tagger energy/mood/vocal-profile/structure/BPM
- source observations from Rekordbox, Songstats, Spotify, tag extraction, and local analysis
- audio features when present

The model must return strict JSON. The result is validated against the taxonomy.
Invalid paths are retried once with validation feedback.

## Failure Behavior

Default behavior is fail-fast:

- first API, parsing, or validation failure stops the run
- the command prints the track-specific error
- exit code is `1`
- no partial blank/error CSV is written for that failed run

Use `--keep-going` only when you intentionally want to keep processing after
failures and write `gpt5_error` rows.

Existing `gpt5_error` rows are never reused. They are retried on the next run.

## Progress Bars

Registry batch workflows now use stderr progress bars by default.

Covered registry workflows include:

- scan
- link
- Rekordbox ingest
- Spotify ISRC enrichment
- Songstats ingest
- local audio analysis
- key and BPM resolution
- review queue generation
- review decision import
- tag sync
- registry export
- taxonomy classification
- taxonomy ground-truth generation
- taxonomy model training and evaluation

Disable progress output with:

```bash
dj-registry --no-progress taxonomy classify
dj-registry taxonomy generate-ground-truth --no-progress
dj run --no-progress
```

## Learned Model

The trained model is stored in:

```text
outputs/registry/taxonomy_model/
```

Artifacts include:

- `model.pkl`
- `training_report.json`
- `training_audit.csv`

The model stores the taxonomy hash, feature schema version, label CSV hash, and
training timestamp. It refuses to load when the taxonomy JSON or feature schema
does not match.

## Feature Groups

The local model uses a single unified feature pipeline:

- provider text features: Rekordbox, Spotify, Songstats, embedded genre, `genres_all`
- local DJ features: energy, mood/vibe, vocal profile, structure, BPM
- audio features: energy, danceability, valence, instrumentalness, acousticness
- context text: artist, remixer, label, title, mix, filename/path, comments
- rule-derived cues: rolling, dark, tribal, breaks, warehouse, acid, dub, vocal, etc.

Provider fields are features only. They are never labels or hard overrides.

## Classification Runtime

`dj-registry taxonomy classify` automatically loads
`outputs/registry/taxonomy_model/model.pkl` when present.

Use deterministic-only classification for debugging:

```bash
dj-registry taxonomy classify --no-model
```

Classification writes these registry columns:

- `genre_family`
- `genre`
- `subgenre`
- `genre_confidence`
- `genre_confidence_level`
- `genre_alternatives`
- `genre_evidence`
- `genre_warnings`
- `genre_taxonomy_version`

Legacy `taxonomy_*` columns are kept populated for downstream compatibility.

## Flat DJ-Functional Category Model

The newer `dj-registry dj-taxonomy` workflow is separate from the 3-level
`family -> genre -> subgenre` classifier. It predicts one `category_id` from
`src/dj_registry/taxonomy/dj_taxonomy.json`.

The selected category controls all exported metadata:

- `category_label`
- `family`
- `moods`
- `grooves`
- `set_roles`
- `bpm_range`
- `energy_range`
- `vocal_profiles`
- `source_genres`
- `keywords`

The model and GPT prompt cannot invent those values; they are expanded from the
static JSON category.

### Methodology

The DJ-taxonomy workflow trains one flat DJ-functional classifier. Ground truth
is created once with GPT/Azure OpenAI using all available evidence, but the
resulting label is constrained to a single active `category_id` from
`dj_taxonomy.json`.

The pipeline is:

1. Collect registry data from the files folder.
2. Run local tagger/librosa analysis and store internal evidence on the registry.
3. Optionally ingest Rekordbox and Songstats/Spotify-style source observations.
4. Build a metadata-only prompt for each track.
5. Ask GPT/Azure OpenAI to choose exactly one allowed `category_id`.
6. Validate the response against `dj_taxonomy.json`.
7. Expand category metadata from JSON into the ground-truth CSV.
8. Train the XGBoost classifier from the label CSV.
9. Evaluate held-out validation metrics and write model reports.

Raw audio is not uploaded to the LLM. The prompt only includes registry metadata,
tagger outputs, source observations, and derived numeric/audio feature values.

### Commands

Validate API credentials:

```bash
dj-registry dj-taxonomy test-api
```

Generate GPT-seeded category labels. This runs registry/tagger collection first
unless `--no-collect` is passed:

```bash
dj-registry dj-taxonomy generate-ground-truth --files ./files
```

The default output is `outputs/dj_taxonomy_ground_truth.csv`. Use
`--out` only when you need a different location.

Train the XGBoost model:

```bash
dj-registry dj-taxonomy train-models --labels outputs/dj_taxonomy_ground_truth.csv
```

Evaluate the trained model against the same labels:

```bash
dj-registry dj-taxonomy evaluate --labels outputs/dj_taxonomy_ground_truth.csv --model-dir outputs/registry/dj_taxonomy_model
```

Classify registry tracks:

```bash
dj-registry dj-taxonomy classify
```

### Model Features

The default model uses the external feature set:

- file and embedded tags
- filename/path/title/artist/mix text
- local tagger mood, vocal profile, energy, structure, BPM, key, and score maps
- local Python/librosa-derived analysis already stored by the registry/tagger
- Rekordbox observations
- Songstats observations and audio features
- Spotify/source IDs and provider metadata where available
- provider genres, labels, comments, release dates, and `genres_all`

The supervised architecture is:

- `DictVectorizer` converts sparse text, categorical, cue, and numeric features.
- `XGBClassifier` predicts the flat `category_id`.
- `predict_proba()` provides the confidence signal.
- Top alternatives come from the next highest probability classes.

The classifier stores one XGB prediction. For compatibility with existing CSV
schemas and tag-writing code, classification mirrors that same prediction into
both internal and external columns:

- `dj_taxonomy_internal_id`
- `dj_taxonomy_internal_confidence`
- `dj_taxonomy_external_id`
- `dj_taxonomy_external_confidence`
- `dj_taxonomy_models_agree`
- `dj_taxonomy_external_evidence_available`

Standalone classification and tag-writing use the same XGB-selected category as
the primary `dj_taxonomy_id`, label, metadata, and confidence.

For COMMENT tag writing, the selected internal model category contributes a
compact code derived from `dj_taxonomy_label`. The tag does not write
`category_id`; IDs remain internal registry/model fields.

### Ground-Truth Guardrails

The LLM prompt is bounded by the full `dj_taxonomy.json` category list. The
response schema requires:

- `category_id`
- `confidence`
- `alternate_category_ids`
- `rationale`
- `warnings`

Every returned ID must exist in `dj_taxonomy.json`. Invalid outputs are retried
once with validation feedback. The prompt explicitly treats Rekordbox, Spotify,
Songstats, and embedded genres as hints rather than truth, because those source
genres are often broad, inconsistent, or wrong for live-DJ use.

`dj_taxonomy.json` may define `deprecated_category_aliases` for merged or renamed
categories. Deprecated IDs are not exposed to the GPT prompt as allowed choices.
When older CSV rows or cached responses contain a deprecated ID, generation reuse
and training normalize it to the canonical active category. Alias normalization is
recorded in model metrics as `label_alias_counts` and `label_alias_targets`; it is
not printed as a training warning.

The ground-truth CSV is fail-fast by default. On the first API, parsing, or
validation error, the command stops and prints the failing track. Use
`--keep-going` only when you intentionally want error rows. Existing error rows
are retried on later runs and are not silently reused as labels.

### Reports

Training/evaluation writes:

```text
outputs/registry/dj_taxonomy_model/
  xgb/model.pkl
  xgb/training_report.json
  xgb/training_audit.csv
```

The training report includes top-1 accuracy, top-3 accuracy, macro/weighted F1,
confidence buckets, per-category support, retained/skipped label counts, alias
normalization counts, and dropped under-supported categories.

Interpret the reports as follows:

- `top1_accuracy`: how often the selected category matched ground truth on the held-out validation split.
- `top3_accuracy`: how often the expected category appeared in the selected category plus two alternatives.
- `confidence_buckets`: calibration check; high-confidence buckets should be more accurate than low-confidence buckets.
- `per_category`: support and accuracy by category, useful for spotting underrepresented or confused categories.

`training_audit.csv` is the most useful audit file when tuning the taxonomy or
feature weights. It contains retained training rows, their expected category,
feature counts, and whether external evidence was available.

### Registry Outputs

After `dj-registry dj-taxonomy classify`, `registry_overview.csv` includes:

- primary category columns: `dj_taxonomy_id`, `dj_taxonomy_label`, `dj_taxonomy_family`
- expanded metadata: `dj_taxonomy_moods`, `dj_taxonomy_grooves`, `dj_taxonomy_set_roles`, `dj_taxonomy_bpm_range`, `dj_taxonomy_energy_range`, `dj_taxonomy_vocal_profiles`, `dj_taxonomy_source_genres`, `dj_taxonomy_keywords`
- compatibility model columns: `dj_taxonomy_internal_id`, `dj_taxonomy_internal_confidence`, `dj_taxonomy_external_id`, `dj_taxonomy_external_confidence`, `dj_taxonomy_models_agree`, `dj_taxonomy_external_evidence_available`
- audit columns: `dj_taxonomy_alternatives`, `dj_taxonomy_evidence`, `dj_taxonomy_version`

When tags are synced, the COMMENT category segment is built from
`dj_taxonomy_label` using 3-4 uppercase characters per word joined with dots,
for example `Dark Tech-House Driver -> DRK.TECH.HOUS.DRV`.
