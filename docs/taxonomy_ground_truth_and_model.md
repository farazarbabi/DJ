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
