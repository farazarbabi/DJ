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
  files/                    sample library (every CLI defaults to D:\Music)
  cache/                    shared caches for the default ./files root
    raw_cache.pkl           (a real library root uses <library>/cache/ instead)
    derived_cache.pkl
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
    playlists/
      groups_coarse/          default
      by_key_coarse/          default
      by_bpm_coarse/          default
      by_subgenre_coarse/     default
      groups/                 optional with --fine-playlists
      by_key/                 optional with --fine-playlists
      by_subgenre/            optional with --fine-playlists
      spotify/                one .m3u8 per fetched Spotify CSV
      collection.xml          Rekordbox XML (tracks + playlist tree + key/BPM/beatgrid/cues), default
  settings.toml             tunable derived-scoring parameters
  src/
  tests/
```

## Quick Start

Run the full local pipeline:

```powershell
dj run
```

Useful variants:

```powershell
dj run "E:\\Music" -w 4
dj run --no-grouping
dj run --no-tags
dj run --no-songstats
dj run --force-extract
dj vibe-audit
```

### Rekordbox collection (get playlists onto CDJ/XDJ)

Every `dj run` writes a single Rekordbox XML at
`outputs/playlists/collection.xml`. It carries every library track with its
canonical key, BPM, a beatgrid (`TEMPO`), and any generated cue points
(`POSITION_MARK`), plus the full playlist tree mirroring `outputs/playlists/`
(categorical, grouper, and Spotify playlists).

Import it once via Rekordbox's XML bridge
(Preferences → Advanced → Database → *rekordbox xml* → set the imported library
to this file), drag the tree into your collection — it imports **pre-analyzed**,
so Rekordbox does not re-analyze — then Export to USB for CDJ/XDJ.

```powershell
dj run "D:\\Music"                                            # writes collection.xml by default
dj run "D:\\Music" --rekordbox-collection "D:\\Music\\rb.xml"  # override output path
dj run "D:\\Music" --no-rekordbox-collection                  # skip it
dj export-rekordbox "D:\\Music"                               # regenerate from an existing registry (no full run)

# Fold an old cache directory into the library cache (D:\Music\cache by default)
dj merge-cache .\\cache
```

The fully standalone route (writing a CDJ USB database with no Rekordbox at all)
is intentionally out of scope here; see `specs/standalone_usb_device_export.md`.

Generate Rekordbox cue points as part of the unified run. Cue generation is
opt-in and requires a Rekordbox XML export, either supplied with
`--rekordbox-xml` or auto-detected from the library folder:

```powershell
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-profile v3-default --cue-force
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-export-xml "D:\\Music\\rekordbox_with_cues.xml"
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-export-xml "D:\\Music\\rekordbox_with_cues.xml" --cue-export-dry-run
```

Find and download tracks from Spotify playlist exports that aren't in your
library yet. This works as a standalone command and now runs by default as
Phase 0 of `dj run`, using `<library>/spotify-playlists` unless you override
it. New tracks are downloaded before analysis, so they are tagged and grouped
in the same run:

```powershell
# Standalone: defaults to D:\Music; pass --library to override
dj fetch-missing playlist.csv
dj fetch-missing ./playlists     # a directory of CSVs
dj fetch-missing playlist.csv --dry-run
dj fetch-missing playlist.csv --format wav
dj fetch-missing playlist.csv --no-soundeo   # YouTube only

# As part of the pipeline: default uses D:\Music\spotify-playlists
dj run "D:\\Music"
dj run "D:\\Music" --fetch-missing ./playlists      # override playlist source
dj run "D:\\Music" --fetch-missing playlist.csv --fetch-format wav
dj run "D:\\Music" --no-fetch-missing               # skip fetching
```

Lower-level commands:

```powershell
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
dj-registry cues analyze --rekordbox-xml files\files_rekordbox_export.xml --registry files\outputs\registry --profile v3-default
dj-registry cues report-quality --registry files\outputs\registry
dj-registry cues validate-rekordbox-xml --input-xml files\files_rekordbox_export.xml
dj-registry cues export-rekordbox --input-xml files\files_rekordbox_export.xml --output-xml files\files_rekordbox_export_with_cues.xml --registry files\outputs\registry

dj-grouper --dry-run
dj-grouper --force-extract
```

## Components

### `dj`

The unified pipeline orchestrates:

1. Spotify playlist gap fill via `<library>/spotify-playlists` unless `--no-fetch-missing` is passed
2. registry scan and file linking
3. Rekordbox ingest plus optional Spotify ISRC and Songstats enrichment
4. local tagger analysis and cache refresh
5. canonical key/BPM resolution and review queue generation
6. flat DJ taxonomy classification for COMMENT category tags when a model exists
7. 3-level genre taxonomy classification and registry report export
8. categorical playlist generation under `outputs/playlists/`
9. optional cue-point generation, cue XML export, XML validation, and cue quality reporting
10. grouping and recommendation generation unless skipped
11. final COMMENT tag writing, delayed until after grouping when group IDs are available

#### Cue Points

Cue-point generation is available from the unified `dj run` command and from
the lower-level `dj-registry cues ...` commands. The workflow is XML-first: it
reads a Rekordbox XML export, writes generated cue rows into the registry, and
optionally writes a copied XML file. It does not write directly to the live
Rekordbox database or USB database.

Most users should start with:

```powershell
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-profile v3-default --cue-quality-report
```

That writes generated cues to:

```text
<registry>/cue_points_master.csv
<registry>/reports/cue_quality_report.csv
```

Cue generation has two layers:

- `cache/raw_cache.pkl` stores the expensive audio-derived cue grid as
  `cue_analysis`, keyed by filename plus duration.
- `<registry>/cue_points_master.csv` stores the generated cue rows: hot cues,
  memory cues, loops, colors, names, review flags, and export statuses.

Fresh cue-grid analyses are checkpointed every 10 tracks and flushed again at
the end of the run. Cue profiles are applied after loading the cached cue grid,
so changing `--cue-profile`, `--cue-loop-bars`, cue names/colors, or review
policy should normally use `--cue-force`, not a full audio recache.

Use `--cue-force` to replace generated `auto_*` cue rows for matched tracks
from the cached cue grid. Use `--cue-force-analysis` when cue-grid analysis
logic changed and you want matched tracks decoded again; it rewrites
`cue_analysis` cache entries and also replaces generated `auto_*` cue rows.

```powershell
# Regenerate cue rows from cached cue analysis.
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-profile v3-default --cue-force

# Recompute cue analysis from audio, then regenerate cue rows.
dj run "D:\\Music" --rekordbox-xml "D:\\Music\\rekordbox.xml" --cues --cue-profile v3-default --cue-force-analysis
```

To also create a copied XML export for Rekordbox import/review:

```powershell
dj run "D:\\Music" `
  --rekordbox-xml "D:\\Music\\rekordbox.xml" `
  --cues `
  --cue-profile v3-default `
  --cue-export-xml "D:\\Music\\rekordbox_with_ai_cues.xml" `
  --cue-export-policy preserve
```

Use `--cue-export-dry-run` first when testing export behavior. Dry-run writes
the export report but does not update cue export statuses or insert markers into
the output XML.

Cue profiles:

| Profile | Generated cues | Source system | Notes |
| --- | --- | --- | --- |
| `v1` | Hot cues only: `MIX IN`, `DROP 1`, `MIX OUT` | `auto_v1` | Backward-compatible V1 behavior. |
| `v2` | V1 hot cues plus memory cues and intro/outro loops | `auto_v2` | Deterministic structural cue set. |
| `v3-default` | Same cue roles as V2 with stricter review flags | `auto_v3` | Phrase-aligns section cues where possible and uses a higher confidence threshold. |

Supported cue roles for custom profiles:

```text
mix_in, drop_1, mix_out,
intro_start, breakdown, peak, outro_start,
intro_loop, outro_loop
```

Custom JSON/YAML profiles can be passed with `--cue-profile-file`. The included
example is `configs/cue_profiles/v3-default.json`.

Cue run options on `dj run`:

| Option | Purpose |
| --- | --- |
| `--cues` | Generate cue rows during the run. |
| `--cue-profile v1|v2|v3-default` | Select the built-in cue profile. Defaults to `v3-default` when cue work is requested through `dj run`. |
| `--cue-profile-file PATH` | Load a custom JSON/YAML cue profile. |
| `--cue-loop-bars N` | Override generated loop length in bars. |
| `--cue-force` | Replace existing generated `auto_*` cue rows for matched files before writing new generated cues. Manual/non-auto rows are preserved. |
| `--cue-force-analysis` | Ignore cached `cue_analysis` grids, decode matched files again, rewrite the raw-cache entries, and replace generated `auto_*` cue rows. |
| `--cue-limit N` | Limit cue analysis to the first N matched Rekordbox tracks. Useful for smoke tests. |
| `--cue-quality-report` | Write `cue_quality_report.csv` without requiring XML export. |
| `--cue-validate-xml` | Validate known Rekordbox marker shapes in the input XML before cue work. |
| `--cue-export-xml PATH` | Write generated cues into a copied Rekordbox XML export at PATH. |
| `--cue-export-policy preserve` | Default export behavior. Preserve existing Rekordbox markers and skip conflicts. |
| `--cue-export-policy review-only` | Write reports/statuses without adding XML markers. |
| `--cue-export-policy replace-generated` | Replace only markers that look like they were generated by this tool. User-created markers are preserved. |
| `--cue-export-policy replace-empty-slot` | Currently equivalent to preserve for hot cues: only empty slots are written. |
| `--cue-export-dry-run` | Preview export statuses without mutating cue rows or XML markers. |

Generated cue outputs:

| Output | Description |
| --- | --- |
| `cue_points_master.csv` | Registry table of generated/reviewed cue rows. |
| `reports/cue_quality_report.csv` | Review flags such as low confidence, fallback used, short loop, possible grid offset, and export conflict. |
| `reports/cue_rekordbox_export_report.csv` | Export statuses for inserted, skipped, invalid, unmatched, dry-run, and review-only cue rows. |
| `raw/cue_analysis/*.json` | Per-track cue-grid audit payloads written from fresh or cached cue analysis. |
| copied XML from `--cue-export-xml` | Rekordbox XML copy containing inserted generated markers when not dry-run/review-only. |

Important safety notes:

- Always export to a new XML file, not over the source XML.
- Existing Rekordbox hot cues, memory cues, and loops are preserved by default.
- Memory and loop XML output is syntactically validated, but your local XML
  exports currently contain only hot-cue examples. Import a copied XML into
  Rekordbox and verify memory/loop behavior before bulk use.

It also provides `dj fetch-missing`, which fills gaps from Spotify playlists:

1. parses one or more Exportify-style Spotify playlist CSVs (deduped by track URI)
2. fuzzy-matches each track against the audio files already in `--library`,
   requiring artist agreement so unrelated same-title tracks and alternate
   remixes of a track you only own the original of count as missing
3. downloads the missing tracks from **Soundeo first, YouTube as fallback**:
   - **Soundeo** (your music-pool subscription) is used when
     `SOUNDEO_USER`/`SOUNDEO_PASS` are set in `.env` and `--format aiff` (the
     default). Per track: search (free); if found and the daily quota (resets
     midnight CET) isn't spent, download the best available Soundeo format in
     AIFF > WAV > MP3 order, converting WAV/MP3 to final AIFF; if the quota is
     spent, defer to tomorrow (status `quota_skip`, not YouTubed). Within the
     best format the cut is chosen by chain: an Extended version, else a
     non-Radio cut in the 6:30-8:00 DJ range, else one within ±3s of the
     Spotify duration, else the best by Extended > Original > plain > Radio Edit.
     `--no-soundeo` forces YouTube-only.
   - **YouTube** via `yt-dlp` is used when a track isn't on Soundeo (or Soundeo
     is off/login fails), **verifying duration**: only results within
     `--duration-tolerance` seconds (default 3) of the Spotify track are
     accepted, trying the closest candidate first and up to `--max-attempts`
     (default 3). Audio is extracted to WAV and converted losslessly to AIFF
     by default (`--format wav` to keep WAV).
4. marks non-original downloads by source quality: YouTube uses `[U]`, Soundeo
   WAV uses `[W]`, and Soundeo MP3 uses `[M]`; only Soundeo AIFF downloads stay
   unmarked like curated originals. An unmarked Soundeo AIFF supersedes and
   prunes any prior marked copy. Present marked tracks are not rechecked against
   Soundeo on every run; use `--check-marked-upgrades` when you want to look for
   new Soundeo AIFF upgrades. Embedded Title/Artist/Album/Genre/Year/Label metadata comes
   from the playlist row.
5. treats curated Original/Extended variants as present for unversioned Spotify
   titles, while true remixes stay distinct
6. prunes marked downloads once a curated unmarked original or Original/Extended
   equivalent appears in the library. A previously downloaded marked file is
   only duration-checked/re-downloaded during an explicit `--check-marked-upgrades`
   run.
7. `--upgrade-soundeo` is the quality pass for tracks you already have: every
   playlist track present in the library is checked against Soundeo's best cut
   and re-downloaded only when it strictly improves — lossless AIFF over
   `[U]`/`[M]`/`[W]`, Extended over Original/plain, either over a Radio Edit (a
   Radio Edit is never a target, and a same-tier hit such as the same remix is
   left alone). The superseded file is deleted if it was a marked download, or
   moved to `<library>/outputs/fetch/replaced/` if it was unmarked so the swap
   can be undone. The Soundeo pick must be a version of the very file on disk
   (same title tokens), so a fuzzy playlist match can never swap a curated
   file for a different remix. It never uses YouTube, skips tracks not yet in
   the library (use plain `fetch-missing` for those), stops at the first quota
   hit, and writes `outputs/fetch/soundeo_upgrade.csv` as it goes; `--dry-run`
   reports only. The pass is resumable: one failing track is logged and
   skipped, and every "nothing better on Soundeo" verdict is remembered in
   `outputs/fetch/soundeo_upgrade_checked.json` so a re-run skips it while the
   library file is unchanged (`--force-lookup` re-checks everything).

Reports and a download log are written to `<library>/outputs/fetch/`
(`matched_report.csv`, `missing_report.csv`, `download_log.csv` with a `source`
column, and `unmatched_report.csv` for tracks with no in-tolerance result).
`--dry-run` reports the missing set without downloading. The YouTube fallback
requires `yt-dlp` and `ffmpeg` on `PATH`; the Soundeo source requires
`SOUNDEO_USER`/`SOUNDEO_PASS` in `.env` (alongside the Spotify/Songstats keys).

The same logic runs inside the pipeline by default via `dj run`, as Phase 0,
downloading into the first `dj run` path. Use `--fetch-missing CSV...` to
override the playlist CSV source
(`--fetch-format` chooses aiff/wav). If `yt-dlp`/`ffmpeg` are missing the
pipeline logs the problem and continues without the fetch step.

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
8. write `groups.csv`, `recommendations.csv`, and Rekordbox-compatible M3U8 playlists

## Cache Model

There are two shared cache files, under `<library>/cache/` when running on a
real library root and under `./cache/` for the default `./files` root.

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
- `cue_analysis`
- analysis observations

Important: raw/data-collection cache keys are identity-only:

- track-scoped data: full filename + rounded duration + layer
- API data: ISRC + layer

If a raw/data-collection entry exists for that identity, downstream changes do
not recollect it. That includes tag formatting, category labels, grouping,
derived scoring, signature metadata, and taxonomy changes. DSP, section DSP,
raw analysis, embedded tags, Spotify lookups, Songstats observations, Songstats
not-found lookups, Rekordbox imports, registry analysis observations, and cue
analysis grids all use this rule. Raw versions/signatures are audit metadata
only; forced recollection requires an explicit force/clear workflow.

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
- `outputs/dj_taxonomy_ground_truth.csv`: GPT/manual seed labels for flat DJ taxonomy training
- `outputs/registry/taxonomy_model/`: trained taxonomy model artifacts
- `outputs/registry/dj_taxonomy_model/`: trained flat DJ taxonomy XGB artifacts
- `outputs/registry/cue_points_master.csv`: generated/reviewed cue-point rows
- `outputs/registry/reports/cue_quality_report.csv`: cue review flags and confidence audit
- `outputs/registry/reports/cue_rekordbox_export_report.csv`: copied-XML cue export report
- `outputs/groups.csv`: grouped tracks
- `outputs/recommendations.csv`: directional recommendations
- `outputs/playlists/*_coarse/`: half-resolution playlist sets (groups_coarse/, by_key_coarse/, by_bpm_coarse/, by_subgenre_coarse/), written by default
- `outputs/playlists/groups/`, `by_key/`, `by_subgenre/`: full-resolution playlists, optional with `--fine-playlists`
- `outputs/playlists/spotify/`: one M3U8 mirroring each fetched Spotify CSV (from standalone `dj fetch-missing` or the default `dj run` fetch phase)

Generated categorical and group playlists are ordered by BPM ascending with Camelot key as the tiebreaker. The registry link step repairs stale `primary_file_id` pointers before playlist generation, and the playlist/XML exporters resolve paths from current `files_master.csv` track links so a stale primary pointer cannot put the wrong audio file into a sorted playlist.

## Testing

```bash
pytest -q
```

Current suite size: `473` tests collected by `pytest --collect-only -q`.

## Documentation

- [docs/technical_spec.md](docs/technical_spec.md): current as-built technical architecture
- [docs/taxonomy_ground_truth_and_model.md](docs/taxonomy_ground_truth_and_model.md): genre taxonomy labels, GPT-5 seeding, learned model, and progress/failure behavior
- [docs/dj_grouping_recommendation_system_spec.md](docs/dj_grouping_recommendation_system_spec.md): current grouping and recommendation behavior
- [docs/tagger_cache_and_experimentation.md](docs/tagger_cache_and_experimentation.md): cache-safe tuning workflow
- [specs/track_registery.md](specs/track_registery.md): current registry reference
