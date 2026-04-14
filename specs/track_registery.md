# track_registry — spec v2

## 1. Purpose

Build a Python module (`dj_registry`) that creates and maintains a **single source of truth** for music track metadata, with **musical key** as the first canonical target field.

The system must:

- ingest metadata from local audio files
- ingest metadata from **Songstats API** (via ISRC lookup)
- ingest metadata exported from **Rekordbox**
- run local audio analysis to infer key (both **librosa** and **Essentia** as equal evidence sources)
- merge all evidence into a canonical registry
- automatically resolve a canonical key whenever evidence is sufficient
- require manual review **only** when programmatic evidence is insufficient
- write resolved key back to both the standard **KEY tag field** and the key portion of the existing **dj-tagger COMMENT tag**
- preserve enough structure to later expand into other metadata domains

This first version uses **CSV** as the registry backend.

---

## 2. Integration with existing codebase

### 2.1 Monorepo package

`dj_registry` is a third package in the existing `dj-tools` monorepo alongside `dj_tagger` and `dj_grouper`. It lives at `src/dj_registry/` and shares the existing `pyproject.toml`. A new CLI entry point `dj-registry` is added.

### 2.2 Modules reused from dj_tagger

| Capability | Existing module | How used |
|---|---|---|
| File scanning | `dj_tagger.cli.find_audio_files()` | Discover audio files in library paths |
| Tag reading | `dj_tagger.metadata.read_existing_tag()` | Read existing dj-tagger COMMENT tag |
| Tag writing | `dj_tagger.metadata.write_tag()` | Update COMMENT tag after key resolution |
| Audio loading + HPSS | `dj_tagger.audio.load_audio_features()` | Provide `TrackAudio` for key analysis |
| Key detection (librosa) | `dj_tagger.analyzers.key.analyze_key()` | Evidence source: `analysis_librosa` |
| Key detection (Essentia) | `dj_tagger.analyzers.key.analyze_key(use_essentia=True)` | Evidence source: `analysis_essentia` |
| Camelot mapping | `dj_tagger.analyzers.key.KEY_TO_CAMELOT` | Basis for key normalization (extended with reverse mapping) |
| BPM from metadata | `dj_tagger.audio._read_native_bpm()` | Extract BPM from file tags |
| Supported formats | `dj_tagger.constants.SUPPORTED_EXTENSIONS` | File discovery filter |
| Tag format parsing | `dj_tagger.formats.parse_tag()` | Parse existing COMMENT tags |

### 2.3 New code required

| Capability | New module |
|---|---|
| Extended tag extraction (all fields including ISRC) | `adapters/tag_extractor.py` |
| Rekordbox XML parser | `adapters/rekordbox_xml.py` |
| Songstats API client (ISRC lookup) | `adapters/songstats.py` |
| Local analysis wrapper (librosa + Essentia) | `adapters/local_analysis.py` |
| Text normalization (artist, title, mix) | `identity/normalize.py` |
| Identity resolution cascade | `identity/matcher.py` |
| Key resolver (weighted scoring) | `resolver/key_resolver.py` |
| Resolution explanation codes | `resolver/explanation.py` |
| CSV registry store (atomic I/O) | `store/csv_store.py` |
| Review queue generation | `review/queue_builder.py` |
| Review CSV import | `review/importer.py` |
| Tag write-back (KEY field + COMMENT update) | `sync/tag_writer.py` |
| Pipeline orchestrator | `pipelines/orchestrator.py` |
| Key normalization utilities | `key_utils.py` |
| CLI entry point | `cli.py` |
| Configuration dataclass | `config.py` |
| Data models | `models.py` |

### 2.4 New function in dj_tagger

`dj_tagger.metadata` needs a new function:

```python
def update_tag_key(path: str, new_camelot: str, dry_run: bool = True) -> bool:
    """Parse existing COMMENT tag, replace the key portion, rewrite."""
```

This parses the existing dj-tagger tag string (e.g., `9A_E3_HYPN_64H_NV_126`), replaces the key component with `new_camelot`, and rewrites. If no existing tag, does nothing (tag creation is dj-tagger's job).

### 2.5 Long-term integration goal

The registry is designed to become the upstream metadata source for `dj_grouper`. In future versions, `dj_grouper` will read from registry CSVs instead of re-scanning file tags. This shapes the data model but is not implemented in v1.

---

## 3. Design goals

### 3.1 Single source of truth
The source of truth is **not** the file tag, Songstats, Rekordbox, filename text, or local analysis output. The source of truth is the project's **canonical registry row per logical track**.

### 3.2 Provenance must be obvious
Every important metadata value must be traceable to its source: file tag, Songstats, Rekordbox, local analysis (librosa), local analysis (Essentia), or manual review. Traceability is visible via the `source_observations.csv` and the `key_evidence_summary` field in `tracks_master.csv`.

### 3.3 Manual review is exception handling
Manual review happens only after automated resolution fails.

### 3.4 Registry first, synchronization second
The system resolves canonical values first, then syncs downstream to file tags, exports, and reports.

### 3.5 Future expansion
The architecture supports future canonical resolution for BPM, genre, mood, energy, label/release metadata, similarity features, and richer audio descriptors.

---

## 4. Scope of v1

### In scope
- scan local audio files (all formats dj_tagger supports)
- extract file tags and technical audio metadata
- extract ISRC from file tags
- ingest Rekordbox XML export
- ingest Songstats metadata via ISRC lookup
- run local key analysis (librosa and Essentia as independent sources)
- resolve canonical key
- generate a manual review queue (CSV) only for unresolved cases
- import manual review decisions from edited CSV
- write canonical key back to file KEY tag field and update COMMENT tag key portion
- export CSV registries and audit outputs
- store raw payload references for traceability

### Out of scope
- direct Rekordbox database editing
- GUI application
- realtime filesystem watchers
- cloud sync
- distributed workers
- automatic canonical resolution for metadata fields other than key
- direct SQL/DB backend
- Songstats search-based matching (ISRC-only in v1)

---

## 5. Supported formats and sources

### 5.1 Audio files

Supported in v1:
- `.mp3` (ID3 tags)
- `.aiff` / `.aif` (ID3 tags)

Other formats supported by dj_tagger (`.flac`, `.wav`, `.m4a`) can be added in future versions.

### 5.2 Rekordbox
Rekordbox XML export file.

### 5.3 Songstats
Songstats Enterprise API v1, ISRC-based track lookup.

### 5.4 Local analysis
Two independent analysis engines:
- **librosa** (existing 4-profile ensemble from `dj_tagger.analyzers.key`)
- **Essentia** (KeyExtractor, optional — gracefully skipped if not installed)

---

## 6. Core data concepts

### 6.1 Logical track
A logical track is a musical work/version/remix identity:
- `Artist + Title + Mix + approximate duration`

A logical track may map to multiple files.

### 6.2 File instance
A file instance is a specific file on disk.

### 6.3 Source observation
A source observation is a single metadata value from a single source for a single track. Multiple observations per track per source are possible (e.g., re-ingestion creates new observations).

### 6.4 Canonical metadata
Canonical metadata is the project-selected "best current truth" after evaluating all source observations.

---

## 7. Key representation rules

### 7.1 Two key representations stored
- `standard key` (e.g., `G# minor`, `C major`, `F# major`)
- `Camelot key` (e.g., `1A`, `8B`, `12A`)

### 7.2 Standard key normalization
One canonical display form:
- `{Note} {mode}` where Note is `C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B` and mode is `major` or `minor`
- Examples: `C major`, `F# major`, `G# minor`, `Eb minor`

### 7.3 Enharmonic handling
Use a deterministic mapping (consistent with `dj_tagger.analyzers.key.NOTE_NAMES`):
- `Db` -> `C#`, `D#` -> `Eb`, `Gb` -> `F#`, `G#` stays `G#` (minor) / `Ab` stays `Ab` (major context), `A#` -> `Bb`

### 7.4 Mandatory Camelot derivation
Whenever a source provides only standard key, normalize it and derive Camelot. Whenever a source provides only Camelot, derive standard. Both forms must always be populated when either is known.

### 7.5 Reverse Camelot mapping
Build a `CAMELOT_TO_KEY` reverse mapping from the existing `KEY_TO_CAMELOT` in `dj_tagger.analyzers.key`. This enables converting Camelot codes (from Rekordbox, DJ software) back to standard key representation.

---

## 8. High-level architecture

Batch pipeline with these layers:

1. **Discovery** — scan files, create file inventory
2. **Extraction** — extract metadata from files, Songstats, Rekordbox, and local analysis
3. **Identity** — map file instances and source records to logical tracks
4. **Resolution** — resolve canonical key from all source observations
5. **Review** — queue unresolved tracks for manual CSV-based review
6. **Sync/export** — write canonical key back to files, generate reports

---

## 9. Registry files

Five CSV files in `outputs/registry/`:

1. `tracks_master.csv` — one row per logical track (canonical fields + evidence summary)
2. `files_master.csv` — one row per file instance
3. `source_observations.csv` — one row per source observation per track
4. `review_queue.csv` — one row per unresolved track (editable by user)
5. `source_payload_index.csv` — one row per raw payload artifact

---

## 10. Output directory layout

```text
outputs/registry/
  tracks_master.csv
  files_master.csv
  source_observations.csv
  review_queue.csv
  source_payload_index.csv
  raw/
    songstats/
    rekordbox_xml/
    analysis/
    file_tag_snapshots/
  reports/
    run_logs/
    audits/
  snapshots/               # pre-mutation CSV backups
```

---

## 11. `tracks_master.csv`

One row per logical track. This is the canonical source of truth.

### 11.1 Columns (~25)

#### Identity
- `track_id` — unique identifier (UUID or sequential)
- `identity_status` — `confirmed` | `provisional` | `ambiguous`
- `artist_canonical`
- `title_canonical`
- `mix_canonical`
- `album_canonical`
- `label_canonical`
- `release_date_canonical`
- `duration_sec_canonical`
- `isrc_canonical`

#### Canonical resolved values
- `canonical_key_standard` — e.g., `G# minor`
- `canonical_key_camelot` — e.g., `1A`
- `canonical_key_confidence` — 0.0-1.0
- `canonical_key_source` — e.g., `analysis_librosa+rekordbox`, `manual`
- `canonical_key_resolution_reason` — e.g., `agreed_analysis_and_rekordbox`
- `key_evidence_summary` — human-readable, e.g., `tag=8A | songstats=8A | rekordbox=7A | librosa=8A | essentia=8A -> canonical=8A`

#### Future canonical fields (populated later)
- `canonical_bpm`
- `canonical_bpm_confidence`
- `canonical_genre`

#### Resolution and control
- `needs_manual_review` — boolean
- `review_reason` — reason code if needs_manual_review=true
- `primary_file_id` — file_id of the primary file for this track
- `linked_file_count` — number of file instances linked to this track
- `last_resolved_at` — ISO timestamp of last resolution
- `registry_notes` — free-text notes

### 11.2 `key_evidence_summary` format
Required, human-readable:
```
tag=8A | songstats=8A | rekordbox=7A | librosa=8A | essentia=8A -> canonical=8A
tag=null | songstats=5A | rekordbox=5A | librosa=6A | essentia=null -> canonical=5A
tag=1A | songstats=null | rekordbox=null | librosa=2A | essentia=1A -> review
```

---

## 12. `source_observations.csv`

One row per source observation per track per field.

### 12.1 Purpose
Stores all evidence from all sources in a normalized, extensible format. Adding a new source or new field type requires no schema change — just new rows.

### 12.2 Columns
- `observation_id` — unique identifier
- `track_id` — FK to tracks_master
- `file_id` — FK to files_master (null for non-file sources)
- `source_system` — `tag` | `songstats` | `rekordbox` | `analysis_librosa` | `analysis_essentia` | `manual`
- `source_object_id` — external ID (songstats track ID, rekordbox track ID, etc.)
- `field_name` — `key` | `bpm` | `genre` | `label` | `release_date` | etc.
- `value_standard` — normalized value (e.g., `G# minor` for key, `128` for BPM)
- `value_display` — display form (e.g., `1A` for Camelot key)
- `confidence` — source-specific confidence (0.0-1.0)
- `payload_ref` — pointer to raw payload file in `raw/`
- `observed_at` — ISO timestamp
- `notes` — free-text

### 12.3 Confidence semantics per source
- `tag`: 1.0 (tag exists and is parseable)
- `songstats`: 1.0 (ISRC match is deterministic)
- `rekordbox`: 1.0 if matched by file path, 0.8 if matched by filename+size
- `analysis_librosa`: the analyzer's reported confidence (0.0-1.0)
- `analysis_essentia`: the analyzer's reported confidence (0.0-1.0)
- `manual`: 1.0 (manual decision is authoritative)

---

## 13. `files_master.csv`

One row per file instance.

### 13.1 Columns

#### File identity
- `file_id`
- `track_id` — FK to tracks_master
- `is_primary_file` — boolean
- `match_method` — `sha256` | `exact_normalized` | `fuzzy` | `manual`
- `match_score` — 0.0-1.0

#### File system metadata
- `path_abs`
- `path_rel`
- `file_name`
- `extension`
- `size_bytes`
- `mtime_utc`
- `sha256`

#### Audio technical metadata
- `audio_duration_sec`
- `sample_rate`
- `bitrate`
- `channels`
- `bits_per_sample`

#### Embedded tag values
- `embedded_title`
- `embedded_artist`
- `embedded_album`
- `embedded_genre`
- `embedded_bpm`
- `embedded_key_standard`
- `embedded_key_camelot`
- `embedded_comment`
- `embedded_isrc`

#### Read/write traceability
- `tag_read_status` — `ok` | `error` | `missing`
- `tag_read_error`
- `tag_write_status` — `ok` | `error` | `pending` | `skipped`
- `tag_write_error`
- `last_scanned_at`
- `last_tag_written_at`

#### Payload reference
- `file_tag_payload_ref`

---

## 14. `review_queue.csv`

One row per unresolved logical track. CSV-based manual review workflow.

### 14.1 Columns
- `review_id`
- `track_id`
- `priority` — `high` | `medium` | `low`
- `reason_code` — e.g., `source_conflict`, `low_confidence`, `no_analysis`, `ambiguous_identity`
- `reason_detail` — human-readable explanation
- `candidate_key_standard_1`, `candidate_key_camelot_1`, `candidate_source_1`, `candidate_score_1`
- `candidate_key_standard_2`, `candidate_key_camelot_2`, `candidate_source_2`, `candidate_score_2`
- `candidate_key_standard_3`, `candidate_key_camelot_3`, `candidate_source_3`, `candidate_score_3`
- `suggested_key_standard` — pre-filled with top candidate
- `suggested_key_camelot` — pre-filled with top candidate
- `primary_file_id`
- `primary_file_path`
- `key_evidence_summary`
- `created_at`

#### User-editable columns (filled during review)
- `reviewer_key_standard` — user fills this with the correct key
- `reviewer_key_camelot` — user fills this with the correct Camelot code
- `reviewer_decision` — `accept` | `override` | `skip`
- `reviewer_note`
- `resolved_at`

### 14.2 Review workflow
1. **Generate:** `dj-registry review-queue` creates/refreshes `review_queue.csv`
2. **Edit:** User opens CSV in Excel or VS Code, fills `reviewer_decision` and optionally `reviewer_key_*` columns
3. **Import:** `dj-registry import-reviews` reads edited CSV:
   - `accept` -> uses suggested key, creates manual observation
   - `override` -> uses reviewer's key, creates manual observation with override_active=true
   - `skip` -> leaves track unresolved
4. **Validation on import:** Verify reviewer keys are valid, Camelot matches standard, no duplicate track_ids

---

## 15. `source_payload_index.csv`

One row per raw payload artifact.

### 15.1 Columns
- `payload_ref` — unique reference ID
- `source_system` — `tag` | `songstats` | `rekordbox` | `analysis_librosa` | `analysis_essentia`
- `source_type` — `json` | `xml` | `pickle`
- `source_object_id`
- `track_id`
- `file_id`
- `payload_path` — relative path under `raw/`
- `fetched_at`
- `notes`

### 15.2 Raw payload storage

```text
raw/
  songstats/          # JSON responses: {isrc}.json
  rekordbox_xml/      # Copy of imported XML file
  analysis/           # JSON: {file_id}_librosa.json, {file_id}_essentia.json
  file_tag_snapshots/ # JSON: {file_id}.json (all extracted tag fields + raw frame data)
```

### 15.3 Rules
- payload files are immutable once written
- payload references are unique and stable
- tag snapshots use JSON format (human-readable, diffable)

---

## 16. Identity resolution

### 16.1 Goal
Map files and external source records to one logical track.

### 16.2 Matching cascade

Each step is tried in order. The first match above the confidence threshold wins.

| Step | Method | Match score | Description |
|---|---|---|---|
| 1 | SHA256 | 1.0 | Exact file content match (duplicate files) |
| 2 | Exact normalized (artist, title, mix, duration +-2s) | 0.95 | All identity fields match after normalization |
| 3 | Exact normalized (title, duration +-2s) when artist missing | 0.80 | Title-only match for tracks with missing artist |
| 4 | Fuzzy title/mix + duration +-5s | 0.50-0.79 | Fuzzy string matching with duration threshold |
| 5 | Unmatched | 0.0 | Track gets its own logical track entry |

### 16.3 Normalization rules
Apply to artist, title, mix fields before matching:
- Unicode NFC normalization
- lowercase
- strip leading/trailing whitespace
- collapse repeated spaces
- normalize featuring syntax: `feat.`, `feat`, `ft.`, `ft`, `featuring` -> `feat`
- normalize artist separators: `&`, `and`, `,`, `vs.`, `vs` -> `&`
- strip surrounding parentheses and brackets from mix names
- normalize mix/version suffixes (see 16.4)
- remove filename artifacts (track numbers, file extensions)

### 16.4 Mix/version normalization
Normalize to canonical forms:
- `Original Mix`, `Original` -> `Original Mix`
- `Extended Mix`, `Extended` -> `Extended Mix`
- `Radio Edit`, `Radio Mix` -> `Radio Edit`
- `Dub Mix`, `Dub` -> `Dub Mix`
- `{Artist} Remix` -> `{Artist} Remix` (preserve remixer name)
- `VIP`, `VIP Mix` -> `VIP`
- `Instrumental`, `Instrumental Mix` -> `Instrumental`
- `Live`, `Live Version` -> `Live`

### 16.5 Cross-source matching

#### Rekordbox -> files
Rekordbox XML contains file paths as `file://localhost/E:/Music/track.mp3`:
1. URL-decode the path
2. Normalize to OS path format (forward slashes to backslashes on Windows)
3. Match to `files_master` by: exact path -> filename+size -> filename-only
4. Drive letter remapping via `RegistryConfig.path_prefix_map` (e.g., `{"E:": "D:"}`)

#### Songstats -> tracks
Match by ISRC: extract ISRC from file tags or Rekordbox XML, look up on Songstats API.

### 16.6 Unmatched records
- Rekordbox tracks not matched to local files: log as warning, do not create file records
- Songstats results not matched (ISRC lookup returned wrong track): log as warning, discard observation
- Local files not matched to any external source: still get their own logical track

---

## 17. File metadata extraction

### 17.1 Library scan
For every discovered file, extract:
- absolute path, relative path, file name, extension
- size in bytes, modification time (UTC)
- SHA256 hash
- audio duration, sample rate, bitrate, channel count, bits per sample

### 17.2 Tag extraction
Extract all readable standard tags via mutagen:
- title, artist, album, genre, comment, BPM, key, ISRC

### 17.3 ISRC extraction
ISRC is critical for Songstats lookup. Extract from:
- ID3 `TSRC` frame (MP3, AIFF)

### 17.4 Key extraction from tags
If tag contains standard key: normalize it, derive Camelot, store both.
If tag contains Camelot only: derive standard via reverse mapping, store both.

### 17.5 Tag payload snapshot
Persist a JSON snapshot of all extracted tag fields per file:
- Path: `raw/file_tag_snapshots/{file_id}.json`
- Referenced via `file_tag_payload_ref` in `files_master.csv`

---

## 18. Rekordbox ingestion

### 18.1 Input
Rekordbox XML export file (specified via `--rekordbox-xml PATH`).

### 18.2 Fields to extract
- track title, artist, album, genre
- BPM, duration, key, rating
- comments
- track location (file URI)
- playlist membership
- Rekordbox internal track ID

### 18.3 Key normalization
Rekordbox provides keys in its own format. Normalize to standard + Camelot.

### 18.4 Path matching
See Section 16.5 for Rekordbox path matching strategy.

### 18.5 Traceability
- Store raw XML file in `raw/rekordbox_xml/`
- Create source observations with `source_system = "rekordbox"`
- Record `observed_at` as the import timestamp

---

## 19. Songstats ingestion

### 19.1 Role
Songstats is a metadata enrichment source, not the canonical owner.

### 19.2 API details
- **Base URL:** `https://api.songstats.com/enterprise/v1`
- **Authentication:** API key in `apikey` header, loaded from `SONGSTATS_API_KEY` environment variable (via `.env`)
- **Primary endpoint:** `GET /enterprise/v1/tracks/info?isrc={isrc}`
- **Rate limiting:** respect `Retry-After` headers; default 1 request/second; configurable via `RegistryConfig.songstats_rate_limit`

### 19.3 Matching strategy
ISRC-only lookup:
1. Extract ISRC from file tags or Rekordbox XML
2. Query Songstats by ISRC
3. If no ISRC available for a track, skip it (no search fallback in v1)
4. If ISRC returns no results, mark observation as `no_match`

### 19.4 Fields to collect
Collect from Songstats response:
- track ID, title, artist(s), remix/version data
- label, release, release date
- BPM / tempo, key, genre
- source/store IDs (Spotify, Beatport, etc.)
- ISRC confirmation

### 19.5 Source observations
Create observations with `source_system = "songstats"`, `confidence = 1.0` (ISRC match is deterministic).

### 19.6 Raw payload storage
Store JSON response in `raw/songstats/{isrc}.json`.

### 19.7 Adapter requirements
- `httpx`-based async-capable client
- retry with exponential backoff on 429/5xx
- partial progress persistence (save observations as they arrive, don't wait for full batch)
- configurable batch size via `--limit N`
- tolerate missing fields in response

---

## 20. Local analysis

### 20.1 Two independent engines

#### librosa (always available)
- Source: `dj_tagger.analyzers.key.analyze_key(track_audio, use_essentia=False)`
- Creates observation with `source_system = "analysis_librosa"`
- Confidence: the analyzer's reported confidence (0.0-1.0)
- Default resolver weight: 0.80

#### Essentia (optional)
- Source: `dj_tagger.analyzers.key.analyze_key(track_audio, use_essentia=True)`
- Creates observation with `source_system = "analysis_essentia"`
- Confidence: the analyzer's reported confidence (0.0-1.0)
- Default resolver weight: 0.85
- Gracefully skipped if Essentia is not installed

### 20.2 Analysis outputs per engine
- `analysis_key_standard`
- `analysis_key_camelot`
- `analysis_confidence`
- `analysis_engine` — `librosa` or `essentia`
- `analysis_engine_version` — matches `ANALYZER_VERSION` from `dj_tagger.cache`
- `analysis_payload_ref`
- `analysis_observed_at`

### 20.3 Future-ready descriptor extraction
Even though key is the only canonical field in v1, analysis payloads should store additional descriptors for future use (loudness, spectral, rhythm, tonal). These remain in raw payloads only.

### 20.4 Analysis caching
Do not recompute analysis if:
- the file SHA256 is unchanged
- an analysis payload already exists for the same file, engine, and engine version

---

## 21. Canonical key resolution

### 21.1 Objective
Automatically determine the canonical key whenever evidence is sufficient.

### 21.2 Candidate sources (in default weight order)
| Source | Default weight | Notes |
|---|---|---|
| `manual` | 1.00 | Always wins when override_active=true |
| `analysis_essentia` | 0.85 | Optional, higher accuracy |
| `analysis_librosa` | 0.80 | Always available |
| `rekordbox` | 0.75 | DJ software analysis |
| `tag` | 0.55 | Embedded file tags |
| `songstats` | 0.40 | External metadata |

Weights are configurable via `RegistryConfig`.

### 21.3 Scoring algorithm

```
resolve_key(observations for one track):
    # Group observations by Camelot key value
    candidates = group_by_camelot(key_observations)

    for each candidate key:
        # Base score: sum of (source_weight * source_confidence)
        base = sum(WEIGHTS[obs.source_system] * obs.confidence for obs in candidate.observations)

        # Agreement boost: +0.15 per additional agreeing source beyond the first
        agreement_boost = 0.15 * (len(candidate.observations) - 1)

        # Cross-type boost: +0.10 if analysis agrees with an external source
        has_analysis = any obs is analysis_librosa or analysis_essentia
        has_external = any obs is rekordbox or songstats
        cross_boost = 0.10 if (has_analysis and has_external) else 0.0

        candidate.score = base + agreement_boost + cross_boost

    # Sort candidates by score descending
    top = highest scoring candidate
    second = second highest (if exists)

    # Auto-resolve if top score > threshold AND lead over second > margin
    if top.score > CONFIDENCE_THRESHOLD and (no second or top.score - second.score > MARGIN_THRESHOLD):
        return resolved(key=top.key, confidence=top.score, source=contributing_sources)
    else:
        return unresolved(reason=classify_conflict(top, second, observations))
```

### 21.4 Default thresholds
- `CONFIDENCE_THRESHOLD = 0.70` — minimum score to auto-resolve
- `MARGIN_THRESHOLD = 0.20` — minimum lead over second candidate

Both configurable via `RegistryConfig`.

### 21.5 Worked examples

**Example 1: Strong agreement**
```
tag=8A (conf=1.0), songstats=8A (conf=1.0), rekordbox=7A (conf=1.0), librosa=8A (conf=0.85)

Candidate 8A:
  base = 0.55*1.0 + 0.40*1.0 + 0.80*0.85 = 1.63
  agreement = 0.15 * 2 = 0.30
  cross_boost = 0.10 (librosa + songstats agree)
  score = 2.03

Candidate 7A:
  base = 0.75*1.0 = 0.75
  score = 0.75

Result: auto-resolve 8A (score=2.03, lead=1.28 > 0.20)
```

**Example 2: Conflict -> review**
```
tag=1A (conf=1.0), librosa=2A (conf=0.60)

Candidate 1A:
  base = 0.55*1.0 = 0.55
  score = 0.55

Candidate 2A:
  base = 0.80*0.60 = 0.48
  score = 0.48

Result: review (top score 0.55 < 0.70 threshold)
```

**Example 3: Analysis + Essentia agree**
```
librosa=5A (conf=0.90), essentia=5A (conf=0.92), rekordbox=5A (conf=1.0)

Candidate 5A:
  base = 0.80*0.90 + 0.85*0.92 + 0.75*1.0 = 0.72 + 0.782 + 0.75 = 2.252
  agreement = 0.15 * 2 = 0.30
  cross_boost = 0.10 (analysis + rekordbox agree)
  score = 2.652

Result: auto-resolve 5A (score=2.652, no second candidate)
```

### 21.6 Manual override rule
If `manual_override_active = true`, manual key overrides all other sources until explicitly changed. The resolver skips scoring and uses the manual value directly.

### 21.7 Relative key handling
In v1, treat relative major/minor (e.g., A minor and C major, which are harmonically related but different keys) as disagreement. Future versions may add "soft agreement" with reduced boost.

### 21.8 Review triggers
Force manual review if:
- no candidate reaches confidence threshold
- top two candidates are within margin of each other
- local analysis failed and metadata-only sources disagree
- identity is ambiguous (`identity_status = "ambiguous"`)
- all sources provide low confidence

### 21.9 Output fields
The resolver populates in `tracks_master.csv`:
- `canonical_key_standard`, `canonical_key_camelot`
- `canonical_key_confidence`, `canonical_key_source`
- `canonical_key_resolution_reason`
- `key_evidence_summary`
- `needs_manual_review`, `review_reason`
- `last_resolved_at`

### 21.10 Resolution reason codes
- `agreed_all_sources`
- `agreed_analysis_and_rekordbox`
- `agreed_analysis_and_songstats`
- `agreed_both_analyses`
- `analysis_overrode_conflicting_tag`
- `single_high_confidence_source`
- `manual_override`
- `metadata_conflict_requires_review`
- `low_confidence_requires_review`
- `no_analysis_available`

### 21.11 Compound source reporting
`canonical_key_source` contains compound values when multiple sources contributed:
- `analysis_librosa+analysis_essentia+rekordbox`
- `tag+songstats`
- `manual`

---

## 22. Write-back strategy

### 22.1 Targets
Write canonical key to **two locations** in each file:

#### Standard KEY tag field
- **MP3/AIFF/WAV**: ID3 `TKEY` frame (e.g., `"Gm"` or `"1A"` — Camelot for DJ software compatibility)
- **FLAC**: Vorbis `KEY` tag
- **M4A**: appropriate iTunes atom

#### dj-tagger COMMENT tag (key portion only)
- Parse the existing dj-tagger COMMENT tag (e.g., `9A_E3_HYPN_64H_NV_126`)
- Replace the key portion (first component) with the canonical Camelot code
- Rewrite the COMMENT tag
- If no existing dj-tagger COMMENT tag exists, do NOT create one

### 22.2 Write-back behavior
- write only if canonical key differs from current embedded key
- support `--dry-run` mode (log what would change without writing)
- snapshot registry CSVs before any write operation
- log before/after values for every file written
- record `tag_write_status` and `last_tag_written_at` in `files_master.csv`

### 22.3 Rekordbox behavior
Do not edit Rekordbox internals directly. Instead:
- ingest Rekordbox XML for evidence
- write corrected keys to file tags
- let Rekordbox re-import from file metadata

---

## 23. CSV store strategy

### 23.1 Backend
CSV for v1 via Python's `csv` module.

### 23.2 Isolation rule
All business logic interacts through a `CsvStore` class with methods like `load_tracks()`, `save_tracks()`, `upsert_track()`, `load_observations()`, `add_observation()`. Only `store/csv_store.py` knows CSV details. Future migration to SQLite requires only replacing this module.

### 23.3 Atomic writes
- Write to temp file in same directory (e.g., `tracks_master.csv.tmp`)
- Rename atomically via `os.replace()` (atomic on NTFS)
- On failure, temp file is cleaned up

### 23.4 Snapshot before mutation
Before any pipeline run that modifies CSVs, copy current state to `outputs/registry/snapshots/{run_id}/`.

### 23.5 Concurrency protection
- Lock file at `outputs/registry/.lock` acquired at pipeline start, released at end
- If stale lock found (>1 hour old), warn and offer to override via `--force`

### 23.6 Required behaviors
- idempotent reruns
- stable column ordering
- tolerant read (ignore unknown columns, fill missing with defaults)

---

## 24. Project structure

```text
src/dj_registry/
  __init__.py
  cli.py
  config.py                    # RegistryConfig dataclass
  models.py                    # LogicalTrack, FileRecord, SourceObservation, ReviewItem
  key_utils.py                 # Standard <-> Camelot bidirectional, normalization

  adapters/
    __init__.py
    file_scanner.py            # Wraps dj_tagger file scanning + extended tag extraction
    tag_extractor.py           # Extract all tag fields including ISRC
    rekordbox_xml.py           # lxml parser for Rekordbox XML
    songstats.py               # httpx client, ISRC-based lookup, retry/rate-limit
    local_analysis.py          # Wraps dj_tagger key analyzers (librosa + Essentia)

  identity/
    __init__.py
    normalize.py               # Text normalization (artist, title, mix, featuring, unicode)
    matcher.py                 # Identity resolution cascade

  resolver/
    __init__.py
    key_resolver.py            # Weighted scoring, agreement boosts, auto-resolution
    explanation.py             # Resolution reason codes and evidence summary

  store/
    __init__.py
    csv_store.py               # Atomic CSV read/write, schema validation, snapshots

  review/
    __init__.py
    queue_builder.py           # Generate review_queue.csv
    importer.py                # Import edited review_queue.csv

  sync/
    __init__.py
    tag_writer.py              # Write KEY field + update COMMENT key portion
    export.py                  # Human-friendly reports and audit outputs

  pipelines/
    __init__.py
    orchestrator.py            # Full pipeline: scan -> link -> ingest -> analyze -> resolve -> sync
```

---

## 25. Configuration

Single `config.py` with a `RegistryConfig` dataclass (consistent with `dj_grouper.config.GrouperConfig` pattern).

```python
@dataclass
class RegistryConfig:
    # Library
    library_roots: list[str] = field(default_factory=lambda: ["./files"])
    supported_extensions: list[str]  # from dj_tagger.constants.SUPPORTED_EXTENSIONS

    # Songstats
    songstats_api_key: str = ""      # loaded from env/dotenv
    songstats_base_url: str = "https://api.songstats.com/enterprise/v1"
    songstats_rate_limit: float = 1.0  # requests per second
    songstats_batch_limit: int = 100   # max tracks per ingest run

    # Rekordbox
    rekordbox_xml_path: str = ""
    path_prefix_map: dict[str, str] = field(default_factory=dict)  # e.g., {"E:": "D:"}

    # Analysis
    run_essentia: bool = True          # try Essentia if installed
    analysis_workers: int = 1          # parallel analysis

    # Resolver
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "manual": 1.00,
        "analysis_essentia": 0.85,
        "analysis_librosa": 0.80,
        "rekordbox": 0.75,
        "tag": 0.55,
        "songstats": 0.40,
    })
    confidence_threshold: float = 0.70
    margin_threshold: float = 0.20
    agreement_boost: float = 0.15
    cross_type_boost: float = 0.10

    # Identity
    duration_tolerance_strong: float = 2.0   # seconds, for exact matches
    duration_tolerance_weak: float = 5.0     # seconds, for fuzzy matches

    # Paths
    output_dir: str = "./outputs/registry"
    raw_dir: str = "./outputs/registry/raw"
    reports_dir: str = "./outputs/registry/reports"
    snapshots_dir: str = "./outputs/registry/snapshots"
```

No YAML config files in v1. CLI flags override defaults.

---

## 26. Dependencies

Added to existing `pyproject.toml`:

```toml
[project.optional-dependencies]
registry = [
    "httpx>=0.25",        # Songstats API
    "lxml>=4.9",          # Rekordbox XML parsing
    "python-dotenv>=1.0", # .env file loading
]
```

Not needed in v1:
- `pydantic` — use dataclasses (consistent with existing codebase)
- `pandas` / `polars` — Python `csv` module suffices for 20K rows
- `typer` / `click` — use `argparse` (consistent with existing CLIs)
- `rich` — use existing logging pattern
- `PyYAML` — no YAML config files
- `rapidfuzz` — add later only if fuzzy matching quality is insufficient

---

## 27. CLI specification

Entry point: `dj-registry`

### Commands

#### `scan`
Scan library files and update `files_master.csv`.

```
dj-registry scan [PATH ...] [--recursive] [--dry-run]
```
- `PATH` — library paths (default: `./files`)
- `--recursive` — recurse into subdirectories (default: true)
- `--dry-run` — report what would be scanned without writing

#### `link`
Resolve file-to-track mappings, update `tracks_master.csv`.

```
dj-registry link
```

#### `ingest-rekordbox`
Parse Rekordbox XML export, create source observations.

```
dj-registry ingest-rekordbox --xml PATH
```

#### `ingest-songstats`
Fetch Songstats metadata for tracks with ISRC, create source observations.

```
dj-registry ingest-songstats [--limit N] [--only-missing]
```
- `--limit N` — max tracks to fetch (default: from config)
- `--only-missing` — skip tracks that already have Songstats observations

#### `analyze`
Run local key analysis (librosa + Essentia).

```
dj-registry analyze [--only-missing] [--track-id ID] [--workers N] [--no-essentia]
```
- `--only-missing` — skip tracks that already have analysis observations
- `--track-id ID` — analyze a specific track
- `--workers N` — parallel analysis workers
- `--no-essentia` — skip Essentia even if installed

#### `resolve`
Run canonical key resolver on all tracks.

```
dj-registry resolve [--force]
```
- `--force` — re-resolve all tracks, not just those with new observations

#### `review-queue`
Generate or refresh `review_queue.csv`.

```
dj-registry review-queue
```

#### `import-reviews`
Import manual review decisions from edited `review_queue.csv`.

```
dj-registry import-reviews [--file PATH]
```

#### `sync-tags`
Write canonical key back to file tags.

```
dj-registry sync-tags [--dry-run] [--only-changed]
```
- `--dry-run` — log what would change without writing
- `--only-changed` — only write files where canonical key differs from embedded key

#### `run`
Full pipeline: scan -> link -> ingest -> analyze -> resolve -> review-queue -> sync-tags.

```
dj-registry run [PATH ...] [--rekordbox-xml PATH] [--songstats] [--dry-run] [--write-tags]
```
- `--rekordbox-xml PATH` — include Rekordbox ingestion
- `--songstats` — include Songstats ingestion
- `--dry-run` — no writes
- `--write-tags` — enable tag writing (default: dry-run)

#### `export`
Generate human-friendly audit reports.

```
dj-registry export [--output PATH]
```

### Global flags
- `-v, --verbose` — debug logging
- `-q, --quiet` — error-only logging
- `--version` — show version

---

## 28. Pipeline order

### Full batch pipeline
1. `scan` — discover files, extract tags, populate files_master
2. `link` — identity resolution, populate tracks_master
3. `ingest-rekordbox` — parse XML, create source observations
4. `ingest-songstats` — ISRC lookup, create source observations
5. `analyze` — run librosa + Essentia, create source observations
6. `resolve` — score candidates, auto-resolve where possible
7. `review-queue` — generate CSV for unresolved tracks
8. `sync-tags` — write canonical key back to files
9. `export` — generate reports

### Incremental rerun behavior
- **Re-scan:** skip files with unchanged mtime. Re-extract if mtime changed.
- **Re-link:** only process new/changed files.
- **Re-ingest Songstats:** skip tracks that already have observations for current date. Fetch new tracks.
- **Re-ingest Rekordbox:** always re-ingest (XML may have changed). Overwrite previous Rekordbox observations.
- **Re-analyze:** skip files with unchanged SHA256 and existing analysis for same engine version.
- **Re-resolve:** re-resolve any track with new/changed observations since `last_resolved_at`. Preserve manual overrides.
- **Re-sync:** only write files where canonical key differs from embedded key.

---

## 29. Error handling

### General rule
No single file, payload, or API error should fail the full run.

### File-level errors
Record in `tag_read_error`, `tag_write_error`, and run logs.

### API errors (Songstats)
- retry with exponential backoff on 429/5xx
- partial progress persistence (save observations as they arrive)
- rate-limit handling (respect `Retry-After` header)

### XML errors (Rekordbox)
- keep raw file
- emit parse report
- do not corrupt registry state

---

## 30. Logging and audit outputs

Each run produces:
- run ID (UUID)
- start/end timestamps
- counts: files scanned, tracks linked, Songstats records ingested, Rekordbox records ingested, analyses completed (per engine), keys auto-resolved, review items created, tag writes attempted, tag writes succeeded
- warning summary
- error summary

### Audit reports
- unresolved-key report
- source-conflict report
- duplicate-file report
- low-confidence report
- tag-write report (before/after values)

---

## 31. Data quality and integrity rules

### Hard checks
- every `track_id` unique in `tracks_master.csv`
- every `file_id` unique in `files_master.csv`
- every file row with a `track_id` must reference an existing track
- canonical standard and Camelot keys must correspond (bidirectional validation)
- review queue must contain only unresolved tracks

### Warning checks
- duplicate logical tracks (possible identity resolution failures)
- malformed key strings
- impossible BPM values (<60 or >200)
- impossible duration values (<0 or >600)
- broken payload references
- missing ISRC on tracks where Songstats ingestion was attempted

---

## 32. Testing requirements

### Unit tests
- key normalization (standard -> Camelot, Camelot -> standard, enharmonic handling)
- text normalization (artist, title, mix, featuring syntax, unicode)
- identity matching cascade (exact, normalized, fuzzy, unmatched)
- resolver scoring (agreement, conflict, single-source, manual override)
- CSV store load/save/upsert round-trip
- Songstats response field normalization
- Rekordbox XML parsing (key formats, path decoding)
- review queue generation and import

### Integration tests
- synthetic library (AIFF + MP3 files with known tags)
- fixture Rekordbox XML
- mock Songstats API responses
- full resolution pipeline end-to-end
- dry-run sync (verify no files changed)
- incremental rerun (verify caching works)

### Golden test cases
- conflicting Songstats vs Rekordbox key
- conflicting librosa vs Essentia key
- missing tags (no key in file)
- ambiguous remixes (same title, different mix)
- duplicate files (same SHA256, different paths)
- analysis disagreement with low confidence
- manual override active
- ISRC present vs absent
- Rekordbox path with different drive letter

---

## 33. Performance expectations

v1 should comfortably support:
- 5,000-20,000 tracks
- incremental reruns in seconds (only process changes)
- cached analysis (skip SHA256-unchanged files)
- parallel analysis execution (configurable workers)

### Optimization priorities
1. Correctness first
2. Avoid re-hashing unchanged files (check mtime first, hash only if mtime changed)
3. Avoid re-analyzing unchanged files (SHA256 + engine version check)
4. Avoid re-fetching unchanged Songstats data (check existing observations)

---

## 34. Security and safety

- API credentials in environment variables or `.env` (loaded via `python-dotenv`)
- never log secrets
- `--dry-run` for all tag writes (default mode)
- snapshot registry before major mutation
- lock file prevents concurrent runs
- raw payloads are immutable once written

---

## 35. Migration path

v1 uses CSV. The `CsvStore` class encapsulates all storage details. Migration to SQLite or Postgres requires only replacing `store/csv_store.py` with an equivalent implementation. All business logic uses the store interface, never raw CSV operations.

---

## 36. Future expansion hooks

The architecture supports future addition of:
- canonical BPM resolution (add `field_name = "bpm"` observations)
- canonical genre resolution
- mood/energy classification
- release/label reconciliation
- similarity embeddings
- recommendation features (registry feeds dj_grouper)
- duplicate detection by audio fingerprint (SHA256 + chromaprint)
- Spotify API as additional enrichment source (credentials already in `.env`)

---

## 37. Acceptance criteria

v1 is complete when it can:

1. scan `.aiff` and `.mp3` audio files
2. extract file tags including ISRC
3. ingest Rekordbox XML (with path matching and drive remapping)
4. ingest Songstats metadata via ISRC lookup
5. run local key analysis with both librosa and Essentia (when available)
6. resolve canonical keys automatically via weighted scoring
7. always store both standard and Camelot key representations
8. create `review_queue.csv` only for unresolved cases
9. import manual review decisions from edited CSV
10. write canonical key to both KEY tag field and COMMENT tag key portion
11. support dry-run mode for all writes
12. preserve raw payload references
13. generate audit reports explaining resolution outcomes
14. run incrementally (only process changes)
15. handle 5,000-20,000 tracks comfortably

---

## 38. Build order

### Phase 1: Foundation
- `config.py` — RegistryConfig dataclass
- `models.py` — LogicalTrack, FileRecord, SourceObservation dataclasses
- `key_utils.py` — standard <-> Camelot bidirectional, normalization, enharmonic map
- `store/csv_store.py` — atomic CSV I/O, schema validation
- Unit tests for key normalization and CSV round-trips

### Phase 2: File Scanning + Identity
- `adapters/file_scanner.py` — wraps dj_tagger file scanning
- `adapters/tag_extractor.py` — extended tag extraction including ISRC
- `identity/normalize.py` — artist/title/mix text normalization
- `identity/matcher.py` — identity resolution cascade
- Populate `files_master.csv` and initial `tracks_master.csv`
- Unit tests for normalization and matching

### Phase 3: External Ingestion
- `adapters/rekordbox_xml.py` — lxml parser with path matching
- `adapters/songstats.py` — httpx client with ISRC lookup
- Source observation creation
- Unit tests with fixture XML and mock API responses

### Phase 4: Analysis + Resolution
- `adapters/local_analysis.py` — wraps both key engines
- `resolver/key_resolver.py` — weighted scoring, auto-resolution
- `resolver/explanation.py` — reason codes, evidence summary
- `review/queue_builder.py` — generate review_queue.csv
- `review/importer.py` — import edited reviews
- Unit tests for scoring edge cases

### Phase 5: Sync + Pipeline
- `sync/tag_writer.py` — KEY field + COMMENT key update
- Add `update_tag_key()` to `dj_tagger.metadata`
- `pipelines/orchestrator.py` — full pipeline
- `cli.py` — entry point with all subcommands
- Integration tests: end-to-end on synthetic library

### Phase 6: Polish
- `sync/export.py` — audit report generation
- Run logging
- Snapshot-before-mutation
- Lock file concurrency protection

---

## 39. Non-negotiable implementation rules

- no external source may directly overwrite canonical registry fields
- no canonical key may exist without a reproducible resolution path
- all source values must be traceable via `source_observations.csv`
- manual review is used only after automatic evidence is insufficient
- standard <-> Camelot conversion is mandatory whenever either form is present
- all tag writes must support dry-run
- snapshots before mutations
- the registry does NOT create dj-tagger COMMENT tags — it only updates the key portion of existing ones

---

## 40. First implementation milestone

Implement a narrow end-to-end slice on a small library:

- scan 100-300 tracks (AIFF and MP3)
- extract file tags including ISRC
- ingest a Rekordbox XML export
- ingest Songstats metadata for tracks with ISRC
- run librosa key analysis (+ Essentia if installed)
- populate `tracks_master.csv`, `files_master.csv`, `source_observations.csv`
- resolve canonical keys
- generate `review_queue.csv` for unresolved cases
- dry-run tag sync (show what would change)

This validates the full backbone before expanding feature breadth.
