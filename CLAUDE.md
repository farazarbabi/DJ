# CLAUDE.md

Project-specific guidance for AI-assisted development.

## Project Overview

DJ music library toolkit with four CLIs:

- `dj`: unified pipeline
- `dj-tagger`: per-track analysis and tag writing
- `dj-registry`: metadata registry, enrichment, canonical resolution
- `dj-grouper`: grouping and recommendations

The registry is the source of truth for canonical key/BPM and stored `tagger_*` outputs. The unified `dj run` command is the preferred end-to-end entry point.

The `dj` CLI also exposes `dj fetch-missing`, which matches Spotify playlist CSV exports against the library and downloads missing tracks (logic in `src/dj_tools/spotify_fetch.py`). Downloads are routed through ordered sources by `spotify_fetch.acquire_track()`: **Soundeo** first (the user's music-pool subscription, original AIFF — `src/dj_tools/soundeo.py`), then **YouTube** via `yt-dlp` as fallback. Soundeo is used only when `SOUNDEO_USER`/`SOUNDEO_PASS` are set in `.env` and `--format aiff` (the default); absent creds, a login failure, or `--no-soundeo` falls back to YouTube-only — exactly the prior behavior. Per-track routing: search Soundeo (free); if found and the daily quota (resets midnight CET) is not exhausted, download the original AIFF **unmarked** (`Artist - Title.aiff`, treated like a curated original); if found but quota is exhausted, skip with status `quota_skip` to retry tomorrow (not downloaded from YouTube); if not on Soundeo, fall back to YouTube (`[U]`-marked). Because Soundeo files are unmarked, a Soundeo download for a track previously held as `[U]` is superseded and pruned (a post-loop `prune_superseded_downloads` runs for this). The work set also includes upgrade candidates — present tracks held as `[U]` are re-checked against Soundeo and upgraded to AIFF when available (missing tracks get quota priority). Soundeo selection (`SoundeoClient.pick`) is suffix-agnostic and **prefers the Extended Mix over the Original Mix** (then the longest cut): since Spotify titles usually omit `(Original Mix)`/`(Extended Mix)` and those words are tokenizer stopwords, all such versions of a song match a bare Spotify title — duration is intentionally **not** a filter (the Extended cut is legitimately longer), and a genuine remix is excluded by its extra remixer-name tokens unless the Spotify title itself names that remix. Tracks found on neither Soundeo nor YouTube are remembered in `<library>/outputs/fetch/not_found_cache.json` (keyed by Spotify URI) and skipped on later runs to avoid re-searching; `--force-lookup` (on `dj fetch-missing` and `dj run`) retries them all, `dj fetch-missing --forget-cached "<substring>"|all` surgically drops matching entries (honors `--dry-run`), and a track becoming present/downloaded clears its own entry. Only a genuine miss is cached: a transient YouTube download error (e.g. HTTP 403) is a `fail` (retried, never cached), and a track Soundeo *listed* but couldn't download (error/quota) is never cached as not-found (`DownloadOutcome.soundeo_listed`). `download_log.csv` records a `source` column (`soundeo`/`youtube`). The Soundeo client drives the site's AJAX-JSON endpoints (`POST /account/logoreg` login, `GET /search?q=`, two-step `GET /download/<id>/3` → tokenized `dl*.sndstatic.com` CDN URL), pinned from a real HAR capture; endpoint paths and the `1`/`2`/`3` MP3/WAV/AIFF format codes are module constants at the top of `soundeo.py`. The playlists path is optional: when omitted it defaults to `<library>/spotify-playlists` (the conventional location for a library's Exportify CSVs, via `spotify_fetch.default_playlists_dir()`); on `dj run`, passing `--fetch-missing` with no value uses the same default (the flag is `nargs="*"`, so an absent flag skips Phase 0 while a bare flag triggers it with the default dir). Downloads are duration-verified: only YouTube results within `--duration-tolerance` seconds (default 3) of the Spotify track are accepted, retrying up to `--max-attempts` (default 3) before a track is written to `unmatched_report.csv`; previously tool-downloaded files that drift out of tolerance are re-downloaded. Matching ignores `(Original Mix/Version)` and `(Extended Mix/Version)` suffixes (via `version_key`), since the AIFF source tags "Original" where Spotify doesn't and an Extended cut is the preferred DJ version of the same song — so an untagged Spotify title is considered present when the library holds any such variant, and no `[U]` copy is downloaded (a true remix stays a distinct track). Before matching, each run prunes superseded downloads: any `[U]`-marked file whose curated unmarked original — named exactly or with one of those equivalent-version suffixes, in any audio format — now exists is deleted, so adding a curated original later cleans up its `[U]` copy on the next run (respects `--dry-run`). `dj fetch-missing --prune-only --library DIR` runs just this prune and exits — no playlist CSVs, matching, downloads, or playlists (also honors `--dry-run` to report without deleting); it's the download-free way to clean up `[U]` copies after dropping in curated originals. Each download gets descriptive metadata embedded from the playlist row (Title/Artist/Album/Genre/Year/Label, via `dj_tagger.metadata.write_track_metadata`) so Rekordbox shows real tags instead of the filename, and its filename is marked with `[U]` (e.g. `Artist - Title[U].aiff`) so YouTube-sourced/converted files can be told apart from higher-quality originally-AIFF tracks — the marker is filename-only and ignored by the matcher. After downloading, each run also writes one representative Rekordbox playlist per CSV: `spotify_fetch.generate_spotify_playlists()` re-scans the library and emits `outputs/playlists/spotify/<csv-name>.m3u8` (UTF-8-sig BOM, absolute paths) listing — **newest-added first** (by the Exportify `Added At` column, ISO-8601 descending; ties keep CSV order, undated rows sort last) — the library tracks that resolve to that playlist's tracks (already-present, an Original/Extended variant, or just-downloaded `[U]` files; unresolvable tracks are skipped and logged). Each resolved library file appears **at most once per playlist** (deduped by absolute path, `normcase`), so duplicate CSV rows or two rows resolving to the same file don't produce duplicate entries. These sit alongside the grouper's `outputs/playlists/` so they import into Rekordbox together. The same step runs as an opt-in Phase 0 of `dj run` via `--fetch-missing` (both share `spotify_fetch.fetch_missing()`), downloading into the run path before analysis so new tracks are tagged and grouped in the same run.

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

# Fetch tracks from Spotify playlist CSVs that aren't in the library yet
dj fetch-missing playlist.csv --library "D:\\Music"
dj fetch-missing ./playlists --library "D:\\Music" --dry-run
dj fetch-missing playlist.csv --library "D:\\Music" --format wav
# Omit the playlists path to use the default <library>/spotify-playlists
dj fetch-missing --library "D:\\Music"
# Soundeo (original AIFF) is preferred when SOUNDEO_USER/SOUNDEO_PASS are in .env;
# force YouTube-only with --no-soundeo
dj fetch-missing --library "D:\\Music" --no-soundeo
# Delete superseded [U] downloads whose curated original now exists (no downloads)
dj fetch-missing --prune-only --library "D:\\Music"
dj fetch-missing --prune-only --library "D:\\Music" --dry-run
# Drop entries from the not-found cache so they're re-searched (or 'all' to clear)
dj fetch-missing --forget-cached "Malevolence" "Kyle Watson - Moments" --library "D:\\Music"
dj fetch-missing --forget-cached all --library "D:\\Music"
# Or fold it into the pipeline (downloads into the run path before analysis)
dj run "D:\\Music" --fetch-missing ./playlists
dj run "D:\\Music" --fetch-missing          # bare flag -> <library>/spotify-playlists

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
KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]
```

Example:

```text
7A|E2|SOUL|FVOC|ORG.HOUS.BUIL|G015
```

`CATEGORY` is the compact no-space DJ taxonomy category label code: each label
word becomes 3-4 uppercase characters separated by dots.

`structure` remains an internal analysis/grouping field, but it is not part of
the COMMENT tag format.

BPM is intentionally omitted from the tag for now but its wiring through
`format_tag` is preserved so it can be reintroduced later. Canonical BPM lives
in the registry (`canonical_bpm` / `tagger_bpm`) regardless.

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
- source payload layers such as `tag`, `songstats`, `songstats_lookup`, `spotify`, `rekordbox`

Relevant derived layer:

- `tagger`

Important current behavior:

- raw/data-collection layers are identity-keyed only
- track-scoped raw keys use full filename + rounded duration + layer
- API raw keys use ISRC + layer
- cached raw layers are reused across downstream tag/category/grouping/signature changes
- tagger results are hydrated with provenance metadata
- registry and grouper reuse the same canonical tagger/raw cache
- grouper preserves richer Songstats-aware tagger entries instead of downgrading them
- grouper uses a lightweight DSP-only path only when tagger analysis is cached but DSP is missing

## Signatures and Derived Refresh

Do not rely on manual version bumps.

`src/dj_tagger/settings.py` computes:

- `raw_version()`
- `dsp_version()`
- `section_dsp_version()`
- `raw_analysis_version()`
- `derived_version()`
- `key_version()`
- `tagger_version()`

These signatures hash both:

- relevant sections of `settings.toml`
- relevant source files listed in:
  - `_DSP_VERSION_FILES`
  - `_SECTION_DSP_VERSION_FILES`
  - `_RAW_ANALYSIS_VERSION_FILES`
  - `_DERIVED_VERSION_FILES`
  - `_KEY_VERSION_FILES`

`_RAW_VERSION_FILES` remains as aggregate tagger provenance metadata. Raw/data
collection cache hits do not depend on these signatures.

Hydrated tagger records carry:

- `_tagger_version`
- `_tagger_raw_sig`
- `_tagger_derived_sig`
- `_tagger_key_sig`
- `_tagger_audio_features_sig`

If you add a brand-new Python file that affects tagger computation, add it to
the appropriate signature list. Otherwise changing that file later may not
refresh derived tagger metadata.

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
- raw/data-collection entries are reused by filename + duration or ISRC identity
- DSP/raw layers are restamped or used to re-derive tagger data without recollection
- DSP is not re-extracted for tag formatting, category label, grouping, derived-scorer, or signature changes
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

Playlist output resolution (both `dj run` and `dj-grouper run`):

- coarse, half-resolution playlists (`groups_coarse/`, `by_key_coarse/`, `by_bpm_coarse/`, `by_subgenre_coarse/`) are written by default
- `--fine-playlists` additionally writes the full-resolution `groups/`, `by_key/`, `by_subgenre/`
- `generate_categorical_playlists(..., fine=, coarse=)` gates the two resolutions independently (function defaults stay `fine=True, coarse=False`; the CLI passes `fine=<flag>, coarse=True`)

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

- current suite size: `555` tests collected by `pytest --collect-only -q`
- tests use synthetic audio fixtures
- registry, tagger, grouper, and cache behaviors all have direct coverage

## Documentation References

- `README.md`
- `docs/technical_spec.md`
- `docs/dj_grouping_recommendation_system_spec.md`
- `docs/tagger_cache_and_experimentation.md`
- `specs/track_registery.md`
