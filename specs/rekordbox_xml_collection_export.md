# Spec 1 — Rekordbox XML collection export

Status: **implement now**
Related: [`standalone_usb_device_export.md`](standalone_usb_device_export.md) (later, fully separate)

## Context

Today the toolkit emits `.m3u8` playlists under `outputs/playlists/`. To use them on
Pioneer hardware the user must, by hand: import each `.m3u8` into Rekordbox desktop, wait
for Rekordbox to **re-analyze** every track (key/BPM/beatgrid/waveform), arrange the tree,
then **Export to USB** — slow, and it discards the key/BPM/cues the toolkit already computed.

A CDJ reads a USB only via Pioneer's proprietary device database, which no library writes
reliably from scratch. But **Rekordbox XML** is fully writable, and Rekordbox's XML-bridge
import carries over key/BPM/beatgrid/cues **without re-analysis**, then does its own
reliable USB export. This spec generates that XML.

## Goal

Emit a single `collection.xml` (Rekordbox XML) from the registry containing **every
library track** (key/BPM/beatgrid/cues) **plus the full playlist tree** (categorical,
grouper, and Spotify playlists as folders). The user's manual work collapses to: point the
Rekordbox *rekordbox xml* bridge at this file once → drag the tree into the collection
(imports pre-analyzed, **no re-analysis**) → Export to USB.

## New module: `src/dj_registry/sync/rekordbox_export.py`

`generate_rekordbox_collection(tracks, files, cue_points, out_path, *, playlist_tree)`
builds the XML with `lxml.etree` and writes UTF-8 with XML declaration (mirror the write
call at `src/dj_registry/cues/export_rekordbox.py:132`).

XML shape (attribute names verified against the ingest parser at
`src/dj_registry/adapters/rekordbox_xml.py:95-111`):

- `<DJ_PLAYLISTS Version="1.0.0">` → `<COLLECTION Entries="N">` → one `<TRACK>` per file:
  - `TrackID` (stable int per track — counter keyed by `track_id`), `Location`
    (`file://localhost/` + URL-encoded abs path — inverse of `_decode_location`), `Name`,
    `Artist`, `Album`, `Genre`, `Label`, `TotalTime` (sec), `Year`, `AverageBpm`
    (`canonical_bpm`), `Tonality` (`canonical_key_*`).
  - **Beatgrid:** one `<TEMPO Inizio="{first_beat_sec}" Bpm="{bpm}" Metro="4/4" Battito="1"/>`
    (constant-tempo grid — correct for four-on-the-floor; multi-TEMPO tempo changes are
    out of scope).
  - **Cues:** one `<POSITION_MARK .../>` per cue — **reuse `_append_position_mark`**
    (`src/dj_registry/cues/export_rekordbox.py:280`), already emitting
    Name/Type/Start/Num/End/RGB for hot/memory/loop.
- `<PLAYLISTS>` → `<NODE Type="0" Name="ROOT">` with folder NODEs (`Type="0"`) and playlist
  NODEs (`Type="1" KeyType="0"`) whose `<TRACK Key="{TrackID}"/>` entries reference the
  collection (pattern at `src/dj_registry/cues/export_rekordbox.py:319-348`).

## Reuse the existing playlist bucketing (don't duplicate)

`sync/playlists.py` couples *which tracks go in which bucket* with *m3u8 writing*. Refactor
the bucketers (`_write_by_key`, `_write_by_key_coarse`, `_write_by_bpm_coarse`,
`_write_by_subgenre`, `_write_by_subgenre_coarse`) to expose a pure
`{playlist_name: [track_id, ...]}` mapping (e.g. `_bucket_by_key(...)`), consumed by both
the m3u8 path and the new XML path. Same for `generate_spotify_playlists()` in
`src/dj_tools/spotify_fetch.py` and the grouper group playlists — build the XML
`<PLAYLISTS>` tree from the same maps so the tree mirrors `outputs/playlists/` exactly
(folders: `by_key`, `by_bpm_coarse`, `by_subgenre`, `groups`, `spotify`).

## Beatgrid input: persist a first-beat offset

The registry has BPM but no beat-1 offset (`Inizio`):
1. **Persist `first_beat_sec`** from the tagger's Librosa `beat_frames[0]` (already computed
   in-memory in `compute_tagger_artifacts()`, `src/dj_tagger/raw_features.py`): add it to
   the raw-analysis output and to `LogicalTrack`/`tracks_master.csv` (follow an existing
   `tagger_*` field; if the new code affects tagger computation, add the file to the
   relevant signature list per CLAUDE.md).
2. **Fallback `Inizio="0.000"`** when unavailable. Cross-check against cue
   `cue_time_sec`/`bar_index`/`beat_index` where cues exist.

## CLI wiring (`src/dj_tools/cli.py`)

- `dj run` writes `<playlists_root>/collection.xml` **by default** (Phase 9, after
  categorical/grouper/Spotify playlists). `--rekordbox-collection PATH` overrides the output
  path; `--no-rekordbox-collection` skips it.
- New `dj export-rekordbox` subcommand to regenerate the XML from an existing registry
  without a full run.

## Tests (mirror existing registry tests)

(a) Round-trip: feed the generated XML through `ingest_rekordbox()`, assert tracks/keys/BPM
match. (b) `<PLAYLISTS>` tree matches the bucket maps. (c) TEMPO/POSITION_MARK attributes
present and well-formed. Run `pytest -q` (currently 555 tests).

## Verification (end-to-end)

1. `pytest -q` green.
2. `dj run "D:\Music" --rekordbox-collection` → `outputs/playlists/collection.xml` written,
   parses, TRACK count == library size.
3. Rekordbox: Preferences → Advanced → Database → *rekordbox xml* → point at file; imported
   tree shows key/BPM populated, imports **without re-analysis**; drag to collection; Export
   to USB.
4. USB into CDJ-3000/XDJ-XZ: playlists browse; beatgrid/cues present.

## Critical files

- `src/dj_registry/sync/rekordbox_export.py` — **new**, XML collection writer
- `src/dj_registry/sync/playlists.py` — refactor bucketers to expose track-id maps
- `src/dj_registry/cues/export_rekordbox.py` — reuse `_append_position_mark`, NODE patterns
- `src/dj_registry/adapters/rekordbox_xml.py` — authoritative TRACK attribute names (invert)
- `src/dj_tools/spotify_fetch.py` / grouper CLI — expose playlist bucket maps
- `src/dj_tagger/raw_features.py` + `src/dj_registry/models.py` — persist `first_beat_sec`
- `src/dj_tools/cli.py` — `--rekordbox-collection` flag + `export-rekordbox` subcommand
