# Cue Points V1 Spec

## Summary

Cue Points V1 generates a small, reviewable set of Rekordbox hot cues for tracks that already exist in the registry and in a supplied Rekordbox XML export.

The implementation is intentionally deterministic and narrow:

- Three generated hot cues per matched track: `MIX IN`, `DROP 1`, `MIX OUT`.
- Rekordbox XML export only; no direct Rekordbox database writes.
- Existing Rekordbox cues are preserved by default.
- The cue/export layer supports hot cue slots A-H, but the default generator writes A-C.
- No memory cues, loops, new MIR dependencies, or ML cue model in V1.

## Commands

Analyze tracks from a Rekordbox XML export and write cue rows:

```powershell
dj-registry cues analyze --rekordbox-xml files/files_rekordbox_export.xml --registry files/outputs/registry
```

Export generated cue rows into a copied Rekordbox XML file:

```powershell
dj-registry cues export-rekordbox --input-xml files/files_rekordbox_export.xml --output-xml files/files_rekordbox_export_with_cues.xml --registry files/outputs/registry
```

Useful analysis options:

- `--limit N` processes only the first N matched XML tracks.
- `--force` replaces existing `auto_v1` cue rows for matched files.
- `--output` is accepted as an alias for `--registry`.

## Data Model

Generated cues are stored in:

```text
outputs/registry/cue_points_master.csv
```

The CSV uses the `CuePoint` model in `dj_registry.models` and stores:

- Registry identity: `track_id`, `file_id`
- Cue identity: `cue_id`, `cue_kind`, `cue_role`, `cue_slot`, `cue_name`
- Cue position: `cue_time_sec`, `cue_end_sec`, `cue_bar_index`, `cue_beat_index`
- Rekordbox export fields: `rekordbox_num`, `rekordbox_type`, `red`, `green`, `blue`
- Review/export metadata: `score`, `confidence`, `selection_reason`, `manual_review_required`, `export_status`, `export_message`

The default profile maps:

| Slot | Rekordbox Num | Cue Name |
| --- | ---: | --- |
| A | 0 | MIX IN |
| B | 1 | DROP 1 |
| C | 2 | MIX OUT |

Slots A-H map to Rekordbox `Num=0..7`.

## Cue Selection

The selector lives in `dj_registry.cues.selection`.

It reuses existing tagger primitives:

- `dj_tagger.audio.load_audio_features`
- existing beat frames and tempo
- `dj_tagger.analyzers.sections.analyze_sections`
- `dj_tagger.raw_features.extract_raw_analysis`

V1 approximates bars as groups of four beats and phrases as eight-bar boundaries. It does not perform true downbeat detection.

Selection rules:

- `MIX IN`: prefer the end of an intro section; fallback to groove start or the first sustained-energy phrase.
- `DROP 1`: prefer the first peak section; fallback to the strongest phrase-level energy rise after the intro.
- `MIX OUT`: prefer an outro section; fallback to a late phrase with remaining transition runway.

Each cue gets a confidence score. Low-confidence and fallback cues set `manual_review_required=true`.

## Rekordbox Export

The exporter lives in `dj_registry.cues.export_rekordbox`.

Export behavior:

- Parses the input XML safely, following the existing Rekordbox adapter style.
- Matches XML tracks to registry files by prior Rekordbox observation, path, then filename plus size.
- Adds generated hot cues as `POSITION_MARK` children on matched `TRACK` elements.
- Writes only hot cues with `Type="0"`.
- Skips generated cues when the target track already has the same hot cue `Num`.
- Updates `export_status` and `export_message` in `cue_points_master.csv`.
- Writes a CSV report to `outputs/registry/reports/cue_rekordbox_export_report.csv`.
- Adds/replaces the XML playlist `AI Generated Cues - Review` with tracks that had generated cue activity.

## Test Coverage

V1 includes focused tests for:

- A-H slot mapping.
- Default A-C cue selection and phrase alignment.
- `CuePoint` CSV round-trip.
- Rekordbox XML insertion.
- Existing hot cue conflict preservation.
- Existing `Num=5`/slot F preservation.
- Review playlist creation.
- CLI export smoke behavior.

## Known Limits

- Downbeat and phrase alignment are approximate.
- Cue quality depends on existing beat and section detection.
- The analysis command expects registry files to exist locally and be matchable from the XML export.
- Review playlist writing targets the standard Rekordbox XML playlist shape used by XML exports.
- V1 is designed as a review workflow, not a fully automatic library overwrite workflow.
