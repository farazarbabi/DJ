# Cue Points V2 Spec

## Summary

Cue Points V2 expands the V1 Rekordbox XML workflow from three hot cues into a fuller structural cue set. V2 adds memory cues and loop cues while preserving the safety constraints of V1: XML export only, no direct Rekordbox database writes, no destructive overwrites by default, and no new heavy MIR or ML dependency.

V2 is the "more cue types" release. Accuracy upgrades, automation policies, and direct Rekordbox integration are deferred to V3.

## V1 Baseline

V1 already provides:

- `dj-registry cues analyze` for tracks present in a Rekordbox XML export.
- `dj-registry cues export-rekordbox` for writing generated cues into a copied XML export.
- `CuePoint` rows in `outputs/registry/cue_points_master.csv`.
- Three hot cues per matched track:
  - `A / Num=0`: `MIX IN`
  - `B / Num=1`: `DROP 1`
  - `C / Num=2`: `MIX OUT`
- Existing hot cue preservation by default.
- A review playlist named `AI Generated Cues - Review`.
- A CSV export report at `outputs/registry/reports/cue_rekordbox_export_report.csv`.

V2 must keep this behavior backward compatible.

## V2 Goals

V2 adds:

- Memory cues for structural navigation:
  - `INTRO START`
  - `BREAKDOWN`
  - `PEAK`
  - `OUTRO START`
- Loop cues:
  - `INTRO LOOP`, default 16 bars.
  - `OUTRO LOOP`, default 16 bars.
- CLI profile control:
  - `--profile v1` keeps current V1 behavior.
  - `--profile v2` generates hot cues, memory cues, and 16-bar loops.
- Optional cue-type flags for targeted runs:
  - `--include-memory`
  - `--include-loops`
  - `--loop-bars N`, default `16` for V2.

## Data Model

Reuse the existing `CuePoint` model and CSV.

Required conventions:

- Hot cues keep `cue_kind="hot"`.
- Memory cues use `cue_kind="memory"`.
- Loop cues use `cue_kind="loop"`.
- Loop cues must set both `cue_time_sec` and `cue_end_sec`.
- Loop cues must set `cue_end_sec > cue_time_sec`.
- Loop cue duration defaults to 16 bars.
- `cue_role` should use stable snake-case values:
  - `intro_start`
  - `breakdown`
  - `peak`
  - `outro_start`
  - `intro_loop`
  - `outro_loop`
- As-built XML conventions:
  - Memory cues use `rekordbox_type="0"` and `rekordbox_num=-1`.
  - Loop cues use `rekordbox_type="4"`, `rekordbox_num=-1`, `cue_time_sec` as `Start`, and `cue_end_sec` as `End`.

No schema migration is required unless fixture validation proves the existing fields cannot represent Rekordbox memory cues or loops.

## Cue Generation Behavior

V2 continues to use the existing V1 analysis inputs:

- Existing beat frames and tempo from `dj_tagger.audio.load_audio_features`.
- Bar energies from `dj_tagger.raw_features.extract_raw_analysis`.
- Coarse sections from `dj_tagger.analyzers.sections.analyze_sections`.
- The existing 4-beat bar and 8-bar phrase approximation.

Memory cue placement:

- `INTRO START`: first usable bar, normally bar 0 or the first detected beat/bar.
- `BREAKDOWN`: first detected breakdown section start; omit if no breakdown is detected.
- `PEAK`: first detected peak section start; fallback to the V1 `DROP 1` bar.
- `OUTRO START`: detected outro section start; fallback to the V1 `MIX OUT` bar.

Loop cue placement:

- `INTRO LOOP`: starts at `MIX IN` or `INTRO START`, whichever is later and has enough runway.
- `OUTRO LOOP`: starts at `OUTRO START` or `MIX OUT`, whichever produces a full 16-bar loop before track end.
- Loop endpoints must snap to bar boundaries.
- If a full 16-bar span is unavailable, generate no loop for that role unless `--loop-bars N` supplies a shorter valid span.
- Any fallback or shortened loop must set `manual_review_required=true`.

## Rekordbox XML Export

V2 extends the existing XML exporter.

Export requirements:

- Preserve existing hot cues, memory cues, and loops by default.
- Insert generated memory cues and loops only when the corresponding Rekordbox marker does not conflict with an existing marker.
- Continue updating `export_status` and `export_message` in `cue_points_master.csv`.
- Continue writing the export report CSV.
- Continue adding generated-cue tracks to `AI Generated Cues - Review`.
- Include skipped memory/loop conflicts in the report.

Implementation note: local Rekordbox XML exports available during V2 implementation contained hot-cue `POSITION_MARK` rows only. No local memory-cue or loop XML examples were found. V2 therefore implements the common XML encoding above and validates that generated XML is well formed and conflict-safe in tests. Before expanding this feature in V3 or using it for bulk library automation, verify these memory/loop markers by importing a copied XML export into Rekordbox.

## CLI Behavior

V2 should keep the existing commands and add options rather than creating new top-level commands.

Analyze examples:

```powershell
dj-registry cues analyze --rekordbox-xml files/files_rekordbox_export.xml --registry files/outputs/registry --profile v2
```

```powershell
dj-registry cues analyze --rekordbox-xml files/files_rekordbox_export.xml --registry files/outputs/registry --include-memory --include-loops --loop-bars 16
```

Export remains:

```powershell
dj-registry cues export-rekordbox --input-xml files/files_rekordbox_export.xml --output-xml files/files_rekordbox_export_with_cues.xml --registry files/outputs/registry
```

Default compatibility:

- If no profile or include flags are passed, keep V1 behavior.
- `--profile v1` must generate exactly the V1 A-C hot cues.
- `--profile v2` must generate hot cues, memory cues, and loops.
- `--loop-bars` only affects loop cues.

## Tests and Acceptance Criteria

Add focused tests for:

- V1 profile compatibility.
- V2 profile cue count and cue kinds.
- Memory cue generation from synthetic section maps.
- Omission of missing breakdown cue when no breakdown section exists.
- 16-bar `INTRO LOOP` and `OUTRO LOOP` generation with correct `cue_end_sec`.
- Loop omission when a full 16-bar span is unavailable.
- `--loop-bars N` override.
- Rekordbox XML export of memory cues and loops using the implemented V2 XML syntax.
- Conflict preservation for generated memory cues and loops.
- Export report rows for inserted, skipped, invalid, and unmatched V2 cue types.

Acceptance:

- Existing V1 cue tests pass unchanged.
- V2 unit tests pass without requiring full audio decoding.
- One smoke run against `files/files_rekordbox_export.xml` succeeds using a temporary registry copy.

## Deferred to V3

Do not implement these in V2:

- True downbeat detection.
- External MIR libraries for section, beat, or phrase detection.
- ML-based cue prediction.
- Direct Rekordbox database writes.
- USB/PDB export.
- Automatic overwrite or slot reassignment policies.
- Large-scale cue quality dashboards.
- Genre-specific cue templates.
- User-authored cue profiles beyond `v1`, `v2`, and `--loop-bars`.
- Full library automation that mutates user Rekordbox data without XML review.

## Implementation Notes

- Keep V2 generation deterministic and auditable.
- Keep XML export preserve-first.
- Treat memory/loop XML fixture validation as the first implementation step.
- Prefer extending `dj_registry.cues.selection` and `dj_registry.cues.export_rekordbox` over adding parallel implementations.
- Keep generated debug payloads under `outputs/registry/raw/cue_analysis/`.
