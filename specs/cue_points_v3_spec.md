# Cue Points V3 Spec

## Summary

Cue Points V3 turns the V2 cue workflow from deterministic structural guesses into a safer, more accurate, and more operationally useful cue system. V3 keeps the XML-first safety model, validates the V2 memory/loop XML behavior against Rekordbox, improves beat/downbeat/phrase accuracy, adds cue quality reporting, and introduces configurable cue profiles.

Direct mutation of a user's live Rekordbox library remains out of scope unless the implementation can prove backup, dry-run, and rollback behavior.

## Current Baseline

V1 provides:

- Three generated hot cues: `MIX IN`, `DROP 1`, `MIX OUT`.
- XML export into a copied Rekordbox XML file.
- Conflict preservation for existing hot cues.

V2 adds:

- Memory cues: `INTRO START`, `BREAKDOWN`, `PEAK`, `OUTRO START`.
- Loop cues: `INTRO LOOP`, `OUTRO LOOP`.
- `--profile v2`, `--include-memory`, `--include-loops`, and `--loop-bars`.
- Conservative XML export for memory cues and loops:
  - Memory: `POSITION_MARK Type="0" Num="-1" Start="..."`
  - Loop: `POSITION_MARK Type="4" Num="-1" Start="..." End="..."`

V2 limitation: local Rekordbox XML fixtures did not include native memory-cue or loop examples. V2 XML is syntactically tested but still needs real Rekordbox import/export verification.

## V3 Goals

V3 should deliver:

- Verified Rekordbox memory/loop XML behavior.
- Better downbeat, phrase, and section placement.
- More accurate cue confidence and review priority.
- User-configurable cue profiles.
- Cue quality reports for library-scale review.
- Safer overwrite/update workflows for generated cues.
- Optional USB/PDB investigation, but not required for the first V3 implementation.

## Priority 1: Rekordbox XML Verification

Before changing cue generation, validate the actual Rekordbox marker formats.

Required workflow:

1. Create a small Rekordbox test library or copied XML export.
2. Manually add in Rekordbox:
   - one memory cue,
   - one active loop,
   - one saved memory loop if Rekordbox distinguishes it,
   - one hot cue with loop behavior if supported.
3. Export XML from Rekordbox.
4. Compare exported `POSITION_MARK` rows against V2 output.
5. Save anonymized fixtures under `tests/fixtures/rekordbox/`.

Acceptance:

- Tests prove memory cue XML round-trips.
- Tests prove loop XML round-trips.
- Exporter uses only fixture-validated attributes.
- V2 assumptions are either confirmed or corrected.
- `specs/cue_points_v2_spec.md` is updated if the V2 XML convention was wrong.

## Priority 2: Better Musical Timing

V3 should improve placement precision without making the feature fragile.

Required improvements:

- Add downbeat-aware bar alignment where analysis data supports it.
- Detect and correct likely one-beat or one-bar offsets.
- Prefer phrase starts over raw section starts.
- Track whether each cue was placed from:
  - verified downbeat,
  - estimated bar grid,
  - section boundary,
  - energy fallback,
  - duration fallback.

Implementation options:

- Use existing `dj_tagger` outputs first.
- Add a small optional analysis adapter only if existing data cannot provide usable downbeat or phrase estimates.
- Keep deterministic fallbacks for tracks where advanced analysis fails.

Acceptance:

- Existing V1/V2 tests continue to pass.
- New synthetic tests cover offset correction.
- New fixture tests cover at least three real tracks with known good cue placements.
- Low-confidence tracks are marked for review rather than silently treated as high confidence.

## Priority 3: Cue Quality Report

Add a report that helps review generated cues before export.

Report output:

- `outputs/registry/reports/cue_quality_report.csv`
- optional human-readable summary printed by CLI.

Required columns:

- `track_id`
- `file_id`
- `file_name`
- `cue_id`
- `cue_kind`
- `cue_role`
- `cue_time_sec`
- `cue_end_sec`
- `cue_bar_index`
- `confidence`
- `manual_review_required`
- `selection_reason`
- `quality_flags`

Quality flags:

- `low_confidence`
- `fallback_used`
- `missing_section`
- `short_loop`
- `loop_near_track_end`
- `cue_order_adjusted`
- `possible_grid_offset`
- `export_conflict`

Acceptance:

- `dj-registry cues report-quality --registry ...` writes the CSV.
- Report can run without a Rekordbox XML file.
- Tracks requiring review are easy to sort by confidence and flags.

## Priority 4: Configurable Cue Profiles

V3 should allow cue templates without hardcoding new CLI flags for every variant.

Profile format:

- YAML or JSON file under `configs/cue_profiles/`.
- Built-in profiles remain available by name: `v1`, `v2`, and `v3-default`.
- User profile path can be passed with `--profile-file`.

Profile settings:

- enabled cue roles,
- hot cue slot mapping,
- memory cue roles,
- loop roles,
- loop length in bars,
- minimum confidence threshold,
- preserve/overwrite policy,
- optional genre or energy hints.

Example roles:

- `mix_in`
- `drop_1`
- `breakdown`
- `peak`
- `mix_out`
- `intro_loop`
- `outro_loop`
- `vocal_in`
- `vocal_out`

Acceptance:

- Built-in `v1` and `v2` behavior is unchanged.
- `v3-default` is documented and tested.
- Invalid profile files produce clear validation errors.
- Profile-generated cues remain auditable in `selection_reason`.

## Priority 5: Safer Update And Overwrite Policies

V1/V2 preserve existing Rekordbox markers and only replace generated `auto_*` rows during forced analysis. V3 should add explicit policies for updating generated cues.

Policies:

- `preserve`: current behavior; never overwrite existing Rekordbox markers.
- `replace-generated`: replace markers previously generated by this tool if identifiable.
- `replace-empty-slot`: only write hot cues into empty slots.
- `review-only`: generate CSV/report but do not export markers.

Requirements:

- Default remains `preserve`.
- Any destructive or overwrite-like behavior requires an explicit flag.
- Export report must identify what would be replaced in dry-run mode.

Acceptance:

- Dry-run tests cover every policy.
- Export never removes user-created markers unless the policy explicitly allows it and tests prove the marker was generated by this tool.

## Priority 6: Optional Rekordbox Database Or USB Investigation

This is exploratory for V3 and can be deferred if XML validation and quality work take priority.

Investigate:

- Rekordbox SQLite/library database schema.
- USB `export.pdb` and related Pioneer files.
- Whether cue writes are feasible without corrupting user libraries.

Required safety gate:

- No write support until backup, dry-run, fixture, and rollback behavior are implemented.
- No write support against a live user library by default.

Acceptance for investigation only:

- Read-only parser or documented findings.
- Clear recommendation: implement later, do not implement, or keep XML-only.

## CLI Changes

Add:

```powershell
dj-registry cues validate-rekordbox-xml --input-xml fixtures.xml
```

```powershell
dj-registry cues report-quality --registry files/outputs/registry
```

Extend analyze:

```powershell
dj-registry cues analyze --rekordbox-xml files/files_rekordbox_export.xml --registry files/outputs/registry --profile v3-default
```

```powershell
dj-registry cues analyze --rekordbox-xml files/files_rekordbox_export.xml --registry files/outputs/registry --profile-file configs/cue_profiles/custom.json
```

Extend export:

```powershell
dj-registry cues export-rekordbox --input-xml in.xml --output-xml out.xml --registry files/outputs/registry --policy preserve
```

## Data Model

Try to reuse `CuePoint` first.

Add fields only if required for auditability:

- `profile_name`
- `profile_version`
- `quality_flags`
- `grid_confidence`
- `section_confidence`
- `generated_marker_id`

If adding fields, CSV loading must remain backward compatible with existing V1/V2 cue CSVs.

## Tests

Required test groups:

- V1/V2 compatibility tests.
- Rekordbox XML fixture tests for memory and loop markers.
- Profile validation tests.
- V3 cue selection tests using synthetic grids.
- Real fixture tests for representative tracks.
- Quality report tests.
- Export policy dry-run tests.

Acceptance:

- Existing cue tests pass unchanged or with only additive assertions.
- New V3 tests do not require access to the user's full local music library.
- Real-track fixtures are small enough to keep in the repo or are represented by stored analysis payloads.

## Non-Goals

Do not implement in the first V3 cut:

- Fully automatic bulk mutation of a live Rekordbox library.
- Unreviewed destructive overwrites.
- Cloud sync with Rekordbox.
- A GUI cue editor.
- Model training on private user libraries without an explicit dataset plan.
- Complex genre-specific cue templates beyond simple profile hints.

## Suggested Execution Order

1. Build Rekordbox memory/loop XML fixtures and validation tests.
2. Correct exporter syntax if fixture validation finds mismatches.
3. Add cue quality reporting.
4. Add `v3-default` selector improvements with better timing confidence.
5. Add profile-file support.
6. Add dry-run export policies.
7. Decide whether USB/PDB or database work belongs in V3 or V4.
