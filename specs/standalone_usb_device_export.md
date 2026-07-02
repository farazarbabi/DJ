# Spec 2 — Standalone USB device export (LATER)

Status: **later / research spike — not near-term work**
Related: [`rekordbox_xml_collection_export.md`](rekordbox_xml_collection_export.md) (implement now)

## Context

A CDJ reads a USB only via Pioneer's proprietary device database — either **Device Library**
(`/PIONEER/rekordbox/export.pdb` DeviceSQL + `ANLZ*.DAT/.EXT/.2EX` beatgrid/waveform files)
or newer **OneLibrary** (`exportLibrary.db`, encrypted SQLite). No mature library writes a
complete, reliable one from scratch (`rex` has no beatgrids/waveforms/cues and warns against
live use; `pyrekordbox`/`rbox` only read ANLZ), and the toolkit computes no waveforms and no
persisted beatgrid.

Spec 1 (Rekordbox XML bridge) is the reliable path and is implemented independently. This
spec is the ambitious "zero Rekordbox" goal, kept fully separate.

## Goal

Write a CDJ-readable USB with **zero Rekordbox**. Experimental. Kept fully isolated (new
`src/dj_tools/usb_export/` package behind an experimental flag) so it never entangles with
Spec 1.

## Scope & unknowns

- **Format A — Device Library** (`export.pdb` DeviceSQL + `ANLZ*.DAT/.EXT/.2EX`): for
  CDJ-2000NXS2/XDJ-1000, CDJ-3000/XDJ-XZ/RX3. References: `rekordcrate`, `rex` (Go),
  `pyrekordbox`, `rbox`. Hard part: generating a **full beatgrid** and **waveform** data the
  toolkit does not compute today (color preview + detail waveforms).
- **Format B — OneLibrary** (`exportLibrary.db`, encrypted SQLite): for
  CDJ-3000X/OPUS-QUAD/XDJ-AZ. Note SQLCipher key/obfuscation research (0xdevalias gist).
- **DeviceSQL is not fully documented** — real risk of "rekordbox database not found" or
  silent re-analysis on hardware.

## Plan of attack (spike)

1. Add waveform + full-beatgrid computation to the tagger (or a dedicated analyzer) — the
   prerequisite that Spec 1 does not need.
2. Prototype writing `export.pdb` + ANLZ for ~5 tracks using the reference libs.
3. Test on the user's actual gear (both a Device Library player and a OneLibrary player).
4. **Go/no-go gate:** only promote to a real feature if waveforms/beatgrids render and the
   player reads it cleanly. Otherwise document findings and stop; Spec 1 remains the
   reliable path.

## Critical files (when/if pursued)

- `src/dj_tools/usb_export/` — **new** isolated package (pdb writer, ANLZ writer, OneLibrary)
- `src/dj_tagger/` — new waveform + full-beatgrid analyzer feeding the ANLZ writer
- `src/dj_tools/cli.py` — experimental `dj export-usb` subcommand behind a flag

## Reference material

- Deep Symmetry — DJ Link Ecosystem Analysis (ANLZ + PDB formats):
  https://djl-analysis.deepsymmetry.org/rekordbox-export-analysis/anlz.html
- `rekordcrate` (Rust parser): https://github.com/Holzhaus/rekordcrate
- `rex` (Go exporter, Mixxx → PDB): https://github.com/kimtore/rex
- `pyrekordbox`: https://github.com/dylanljones/pyrekordbox
- `rbox` (Rust + Python, read/write XML + ANLZ): https://pypi.org/project/rbox/
- OneLibrary / exportLibrary.db SQLCipher notes:
  https://gist.github.com/0xdevalias/b803476793b56f7c45e6361799168eb0
