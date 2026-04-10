# DJ Track Grouping and Recommendation System

## Overview

A local system that organizes a DJ music library into similarity-based groups and generates per-track recommendations. Built for a Windows/Rekordbox workflow targeting techno, minimal tech, deep house, and downtempo.

Builds on `dj-tagger` (per-track audio analysis and tagging). Reads from `./files`, writes all output to `./outputs`.

## Tag Format

```
E# | KEY | BPM | STRUCT | VIBE | VOC | GID
```

Example: `E3 | 9A | 126 | 64H | HYPN | NV | G017`

Only the main group ID is written into file metadata. Recommendations and detailed data live in CSV files under `./outputs`.

## Feature Architecture

Three layers, computed independently, combined via blended distance:

1. **Tags** (~19 dims) — encoded from existing dj-tagger output: energy (ordinal), BPM (continuous, non-linear), key (circular sin/cos on Camelot wheel), structure (ordinal + one-hot), vibe (one-hot), vocal (binary)

2. **DSP** (~45 dims raw, PCA-reduced to 18) — extracted from audio: onset density, beat strength, perc/harmonic ratio, spectral centroid/rolloff/flatness/bandwidth, MFCCs, RMS, low-freq ratio, energy trend, flux, chroma strength/variance, tonal stability. Z-score normalized, then PCA-reduced to 18 dims to prevent MFCC dominance.

3. **CLAP** (optional, 64 dims after PCA) — perceptual audio embeddings via CLAP model, PCA-reduced

## Distance

```
d = w_tags * d_tags + w_dsp * d_dsp + w_embed * d_embed
```

Weights: 0.25 / 0.30 / 0.45 (with CLAP), 0.40 / 0.60 (without CLAP).

All tag sub-distances normalized to [0, 1] before weighting:
- Energy: 0.30 (strongest differentiator)
- Vibe: 0.25
- BPM: 0.20 (non-linear: `d ** 1.5` to amplify genre-boundary gaps)
- Key: 0.10 * vibe-conditional weight (MEL 0.8 down to RAW 0.1, +0.2 for vocals)
- Flow type: 0.05
- Intro bars: 0.05
- Vocal: 0.05

## Caching

Feature extraction is cached in `outputs/features_cache.pkl`. Subsequent runs load from cache and skip extraction. Use `--force-extract` to regenerate. Clustering and recommendations always recompute.

## Clustering

1. Split by vocal presence (V / NV)
2. Agglomerative clustering with `complete` linkage (conservative — won't merge unless all cross-pairs are close)
3. Auto-threshold to target group sizes of 2–10
4. Post-process: merge <2, bisect >20
5. Assign stable group IDs (G001–G999)
6. Compute medoid (most central real track) per group

## Stable Mode

New tracks assigned to nearest existing medoid. If too distant (>0.8), a new group is created.

## Recommendations

Single similarity score per track pair:

```
score = (1 - blended_distance) - key_penalty + structure_compatibility
```

Hard filters: BPM within 4%, same vocal class.

Top 10 recommendations per track. Structure compatibility via flow type pairing matrix and intro bar matching.

## Feedback

Three types in `outputs/feedback.csv`:
- **good_pair**: reduce pairwise distance by 30%
- **bad_pair**: increase pairwise distance by 50%
- **group_override**: force assignment to specific group

## Output

| Output | Location |
|--------|----------|
| Group assignments | `outputs/groups.csv` |
| Recommendations | `outputs/recommendations.csv` |
| Feature cache | `outputs/features_cache.pkl` |
| Group folders | `outputs/Grouped/G001_E3HYP_126_64H/` |
| Group playlists | `outputs/playlists/groups/*.m3u8` |
| Rec playlists | `outputs/playlists/recommendations/*.m3u8` |

Group folders contain hard-linked files (zero extra space on NTFS).

## Folder Naming

```
G017_E3HYP_126_64H
```

Group ID + dominant energy + vibe abbreviation + median BPM + dominant structure.

## CLI

```bash
# Full pipeline (loads from cache if available)
dj-grouper --write-tags

# Force re-extraction
dj-grouper --write-tags --force-extract

# Step by step
dj-grouper extract              # skips if cache exists
dj-grouper extract --force      # regenerate cache
dj-grouper cluster
dj-grouper review
dj-grouper apply --write-tags
dj-grouper recommend

# Feedback
dj-grouper feedback --good "track_a.aiff" "track_b.aiff"
```

All commands read from `./files` and write to `./outputs` by default.

## Review Mode

Interactive text-based review before writing. Shows proposed groups with tracks, medoids, and descriptors. Supports move and merge overrides.
