# DJ Tools

Local Python toolkit for DJ music library management. Two CLI tools:

- **dj-tagger** -- Analyze audio files and write structured tags into metadata
- **dj-grouper** -- Group tracks by similarity and generate per-track recommendations

Built for techno, minimal tech, deep house, and downtempo. Windows-first, Rekordbox-compatible.

---

## Tag Format

```
E# | KEY | BPM | STRUCT | VIBE | VOC
```

With grouping:

```
E# | KEY | BPM | STRUCT | VIBE | VOC | GID
```

| Field | Values |
|-------|--------|
| E# | E1 (ambient) -- E5 (aggressive) |
| KEY | 1A--12A, 1B--12B (Camelot) |
| BPM | Integer tempo |
| STRUCT | 16/32/64 + G/H/D/B/L |
| VIBE | HYPN, DRK, RAW, DEEP, TRIB, MEL, ACID, ATM |
| VOC | V or NV |
| GID | G001--G999 |

**Examples:**
```
E3 | 9A | 126 | 64H | HYPN | NV
E4 | 5A | 130 | 32D | RAW | V
E2 | 8A | 122 | 64L | DEEP | NV | G003
```

---

## Installation

```bash
cd DJ
pip install -e ".[dev]"
```

Optional:
```bash
pip install -e ".[essentia]"   # better key detection
pip install -e ".[clap]"       # CLAP embeddings for grouping (requires PyTorch)
```

Requires Python 3.10+. Windows (primary), macOS/Linux supported.

---

## Directory Layout

```
DJ/
+-- files/              # put your music here (default input)
+-- outputs/            # all generated files go here
|   +-- features_cache.pkl
|   +-- groups.csv
|   +-- recommendations.csv
|   +-- feedback.csv
|   +-- Grouped/        # group folders with hard-linked tracks
|   |   +-- G001_E3HYP_126_64H/
|   |   +-- G002_E4MEL_130_32G/
|   +-- playlists/
|       +-- groups/     # one .m3u8 per group
|       +-- recommendations/  # one .m3u8 per track
+-- src/
|   +-- dj_tagger/      # audio analysis
|   +-- dj_grouper/     # grouping + recommendations
+-- tests/
```

No generated files land in the project root. Input defaults to `./files`, output defaults to `./outputs`.

---

## dj-tagger

Analyzes audio files and writes per-track tags into the COMMENT metadata field.

### Usage

```bash
# Preview (dry run, default) -- reads from ./files
dj-tagger files -r

# Write tags
dj-tagger files -r --write-tags

# Export to CSV
dj-tagger files -r --csv outputs/results.csv

# JSON output, 4 workers, first 20 files
dj-tagger files -r --json -w 4 --limit 20
```

### Options

```
dj-tagger PATH [PATH ...] [OPTIONS]

  -r, --recursive         Recurse into subdirectories
  --write-tags            Write tags to file metadata
  --overwrite             Overwrite existing comment tags
  -w, --workers N         Parallel workers (default: cpu_count / 2)
  --limit N               Process only first N files
  --csv FILE              Export results to CSV
  --json                  Output as JSON lines
  -v, --verbose           Per-analyzer detail
  --use-essentia          Use Essentia for key detection
```

### Supported Formats

MP3, FLAC, AIFF, WAV, M4A

### Analyzers

| Analyzer | Output | Method |
|----------|--------|--------|
| Energy | E1--E5 + confidence | Weighted composite: RMS, spectral centroid, flux, onset density, low-freq ratio |
| Key | Camelot (e.g., 9A) + confidence | Chroma + Krumhansl-Schmuckler profiles. Optional Essentia EDMA backend |
| BPM | Integer | Librosa beat tracking |
| Structure | e.g., 64H + confidence | Intro bars from onset energy envelope + flow type from energy variance |
| Vibe | Label + continuous scores + confidence | Heuristic scoring from spectral features; all 8 scores preserved |
| Vocals | V/NV + confidence | Multi-stage spectral analysis of harmonic component in 300--3000 Hz band |
| Sections | intro/groove/peak/breakdown | Energy envelope segmentation for section boundary detection |

Every analyzer now reports a confidence value (0.0--1.0). Vibe produces continuous scores for all 8 categories in addition to the winning label. Section detection identifies structural regions (intro, groove, peak, breakdown) used downstream by dj-grouper for section-aware DSP.

### Energy Scale

| Level | Description |
|-------|-------------|
| E1 | Ambient / intro tool / very sparse |
| E2 | Deep groove / low tension |
| E3 | Driving / stable dance floor |
| E4 | Peak-time hypnotic / high density |
| E5 | Aggressive / industrial / hard |

### Flow Types

| Code | Description |
|------|-------------|
| G | Groove-focused, steady rhythm |
| H | Hypnotic, loop-based |
| D | Drop-focused, impact-oriented |
| B | Breakdown-heavy |
| L | Layer tool, minimal percussion |

### Vibe Categories

| Code | Key audio drivers |
|------|-------------------|
| HYPN | Spectral stability, low onset variance, repetitive |
| DRK | Low centroid, high low-freq ratio, high flux |
| RAW | High spectral flatness, high RMS |
| DEEP | High low-freq, low centroid, moderate RMS |
| TRIB | High percussive ratio, high onset density |
| MEL | Chroma variance, tonal content |
| ACID | High centroid variance (filter sweeps), spectral peakiness |
| ATM | Low onset density, high bandwidth, low RMS |

---

## dj-grouper

Groups tracks into similarity-based families and recommends similar tracks for each.

### Quick Start

```bash
# Dry run -- see results without writing anything
dj-grouper run -r --dry-run

# Full run -- everything at once
dj-grouper run -r --write-tags

# Force re-extraction of features
dj-grouper run -r --force-extract
```

That single `run` command:
1. Scans `./files` for audio files
2. Extracts features (incremental: only new/changed files, skips cached ones)
3. Clusters tracks into groups (soft vocal partitioning, agglomerative with average linkage)
4. Infers track roles (TOOL, DRIVER, PEAK, RESET, BREAKDOWN, BRIDGE)
5. Computes top-10 directional recommendations per track
6. Exports `groups.csv` and `recommendations.csv` to `./outputs`
7. Creates group folders with hard-linked files and `_group_info.txt` in `./outputs/Grouped`
8. Writes group IDs into file metadata (if `--write-tags`)
9. Generates `.m3u8` playlists in `./outputs/playlists`

Subsequent runs only extract features for new or changed files (incremental cache). Use `--force-extract` to regenerate everything.

### Caching

Feature extraction is the slow step. Results are cached incrementally in `outputs/features_cache.pkl`.

```bash
# First run: extracts features (~10s per track), saves cache
dj-grouper run -r

# Subsequent runs: only extracts new/changed files, loads rest from cache
dj-grouper run -r

# Force full re-extraction
dj-grouper run -r --force-extract
```

### Step-by-Step

```bash
# Extract features only (incremental by default, --force to regenerate all)
dj-grouper extract -r
dj-grouper extract -r --force

# Cluster and review interactively
dj-grouper cluster
dj-grouper review

# Write output
dj-grouper apply --write-tags

# Compute recommendations
dj-grouper recommend

# Evaluate against known pairs
dj-grouper evaluate --eval-file pairs.csv

# Add feedback
dj-grouper feedback --good "track_a.aiff" "track_b.aiff"
dj-grouper feedback --bad "track_a.aiff" "track_b.aiff"
dj-grouper feedback --override "track.aiff" G017
```

### Subcommands

| Command | Description |
|---------|-------------|
| `run` | Full pipeline in one shot |
| `extract` | Extract features, save to cache (incremental) |
| `cluster` | Cluster tracks into groups |
| `review` | Interactive review of proposed groupings |
| `apply` | Write tags, create folders, generate playlists |
| `recommend` | Compute similar-track recommendations |
| `feedback` | Add good_pair, bad_pair, or group_override |
| `evaluate` | Score system against known good/bad pairs from a CSV |

### Output Files

All generated files live under `./outputs/`:

| File | Description |
|------|-------------|
| `outputs/groups.csv` | One row per track: group ID, folder, tags, medoid flag |
| `outputs/recommendations.csv` | One row per recommendation: source, target, rank, score |
| `outputs/features_cache.pkl` | Cached feature vectors (incremental) |
| `outputs/feedback.csv` | User feedback entries |
| `outputs/Grouped/` | Group folders with hard-linked track files |
| `outputs/playlists/groups/` | One `.m3u8` per group |
| `outputs/playlists/recommendations/` | One `.m3u8` per track |

### Group Folders

```
outputs/Grouped/
  G001_E3HYP_126_64H/
    _group_info.txt
    Brunello - Jester's Privilege.aiff
    Malandra Jr. - Sveva.aiff
  G002_E2DEP_122_64L/
    _group_info.txt
    Frivolous - Back Into The Deep.aiff
```

Folder names encode: group ID, representative energy + vibe, median BPM, representative structure.

Each folder contains a `_group_info.txt` with a summary of the group, its members, and inferred track roles.

Files are **hard-linked** (zero extra disk space on NTFS). Falls back to copy if hard links fail.

### Recommendations

Recommendations are **directional** -- the score from track A to track B differs from B to A. This reflects real DJ mixing: what matters is how well B works as a destination when mixing out of A.

Scoring factors:

| Factor | Description |
|--------|-------------|
| Blended distance | Weighted combination of tag, DSP, and optional CLAP distances |
| Intro usability | How usable the destination track's intro is for mixing in |
| Bass conflict risk | Penalty when both tracks have heavy bass that would clash |
| Groove compatibility | Similarity of section-level DSP features (intro, groove, peak) |
| Energy direction bonus | Reward for musically sensible energy progressions |
| Breakdown risk | Penalty when mixing into a track with an early breakdown |
| Soft BPM penalty | Gradual penalty starting at 4% BPM difference, hard cutoff at 8% |
| Key penalty | Conditional on vibe -- high weight for MEL, low for RAW |
| Structure compatibility | Flow type pairing + intro length matching |

### Track Roles

Each track in a group is assigned an inferred role based on its characteristics:

| Role | Description |
|------|-------------|
| TOOL | Low energy, minimal -- used for layering and transitions |
| DRIVER | Steady mid-energy groove carrier |
| PEAK | High energy, high density -- the climax track |
| RESET | Energy drop after a peak, palette cleanser |
| BREAKDOWN | Breakdown-heavy, creates tension/release |
| BRIDGE | Transitional -- connects different energy levels or vibes |

Roles appear in `_group_info.txt` inside each group folder.

### Evaluation

Test the system against known good and bad pairs:

```bash
dj-grouper evaluate --eval-file pairs.csv
```

The evaluation framework (`evaluation.py`) reads a CSV of known-good and known-bad track pairs and measures how well the recommendation scoring ranks good pairs above bad pairs.

### Feedback

```bash
dj-grouper feedback --good "Malandra Jr. - Sveva.aiff" "Malandra Jr. - Adria.aiff"
dj-grouper feedback --bad "Bob Moses - All I Want.aiff" "HardTechno.aiff"
dj-grouper feedback --override "SomeTrack.aiff" G017
```

| Type | Effect |
|------|--------|
| good_pair | Distance multiplied by `(1 - 0.3)` -- 30% closer |
| bad_pair | Distance multiplied by `(1 + 0.5)` -- 50% farther |
| group_override | Hard assignment to specified group post-clustering |

Feedback is stored in `outputs/feedback.csv` and persists across runs.

---

## Calibration

### dj-tagger -- `src/dj_tagger/constants.py`

| Constant | Default | Tunes |
|----------|---------|-------|
| `ENERGY_WEIGHTS` | rms:0.30, centroid:0.15, flux:0.20, onset:0.20, low_freq:0.15 | Energy feature weighting |
| `ENERGY_THRESHOLDS` | [0.20, 0.40, 0.60, 0.80] | E1--E5 boundaries |
| `INTRO_ENERGY_RATIO` | 0.80 | Intro end detection sensitivity |
| `INTRO_SUSTAIN_BARS` | 8 | Bars above threshold for intro end |
| `VOCAL_ENERGY_RATIO` | 0.15 | Vocal band energy threshold |
| `VOCAL_FLATNESS_MAX` | 0.40 | Spectral flatness threshold |
| `VOCAL_FRAME_THRESHOLD` | 0.08 | Fraction of vocal frames for V |

### dj-grouper -- `src/dj_grouper/config.py`

| Field | Default | Tunes |
|-------|---------|-------|
| `w_tags` / `w_dsp` / `w_embed` | 0.25 / 0.30 / 0.45 | Layer weights (with CLAP) |
| `w_tags_no_embed` / `w_dsp_no_embed` | 0.60 / 0.40 | Layer weights (without CLAP) |
| `key_weight_by_vibe` | MEL:0.8 ... RAW:0.1 | Key importance per vibe |
| `key_weight_vocal_boost` | 0.20 | Extra key weight when vocals present |
| `bpm_soft_penalty_pct` | 0.04 | BPM penalty starts at 4% difference |
| `bpm_hard_cutoff_pct` | 0.08 | BPM hard cutoff at 8% difference |
| `bpm_penalty_weight` | 0.15 | Max BPM penalty contribution |
| `linkage` | "average" | Clustering linkage method |
| `vocal_confidence_threshold` | 0.5 | Only hard-split vocals above this confidence |
| `target_group_size` | (2, 10) | Ideal group size range |
| `min_group_size` / `max_group_size` | 2 / 20 | Hard group size limits |
| `clap_pca_dims` | 64 | PCA dimensions for CLAP embeddings |
| `new_group_distance_threshold` | 0.80 | Distance beyond which a new group is created |
| `n_recommendations` | 10 | Recommendations per track |
| `good_pair_factor` / `bad_pair_factor` | 0.30 / 0.50 | Feedback strength |

---

## Limitations

- **Key detection**: chroma-based, can miss on heavily percussive tracks. Use `--use-essentia` for better accuracy.
- **Vocal detection**: multi-stage spectral heuristics, not ML. Resonant synth leads may false positive.
- **Structure**: assumes 4/4 time.
- **Vibe**: heuristic, consistent but not always matching subjective judgment.
- **Section detection**: energy-envelope based; abrupt style changes within a track may confuse boundaries.
- **Grouping without CLAP**: tags + DSP only gives approximate perceptual similarity. CLAP significantly improves results.
- **Hard links**: require same NTFS volume. Falls back to copy otherwise.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v       # 80 tests
```

---

## Project Structure

```
src/
+-- dj_tagger/                    # Audio analysis and tagging
|   +-- cli.py                    # CLI entry point
|   +-- pipeline.py               # Per-track analysis orchestrator
|   +-- audio.py                  # Audio loading + HPSS + beat tracking
|   +-- constants.py              # All tunable thresholds
|   +-- formats.py                # Tag formatting/parsing (v1 + v2)
|   +-- metadata.py               # Mutagen read/write (MP3/FLAC/AIFF/WAV/M4A)
|   +-- analyzers/
|       +-- energy.py             # E1--E5 composite scoring (with confidence)
|       +-- key.py                # Camelot key detection (with confidence)
|       +-- structure.py          # Intro bars + flow type (with confidence)
|       +-- vibe.py               # 8-category scorer (continuous scores + confidence)
|       +-- vocal.py              # Multi-stage V/NV detection (with confidence)
|       +-- sections.py           # Section detection: intro/groove/peak/breakdown
+-- dj_grouper/                   # Grouping and recommendations
    +-- cli.py                    # CLI: run/extract/cluster/review/apply/recommend/feedback/evaluate
    +-- config.py                 # All weights and thresholds
    +-- scanner.py                # Library scan + tag reading
    +-- review.py                 # Interactive group review
    +-- evaluation.py             # Evaluation framework for known pairs
    +-- features/
    |   +-- dsp.py                # 10 curated DSP features + section-aware extraction
    |   +-- embeddings.py         # CLAP extraction + PCA (optional)
    |   +-- builder.py            # Tag encoding (continuous vibe, confidence-weighted) + cache
    |   +-- role.py               # Track role inference (TOOL/DRIVER/PEAK/RESET/BREAKDOWN/BRIDGE)
    +-- grouping/
    |   +-- distance.py           # 3-layer blended distance
    |   +-- medoid.py             # Medoid computation
    |   +-- clustering.py         # Agglomerative clustering + size constraints
    |   +-- assignment.py         # Stable group IDs + nearest-medoid
    +-- recommend/
    |   +-- scoring.py            # Directional scoring with DJ usability features
    |   +-- transition.py         # Structure compatibility matrix
    |   +-- neighbors.py          # Top-N computation
    +-- output/
    |   +-- folders.py            # Group folders + hard links + _group_info.txt
    |   +-- csv_export.py         # groups.csv, recommendations.csv
    |   +-- playlists.py          # .m3u8 generation
    +-- feedback/
        +-- store.py              # feedback.csv CRUD
        +-- apply.py              # Distance matrix modification
```
