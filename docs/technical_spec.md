# Technical Specification

## Architecture

Two packages sharing a codebase, installed from a single `pyproject.toml`:

```
dj-tagger  (per-track audio analysis)  ->  dj-grouper (library grouping + recommendations)
```

`dj-grouper` depends on `dj-tagger` for audio loading, feature precomputation, tag parsing, and metadata writing. Both are registered as CLI entry points.

### Default Paths

```
./files/        input audio files
./outputs/      all generated artifacts:
  features_cache.pkl
  groups.csv
  recommendations.csv
  feedback.csv
  Grouped/      group folders with hard-linked files + _group_info.txt
  playlists/    .m3u8 files
```

---

## dj-tagger: Per-Track Analysis

### Data Flow

```
audio file
  -> librosa.load(sr=22050)
  -> HPSS separation (y_harmonic, y_percussive)
  -> beat tracking (tempo, beat_frames)
  -> TrackAudio dataclass
       |
       |-> analyze_energy()     -> EnergyResult(level=1-5, confidence=0.85)
       |-> analyze_key()        -> KeyResult(camelot="9A", confidence=0.80)
       |-> analyze_structure()  -> StructureResult(intro_bars=64, flow_type="H", confidence=0.75)
       |-> analyze_vibe()       -> VibeResult(label="HYPN", scores={...}, confidence=0.70)
       |-> analyze_vocal()      -> VocalResult(has_vocals=False, confidence=0.90)
       |-> detect_sections()    -> [Section(type="intro", start=0, end=32), ...]
       |
       v
  format_tag() -> "E3 | 9A | 126 | 64H | HYPN | NV"
       |
       v
  write_tag() via mutagen
```

All analyzers receive the same `TrackAudio`. Each is independently fault-tolerant. Parallelism at the track level via `ProcessPoolExecutor`.

### TrackAudio

```python
@dataclass
class TrackAudio:
    path: str
    y: NDArray             # raw signal
    sr: int                # 22050
    y_harmonic: NDArray    # HPSS harmonic
    y_percussive: NDArray  # HPSS percussive
    tempo: float           # BPM
    beat_frames: NDArray   # beat frame indices
    duration: float        # seconds
```

### Energy (E1--E5)

Weighted composite of five spectral/rhythmic features, each normalized to [0, 1]:

| Feature | Source | Normalization | Weight |
|---------|--------|---------------|--------|
| RMS | `librosa.feature.rms` mean | (x - 0.04) / 0.08 | 0.30 |
| Spectral centroid | `spectral_centroid` mean | (x - 1500) / 2500 | 0.15 |
| Spectral flux | RMS of STFT diffs | (x - 0.5) / 2.5 | 0.20 |
| Onset density | onset count / duration | (x - 1.5) / 4.0 | 0.20 |
| Low-freq ratio | power <150 Hz / total | (x - 0.15) / 0.30 | 0.15 |

Thresholds: E1 < 0.20, E2 < 0.40, E3 < 0.60, E4 < 0.80, E5 >= 0.80.

**Confidence**: derived from distance to nearest threshold boundary. Scores near a boundary get lower confidence; scores firmly within a level get higher confidence.

### Key Detection

**Librosa** (default): `chroma_cqt` on harmonic component -> time-averaged chroma vector -> correlate with Krumhansl-Schmuckler major and minor profiles across all 12 pitch classes -> best correlation wins -> map to Camelot notation.

**Essentia** (optional, `--use-essentia`): `KeyExtractor(profileType='edma')` tuned for electronic music. Generally more accurate on percussive material.

**Confidence**: the correlation coefficient of the winning key profile. Higher correlation = more tonal content = higher confidence.

### Structure

**Intro bars**: onset energy computed per bar (4 beats/bar) -> find first bar exceeding 80% of median energy sustained for 8+ consecutive bars -> snap result to nearest of 16, 32, or 64.

**Flow type**: track divided into 8 equal segments -> compute energy per segment -> derive variance, max jump, trend, plateau count:

| Type | Condition | Priority |
|------|-----------|----------|
| L | variance < 0.005 | 1 (checked first) |
| H | variance < 0.02 AND max_jump < 0.15 | 2 |
| D | max_jump > 0.40 | 3 |
| B | trend > 0.05 AND plateaus >= 2 | 4 |
| G | trend > 0.03 | 5 |
| H | fallback | 6 |

**Confidence**: based on how decisively the conditions are met. Ambiguous tracks (close to multiple thresholds) get lower confidence.

### Vibe

Eight labels scored from spectral features. All scores are computed; the highest wins. Continuous scores (8 floats, one per vibe) are preserved for downstream use.

| Label | Formula |
|-------|---------|
| HYPN | 0.5 * stability + 0.3 * (1 - onset_var) + 0.1 * driving_rhythm + 0.1 * (1 - chroma_var) |
| DRK | 0.30 * (1 - centroid/4k) + 0.30 * low_ratio + 0.25 * flux + 0.15 * rms |
| RAW | 0.25 * flatness + 0.25 * rms + 0.20 * flux + 0.15 * centroid + 0.15 * onset_density |
| DEEP | 0.30 * low_ratio + 0.25 * (1 - centroid/3k) + 0.25 * (1 - rms) + 0.20 * (1 - onset_density) |
| TRIB | 0.40 * perc_ratio + 0.35 * onset_density + 0.25 * (1 - harmonic_e) |
| MEL | 0.30 * (chroma_var * 30) + 0.30 * chroma_strength + 0.20 * (1 - flatness) + 0.20 * (1 - perc_ratio) |
| ACID | 0.40 * centroid_var + 0.35 * peakiness + 0.25 * (1 - chroma_strength) |
| ATM | 0.30 * (1 - onset_density) + 0.30 * bandwidth + 0.25 * (1 - rms) + 0.15 * smoothness |

**Confidence**: gap between the top score and the second-highest score. A large gap means unambiguous classification.

### Vocals

Multi-stage detection:

1. STFT of harmonic component
2. Isolate vocal band (300--3000 Hz)
3. Per-frame: compute energy ratio (vocal band / total) and spectral flatness
4. Frame classified as "vocal-like" if ratio > 0.15 AND flatness < 0.40
5. Track classified as V if > 8% of frames qualify

**Confidence**: derived from how far the vocal frame percentage is from the 8% threshold. A track at 2% or 40% gets high confidence; a track at 7% or 9% gets low confidence.

### Section Detection (sections.py)

Identifies structural regions within a track:

- **intro**: low-energy opening before the main groove establishes
- **groove**: steady-state rhythmic sections
- **peak**: highest-energy sections
- **breakdown**: energy drops within the body of the track

Detection uses the energy envelope (computed per bar from onset strength), segmented by significant energy transitions. Section boundaries are quantized to bar positions.

Output: list of Section objects with type, start bar, and end bar. Used by dj-grouper for section-aware DSP feature extraction.

### Tag Format Versions

```
v1 (legacy):  E# | KEY | STRUCT | VIBE | VOC
v2 (current): E# | KEY | BPM | STRUCT | VIBE | VOC [| GID]
```

Parser auto-detects by checking if the third field is a pure integer (v2) or not (v1).

### Metadata Writing

| Format | Library | Field |
|--------|---------|-------|
| MP3 | mutagen ID3 | COMM frame (desc="" for DJ software + desc="DJTAGGER" for detection) |
| FLAC | mutagen Vorbis | `comment` + `djtagger` field |
| AIFF | mutagen ID3 | COMM frame |
| WAV | mutagen ID3 | COMM frame |
| M4A | mutagen MP4 | `\xa9cmt` atom |

---

## dj-grouper: Grouping and Recommendations

### Three Feature Layers

#### Layer 1: Tags (19 dimensions with continuous vibe)

| Feature | Encoding | Dims |
|---------|----------|------|
| Energy | Ordinal: (E - 1) / 4 | 1 |
| BPM | (BPM - 100) / 40 | 1 |
| Key | sin/cos of Camelot wheel position (24 positions) | 2 |
| Intro bars | 16 -> 0, 32 -> 0.5, 64 -> 1 | 1 |
| Flow type | One-hot: G, H, D, B, L | 5 |
| Vibe | Continuous scores (8 floats from analyzer) | 8 |
| Vocal | Binary: 0 or 1 | 1 |

**Total: 19 dimensions.**

Key difference from v1: vibe encoding uses the continuous scores (8 floats) from the vibe analyzer rather than a one-hot vector. This captures that a track can be 0.7 HYPN and 0.5 DRK simultaneously, producing smoother distance gradients.

**Confidence weighting**: when an analyzer reports low confidence, the corresponding tag dimensions are pushed toward neutral values (0.5 for ordinal, uniform for one-hot/continuous). This prevents low-confidence tags from creating misleading distances.

#### Layer 2: DSP (10 curated features + section-aware extraction)

Instead of the original 45-dim PCA-reduced approach, DSP now uses 10 curated features chosen to be orthogonal and interpretable:

| Feature | Source | Captures |
|---------|--------|----------|
| onset_density | onset count / duration | Rhythmic activity |
| beat_strength | mean onset strength at beat positions | Groove strength |
| perc_harmonic_ratio | percussive RMS / harmonic RMS | Timbral character |
| centroid_mean | mean spectral centroid | Brightness |
| flatness_mean | mean spectral flatness | Noisiness |
| bandwidth_mean | mean spectral bandwidth | Spectral spread |
| low_freq_ratio | power <150 Hz / total | Bass weight |
| rms_mean | mean RMS energy | Loudness |
| chroma_strength | max chroma correlation | Tonal content |
| tonal_stability | std of chroma over time | Harmonic consistency |

All features are z-score normalized before distance computation.

**Section-aware DSP**: features are extracted separately for each detected section (intro, groove, peak). This lets dj-grouper compare how two tracks' grooves sound, independent of their intros or breakdowns. Section-level features feed into the groove compatibility score in recommendations.

**Distance metric**: cosine distance on the 10-dimensional feature vector.

#### Layer 3: CLAP (optional, 64 dims after PCA)

512-dim CLAP embeddings from `laion-clap`, PCA-reduced to 64 dims (fitted on the library). Captures perceptual "sounds like" similarity that spectral features miss.

### Distance Computation

Each layer produces a distance in [0, 1], then blended:

```
d = w_tags * d_tags + w_dsp * d_dsp [+ w_embed * d_embed]
```

| Layer | Metric | Weight (with CLAP) | Weight (no CLAP) |
|-------|--------|--------------------|--------------------|
| Tags | Custom (see sub-weights below) | 0.25 | 0.60 |
| DSP | Cosine distance | 0.30 | 0.40 |
| CLAP | Cosine distance | 0.45 | -- |

#### Tag Distance Sub-Weights

All sub-distances are normalized to [0, 1] before weighting:

| Sub-distance | Weight | Metric | Notes |
|---|---|---|---|
| Energy | 0.25 | Absolute difference / 4 | |
| BPM | 0.30 | Non-linear: `d_bpm_raw^2.0` | Strongest differentiator; amplifies genre-boundary gaps |
| Vibe | 0.20 | Euclidean on continuous scores | Smooth gradient across vibe space |
| Key | 0.10 * vibe_conditional_weight | Circular (Camelot wheel) | Weight varies: MEL 0.8, RAW 0.1; +0.2 for vocals |
| Intro bars | 0.05 | Ordinal distance | |
| Flow type | 0.05 | Binary match/mismatch | |
| Vocal | 0.05 | Binary match/mismatch | Reduced weight here; vocals handled more in clustering |

### Soft Vocal Partitioning

Vocals are no longer a hard binary split in clustering. The vocal confidence from the analyzer determines behavior:

- **confidence > 0.5**: hard partition (V tracks only cluster with V, NV with NV)
- **confidence <= 0.5**: track can cluster with either vocal class; the vocal dimension contributes to distance normally but does not force a partition

This prevents borderline tracks (faint vocal samples, vocal-like synths) from being isolated in the wrong partition.

### Clustering

1. Soft vocal partitioning (hard-split only when vocal confidence > 0.5)
2. Pairwise distance matrix per partition
3. Apply feedback (multiplicative adjustments to distance matrix)
4. Agglomerative clustering with **average linkage** (balances cohesion; less aggressive than complete, less permissive than single)
5. Auto-threshold: sweep 30 candidate thresholds, minimize deviation from target group size (1, 6)
6. Post-process: bisect groups larger than 20, re-number labels (singletons allowed, min_group_size = 1)
7. Post-clustering BPM validation: groups with >6% BPM spread (`bpm_group_max_spread_pct = 0.06`) are force-split
8. Assign stable group IDs (G001, G002, ...)

### Medoid

The actual track with minimum total distance to all cluster members. Used for:
- Stable-mode new-track assignment (assign to nearest medoid)
- Group anchor in review
- Representative track for folder naming

```
medoid = argmin_i( sum_j(d(i, j)) for j in cluster )
```

New tracks assigned to nearest medoid. If distance > 0.8, a new group is created.

### Track Role Inference (role.py)

Each track in a group is assigned a role based on its tags and features:

| Role | Inference Logic |
|------|----------------|
| TOOL | E1--E2, flow L, low onset density |
| DRIVER | E3, flow G or H, moderate energy variance |
| PEAK | E4--E5, high RMS, high onset density |
| RESET | Energy significantly lower than group median, follows peak-level tracks |
| BREAKDOWN | Flow B, or detected breakdown sections dominate |
| BRIDGE | Energy between group extremes, vibe overlaps with multiple group members |

Roles are written to `_group_info.txt` in each group folder.

### Directional Recommendation Scoring

Recommendations are **directional**: the score from track A to track B is not the same as B to A. This models real DJ mixing -- what matters is how well B works as a follow-up to A.

For each candidate pair (source -> destination):

```
base_score = 1 - blended_distance

score = base_score
      + intro_usability(dest)
      - bass_conflict_risk(source, dest)
      + groove_compatibility(source, dest)
      + energy_direction_bonus(source, dest)
      - breakdown_risk(dest)
      - bpm_penalty(source, dest)
      - key_penalty(source, dest)
      + structure_compatibility(source, dest)
```

#### DJ Usability Features

**Intro usability**: rewards destination tracks with longer, cleaner intros (easier to mix into). Based on intro bars and flow type of the destination.

**Bass conflict risk**: penalty when both source and destination have high low_freq_ratio. Two bass-heavy tracks playing simultaneously creates mud.

**Groove compatibility**: cosine similarity between section-level DSP features. Compared at the groove section level -- two tracks whose grooves have similar spectral character will blend well.

**Energy direction bonus**: small reward for energy progressions that make musical sense (e.g., E3 -> E4 for building, E4 -> E2 for a reset). Penalizes jarring jumps (E1 -> E5).

**Breakdown risk**: penalty when the destination track has an early breakdown. Starting a mix and immediately hitting a breakdown disrupts flow.

**Soft BPM penalty**: gradual penalty starting at 4% BPM difference (`bpm_soft_penalty_pct`), increasing to the hard cutoff at 8% (`bpm_hard_cutoff_pct`). Replaces the old hard 4% filter. Tracks beyond 8% are excluded entirely.

```
if bpm_diff_pct <= 0.04:  penalty = 0
elif bpm_diff_pct <= 0.08: penalty = bpm_penalty_weight * (bpm_diff_pct - 0.04) / 0.04
else: exclude pair
```

**Key penalty**: weighted by vibe category. Melodic tracks (MEL) are penalized heavily for key clashes; percussive tracks (RAW) barely at all.

| Vibe | Key Weight |
|------|------------|
| MEL | 0.8 |
| ACID | 0.6 |
| DEEP | 0.4 |
| ATM | 0.3 |
| HYPN | 0.2 |
| TRIB | 0.2 |
| DRK | 0.15 |
| RAW | 0.1 |

Vocal tracks get +0.2 added to their vibe key weight.

**Structure compatibility**: flow type pairing matrix (bonus/penalty):

| From \ To | G | H | D | B | L |
|-----------|-----:|-----:|------:|-----:|-----:|
| G | +0.15 | +0.10 | 0.00 | +0.05 | +0.10 |
| H | +0.10 | +0.15 | -0.10 | 0.00 | +0.10 |
| D | 0.00 | -0.10 | +0.05 | 0.00 | -0.10 |
| B | +0.05 | 0.00 | 0.00 | +0.10 | 0.00 |
| L | +0.10 | +0.10 | -0.10 | 0.00 | +0.15 |

Intro bars: same length +0.05, difference > 16 bars -0.05.

Top 10 per source track, sorted by score descending.

### Evaluation Framework (evaluation.py)

```bash
dj-grouper evaluate --eval-file pairs.csv
```

Input CSV format:
```csv
track_a,track_b,label
"Track A.aiff","Track B.aiff",good
"Track C.aiff","Track D.aiff",bad
```

The framework:
1. Loads the feature cache
2. Computes recommendation scores for all evaluation pairs
3. Measures ranking quality: do known-good pairs score higher than known-bad pairs?
4. Reports precision, recall, and average score differential

Used for iterating on weights and scoring features without manual listening.

### Feedback

| Type | Effect on Distance Matrix |
|------|---------------------------|
| good_pair | `d *= (1 - 0.3)` -- 30% closer |
| bad_pair | `d *= (1 + 0.5)` -- 50% farther |
| group_override | Hard assignment to specified group post-clustering |

Stored in `outputs/feedback.csv`. Applied to the distance matrix before clustering on every run.

---

## Configuration Reference

### dj_tagger/constants.py

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | 22050 | Audio sample rate for analysis |
| `ENERGY_WEIGHTS` | rms:0.30, centroid:0.15, flux:0.20, onset:0.20, low_freq:0.15 | Energy composite weights |
| `ENERGY_THRESHOLDS` | [0.20, 0.40, 0.60, 0.80] | E1--E5 level boundaries |
| `INTRO_ENERGY_RATIO` | 0.80 | Fraction of median energy for intro end |
| `INTRO_SUSTAIN_BARS` | 8 | Consecutive bars above threshold |
| `VOCAL_ENERGY_RATIO` | 0.15 | Vocal band energy ratio threshold |
| `VOCAL_FLATNESS_MAX` | 0.40 | Max spectral flatness for vocal frame |
| `VOCAL_FRAME_THRESHOLD` | 0.08 | Fraction of frames for V classification |

### dj_grouper/config.py (GrouperConfig dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `w_tags` | 0.25 | Tag layer weight (with CLAP) |
| `w_dsp` | 0.30 | DSP layer weight (with CLAP) |
| `w_embed` | 0.45 | CLAP layer weight |
| `w_tags_no_embed` | 0.60 | Tag layer weight (no CLAP) |
| `w_dsp_no_embed` | 0.40 | DSP layer weight (no CLAP) |
| `key_weight_by_vibe` | MEL:0.8, ACID:0.6, DEEP:0.4, ATM:0.3, HYPN:0.2, TRIB:0.2, DRK:0.15, RAW:0.1 | Key penalty weight per vibe |
| `key_weight_vocal_boost` | 0.20 | Added to key weight when vocals present |
| `bpm_hard_cutoff_pct` | 0.08 | 8% BPM difference = excluded |
| `bpm_soft_penalty_pct` | 0.04 | BPM penalty starts at 4% |
| `bpm_penalty_weight` | 0.15 | Max BPM penalty contribution |
| `linkage` | "average" | Agglomerative linkage method |
| `min_group_size` | 1 | Minimum tracks per group (singletons allowed) |
| `max_group_size` | 20 | Maximum tracks per group |
| `target_group_size` | (1, 6) | Ideal group size for threshold tuning |
| `bpm_group_max_spread_pct` | 0.06 | Post-clustering BPM validation: force-split groups with >6% spread |
| `vocal_confidence_threshold` | 0.5 | Hard vocal partition above this |
| `clap_pca_dims` | 64 | PCA dims for CLAP embeddings |
| `new_group_distance_threshold` | 0.80 | Distance for creating new group in stable mode |
| `n_recommendations` | 10 | Recommendations per track |
| `good_pair_factor` | 0.30 | Good feedback distance multiplier |
| `bad_pair_factor` | 0.50 | Bad feedback distance multiplier |
| `input_dir` | "./files" | Default input directory |
| `output_dir` | "./outputs" | Default output directory |

---

## Dependencies

**Required**: librosa >=0.10, mutagen >=1.47, numpy >=1.24, scipy >=1.10, soundfile >=0.12, scikit-learn >=1.3

**Optional**: essentia >=2.1b6 (key detection), laion-clap >=1.0 + torch >=2.0 (embeddings)

**Dev**: pytest >=7.0, pytest-cov >=4.0

---

## Test Coverage

80 tests across all modules:

| Area | Tests | Covers |
|------|-------|--------|
| Tag formatting/parsing | 10 | v1/v2 format, round-trip, edge cases |
| Energy analyzer | 5 | Composite scoring, thresholds, confidence |
| Key analyzer | 4 | Chroma correlation, Camelot mapping, confidence |
| Structure analyzer | 4 | Intro detection, flow classification, confidence |
| Vibe analyzer | 3 | Scoring, continuous output, confidence |
| Vocal analyzer | 3 | Multi-stage detection, confidence |
| Section detection | 3 | Boundary detection, section types |
| Metadata read/write | 5 | MP3, FLAC, AIFF, WAV, M4A |
| Pipeline + CLI (tagger) | 10 | End-to-end, options, error handling |
| Tag encoding + builder | 4 | Continuous vibe, confidence weighting |
| Distance computation | 5 | Sub-weights, normalization, blending |
| Clustering | 3 | Average linkage, soft vocal partition, size constraints |
| Track roles | 2 | Role inference logic |
| Recommendation scoring | 6 | Directional scoring, DJ usability features, soft BPM |
| Transition compatibility | 6 | Flow pairing matrix, intro matching |
| Feedback | 4 | Good/bad pair, override, persistence |
| CSV/playlist export | 4 | groups.csv, recommendations.csv, m3u8 |
| Evaluation | 3 | Known pair ranking, metrics |
