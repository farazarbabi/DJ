# Genre Detection and 3-Level Classification System for DJ Track Libraries

## 1. Objective

Build a genre detection and classification module for a larger DJ-focused music intelligence system.

The module must assign each track to a normalized 3-level taxonomy:

```json
{
  "family": "House",
  "genre": "Tech House",
  "subgenre": "Rolling Tech House"
}
```

The classifier must be constrained to a fixed reference taxonomy stored in JSON. It should not freely invent genres. When evidence is weak or ambiguous, it should return lower confidence, alternatives, warnings, or null values rather than forcing a bad classification.

The output should support practical DJ use cases:

- crate building
- library cleanup
- filtering by style
- grouping similar tracks
- improving recommendations
- supporting transition planning
- identifying set roles such as opener, builder, driver, or peak-time weapon

This is not intended to be an academic musicology classifier. It should be deterministic, explainable, configurable, and robust to noisy real-world music metadata.

---

## 2. Core Classification Target

The target taxonomy has exactly three conceptual levels:

```text
family → genre → subgenre
```

Example:

```json
{
  "family": "Techno",
  "genre": "Hypnotic Techno",
  "subgenre": "Rolling Hypnotic Techno"
}
```

Another example:

```json
{
  "family": "Rock / Metal / Guitar-Based",
  "genre": "Alternative Rock",
  "subgenre": "Indie Rock"
}
```

The module must validate every output path against the reference taxonomy.

Invalid output example:

```json
{
  "family": "House",
  "genre": "Techno",
  "subgenre": "Rolling Hypnotic Techno"
}
```

This must be rejected because the path is inconsistent.

---

## 3. Required Inputs

The classifier receives a consolidated track record. Each record may include file metadata, registry identity, source observations, audio features, local tagger fields, and derived DJ signals.

Fields may be missing. The classifier must handle sparse inputs gracefully.

---

### 3.1 File Metadata

```json
{
  "file_path": "D:/Music/Artist - Track Name (Extended Mix).aiff",
  "filename": "Artist - Track Name (Extended Mix).aiff",
  "embedded_title": "Track Name",
  "embedded_artist": "Artist",
  "embedded_album": "Release Name",
  "embedded_genre": "Tech House",
  "embedded_comment": "E4 HYP 16H NV"
}
```

Useful signals:

- filename may contain artist, title, remix, edit, bootleg, and genre hints
- embedded genre may be useful but can be noisy
- embedded comments may contain DJ tags, source notes, energy tags, or imported classification data
- file path may contain folder-level genre hints

---

### 3.2 Registry Identity

```json
{
  "artist": "Artist Name",
  "title": "Track Name",
  "mix": "Extended Mix",
  "remix_artist": "Remixer Name",
  "label": "Label Name",
  "release_year": 2024
}
```

Useful signals:

- artist identity can provide a genre prior
- label identity can provide a scene/style prior
- remix artist may be more important than original artist for remixes
- mix names can provide strong hints:
  - Dub Mix
  - Tribal Mix
  - Afro Mix
  - Acid Mix
  - Warehouse Mix
  - Progressive Mix
  - Organic Mix
  - Club Mix
  - Extended Mix
  - Instrumental Mix

---

### 3.3 Source Observations

```json
{
  "rekordbox_genre": "Melodic Techno",
  "songstats_genre": "Melodic House & Techno",
  "spotify_genres": ["melodic techno", "indie dance"],
  "genres_all": ["melodic techno", "dark disco", "indie dance"],
  "source": "Soundeo",
  "source_comments": "dark rolling indie dance track with hypnotic arps",
  "source_labels": ["Indie Dance", "Dark Disco"]
}
```

Useful signals:

- Rekordbox genre may be manually curated or inherited from file tags
- Songstats and Spotify genres may be broad, artist-level, or inconsistent
- `genres_all` may contain multiple noisy candidates
- source comments may include high-quality DJ-store descriptions
- source labels may be closer to DJ-store classification than streaming metadata

---

### 3.4 Songstats / Streaming Audio Features

```json
{
  "danceability": 0.78,
  "valence": 0.25,
  "instrumentalness": 0.91,
  "acousticness": 0.04,
  "energy": 0.84
}
```

Interpretation guidance:

- high danceability supports club/electronic styles
- high instrumentalness supports techno, house tools, downtempo, ambient, post-rock, or instrumental electronic music
- low valence supports dark, melancholic, industrial, deep, gothic, noir, or tense subgenres
- high energy supports peak-time, hard, driving, rave, DnB, bass, rock, or metal
- high acousticness supports rock, alternative, classic rock, folk-adjacent, organic, acoustic, jazz/soul, or singer-songwriter-adjacent material

Audio features should not override strong explicit metadata by themselves. They are best used for tie-breaking, subgenre selection, and confidence calibration.

---

### 3.5 Local Tagger Fields

```json
{
  "tagger_energy": "E4",
  "tagger_vibe": "DRK,HYP",
  "tagger_vocal": "NV",
  "tagger_structure": "16H",
  "tagger_bpm": 126.0
}
```

Possible values:

#### Energy

```text
E1 → intro, ambient, chill, background, low intensity
E2 → warm-up, opener, relaxed groove
E3 → builder, groove, bridge
E4 → driver, builder, main-set track
E5 → peak, weapon, high-intensity
```

#### Vibe

```text
DRK → dark, tense, nocturnal, gothic, industrial
HYP → hypnotic, repetitive, meditative
MEL → melodic, emotional, cinematic
ROM → romantic, sensual, melancholic
PSY → psychedelic, trippy, acid, mental
ORG → organic, earthy, tribal, desert
RAW → raw, warehouse, gritty
EUP → euphoric, uplifting, festival
FUN → funky, playful, bright
```

#### Vocal profile

```text
V     → vocal-led
LV    → low-vocal / vocal fragments
NV    → instrumental / no vocal
SPK   → spoken word
CHANT → chant / ritual vocal
DUB   → dub mix / reduced vocal / echo-heavy
```

#### Structure

```text
16H     → rolling, steady, hypnotic
16D     → driving, linear, forward-motion
32H     → extended hypnotic phrasing
32D     → long driving progression
BREAKS  → broken rhythm, breakbeat, garage, jungle, bass
ROLLING → rolling groove
LINEAR  → functional, driving, tool-like
```

---

## 4. Required Output

For each track, return a structured result:

```json
{
  "track_id": "optional-track-id",
  "family": "Techno",
  "genre": "Hypnotic Techno",
  "subgenre": "Rolling Hypnotic Techno",
  "confidence": 0.84,
  "confidence_level": "high",
  "evidence": {
    "matched_terms": ["techno", "hypnotic", "rolling"],
    "source_genres_used": ["hypnotic techno", "raw techno"],
    "metadata_signals": ["rekordbox_genre=Hypnotic Techno"],
    "audio_feature_signals": ["energy high", "valence low", "instrumentalness high"],
    "tagger_signals": ["tagger_vibe=HYP", "tagger_structure=16H", "tagger_energy=E4"]
  },
  "alternatives": [
    {
      "family": "Techno",
      "genre": "Driving Techno",
      "subgenre": "Rolling Techno",
      "score": 0.72,
      "evidence": ["tagger_structure=16H", "energy high"]
    },
    {
      "family": "Techno",
      "genre": "Minimal Techno",
      "subgenre": "Loop Techno",
      "score": 0.65,
      "evidence": ["hypnotic", "instrumental", "minimal cue"]
    }
  ],
  "warnings": [
    "Source genres disagree: Spotify suggests melodic techno, Rekordbox suggests minimal techno."
  ]
}
```

---

## 5. Confidence Levels

Map numeric confidence to labels:

```text
0.85–1.00 → high
0.70–0.84 → medium-high
0.55–0.69 → medium
0.35–0.54 → low
0.00–0.34 → unknown
```

Confidence should reflect:

- strength of explicit metadata
- agreement across sources
- exact taxonomy matches
- consistency of family, genre, and subgenre
- margin between top and second candidate
- amount of missing data
- ambiguity of terms such as deep, dark, melodic, progressive, garage
- whether the subgenre is inferred from weak or strong evidence

---

## 6. Taxonomy Loading

The classifier must load a JSON taxonomy with this shape:

```json
{
  "House": {
    "Classic House": [
      "Chicago House",
      "New York House",
      "Piano House",
      "Vocal House"
    ],
    "Tech House": [
      "Rolling Tech House",
      "Minimal Tech House",
      "Peak-Time Tech House"
    ]
  },
  "Techno": {
    "Minimal Techno": [
      "Berlin Minimal",
      "Loop Techno",
      "Reduced Techno"
    ],
    "Hypnotic Techno": [
      "Deep Hypnotic Techno",
      "Rolling Hypnotic Techno",
      "Mental Techno"
    ]
  }
}
```

The taxonomy module must expose:

```python
class GenreTaxonomy:
    def families(self) -> list[str]: ...
    def genres(self, family: str | None = None) -> list[str]: ...
    def subgenres(self, family: str | None = None, genre: str | None = None) -> list[str]: ...
    def validate_path(self, family: str | None, genre: str | None, subgenre: str | None) -> bool: ...
    def find_paths_for_label(self, label: str) -> list[TaxonomyPath]: ...
```

Validation rules:

- family may be null only when classification is unknown
- genre may be null if family is known but genre confidence is low
- subgenre may be null if genre is known but subgenre confidence is low
- genre must belong to family
- subgenre must belong to genre under family

---

## 7. Text Normalization

Create robust text normalization utilities.

The normalizer should:

- lowercase text
- Unicode normalize
- strip leading/trailing spaces
- collapse multiple spaces
- normalize separators: `/`, `-`, `_`, `|`, `,`, `;`, parentheses
- preserve meaningful genre tokens like `r&b`, `2-step`, `drum & bass`
- remove file extensions from filename text
- split camel case where useful
- normalize common spelling variants
- normalize apostrophes and quotes

Examples:

```text
"Tech-House" → "tech house"
"techhouse" → "tech house"
"Melodic Techno / Progressive House" → ["melodic techno", "progressive house"]
"indie-dance" → "indie dance"
"nu disco" → "nu disco"
"UKG" → "ukg"
"DNB" → "dnb"
"drum n bass" → "drum and bass"
"RNB" → "r&b"
"rock'n'roll" → "rock and roll"
"Post Rock" → "post rock"
```

Suggested libraries:

- Python `re`
- `unicodedata`
- optional `unidecode`
- optional `rapidfuzz` for fuzzy matching

---

## 8. Alias and Synonym Mapping

Create a configurable alias system.

Example alias file:

```json
{
  "techhouse": "Tech House",
  "tech-house": "Tech House",
  "tech house": "Tech House",
  "deep tech": "Deep Tech",
  "melodic techno": "Melodic Techno",
  "melodic house and techno": "Melodic Techno",
  "afro house": "Afro House",
  "organic house": "Organic House",
  "indie dance": "Indie Dance",
  "dark disco": "Dark Disco",
  "nu disco": "Nu-Disco",
  "ukg": "UK Garage",
  "2step": "2-Step Garage",
  "2 step": "2-Step Garage",
  "dnb": "Drum & Bass",
  "drum n bass": "Drum & Bass",
  "drum and bass": "Drum & Bass",
  "jungle": "Jungle",
  "donk": "Donk",
  "garage rock": "Garage Rock",
  "alternative rock": "Alternative Rock",
  "alt rock": "Alternative Rock",
  "classic rock": "Classic Rock",
  "post rock": "Post-Rock",
  "post-rock": "Post-Rock"
}
```

The alias matcher should support:

1. exact normalized match
2. known alias match
3. phrase containment
4. conservative fuzzy match
5. source-weighted match

Fuzzy matching thresholds:

```text
>= 92 for taxonomy labels
>= 88 for known aliases
```

Avoid aggressive fuzzy matching because genre labels are short and easy to misclassify.

---

## 9. Evidence Model

Represent every piece of evidence as a structured object.

```python
class EvidenceItem(BaseModel):
    source_field: str
    raw_value: str | float | int | None
    normalized_value: str | None = None
    matched_label: str | None = None
    matched_path: TaxonomyPath | None = None
    evidence_type: Literal[
        "explicit_genre",
        "alias",
        "artist_prior",
        "label_prior",
        "title_cue",
        "mix_cue",
        "filename_cue",
        "audio_feature",
        "tagger_signal",
        "derived_signal",
        "disambiguation_rule"
    ]
    weight: float
    score_delta: float
    explanation: str
```

This allows the classifier to explain why it chose a given path.

---

## 10. Source Weights

Use configurable source weights.

Recommended defaults:

```text
trusted_dj_source_genre          1.00
rekordbox_genre_manual          0.90
rekordbox_genre_unknown_origin  0.75
embedded_genre                  0.60
songstats_track_genre           0.70
spotify_track_genre             0.65
spotify_artist_genre            0.45
genres_all                      0.70
source_label                    0.80
source_comment                  0.55
label_prior                     0.65
artist_prior                    0.55
remix_artist_prior              0.65
title_or_mix_text_cue           0.50
filename_cue                    0.35
audio_feature                   0.35
local_tagger_field              0.50
derived_dj_signal               0.45
```

All weights should be configurable through a YAML or JSON config file.

---

## 11. Candidate Generation

Generate classification candidates from multiple sources.

Candidate sources:

1. direct taxonomy label matches
2. alias matches
3. explicit genre phrases
4. source labels
5. Rekordbox genre
6. embedded genre
7. Spotify/Songstats genres
8. artist priors
9. label priors
10. remix artist priors
11. title/mix/remix text cues
12. filename cues
13. audio-feature heuristics
14. local tagger fields
15. derived DJ signals
16. disambiguation rules

Candidate object:

```python
class GenreCandidate(BaseModel):
    family: str | None
    genre: str | None
    subgenre: str | None
    score: float = 0.0
    evidence: list[EvidenceItem] = []
    warnings: list[str] = []
```

The classifier should aggregate evidence into candidate scores.

---

## 12. Hierarchical Scoring

Perform scoring hierarchically.

### 12.1 Family scoring

Infer broad family first.

Examples:

```text
tech house, deep house, afro house → House
minimal techno, dub techno, hard techno → Techno
indie dance, dark disco, nu-disco → Indie Dance / Dark Disco / Nu-Disco
uk garage, 2-step, bassline, breaks → Garage / UK Bass / Breaks
alternative rock, classic rock, post-rock, garage rock → Rock / Metal / Guitar-Based
```

### 12.2 Genre scoring

Within the top family, infer genre.

Examples:

```text
House + tribal/percussion/chant → Tribal House
House + afro/African percussion → Afro House
House + organic/desert/earthy → Organic House
House + tech/rolling/bassline → Tech House
Techno + hypnotic/rolling/loop → Hypnotic Techno
Techno + melodic/emotional/cinematic/arpeggio → Melodic Techno
Techno + industrial/metallic/noise/raw → Industrial Techno
Rock + classic/70s/blues/arena → Classic Rock
Rock + indie/alternative/90s/college → Alternative Rock
Rock + instrumental/crescendo/cinematic → Post-Rock
```

### 12.3 Subgenre scoring

Infer subgenre only if there is enough evidence.

Examples:

```text
Tech House + rolling/16H/E4 → Rolling Tech House
Tech House + peak/E5/bass hook → Peak-Time Tech House
Tribal House + chant/ORG/percussion → Shamanic Tribal House
Organic House + desert/middle eastern/ORG → Desert House
Techno + HYP + 16H + low vocal → Rolling Hypnotic Techno
Techno + DRK + RAW + E5 → Dark Driving Techno or Industrial Techno
Melodic Techno + ROM/MEL + low valence → Dark Melodic Techno or Romantic Techno
Garage + BREAKS + vocal chops → 2-Step Garage
Rock + classic + blues/70s cues → Blues Rock or Classic Rock
Rock + instrumental + long crescendo → Post-Rock
```

Subgenre should remain null when the margin between top candidates is too small.

---

## 13. BPM-Based Priors

BPM is a soft prior, not a rule.

Suggested priors:

```text
60–95 BPM:
  Downtempo, Trip-Hop, Hip-Hop, Slow Electronic, Ambient, Rock Ballads

95–115 BPM:
  Slow House, Slow Disco, Indie Dance, Nu-Disco, Midtempo Bass, Funk, Rock

115–124 BPM:
  Deep House, Organic House, Afro House, Indie Dance, Nu-Disco, Progressive House

122–128 BPM:
  House, Tech House, Minimal House, Melodic House, Progressive House

126–134 BPM:
  Techno, Driving Techno, Peak-Time Techno, Hardgroove, Trance

130–145 BPM:
  Hard Techno, Trance, Hard Dance, Breaks, Garage, Donk

135–150 BPM:
  Garage, Bassline, Donk, Hard Trance, Hardstyle, some Techno

160–180 BPM:
  Drum & Bass, Jungle, Footwork Jungle

170–180 BPM:
  Liquid DnB, Neurofunk, Jungle, Jump-Up
```

Implementation guidance:

- use BPM to increase or decrease candidate scores mildly
- do not reject a candidate only because BPM is outside the expected range
- treat half-time and double-time ambiguity carefully
- allow configurable BPM ranges per family/genre

---

## 14. Audio Feature Heuristics

Audio features should influence candidate scoring but not dominate explicit metadata.

### 14.1 Energy

```text
Low energy:
  Ambient, Downtempo, Chillout, Deep House, Organic House

Medium energy:
  Deep House, Indie Dance, Organic House, Nu-Disco, Progressive House

High energy:
  Tech House, Driving Techno, Peak-Time Techno, Hard Dance, DnB, Bass, Rock

Very high energy:
  Hard Techno, Industrial Techno, Hardstyle, Gabber, Metal, Neurofunk
```

### 14.2 Valence

```text
Low valence:
  Dark Techno, Industrial Techno, Dark Indie Dance, Post-Punk, Darkwave, Trip-Hop, Doom, Dark R&B

Medium valence:
  Deep House, Minimal Techno, Progressive House, Indie Dance

High valence:
  Disco House, Funky House, Soulful House, Nu-Disco, Happy Hardcore, Pop
```

### 14.3 Instrumentalness

```text
High instrumentalness:
  Techno, House tools, Downtempo, Ambient, Post-Rock, Electro

Low instrumentalness:
  Pop, R&B, Vocal House, Rap, Rock, Vocal Garage
```

### 14.4 Acousticness

```text
High acousticness:
  Rock, Alternative Rock, Classic Rock, Folk-adjacent, Jazz/Soul, Organic Downtempo

Low acousticness:
  Techno, House, Bass, Electro, Synthwave
```

---

## 15. Derived DJ Signals

Derive DJ-relevant signals from tagger fields, filename, mix name, title, comments, and source text.

### 15.1 Mood derivation

```text
DRK → dark, tense, nocturnal, gothic, industrial
HYP → hypnotic, repetitive, meditative, rolling
MEL → melodic, emotional, cinematic
ROM → romantic, sensual, melancholic
PSY → psychedelic, trippy, acid, mental
ORG → organic, earthy, tribal, desert
RAW → raw, warehouse, gritty
EUP → euphoric, uplifting, festival
FUN → funky, playful, bright
```

### 15.2 Groove derivation

```text
16H → rolling, steady, hypnotic
16D → driving, linear, forward-motion
32H → extended hypnotic phrasing
32D → long driving progression
BREAKS → broken rhythm, breakbeat, garage, jungle, bass
ROLLING → rolling groove
LINEAR → functional, driving, tool-like
```

### 15.3 Set role derivation

```text
E1 → intro, ambient, warm-up, chill
E2 → warm-up, opener, groove
E3 → builder, groove, bridge
E4 → driver, builder, main set
E5 → peak, weapon, high-intensity
```

### 15.4 Vocal profile derivation

```text
V → vocal-led
LV → low-vocal / vocal fragments
NV → instrumental / no vocal
SPK → spoken word
CHANT → chant / ritual vocal
DUB → dub mix / reduced vocal / echo-heavy
```

Also parse text cues:

```text
feat, featuring, ft. → vocal or collaboration cue
dub, dub mix → dubby / reduced vocal cue
instrumental → no-vocal cue
chant, ritual, prayer, mantra → chant / tribal / spiritual cue
spoken, speech → spoken vocal cue
```

---

## 16. Disambiguation Rules

Explicit disambiguation rules are critical because many genre terms overlap.

---

### 16.1 Garage ambiguity

The word `garage` can mean:

1. UK Garage / 2-Step / Bassline
2. Garage House
3. Garage Rock

Rules:

- If terms include `ukg`, `2-step`, `2 step`, `bassline`, `speed garage`, `breaks`, or BPM is around 125–140 with electronic metadata, prefer `Garage / UK Bass / Breaks`.
- If terms include `rock`, `punk`, `psych`, `lo-fi guitar`, or audio features indicate guitar/acoustic content, prefer `Rock / Metal / Guitar-Based → Garage Rock`.
- If terms include `house`, `soulful`, `vocal house`, or 4/4 house metadata, prefer House.

---

### 16.2 Progressive ambiguity

`Progressive` can mean:

- Progressive House
- Progressive Trance
- Progressive Rock
- Progressive Metal

Rules:

- BPM 120–128 + house metadata → Progressive House
- BPM 128–140 + trance metadata → Progressive Trance
- guitar/acoustic/rock metadata → Progressive Rock
- metal cues → Progressive Metal or Metal branch

---

### 16.3 Melodic ambiguity

`Melodic` can apply to many families.

Rules:

- `melodic techno` explicit → Techno → Melodic Techno
- `melodic house` explicit → House → Melodic House
- `melodic dubstep` explicit → Bass / Dubstep / Trap → Dubstep → Melodic Dubstep
- if only `melodic` appears, use family-level evidence first

---

### 16.4 Deep ambiguity

`Deep` can apply to house, techno, dubstep, DnB, progressive, etc.

Rules:

- `deep house` explicit → House → Deep House
- `deep techno` explicit → Techno → Deep Techno
- `deep dubstep` explicit → Bass / Dubstep / Trap → Dubstep → Deep Dubstep
- `deep dnb` explicit → Drum & Bass / Jungle → Drum & Bass → Deep DnB
- if only `deep` appears, use BPM, source genre, and family evidence

---

### 16.5 Dark ambiguity

`Dark` is primarily a mood, not a genre.

Rules:

- do not classify as `Dark X` unless family or genre is already known
- use `dark` mostly for subgenre inference
- explicit `techno` + `dark` → Dark Techno
- explicit `indie dance` + `dark` → Dark Indie Dance
- explicit `r&b` + `dark` → Dark R&B

---

### 16.6 Vocal cues

Vocals should not dominate genre classification.

Rules:

- vocal-heavy + house metadata → Vocal House or Soulful House
- low-vocal + techno metadata → Techno remains likely
- `dub mix` should reduce vocal confidence and increase dubby/deep/minimal likelihood
- `chant` + organic/tribal/afro metadata supports Organic House, Tribal House, Afro House, or Ethno-Downtempo

---

### 16.7 Rock versus electronic ambiguity

Rules:

- high acousticness, low synthetic metadata, and rock source tags → Rock / Metal / Guitar-Based
- explicit remix/edit/club mix plus house/techno labels may indicate electronic remix of a rock song
- if artist is rock but remix artist is electronic, remix artist prior should be stronger
- if the track title contains `remix`, `club edit`, `extended mix`, or DJ-store electronic metadata, do not over-weight original artist genre

---

## 17. Artist and Label Priors

Support optional prior maps.

Example:

```json
{
  "artist_priors": {
    "Malandra Jr.": [
      {
        "family": "Indie Dance / Dark Disco / Nu-Disco",
        "genre": "Indie Dance",
        "weight": 0.75
      },
      {
        "family": "Techno",
        "genre": "Melodic Techno",
        "weight": 0.45
      }
    ]
  },
  "label_priors": {
    "Diynamic": [
      {
        "family": "House",
        "genre": "Melodic House",
        "weight": 0.65
      },
      {
        "family": "Techno",
        "genre": "Melodic Techno",
        "weight": 0.55
      }
    ]
  }
}
```

Rules:

- artist priors should be weaker than track-level genre metadata
- label priors are useful but should not override explicit genre labels
- remix artist priors may be stronger than original artist priors
- priors should contribute to candidates, not directly determine the answer

---

## 18. Classification Algorithm

Recommended deterministic pipeline:

```text
Input track record
  ↓
Normalize all text fields
  ↓
Extract explicit genre strings
  ↓
Apply alias mapping
  ↓
Generate taxonomy candidates
  ↓
Extract title/mix/filename/comment cues
  ↓
Apply artist, label, and remix artist priors
  ↓
Apply BPM, audio-feature, and tagger priors
  ↓
Apply disambiguation rules
  ↓
Aggregate weighted evidence
  ↓
Rank candidates
  ↓
Validate taxonomy path
  ↓
Compute confidence
  ↓
Return best classification, alternatives, evidence, and warnings
```

---

## 19. Scoring Model

Implement candidate scoring as an additive weighted evidence model.

Simplified formula:

```text
candidate_score =
    sum(weighted_evidence_scores)
    + source_agreement_bonus
    + exact_match_bonus
    + taxonomy_consistency_bonus
    + bpm_compatibility_bonus
    + tagger_consistency_bonus
    - ambiguity_penalty
    - contradiction_penalty
    - weak_subgenre_penalty
```

Suggested bonuses and penalties:

```text
exact taxonomy label match: +0.20
multiple independent source agreement: +0.15 to +0.35
family/genre/subgenre path consistency: +0.10
BPM strongly compatible: +0.05 to +0.12
tagger signals consistent with subgenre: +0.05 to +0.15
ambiguous keyword penalty: -0.05 to -0.20
source contradiction penalty: -0.10 to -0.30
missing data penalty: -0.05 to -0.20
weak subgenre evidence penalty: -0.10 to -0.25
small top-two margin penalty: -0.10 to -0.25
```

Normalize final candidate scores to confidence values between 0 and 1.

The exact math can be adjusted, but it must be deterministic and explainable.

---

## 20. Handling Unknowns

The classifier must support null outputs.

### 20.1 Fully unknown

```json
{
  "family": null,
  "genre": null,
  "subgenre": null,
  "confidence": 0.22,
  "confidence_level": "unknown",
  "warnings": ["Insufficient genre evidence."]
}
```

### 20.2 Family known, genre unknown

```json
{
  "family": "House",
  "genre": null,
  "subgenre": null,
  "confidence": 0.46,
  "confidence_level": "low",
  "warnings": ["Family inferred, but genre/subgenre evidence is weak."]
}
```

### 20.3 Genre known, subgenre unknown

```json
{
  "family": "Techno",
  "genre": "Minimal Techno",
  "subgenre": null,
  "confidence": 0.68,
  "confidence_level": "medium",
  "warnings": ["Subgenre left blank because candidate margin was too small."]
}
```

Rules:

- do not force subgenre when evidence is weak
- do not force genre if only family is supported
- prefer null plus warnings over false precision

---

## 21. Python Project Structure

Suggested module layout:

```text
genre_classifier/
  __init__.py
  taxonomy.py
  normalizer.py
  aliases.py
  evidence.py
  scoring.py
  disambiguation.py
  priors.py
  features.py
  classifier.py
  schemas.py
  config.py
  batch.py
  io.py
  tests/
    test_normalizer.py
    test_aliases.py
    test_taxonomy_validation.py
    test_evidence_extraction.py
    test_disambiguation.py
    test_scoring.py
    test_classifier_examples.py
examples/
  classify_one_track.py
  classify_batch.py
  inspect_evidence.py
config/
  source_weights.yaml
  aliases.json
  artist_priors.json
  label_priors.json
  bpm_priors.yaml
  audio_feature_priors.yaml
README.md
```

---

## 22. Pydantic Schemas

Use Pydantic for input and output validation.

### 22.1 Taxonomy path

```python
from pydantic import BaseModel
from typing import Literal

class TaxonomyPath(BaseModel):
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None
```

### 22.2 Input schema

```python
class TrackGenreInput(BaseModel):
    track_id: str | None = None

    file_path: str | None = None
    filename: str | None = None
    embedded_title: str | None = None
    embedded_artist: str | None = None
    embedded_album: str | None = None
    embedded_genre: str | None = None
    embedded_comment: str | None = None

    artist: str | None = None
    title: str | None = None
    mix: str | None = None
    remix_artist: str | None = None
    label: str | None = None
    release_year: int | None = None

    rekordbox_genre: str | None = None
    songstats_genre: str | None = None
    spotify_genres: list[str] = []
    genres_all: list[str] = []
    source: str | None = None
    source_comments: str | None = None
    source_labels: list[str] = []

    danceability: float | None = None
    valence: float | None = None
    instrumentalness: float | None = None
    acousticness: float | None = None
    energy: float | None = None

    tagger_energy: str | None = None
    tagger_vibe: str | None = None
    tagger_vocal: str | None = None
    tagger_structure: str | None = None
    tagger_bpm: float | None = None
```

### 22.3 Evidence schema

```python
class EvidenceItem(BaseModel):
    source_field: str
    raw_value: str | float | int | None = None
    normalized_value: str | None = None
    matched_label: str | None = None
    matched_path: TaxonomyPath | None = None
    evidence_type: str
    weight: float
    score_delta: float
    explanation: str
```

### 22.4 Candidate schema

```python
class GenreCandidate(BaseModel):
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None
    score: float
    evidence: list[EvidenceItem] = []
```

### 22.5 Output schema

```python
class GenreClassificationResult(BaseModel):
    track_id: str | None = None
    family: str | None = None
    genre: str | None = None
    subgenre: str | None = None
    confidence: float
    confidence_level: Literal["high", "medium-high", "medium", "low", "unknown"]
    evidence: dict[str, list[str]]
    alternatives: list[GenreCandidate]
    warnings: list[str]
```

---

## 23. Batch Processing

The classifier must support batch processing:

```python
results = classifier.classify_many(track_records)
```

Batch output must be serializable to JSON and optionally CSV.

CSV columns:

```text
track_id
artist
title
mix
family
genre
subgenre
confidence
confidence_level
top_alternatives
warnings
evidence_summary
```

Batch mode should support:

- progress logging
- error handling per track
- output continuation if one track fails
- optional debug mode to save candidate scores
- optional evidence export for manual review

---

## 24. Optional LLM Fallback

Do not implement the first version as an LLM-only classifier.

Optional later fallback:

- only call LLM when deterministic confidence is below threshold
- constrain LLM to the taxonomy
- require JSON-only output
- validate LLM output against taxonomy
- ask LLM to cite input fields used
- never allow LLM to invent taxonomy labels
- deterministic classifier remains source of validation

Suggested fallback prompt behavior:

```text
Given this track metadata and this allowed taxonomy, choose the best family, genre, and optional subgenre. Return JSON only. If evidence is weak, use null for uncertain levels. Do not invent labels.
```

---

## 25. Unit Tests

Create strong unit tests around ambiguity.

### Test 1: Clear Tech House

Input:

```json
{
  "embedded_genre": "Tech House",
  "rekordbox_genre": "Tech House",
  "tagger_energy": "E4",
  "tagger_structure": "16H",
  "tagger_bpm": 126
}
```

Expected:

```json
{
  "family": "House",
  "genre": "Tech House",
  "subgenre": "Rolling Tech House"
}
```

---

### Test 2: Dark Hypnotic Techno

Input:

```json
{
  "genres_all": ["Techno", "Hypnotic Techno"],
  "tagger_vibe": "DRK,HYP",
  "tagger_vocal": "NV",
  "tagger_structure": "16H",
  "tagger_energy": "E4",
  "tagger_bpm": 132,
  "energy": 0.82,
  "valence": 0.21,
  "instrumentalness": 0.91
}
```

Expected:

```json
{
  "family": "Techno",
  "genre": "Hypnotic Techno",
  "subgenre": "Rolling Hypnotic Techno"
}
```

---

### Test 3: Garage ambiguity — UK Garage

Input:

```json
{
  "embedded_genre": "Garage",
  "genres_all": ["UKG", "2 Step", "Bassline"],
  "tagger_structure": "BREAKS",
  "tagger_bpm": 134
}
```

Expected:

```json
{
  "family": "Garage / UK Bass / Breaks",
  "genre": "UK Garage",
  "subgenre": "2-Step Garage"
}
```

---

### Test 4: Garage ambiguity — Garage Rock

Input:

```json
{
  "embedded_genre": "Garage Rock",
  "genres_all": ["garage punk", "psych rock"],
  "acousticness": 0.74,
  "energy": 0.71
}
```

Expected:

```json
{
  "family": "Rock / Metal / Guitar-Based",
  "genre": "Garage Rock"
}
```

---

### Test 5: Organic / Desert House

Input:

```json
{
  "genres_all": ["Organic House", "Downtempo"],
  "source_comments": "middle eastern percussion desert sunset vibe",
  "tagger_vibe": "ORG,MEL",
  "tagger_energy": "E2",
  "tagger_bpm": 120,
  "instrumentalness": 0.84
}
```

Expected:

```json
{
  "family": "House",
  "genre": "Organic House",
  "subgenre": "Desert House"
}
```

---

### Test 6: Rock expansion — Alternative Rock

Input:

```json
{
  "embedded_genre": "Alternative Rock",
  "spotify_genres": ["modern rock", "indie rock"],
  "acousticness": 0.48,
  "energy": 0.76
}
```

Expected:

```json
{
  "family": "Rock / Metal / Guitar-Based",
  "genre": "Alternative Rock"
}
```

---

### Test 7: Rock expansion — Post-Rock

Input:

```json
{
  "embedded_genre": "Post-Rock",
  "source_comments": "instrumental, cinematic, long crescendo",
  "instrumentalness": 0.92,
  "energy": 0.61,
  "acousticness": 0.43
}
```

Expected:

```json
{
  "family": "Rock / Metal / Guitar-Based",
  "genre": "Post-Rock"
}
```

---

### Test 8: Insufficient evidence

Input:

```json
{
  "filename": "track01.mp3",
  "energy": 0.58
}
```

Expected:

```json
{
  "family": null,
  "genre": null,
  "subgenre": null,
  "confidence_level": "unknown"
}
```

---

### Test 9: Progressive ambiguity — Progressive House

Input:

```json
{
  "genres_all": ["progressive", "house"],
  "tagger_bpm": 124,
  "danceability": 0.82,
  "energy": 0.68,
  "instrumentalness": 0.76
}
```

Expected:

```json
{
  "family": "House",
  "genre": "Progressive House"
}
```

---

### Test 10: Progressive ambiguity — Progressive Rock

Input:

```json
{
  "embedded_genre": "Progressive Rock",
  "genres_all": ["classic rock", "prog rock"],
  "acousticness": 0.61,
  "energy": 0.64
}
```

Expected:

```json
{
  "family": "Rock / Metal / Guitar-Based",
  "genre": "Progressive Rock"
}
```

---

## 26. README Requirements

Create a README explaining:

1. What the classifier does
2. Input schema
3. Output schema
4. Taxonomy format
5. How source weights work
6. How confidence is calculated
7. How aliases work
8. How artist and label priors work
9. How BPM and audio features are used
10. How disambiguation rules work
11. How to classify one track
12. How to classify a batch
13. How to export results
14. How to inspect evidence and warnings
15. How to add new genres safely

---

## 27. Implementation Priorities

Build in this order:

1. Taxonomy loader and validation
2. Normalizer
3. Alias matcher
4. Evidence extraction
5. Candidate generation
6. Basic scoring
7. BPM/audio/tagger priors
8. Disambiguation rules
9. Confidence calculation
10. Batch processing
11. Tests
12. README
13. Optional LLM fallback

---

## 28. Design Principles

The module must be:

- deterministic by default
- explainable
- taxonomy-constrained
- configurable
- robust to missing data
- conservative under uncertainty
- easy to test
- practical for DJ workflows
- capable of returning alternatives
- able to expose evidence and warnings

Avoid:

- overconfident classification from weak metadata
- inventing non-taxonomy genres
- overusing mood words as genres
- letting Spotify artist-level genres dominate track-level evidence
- forcing subgenres when uncertain
- treating BPM as a hard rule
- treating words like dark, deep, melodic, or progressive as sufficient by themselves
- creating too many rare one-off categories

---

## 29. Final Deliverables

Produce:

1. Python implementation of the classifier
2. Taxonomy loader
3. Alias system
4. Normalization utilities
5. Evidence extraction module
6. Candidate scoring module
7. Disambiguation rule module
8. Pydantic schemas
9. Batch classification interface
10. JSON and CSV export
11. Unit tests
12. Example scripts
13. README
14. Config files for weights, aliases, BPM priors, artist priors, and label priors

---

## 30. Final Guidance for the Coding Agent

Start with a deterministic classifier. Do not implement an LLM-only genre classifier.

The best first version is:

```text
rules + taxonomy + weighted metadata + DJ tagger signals + BPM/audio priors + explicit disambiguation
```

A later version may add:

```text
optional constrained LLM fallback
```

However, all outputs must remain validated against the reference taxonomy.

The classifier should prefer reliable, explainable, and conservative results over false precision.
