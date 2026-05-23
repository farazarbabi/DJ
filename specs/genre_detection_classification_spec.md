# DJ Taxonomy Classifier — v2 Improvement Spec

Version: dj-taxonomy-v2.0
Status: draft (replaces all prior genre-classification specs)
Date: 2026-05-23

---

## 1. Objective and Non-Goals

### Objective

Improve the existing flat-label DJ-taxonomy classifier (`src/dj_registry/taxonomy/dj_model.py`) so that:

1. Over-broad catch-all buckets (notably `tribal_afro_driver` at 28.1% of the library, `organic_house_builder` at 22.8%) are split into accurate sub-types.
2. The catalog covers modern DJ-relevant genres absent today (Afro-Tech, 3-Step, bass house, hardgroove, cosmic indie, lo-fi deep tech, etc.).
3. Classification accuracy improves measurably against a refreshed ground-truth set: top-1 ≥ 50%, top-3 ≥ 75%, macro F1 ≥ 0.30, no single bucket > 20% of library inference.
4. Confidence-aware null fallback prevents force-fitting tracks with weak evidence into wrong categories.

### Non-Goals (explicit)

- **No 3-level hierarchy.** Labels stay flat: `id` + `label` (e.g. `afro_cinematic_builder` / "Afro Cinematic Builder"). No `family → genre → subgenre` nesting.
- **No replacement of the LR model.** Keep `scikit-learn` LogisticRegression + DictVectorizer in `src/dj_registry/taxonomy/dj_model.py`. Improvement is via taxonomy expansion + feature engineering + retraining, not architecture change.
- **No new ML stack.** No deep learning, no embeddings, no LLM at inference time. GPT remains used only for ground-truth seeding via `src/dj_registry/taxonomy/dj_ground_truth.py`.
- **No new top-level families.** The existing categories already cover Soul/R&B/Hip-Hop/Reggae/Rock placeholders for the user's library's small non-EDM tail. Additions go inside existing electronic families.
- **No backward-incompatible changes to the COMMENT tag format.** `KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]` stays. The CATEGORY code derives from `dj_taxonomy_label` via `compact_category_label()` unchanged.

---

## 2. Baseline State

Measured before this spec, as reported by the LR model's own training_report.json and a library distribution audit:

| Metric | Current value | Target |
|---|---:|---:|
| Categories total | 70 | ~100 |
| Top-1 accuracy (held-out) | 16.7% | ≥ 50% |
| Top-3 accuracy (held-out) | 41.7% | ≥ 75% |
| Macro F1 | 0.042 | ≥ 0.30 |
| Largest single-bucket share on full library | 28.1% (`tribal_afro_driver`, 95/338) | ≤ 20% |
| Training rows used | 60 of 247 (24%) | 247 of 247 (100%) |
| Categories seen during training | 20 of 70 | every category ≥ 3 examples |
| Mean confidence on top-3 buckets | ~0.50 (coin-flip) | ≥ 0.55 average; ≥ 0.70 on confident picks |

### Why the LR is broken (root cause)

Two compounding causes:

1. **Training utilization gap.** 187 of 247 ground-truth rows were never shown to the model. Of 70 categories, 50 had zero training examples — the model literally cannot output them.
2. **Over-broad catch-all categories.** `tribal_afro_driver` collapses any percussion-heavy mid-BPM track regardless of whether real afro/tribal genre signals are present in the providers. Same shape for `organic_house_builder`. The classifier's percussion + mid-BPM features over-fire and dominate the limited training signal.

These two interact: a bigger catalog without retraining-discipline would make the collapse worse (more low-training categories). Hence the spec's joint requirements: bigger catalog **and** full training utilization **and** anti-collapse feature constraints.

### Two named regression tracks

The new classifier must pass these specifically:

- **Nina Simone — *I Put a Spell on You*** (ISRC `USPR36600001`)
  - Providers report `["Jazz", "Vocal Jazz", "Vocal", "Soundtrack", "Dance"]`, BPM 89, low instrumentalness, acousticness 0.17, vocal-led tagger.
  - Current label: `organic_house_builder` (wrong).
  - Required: `warm_rnb_warmup` (or any non-electronic catch-all label — `alternative_rock_reset`, `afrobeat_funk_groove`, `reggae_dub_warmup` are also acceptable since the library's non-EDM coverage is minimal). Must NOT be classified as any house/techno/afro/tribal category.

- **Daval — *Robots (Original Mix)*** (ISRC `GBKQU2582247`)
  - Providers report `["Dance", "House", "Melodic / Progressive House", "Organic House"]`, BPM 114, danceability 0.804, energy 0.514, valence 0.856.
  - Current label: `organic_house_builder` (low confidence default).
  - Required: an `organic_house_*` or `melodic_house_*` sub-type — the providers explicitly say Organic House. Must NOT be classified as anything in the Tribal/Afro family.

---

## 3. Taxonomy Redesign — Splitting Rules

The 5 most over-populated buckets are split. Each split keeps the original `id` for the narrowest "central" variant of its old definition; sibling sub-types take new IDs. This minimizes data migration for tracks already labeled with the old ID — those tracks stay valid under the narrower kept variant, then re-classification rebalances naturally.

### 3.1 `tribal_afro_driver` (95 tracks → 6 sub-types)

Split axis: **BPM band + mood + vocal profile**.

| New / kept ID | Old? | Defines |
|---|---|---|
| `afro_tribal_warmup` | NEW | Lower BPM (108-116), warmup energy, organic warm mood |
| `afro_tribal_builder` | NEW | Mid BPM (116-120), building energy |
| `tribal_afro_driver` | kept (narrower) | Driving (120-126), peak-leaning percussion, instrumental |
| `spiritual_afro_chant` | NEW | Chant-led vocal profile, ritual/sacred mood |
| `afro_cinematic_builder` | NEW | Cinematic mood (atmospheric, soundtrack-like), builder role |
| `afro_3_step` | NEW | Modern 3-step rhythm pattern (afro-tech crossover) |

### 3.2 `organic_house_builder` (77 tracks → 5 sub-types)

Split axis: **mood + structure**.

| New / kept ID | Old? | Defines |
|---|---|---|
| `organic_house_warmup` | NEW | Slower (108-115), ambient leaning, warmup |
| `organic_house_builder` | kept (narrower) | Mid-tempo (115-122), building energy, warm/organic moods |
| `desert_organic_house` | NEW | Middle-Eastern / ethnic instrumentation, desert mood |
| `balearic_organic_house` | NEW | Sunset, warm, Ibiza-leaning |
| `organic_downtempo_crossover` | NEW | Sub-110 BPM crossover with downtempo / chillout |

### 3.3 `melodic_house_builder` (49 tracks → 3 sub-types)

Split axis: **emotional intensity**.

| New / kept ID | Old? | Defines |
|---|---|---|
| `warm_melodic_house` | NEW | Warm, gentle, low energy (E1-E2) |
| `melodic_house_builder` | kept (narrower) | Mid-energy (E2-E3) classic builder shape |
| `cinematic_melodic_house` | NEW | Atmospheric / dramatic / soundtrack-leaning |

### 3.4 `dark_indie_tech_house` (42 tracks → 3 sub-types)

Split axis: **structure / groove**.

| New / kept ID | Old? | Defines |
|---|---|---|
| `rolling_dark_indie_tech` | NEW | Rolling 16H structure, hypnotic |
| `driving_dark_indie_tech` | kept (narrower)* | Driving 16D structure, peak-leaning |
| `hypnotic_dark_indie_tech` | NEW | Slower hypnotic, late-night |

*Note: the original `dark_indie_tech_house` ID is retired in favor of `driving_dark_indie_tech` as the "default" interpretation, because most existing tagging used the driving structure. Migration path documented in §10.

### 3.5 `hypnotic_indie_dance` (~25 tracks → 3 sub-types)

Split axis: **mood**.

| New / kept ID | Old? | Defines |
|---|---|---|
| `warm_hypnotic_indie` | NEW | Warm, sunlit, balearic-adjacent |
| `hypnotic_indie_dance` | kept (narrower) | Default hypnotic indie dance |
| `psychedelic_hypnotic_indie` | NEW | Psychedelic / kraut-leaning |

---

## 4. Taxonomy Redesign — Modern Additions

15 new categories beyond the splits, covering current-generation DJ vocabulary that the legacy catalog ignored:

| ID | Label | Rationale |
|---|---|---|
| `afro_tech_driver` | Afro-Tech Driver | Afro House + Tech fusion is now a major scene; previously collapsed into tribal_afro_driver or tech_house_driver |
| `afro_house_peak` | Afro House Peak | Peak-time afro variant, distinct from driver-tier |
| `micro_house` | Micro House | Modern minimal house with micro-detail and ambient texture |
| `minimal_dub_tool` | Minimal Dub Tool | Dubby minimal tool tracks (Perlon-school) |
| `bass_tech_house` | Bass Tech-House | Bass house / G-house leaning, prominent bassline |
| `driving_bass_house` | Driving Bass House | Heavier bass-led house, festival-leaning |
| `hardgroove_techno_driver` | Hardgroove Techno Driver | Hardgroove resurgence; tribal-percussive techno |
| `cosmic_indie_dance` | Cosmic Indie Dance | Cosmic disco / space disco crossover with indie dance |
| `synth_wave_indie` | Synth-Wave Indie | Synth-wave / italo-disco leaning indie |
| `darkwave_indie_crossover` | Darkwave Indie Crossover | Modern darkwave / post-punk crossover into dance |
| `lo_fi_deep_tech` | Lo-Fi Deep Tech | Lo-fi aesthetic deep tech house |
| `dub_deep_tech_house` | Dub Deep Tech-House | Dub-influenced deep tech house |
| `sunset_balearic_house` | Sunset Balearic House | Sunset-leaning balearic; warm closer/opener |
| `slow_melodic_house` | Slow Melodic House | Slow tempo (95-110) melodic house |
| `psy_organic_crossover` | Psy-Organic Crossover | Psy-trance × organic house modern hybrid |

---

## 5. Full Taxonomy Enumeration

The complete v2 catalog (~100 entries) lives at `src/dj_registry/taxonomy/dj_taxonomy.json`. Schema for every entry — all fields are required:

```json
{
  "id": "<snake_case_id>",
  "label": "<Title Case Label>",
  "family": "<Family String>",
  "source_genres": ["<lowercase genre>", ...],
  "moods": ["<mood>", ...],
  "grooves": ["<groove>", ...],
  "set_roles": ["<role>", ...],
  "bpm_range": [<low>, <high>],
  "energy_range": [<low>, <high>],
  "vocal_profiles": ["<profile>", ...],
  "keywords": ["<keyword>", ...]
}
```

Field rules:
- `id` is unique, snake_case, stable (do NOT change after first ship — downstream tags/playlists reference it).
- `label` is the user-facing string (used in playlists, COMMENT tag CATEGORY field via `compact_category_label`).
- `family` is a free-text string (NOT enforced against an enum — kept for backward compat with the legacy schema).
- `source_genres` are normalized lowercase genre tokens. The classifier compares against `genres_all` from observations using fuzzy match.
- `moods` and `grooves` lowercase strings from a controlled vocabulary (see §15 of the original spec for vocabularies — kept verbatim).
- `set_roles` from {warmup, opener, builder, driver, peak, bridge, reset, tool}.
- `bpm_range` is `[low, high]` inclusive, integers.
- `energy_range` is `[low, high]` from 1-5.
- `vocal_profiles` from {instrumental, vocal, featured_vocal, spoken, chant, dub, tool}.
- `keywords` are matching tokens in title / mix / filename / comments.

### 5.1 Catalog (grouped by family)

#### Indie Dance / Dark Disco / Nu-Disco / Electro (12)

```json
[
  {"id":"hypnotic_indie_dance","label":"Hypnotic Indie Dance","family":"Indie Dance","source_genres":["indie dance","electronica","melodic house"],"moods":["hypnotic","melodic"],"grooves":["rolling","steady"],"set_roles":["builder","driver","bridge"],"bpm_range":[116,126],"energy_range":[2,4],"vocal_profiles":["instrumental","vocal"],"keywords":["indie","hypnotic","synth"]},
  {"id":"warm_hypnotic_indie","label":"Warm Hypnotic Indie","family":"Indie Dance","source_genres":["indie dance","balearic","melodic house"],"moods":["warm","sunlit","hypnotic"],"grooves":["rolling","steady"],"set_roles":["builder","warmup","bridge"],"bpm_range":[112,122],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["warm","sunset","balearic","indie"]},
  {"id":"psychedelic_hypnotic_indie","label":"Psychedelic Hypnotic Indie","family":"Indie Dance / Psychedelic","source_genres":["indie dance","psychedelic","krautrock"],"moods":["psychedelic","hypnotic","tense"],"grooves":["rolling","driving"],"set_roles":["driver","builder"],"bpm_range":[116,128],"energy_range":[2,4],"vocal_profiles":["instrumental","spoken"],"keywords":["psychedelic","kraut","indie","trippy"]},
  {"id":"indie_disco_groove","label":"Indie Disco Groove","family":"Indie Dance / Nu-Disco","source_genres":["indie dance","nu disco","disco"],"moods":["playful","warm","gritty"],"grooves":["groovy","swinging","dancefloor"],"set_roles":["builder","driver"],"bpm_range":[112,124],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["disco","funk","indie"]},
  {"id":"psychedelic_indie_driver","label":"Psychedelic Indie Driver","family":"Indie Dance","source_genres":["indie dance","psychedelic","electronica"],"moods":["psychedelic","tense","hypnotic"],"grooves":["driving","rolling"],"set_roles":["driver","peak"],"bpm_range":[120,128],"energy_range":[3,5],"vocal_profiles":["instrumental","spoken","vocal"],"keywords":["psychedelic","acid","indie","driver"]},
  {"id":"vocal_indie_dance","label":"Vocal Indie Dance","family":"Indie Dance","source_genres":["indie dance","electronica","melodic house"],"moods":["emotional","melodic"],"grooves":["rolling","steady"],"set_roles":["builder","bridge","reset"],"bpm_range":[112,124],"energy_range":[2,4],"vocal_profiles":["vocal","featured_vocal"],"keywords":["vocal","indie","song"]},
  {"id":"indie_electro_driver","label":"Indie Electro Driver","family":"Indie Dance / Electro","source_genres":["indie dance","electro","techno"],"moods":["raw","warehouse","gritty"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[122,132],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["warehouse","raw","indie","electro"]},
  {"id":"minimal_indie_chugger","label":"Minimal Indie Chugger","family":"Indie Dance","source_genres":["indie dance","minimal","electronica"],"moods":["minimal","hypnotic","dark"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","bridge"],"bpm_range":[110,122],"energy_range":[2,4],"vocal_profiles":["instrumental","dub"],"keywords":["chug","minimal","indie","slow"]},
  {"id":"acid_indie_dance","label":"Acid Indie Dance","family":"Indie Dance / Acid","source_genres":["indie dance","acid house","electronica"],"moods":["acidic","psychedelic","tense"],"grooves":["rolling","driving"],"set_roles":["driver","peak"],"bpm_range":[120,130],"energy_range":[3,5],"vocal_profiles":["instrumental","vocal"],"keywords":["acid","303","indie"]},
  {"id":"cosmic_indie_dance","label":"Cosmic Indie Dance","family":"Indie Dance / Cosmic Disco","source_genres":["indie dance","cosmic disco","space disco"],"moods":["sunlit","psychedelic","warm"],"grooves":["rolling","groovy"],"set_roles":["builder","bridge"],"bpm_range":[110,120],"energy_range":[2,3],"vocal_profiles":["instrumental","vocal"],"keywords":["cosmic","space","disco","cosmic"]},
  {"id":"synth_wave_indie","label":"Synth-Wave Indie","family":"Indie Dance / Synth-Wave","source_genres":["indie dance","synthwave","italo disco"],"moods":["warm","emotional","psychedelic"],"grooves":["steady","driving"],"set_roles":["bridge","builder"],"bpm_range":[108,122],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["synth","italo","retro","wave"]},
  {"id":"darkwave_indie_crossover","label":"Darkwave Indie Crossover","family":"Indie Dance / Darkwave","source_genres":["indie dance","darkwave","post-punk"],"moods":["dark","gothic","tense"],"grooves":["driving","steady"],"set_roles":["driver","peak","bridge"],"bpm_range":[110,128],"energy_range":[3,5],"vocal_profiles":["vocal","spoken"],"keywords":["darkwave","goth","post punk","crossover"]}
]
```

#### Tech-House (10)

```json
[
  {"id":"driving_dark_indie_tech","label":"Driving Dark Indie Tech-House","family":"Indie Dance / Tech-House","source_genres":["indie dance","tech house","melodic house"],"moods":["dark","tense","hypnotic"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[121,128],"energy_range":[3,5],"vocal_profiles":["instrumental","spoken"],"keywords":["dark","indie","tech","driving","club"]},
  {"id":"rolling_dark_indie_tech","label":"Rolling Dark Indie Tech-House","family":"Indie Dance / Tech-House","source_genres":["indie dance","tech house","minimal tech"],"moods":["dark","hypnotic","tense"],"grooves":["rolling","steady"],"set_roles":["driver","builder","bridge"],"bpm_range":[120,126],"energy_range":[3,4],"vocal_profiles":["instrumental","vocal","spoken"],"keywords":["rolling","dark","indie","tech"]},
  {"id":"hypnotic_dark_indie_tech","label":"Hypnotic Dark Indie Tech-House","family":"Indie Dance / Tech-House","source_genres":["indie dance","tech house","minimal"],"moods":["dark","hypnotic","minimal"],"grooves":["rolling","steady"],"set_roles":["builder","driver","tool"],"bpm_range":[118,124],"energy_range":[2,4],"vocal_profiles":["instrumental","dub","spoken"],"keywords":["hypnotic","late night","indie","tech"]},
  {"id":"dark_tech_house_driver","label":"Dark Tech-House Driver","family":"Tech-House","source_genres":["tech house","house","minimal tech"],"moods":["dark","tense","hypnotic"],"grooves":["driving","rolling","dancefloor"],"set_roles":["driver","peak"],"bpm_range":[123,130],"energy_range":[3,5],"vocal_profiles":["instrumental","vocal","spoken"],"keywords":["tech house","dark","driver"]},
  {"id":"percussive_tech_house","label":"Percussive Tech-House","family":"Tech-House","source_genres":["tech house","minimal tech","house"],"moods":["raw","gritty","warehouse"],"grooves":["groovy","swinging","dancefloor"],"set_roles":["driver","peak"],"bpm_range":[124,130],"energy_range":[3,5],"vocal_profiles":["instrumental","tool"],"keywords":["percussion","drums","tech house"]},
  {"id":"rolling_tech_house_tool","label":"Rolling Tech-House Tool","family":"Tech-House","source_genres":["tech house","minimal tech"],"moods":["minimal","hypnotic"],"grooves":["rolling","steady","dancefloor"],"set_roles":["tool","builder","driver"],"bpm_range":[123,130],"energy_range":[3,4],"vocal_profiles":["instrumental","tool","dub"],"keywords":["tool","rolling","tech house"]},
  {"id":"vocal_hook_tech_house","label":"Vocal Hook Tech-House","family":"Tech-House","source_genres":["tech house","house"],"moods":["playful","warm"],"grooves":["groovy","dancefloor"],"set_roles":["driver","peak","bridge"],"bpm_range":[123,130],"energy_range":[3,5],"vocal_profiles":["vocal","featured_vocal"],"keywords":["vocal","hook","extended","club"]},
  {"id":"acid_tech_house_peak","label":"Acid Tech-House Peak","family":"Acid Tech-House","source_genres":["acid house","tech house"],"moods":["acidic","tense","warehouse"],"grooves":["driving","dancefloor"],"set_roles":["peak","driver"],"bpm_range":[124,132],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["acid","303","peak","tech"]},
  {"id":"latin_tech_house","label":"Latin Tech-House","family":"Latin Tech-House","source_genres":["latin house","tech house","house"],"moods":["warm","playful","tribal"],"grooves":["groovy","swinging","percussive"],"set_roles":["builder","driver"],"bpm_range":[122,128],"energy_range":[3,5],"vocal_profiles":["vocal","chant","instrumental"],"keywords":["latin","tribal","percussion"]},
  {"id":"warehouse_tech_house_peak","label":"Warehouse Tech-House Peak","family":"Tech-House","source_genres":["tech house","techno"],"moods":["warehouse","raw","dark"],"grooves":["driving","linear"],"set_roles":["peak","driver"],"bpm_range":[126,133],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["warehouse","peak","raw"]}
]
```

#### Deep Tech / Bass Tech (5)

```json
[
  {"id":"minimal_deep_tech","label":"Minimal Deep Tech","family":"Minimal Deep Tech","source_genres":["minimal tech","tech house","deep tech"],"moods":["minimal","deep","hypnotic"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","tool"],"bpm_range":[120,127],"energy_range":[2,4],"vocal_profiles":["dub","instrumental","tool"],"keywords":["dub","minimal","deep tech"]},
  {"id":"lo_fi_deep_tech","label":"Lo-Fi Deep Tech","family":"Deep Tech / Lo-Fi","source_genres":["deep tech","minimal","lo-fi house"],"moods":["deep","warm","gritty"],"grooves":["rolling","loose"],"set_roles":["warmup","builder","bridge"],"bpm_range":[118,126],"energy_range":[2,3],"vocal_profiles":["instrumental","dub"],"keywords":["lofi","dusty","deep tech"]},
  {"id":"dub_deep_tech_house","label":"Dub Deep Tech-House","family":"Deep Tech / Dub","source_genres":["deep tech","tech house","dub house"],"moods":["deep","dub","minimal"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","tool"],"bpm_range":[120,128],"energy_range":[2,4],"vocal_profiles":["dub","instrumental","tool"],"keywords":["dub","deep tech","echo"]},
  {"id":"bass_tech_house","label":"Bass Tech-House","family":"Tech-House / Bass","source_genres":["tech house","bass house","g-house"],"moods":["raw","gritty","dark"],"grooves":["driving","dancefloor"],"set_roles":["driver","peak"],"bpm_range":[124,130],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken","vocal"],"keywords":["bass","g-house","tech house"]},
  {"id":"driving_bass_house","label":"Driving Bass House","family":"Bass House","source_genres":["bass house","house","g-house"],"moods":["raw","dark","tense"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[124,130],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["bass","driving","festival"]}
]
```

#### Organic House (5)

```json
[
  {"id":"organic_house_warmup","label":"Organic House Warmup","family":"Organic House","source_genres":["organic house","downtempo","ambient"],"moods":["warm","organic","atmospheric"],"grooves":["rolling","loose"],"set_roles":["warmup","opener"],"bpm_range":[108,116],"energy_range":[1,2],"vocal_profiles":["instrumental","vocal","chant"],"keywords":["organic","warmup","opener","slow"]},
  {"id":"organic_house_builder","label":"Organic House Builder","family":"Organic House","source_genres":["organic house","afro house","downtempo"],"moods":["warm","organic","melodic"],"grooves":["rolling","groovy"],"set_roles":["builder"],"bpm_range":[116,122],"energy_range":[2,3],"vocal_profiles":["instrumental","vocal","chant"],"keywords":["organic","builder"]},
  {"id":"desert_organic_house","label":"Desert Organic House","family":"Organic House / Desert","source_genres":["organic house","middle eastern","downtempo"],"moods":["organic","cinematic","warm"],"grooves":["rolling","groovy"],"set_roles":["builder","driver","bridge"],"bpm_range":[114,124],"energy_range":[2,3],"vocal_profiles":["chant","instrumental","vocal"],"keywords":["desert","middle eastern","ethnic","oud","duduk"]},
  {"id":"balearic_organic_house","label":"Balearic Organic House","family":"Organic House / Balearic","source_genres":["organic house","balearic","downtempo"],"moods":["sunlit","warm","organic"],"grooves":["loose","groovy","steady"],"set_roles":["warmup","builder","reset"],"bpm_range":[100,118],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["balearic","sunset","ibiza","warm"]},
  {"id":"organic_downtempo_crossover","label":"Organic Downtempo Crossover","family":"Organic House / Downtempo","source_genres":["organic house","downtempo","chillout"],"moods":["deep","atmospheric","warm"],"grooves":["loose","steady"],"set_roles":["warmup","reset","opener"],"bpm_range":[85,108],"energy_range":[1,2],"vocal_profiles":["instrumental","vocal","chant"],"keywords":["downtempo","crossover","organic","slow"]}
]
```

#### Afro House / Tribal (8)

```json
[
  {"id":"afro_tribal_warmup","label":"Afro Tribal Warmup","family":"Afro House","source_genres":["afro house","tribal house","organic house"],"moods":["warm","organic","tribal"],"grooves":["percussive","rolling","groovy"],"set_roles":["warmup","builder"],"bpm_range":[108,116],"energy_range":[1,3],"vocal_profiles":["chant","instrumental","vocal"],"keywords":["afro","warmup","slow","percussion"]},
  {"id":"afro_tribal_builder","label":"Afro Tribal Builder","family":"Afro House","source_genres":["afro house","tribal house","organic house"],"moods":["tribal","organic","warm"],"grooves":["percussive","rolling","groovy"],"set_roles":["builder","bridge"],"bpm_range":[116,120],"energy_range":[2,4],"vocal_profiles":["chant","instrumental"],"keywords":["afro","builder","percussion","tribal"]},
  {"id":"tribal_afro_driver","label":"Tribal Afro Driver","family":"Afro House","source_genres":["afro house","tribal house"],"moods":["tribal","hypnotic","raw"],"grooves":["percussive","driving","dancefloor"],"set_roles":["driver","peak"],"bpm_range":[120,126],"energy_range":[3,5],"vocal_profiles":["chant","instrumental"],"keywords":["afro","tribal","drums","driver"]},
  {"id":"afro_house_peak","label":"Afro House Peak","family":"Afro House","source_genres":["afro house","tribal house","house"],"moods":["tribal","euphoric","warm"],"grooves":["driving","groovy","dancefloor"],"set_roles":["peak","driver"],"bpm_range":[122,128],"energy_range":[4,5],"vocal_profiles":["vocal","chant","instrumental"],"keywords":["afro","peak","drums"]},
  {"id":"spiritual_afro_chant","label":"Spiritual Afro Chant","family":"Afro House / Spiritual","source_genres":["afro house","tribal house","spiritual"],"moods":["tribal","organic","cinematic"],"grooves":["percussive","rolling"],"set_roles":["builder","bridge","reset"],"bpm_range":[110,124],"energy_range":[2,4],"vocal_profiles":["chant","vocal"],"keywords":["spiritual","ritual","chant","ceremony"]},
  {"id":"afro_cinematic_builder","label":"Afro Cinematic Builder","family":"Afro House / Cinematic","source_genres":["afro house","melodic house","cinematic"],"moods":["cinematic","warm","organic"],"grooves":["rolling","groovy"],"set_roles":["builder","warmup"],"bpm_range":[114,122],"energy_range":[2,3],"vocal_profiles":["instrumental","chant","vocal"],"keywords":["cinematic","afro","builder","soundtrack"]},
  {"id":"afro_3_step","label":"Afro 3-Step","family":"Afro House / 3-Step","source_genres":["afro house","3-step","tech house"],"moods":["tribal","raw","playful"],"grooves":["broken","percussive"],"set_roles":["driver","bridge","peak"],"bpm_range":[120,128],"energy_range":[3,5],"vocal_profiles":["chant","vocal","instrumental"],"keywords":["3-step","afro","syncopated"]},
  {"id":"afro_tech_driver","label":"Afro-Tech Driver","family":"Afro-Tech","source_genres":["afro house","tech house","afro tech"],"moods":["tribal","driving","warm"],"grooves":["driving","percussive"],"set_roles":["driver","peak"],"bpm_range":[122,128],"energy_range":[3,5],"vocal_profiles":["instrumental","chant","spoken"],"keywords":["afro tech","tech","driver"]}
]
```

#### Tribal / Organic / Latin Crossover (3)

```json
[
  {"id":"deep_afro_house","label":"Deep Afro House","family":"Afro / Deep House","source_genres":["afro house","deep house","organic house"],"moods":["deep","warm","sunlit"],"grooves":["groovy","rolling"],"set_roles":["warmup","builder","bridge"],"bpm_range":[112,122],"energy_range":[1,3],"vocal_profiles":["vocal","chant","instrumental"],"keywords":["afro","deep"]},
  {"id":"organic_chant_house","label":"Organic Chant House","family":"Organic / Tribal","source_genres":["organic house","tribal house","afro house"],"moods":["tribal","organic"],"grooves":["percussive","rolling"],"set_roles":["builder","driver","reset"],"bpm_range":[108,124],"energy_range":[2,4],"vocal_profiles":["chant","vocal"],"keywords":["ritual","chant","ceremony","organic"]},
  {"id":"psy_organic_crossover","label":"Psy-Organic Crossover","family":"Psy / Organic House","source_genres":["organic house","psytrance","progressive trance"],"moods":["psychedelic","organic","hypnotic"],"grooves":["rolling","driving"],"set_roles":["builder","driver"],"bpm_range":[118,126],"energy_range":[3,4],"vocal_profiles":["chant","instrumental"],"keywords":["psy organic","crossover","tribal","trippy"]}
]
```

#### Progressive / Melodic House (7)

```json
[
  {"id":"dark_progressive_house","label":"Dark Progressive House","family":"Progressive House","source_genres":["progressive house","melodic house","melodic techno"],"moods":["dark","melodic","hypnotic"],"grooves":["rolling","driving"],"set_roles":["builder","driver","peak"],"bpm_range":[120,128],"energy_range":[3,5],"vocal_profiles":["instrumental","vocal"],"keywords":["progressive","dark","melodic"]},
  {"id":"progressive_house_peak","label":"Progressive House Peak","family":"Progressive House","source_genres":["progressive house","melodic house","trance"],"moods":["euphoric","melodic","emotional"],"grooves":["driving","linear"],"set_roles":["peak","driver"],"bpm_range":[124,132],"energy_range":[4,5],"vocal_profiles":["instrumental","vocal"],"keywords":["progressive","peak"]},
  {"id":"deep_progressive_warmup","label":"Deep Progressive Warmup","family":"Progressive House","source_genres":["progressive house","deep house"],"moods":["deep","warm","melodic"],"grooves":["rolling","steady"],"set_roles":["warmup","builder"],"bpm_range":[112,122],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["deep","progressive","warmup"]},
  {"id":"warm_melodic_house","label":"Warm Melodic House","family":"Melodic House","source_genres":["melodic house","progressive house"],"moods":["warm","melodic","emotional"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","bridge"],"bpm_range":[112,122],"energy_range":[1,2],"vocal_profiles":["vocal","instrumental"],"keywords":["warm","melodic","emotional"]},
  {"id":"melodic_house_builder","label":"Melodic House Builder","family":"Melodic House","source_genres":["melodic house","progressive house","electronica"],"moods":["melodic","warm"],"grooves":["rolling","steady"],"set_roles":["builder","bridge"],"bpm_range":[116,124],"energy_range":[2,3],"vocal_profiles":["instrumental","vocal"],"keywords":["melodic","house"]},
  {"id":"cinematic_melodic_house","label":"Cinematic Melodic House","family":"Melodic House / Cinematic","source_genres":["melodic house","cinematic","soundtrack"],"moods":["cinematic","melodic","emotional"],"grooves":["rolling","steady","loose"],"set_roles":["builder","bridge","reset"],"bpm_range":[112,124],"energy_range":[2,3],"vocal_profiles":["instrumental","vocal"],"keywords":["cinematic","melodic","soundtrack"]},
  {"id":"slow_melodic_house","label":"Slow Melodic House","family":"Melodic House / Slow","source_genres":["melodic house","slow house","downtempo"],"moods":["warm","melodic","atmospheric"],"grooves":["loose","steady"],"set_roles":["warmup","reset"],"bpm_range":[95,112],"energy_range":[1,2],"vocal_profiles":["vocal","instrumental"],"keywords":["slow","melodic","downtempo"]}
]
```

#### Melodic Techno (3)

```json
[
  {"id":"melodic_techno_driver","label":"Melodic Techno Driver","family":"Melodic Techno","source_genres":["melodic techno","techno","progressive house"],"moods":["melodic","dark","emotional"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[124,132],"energy_range":[3,5],"vocal_profiles":["instrumental","vocal"],"keywords":["melodic techno","driver"]},
  {"id":"emotional_melodic_techno","label":"Emotional Melodic Techno","family":"Melodic Techno","source_genres":["melodic techno","cinematic","melodic house"],"moods":["emotional","cinematic","melodic"],"grooves":["rolling","steady"],"set_roles":["builder","bridge"],"bpm_range":[120,128],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["emotional","cinematic","melodic techno"]},
  {"id":"dark_melodic_techno_driver","label":"Dark Melodic Techno Driver","family":"Melodic Techno / Dark","source_genres":["melodic techno","techno","dark"],"moods":["dark","melodic","tense"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[124,134],"energy_range":[3,5],"vocal_profiles":["instrumental","spoken"],"keywords":["dark","melodic techno","driver"]}
]
```

#### Techno (7)

```json
[
  {"id":"raw_warehouse_techno","label":"Raw Warehouse Techno","family":"Techno","source_genres":["techno","raw techno","industrial"],"moods":["raw","warehouse","dark"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[128,140],"energy_range":[4,5],"vocal_profiles":["instrumental","tool"],"keywords":["warehouse","raw","techno"]},
  {"id":"hypnotic_deep_techno","label":"Hypnotic Deep Techno","family":"Deep Techno","source_genres":["deep techno","techno","dub techno"],"moods":["hypnotic","deep","minimal"],"grooves":["rolling","steady"],"set_roles":["builder","driver","tool"],"bpm_range":[124,134],"energy_range":[2,4],"vocal_profiles":["instrumental","dub","tool"],"keywords":["hypnotic","deep techno","dub"]},
  {"id":"peak_time_techno","label":"Peak-Time Techno","family":"Techno","source_genres":["techno","peak time techno"],"moods":["dark","tense","warehouse"],"grooves":["driving","linear"],"set_roles":["peak","driver"],"bpm_range":[130,142],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["peak","techno"]},
  {"id":"tribal_techno","label":"Tribal Techno","family":"Techno / Tribal","source_genres":["techno","tribal techno"],"moods":["tribal","raw","hypnotic"],"grooves":["percussive","groovy","driving"],"set_roles":["driver","peak"],"bpm_range":[126,136],"energy_range":[3,5],"vocal_profiles":["instrumental","chant"],"keywords":["tribal","techno","percussion"]},
  {"id":"dub_techno","label":"Dub Techno","family":"Dub Techno","source_genres":["dub techno","deep techno","ambient techno"],"moods":["deep","atmospheric","minimal"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","reset"],"bpm_range":[112,128],"energy_range":[1,3],"vocal_profiles":["dub","instrumental"],"keywords":["dub techno","echo"]},
  {"id":"acid_techno","label":"Acid Techno","family":"Acid Techno","source_genres":["acid techno","techno"],"moods":["acidic","warehouse","tense"],"grooves":["driving","linear"],"set_roles":["peak","driver"],"bpm_range":[130,145],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["acid","303","warehouse"]},
  {"id":"minimal_techno_tool","label":"Minimal Techno Tool","family":"Minimal Techno","source_genres":["minimal techno","techno"],"moods":["minimal","hypnotic"],"grooves":["steady","tool","rolling"],"set_roles":["tool","builder","driver"],"bpm_range":[124,134],"energy_range":[2,4],"vocal_profiles":["instrumental","tool"],"keywords":["minimal","tool","techno"]}
]
```

#### Hardgroove + Modern Minimal (3)

```json
[
  {"id":"hardgroove_techno_driver","label":"Hardgroove Techno Driver","family":"Hardgroove","source_genres":["hardgroove","techno","tribal techno"],"moods":["raw","tribal","driving"],"grooves":["driving","percussive","linear"],"set_roles":["driver","peak"],"bpm_range":[132,142],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken","chant"],"keywords":["hardgroove","drums","driving"]},
  {"id":"micro_house","label":"Micro House","family":"Micro House","source_genres":["micro house","minimal","tech house"],"moods":["minimal","warm","playful"],"grooves":["steady","groovy"],"set_roles":["builder","tool"],"bpm_range":[118,126],"energy_range":[2,3],"vocal_profiles":["instrumental","tool"],"keywords":["micro","minimal","clicks","textures"]},
  {"id":"minimal_dub_tool","label":"Minimal Dub Tool","family":"Minimal / Dub","source_genres":["minimal","dub techno","minimal house"],"moods":["minimal","dub","atmospheric"],"grooves":["steady","tool"],"set_roles":["tool","reset"],"bpm_range":[120,128],"energy_range":[2,3],"vocal_profiles":["dub","tool","instrumental"],"keywords":["dub tool","minimal","perlon"]}
]
```

#### Deep House / House (7)

```json
[
  {"id":"warm_deep_house","label":"Warm Deep House","family":"Deep House","source_genres":["deep house","house"],"moods":["warm","deep","soulful"],"grooves":["groovy","rolling"],"set_roles":["warmup","builder"],"bpm_range":[112,124],"energy_range":[1,3],"vocal_profiles":["vocal","instrumental"],"keywords":["deep house","warm","soul"]},
  {"id":"dub_deep_house","label":"Dub Deep House","family":"Deep House / Dub","source_genres":["deep house","dub house"],"moods":["deep","minimal","warm"],"grooves":["rolling","steady"],"set_roles":["warmup","builder","tool"],"bpm_range":[112,124],"energy_range":[1,3],"vocal_profiles":["dub","instrumental"],"keywords":["dub","deep house","echo"]},
  {"id":"soulful_vocal_house","label":"Soulful Vocal House","family":"House","source_genres":["house","soulful house","deep house"],"moods":["warm","soulful","euphoric"],"grooves":["groovy","dancefloor"],"set_roles":["builder","bridge","peak"],"bpm_range":[116,126],"energy_range":[2,4],"vocal_profiles":["vocal","featured_vocal"],"keywords":["soulful","vocal","house"]},
  {"id":"garage_house","label":"Garage House","family":"House / Garage","source_genres":["garage","house","uk garage"],"moods":["warm","playful"],"grooves":["swinging","broken","dancefloor"],"set_roles":["builder","driver","bridge"],"bpm_range":[122,132],"energy_range":[2,5],"vocal_profiles":["vocal","dub"],"keywords":["garage","swing","ukg"]},
  {"id":"raw_acid_house","label":"Raw Acid House","family":"Acid House","source_genres":["acid house","house"],"moods":["acidic","raw","warehouse"],"grooves":["rolling","dancefloor"],"set_roles":["driver","peak"],"bpm_range":[120,132],"energy_range":[3,5],"vocal_profiles":["instrumental","spoken"],"keywords":["acid house","303","raw"]},
  {"id":"classic_house","label":"Classic House","family":"House","source_genres":["house","classic house"],"moods":["warm","euphoric","playful"],"grooves":["groovy","dancefloor"],"set_roles":["builder","peak","bridge"],"bpm_range":[118,128],"energy_range":[2,5],"vocal_profiles":["vocal","instrumental"],"keywords":["classic","house","piano"]},
  {"id":"tribal_house","label":"Tribal House","family":"Tribal House","source_genres":["tribal house","house"],"moods":["tribal","organic","raw"],"grooves":["percussive","driving"],"set_roles":["driver","peak"],"bpm_range":[122,130],"energy_range":[3,5],"vocal_profiles":["chant","instrumental"],"keywords":["tribal house","drums"]}
]
```

#### Low-BPM Deep House / Slow House (5)

```json
[
  {"id":"low_slung_deep_house","label":"Low-Slung Deep House","family":"Low-BPM Deep House","source_genres":["deep house","slow house","low slung house"],"moods":["deep","warm","hypnotic"],"grooves":["loose","rolling","groovy"],"set_roles":["warmup","builder","bridge"],"bpm_range":[96,114],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal","dub"],"keywords":["slow house","deep","low bpm","warmup"]},
  {"id":"balearic_deep_house","label":"Balearic Deep House","family":"Balearic / Deep House","source_genres":["balearic","deep house","downtempo house"],"moods":["sunlit","warm","deep"],"grooves":["loose","steady","groovy"],"set_roles":["warmup","builder","reset"],"bpm_range":[92,112],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["balearic","sunset","deep house","warm"]},
  {"id":"sunset_balearic_house","label":"Sunset Balearic House","family":"Balearic / Sunset","source_genres":["balearic","deep house","ambient"],"moods":["sunlit","warm","atmospheric"],"grooves":["loose","steady"],"set_roles":["warmup","reset","opener"],"bpm_range":[88,108],"energy_range":[1,2],"vocal_profiles":["instrumental","vocal"],"keywords":["sunset","balearic","golden hour","ibiza"]},
  {"id":"dubbed_out_slow_house","label":"Dubbed-Out Slow House","family":"Low-BPM Deep House / Dub","source_genres":["dub house","slow house","deep house"],"moods":["deep","minimal","atmospheric"],"grooves":["rolling","steady","loose"],"set_roles":["warmup","tool","bridge"],"bpm_range":[98,118],"energy_range":[1,3],"vocal_profiles":["dub","instrumental","tool"],"keywords":["dub","slow house","echo","deep"]},
  {"id":"lo_fi_deep_house","label":"Lo-Fi Deep House","family":"Lo-Fi / Deep House","source_genres":["lo-fi house","deep house","house"],"moods":["warm","gritty","deep"],"grooves":["groovy","loose","steady"],"set_roles":["warmup","builder","bridge"],"bpm_range":[108,122],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["lofi","dusty","deep house","warm"]}
]
```

#### Nu-Disco / Funky (3)

```json
[
  {"id":"nu_disco_house","label":"Nu-Disco House","family":"Nu-Disco","source_genres":["nu disco","disco","indie dance"],"moods":["sunlit","warm","playful"],"grooves":["groovy","swinging","dancefloor"],"set_roles":["warmup","builder","bridge"],"bpm_range":[106,122],"energy_range":[1,4],"vocal_profiles":["vocal","instrumental"],"keywords":["disco","funk"]},
  {"id":"dark_nu_disco_chug","label":"Dark Nu-Disco Chug","family":"Nu-Disco / Indie Dance","source_genres":["nu disco","indie dance"],"moods":["dark","hypnotic","playful"],"grooves":["rolling","groovy"],"set_roles":["builder","driver"],"bpm_range":[108,122],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["disco","chug","dark"]},
  {"id":"funky_disco_house","label":"Funky Disco House","family":"Disco / House","source_genres":["disco","funky house","house"],"moods":["euphoric","playful","warm"],"grooves":["groovy","dancefloor"],"set_roles":["peak","driver","bridge"],"bpm_range":[118,126],"energy_range":[3,5],"vocal_profiles":["vocal","instrumental"],"keywords":["funk","disco","peak"]}
]
```

#### Breakbeat / Electro / Leftfield (4)

```json
[
  {"id":"breakbeat_house","label":"Breakbeat House","family":"Breakbeat / House","source_genres":["breakbeat","breaks","house"],"moods":["playful","warm"],"grooves":["broken","bridge","dancefloor"],"set_roles":["bridge","builder","driver"],"bpm_range":[118,132],"energy_range":[2,5],"vocal_profiles":["vocal","instrumental"],"keywords":["breakbeat","breaks","house"]},
  {"id":"dark_electro_breaks","label":"Dark Electro Breaks","family":"Electro / Breaks","source_genres":["electro","breaks","breakbeat"],"moods":["dark","raw","warehouse"],"grooves":["broken","driving"],"set_roles":["driver","peak","bridge"],"bpm_range":[120,135],"energy_range":[3,5],"vocal_profiles":["instrumental","spoken"],"keywords":["electro","breaks","dark"]},
  {"id":"acid_breaks","label":"Acid Breaks","family":"Acid Breaks","source_genres":["acid","breaks","electro"],"moods":["acidic","psychedelic","tense"],"grooves":["broken","driving"],"set_roles":["peak","driver"],"bpm_range":[124,136],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["acid","breaks","303"]},
  {"id":"leftfield_bass_bridge","label":"Leftfield Bass Bridge","family":"Leftfield Bass","source_genres":["leftfield","bass","electronica"],"moods":["cinematic","deep","raw"],"grooves":["broken","reset"],"set_roles":["reset","bridge"],"bpm_range":[80,122],"energy_range":[1,4],"vocal_profiles":["instrumental","vocal"],"keywords":["leftfield","bass","reset"]}
]
```

#### Downtempo / Electronica / Ambient (3)

```json
[
  {"id":"downtempo_opener","label":"Downtempo Opener","family":"Downtempo","source_genres":["downtempo","electronica","chillout"],"moods":["deep","warm","atmospheric"],"grooves":["loose","steady"],"set_roles":["warmup","reset"],"bpm_range":[70,105],"energy_range":[1,2],"vocal_profiles":["instrumental","vocal"],"keywords":["downtempo","opener"]},
  {"id":"cinematic_electronica","label":"Cinematic Electronica","family":"Electronica","source_genres":["electronica","ambient","downtempo"],"moods":["cinematic","atmospheric"],"grooves":["ambient","reset","loose"],"set_roles":["reset","warmup"],"bpm_range":[60,112],"energy_range":[1,2],"vocal_profiles":["instrumental","vocal"],"keywords":["cinematic","electronica","ambient"]},
  {"id":"deep_bass_electronica","label":"Deep Bass Electronica","family":"Electronica / Bass","source_genres":["electronica","bass","downtempo"],"moods":["deep","subby","dark"],"grooves":["loose","broken"],"set_roles":["warmup","reset","bridge"],"bpm_range":[70,115],"energy_range":[1,3],"vocal_profiles":["instrumental","vocal"],"keywords":["sub","bass","electronica"]}
]
```

#### Trance / Psy (2)

```json
[
  {"id":"psy_trance_driver","label":"Psy-Trance Driver","family":"Psy / Trance","source_genres":["psytrance","trance","progressive trance"],"moods":["psychedelic","hypnotic","tense"],"grooves":["driving","linear"],"set_roles":["driver","peak"],"bpm_range":[132,145],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["psy","trance","driver"]},
  {"id":"progressive_trance_builder","label":"Progressive Trance Builder","family":"Progressive Trance","source_genres":["progressive trance","trance","progressive house"],"moods":["euphoric","melodic","hypnotic"],"grooves":["rolling","driving"],"set_roles":["builder","driver","peak"],"bpm_range":[126,138],"energy_range":[3,5],"vocal_profiles":["instrumental","vocal"],"keywords":["trance","progressive","builder"]}
]
```

#### Drum & Bass / Jungle / Bass (4)

```json
[
  {"id":"liquid_drum_and_bass","label":"Liquid Drum & Bass","family":"Drum & Bass","source_genres":["drum and bass","liquid drum and bass","liquid funk"],"moods":["melodic","warm","euphoric"],"grooves":["broken","rolling","dancefloor"],"set_roles":["builder","driver","bridge"],"bpm_range":[160,176],"energy_range":[3,5],"vocal_profiles":["vocal","instrumental"],"keywords":["liquid","dnb","rolling","melodic"]},
  {"id":"dark_dnb_roller","label":"Dark DnB Roller","family":"Drum & Bass","source_genres":["drum and bass","neurofunk","techstep"],"moods":["dark","tense","raw"],"grooves":["broken","driving","rolling"],"set_roles":["driver","peak"],"bpm_range":[168,176],"energy_range":[4,5],"vocal_profiles":["instrumental","spoken"],"keywords":["dnb","roller","dark","bass"]},
  {"id":"jungle_breaks","label":"Jungle Breaks","family":"Jungle / Breakbeat","source_genres":["jungle","breakbeat","drum and bass"],"moods":["raw","playful","gritty"],"grooves":["broken","swinging","dancefloor"],"set_roles":["driver","bridge","peak"],"bpm_range":[155,175],"energy_range":[3,5],"vocal_profiles":["vocal","spoken","instrumental"],"keywords":["jungle","breaks","amen","ragga"]},
  {"id":"halftime_bass_weight","label":"Halftime Bass Weight","family":"Halftime Bass","source_genres":["halftime","bass music","dubstep"],"moods":["subby","dark","minimal"],"grooves":["broken","loose","reset"],"set_roles":["bridge","reset","driver"],"bpm_range":[80,92],"energy_range":[2,4],"vocal_profiles":["instrumental","spoken","dub"],"keywords":["halftime","bass","sub","weight"]}
]
```

#### Rock / Indie / Post-Punk (5)

```json
[
  {"id":"indie_rock_club_groove","label":"Indie Rock Club Groove","family":"Rock / Indie","source_genres":["indie rock","alternative rock","dance rock"],"moods":["playful","warm","gritty"],"grooves":["groovy","steady","dancefloor"],"set_roles":["bridge","builder","driver"],"bpm_range":[95,128],"energy_range":[2,4],"vocal_profiles":["vocal","featured_vocal"],"keywords":["indie rock","guitar","club","dance rock"]},
  {"id":"dark_post_punk_dance","label":"Dark Post-Punk Dance","family":"Post-Punk / Darkwave","source_genres":["post-punk","darkwave","new wave"],"moods":["dark","tense","gritty"],"grooves":["driving","steady","dancefloor"],"set_roles":["bridge","driver","peak"],"bpm_range":[105,132],"energy_range":[3,5],"vocal_profiles":["vocal","spoken"],"keywords":["post punk","darkwave","goth","guitar"]},
  {"id":"psychedelic_rock_groove","label":"Psychedelic Rock Groove","family":"Rock / Psychedelic","source_genres":["psychedelic rock","krautrock","indie rock"],"moods":["psychedelic","hypnotic","raw"],"grooves":["rolling","steady","groovy"],"set_roles":["bridge","builder","reset"],"bpm_range":[85,124],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["psychedelic rock","kraut","guitar","trip"]},
  {"id":"alternative_rock_reset","label":"Alternative Rock Reset","family":"Rock / Alternative","source_genres":["alternative rock","indie rock","modern rock"],"moods":["emotional","raw","melodic"],"grooves":["steady","reset"],"set_roles":["reset","bridge","warmup"],"bpm_range":[75,118],"energy_range":[1,4],"vocal_profiles":["featured_vocal","vocal"],"keywords":["alternative rock","song","reset","guitar"]},
  {"id":"funk_rock_disco_bridge","label":"Funk-Rock Disco Bridge","family":"Funk / Rock / Disco","source_genres":["funk rock","disco rock","dance rock"],"moods":["playful","warm","soulful"],"grooves":["groovy","swinging","dancefloor"],"set_roles":["bridge","builder","driver"],"bpm_range":[95,122],"energy_range":[2,4],"vocal_profiles":["vocal","instrumental"],"keywords":["funk rock","disco","guitar","groove"]}
]
```

#### Hip-Hop / R&B / Soul / Funk / Reggae / Leftfield (6)

```json
[
  {"id":"golden_era_hip_hop_groove","label":"Golden-Era Hip-Hop Groove","family":"Hip-Hop","source_genres":["hip hop","boom bap","rap"],"moods":["warm","soulful","gritty"],"grooves":["headnod","steady","loose"],"set_roles":["warmup","bridge","reset"],"bpm_range":[78,104],"energy_range":[1,3],"vocal_profiles":["spoken","vocal"],"keywords":["hip hop","boom bap","rap","break"]},
  {"id":"trap_bass_bridge","label":"Trap Bass Bridge","family":"Hip-Hop / Trap","source_genres":["trap","hip hop","bass music"],"moods":["dark","subby","raw"],"grooves":["loose","broken","reset"],"set_roles":["bridge","reset","driver"],"bpm_range":[65,85],"energy_range":[2,4],"vocal_profiles":["spoken","vocal","instrumental"],"keywords":["trap","808","rap","bass"]},
  {"id":"warm_rnb_warmup","label":"Warm R&B Warmup","family":"R&B / Soul","source_genres":["r&b","soul","neo soul"],"moods":["warm","soulful","emotional"],"grooves":["loose","groovy","steady"],"set_roles":["warmup","reset","bridge"],"bpm_range":[68,108],"energy_range":[1,3],"vocal_profiles":["featured_vocal","vocal"],"keywords":["rnb","soul","warmup","vocal"]},
  {"id":"afrobeat_funk_groove","label":"Afrobeat Funk Groove","family":"Afrobeat / Funk","source_genres":["afrobeat","funk","world"],"moods":["warm","organic","playful"],"grooves":["percussive","groovy","dancefloor"],"set_roles":["warmup","builder","bridge"],"bpm_range":[95,118],"energy_range":[2,4],"vocal_profiles":["vocal","chant","instrumental"],"keywords":["afrobeat","funk","horns","percussion"]},
  {"id":"reggae_dub_warmup","label":"Reggae Dub Warmup","family":"Reggae / Dub","source_genres":["reggae","dub","roots reggae"],"moods":["warm","deep","sunlit"],"grooves":["loose","steady","skank"],"set_roles":["warmup","reset","bridge"],"bpm_range":[68,100],"energy_range":[1,3],"vocal_profiles":["dub","vocal","instrumental"],"keywords":["reggae","dub","skank","roots"]},
  {"id":"leftfield_club_tool","label":"Leftfield Club Tool","family":"Leftfield Club","source_genres":["leftfield","electronica","club"],"moods":["raw","minimal","hypnotic"],"grooves":["broken","rolling","tool"],"set_roles":["tool","bridge","driver"],"bpm_range":[100,132],"energy_range":[2,5],"vocal_profiles":["instrumental","tool"],"keywords":["leftfield","club","tool"]}
]
```

#### Dub Indie crossover (1)

```json
[
  {"id":"dub_indie_dance","label":"Dub Indie Dance","family":"Indie Dance / Dub","source_genres":["indie dance","dub","electronica"],"moods":["dub","hypnotic","deep"],"grooves":["rolling","steady"],"set_roles":["bridge","tool","builder"],"bpm_range":[112,124],"energy_range":[2,4],"vocal_profiles":["dub","instrumental"],"keywords":["dub","indie","echo"]}
]
```

### 5.2 Category totals and migration

- **Final count**: 99 categories (within the 90-110 target)
- **Kept unchanged** (narrowed scope only): 65 IDs from v1 retained verbatim
- **Kept with narrower scope (same ID, redefined boundary)**: 5 — `tribal_afro_driver`, `organic_house_builder`, `melodic_house_builder`, `hypnotic_indie_dance`, plus implicit `melodic_techno_driver` (kept as-is, but new sibling `dark_melodic_techno_driver` and `emotional_melodic_techno` added)
- **Retired**: 1 — `dark_indie_tech_house` (split into 3 new IDs; existing ground-truth rows mapped to `driving_dark_indie_tech` as the default heir)
- **New from splits**: 14
  - From `organic_house_builder`: 4 (warmup, desert, balearic, downtempo_crossover)
  - From `tribal_afro_driver`: 5 (warmup, builder, spiritual, cinematic, 3_step)
  - From `melodic_house_builder`: 2 (warm_melodic, cinematic_melodic)
  - From `dark_indie_tech_house`: 3 (rolling, driving, hypnotic)
  - From `hypnotic_indie_dance`: 2 (warm, psychedelic)  *(hypnotic_indie_dance itself stays as the central variant)*
  - Wait — actual new IDs from splits per §3 above: 4 + 5 + 2 + 3 + 2 = **16**
- **Net-new modern additions**: 14 (one — `psy_organic_crossover` — appears under "Organic Crossover" in §5.1)

Migration table (legacy ID → v2 default mapping for existing ground-truth rows):

| Legacy ID | v2 default mapping for re-classification |
|---|---|
| `dark_indie_tech_house` | `driving_dark_indie_tech` |
| (all others) | unchanged |

GPT re-labeling (§8) is allowed to override defaults when evidence clearly points to a sibling sub-type.

---

## 6. Feature Engineering Upgrades

Requirements for `src/dj_registry/taxonomy/features.py`. The classifier's feature dict must be enriched and re-weighted to fix the tribal_afro_driver collapse and improve overall accuracy.

### 6.1 Up-weight provider genres

Provider genre fields (`genres_all`, `rekordbox_genre`, `songstats_genre`, `embedded_genre`) are the strongest evidence source available. They are *underweighted* relative to tagger fields in the current code.

Required changes:
- Source weight for `genres_all` raised to **0.85** (was 0.70 per spec §10)
- Source weight for `rekordbox_genre` raised to **0.90** if manually-tagged status known, **0.75** if origin unclear (current: 0.90 and 0.75 — but actually wired at lower effective weight; verify in features.py)
- When provider returns an unambiguous specific genre (e.g. "Organic House") and no contradicting signal, the matched candidate must score *above* any pure tagger-based candidate. The LR can re-learn this with proper feature scaling.

Implementation hook: `features.py:build_track_features()`, the section currently labeled "External-only features (lines 219-268)".

### 6.2 Anti-collapse rule for tribal/afro classification (HARD RULE)

The fundamental fix for the tribal_afro_driver collapse. This is a feature-level constraint, not a soft prior.

A track may score for any category in the `Afro House`, `Tribal House`, `Afro-Tech`, or `Tribal Techno` families **only if at least one** of these conditions holds:

(a) `genres_all`, `rekordbox_genre`, `songstats_genre`, `spotify_genres`, or `embedded_genre` contains a token matching `/(afro|tribal|spiritual house|3-step)/i`, OR
(b) the track's artist is in `data/artist_priors.json` with an explicit afro/tribal prior, OR
(c) the track's label (`label_canonical`) is in `data/label_priors.json` with an explicit afro/tribal prior, OR
(d) the track's `mix_canonical` or title contains a recognized "afro" / "tribal" / "shamanic" / "ritual" mix-name token.

If NONE of these hold, the classifier must zero out scores for all afro/tribal categories at the candidate-generation stage, regardless of how strongly the tagger's `TRIB` mood or `chant` vocal cue fires. Percussion-heavy mid-BPM tracks lacking explicit afro/tribal provider signals are NOT afro.

Implementation: add a `_is_eligible_for_afro_tribal(track)` helper in `features.py`. After candidate generation, filter the candidate set:

```python
if not _is_eligible_for_afro_tribal(track):
    candidates = [c for c in candidates if c.matched_path.family not in AFRO_TRIBAL_FAMILIES]
```

`AFRO_TRIBAL_FAMILIES` is a module constant listing the family strings used by the eligible categories.

### 6.3 Mix-name parsing

Extract recognized tokens from `mix_canonical` and surface each as its own categorical feature. Recognized tokens (case-insensitive substring match, normalized first):
- `dub mix`, `dub remix` → `mix:dub`
- `extended mix`, `extended version` → `mix:extended`
- `tribal mix` → `mix:tribal`
- `afro mix` → `mix:afro`
- `club mix`, `club edit` → `mix:club`
- `rework`, `vip mix`, `vip edit` → `mix:rework`
- `instrumental mix` → `mix:instrumental`
- `radio edit`, `radio mix` → `mix:radio` (down-weights driver/peak roles)

The `mix:dub`, `mix:tribal`, `mix:afro` tokens are inputs to §6.2's eligibility test.

### 6.4 Songstats audio feature normalization

Existing handling in `features.py:226-268` emits raw values plus low/mid/high bands. Keep both. Additional requirements:

- All audio features must be parsed to float once and validated. Empty/invalid values use a sentinel rather than `NaN`/0.0 to avoid contaminating the LR.
- Band cutoffs (`<0.33` / `0.33-0.66` / `>0.66`) made consistent across features.
- A 4th "very high" band for `instrumentalness ≥ 0.85` (strong techno/electronica signal).
- A 4th "very low" band for `valence ≤ 0.20` (strong dark mood signal).

### 6.5 BPM banding

Add the following banded features alongside raw BPM:
- `bpm_band:slow` for BPM < 100
- `bpm_band:warmup` for 100-115
- `bpm_band:builder` for 115-122
- `bpm_band:driver` for 122-128
- `bpm_band:peak` for 128-134
- `bpm_band:peak_plus` for 134-145
- `bpm_band:dnb` for 150-180

A track gets exactly one BPM band (highest BPM that fits).

### 6.6 Disambiguation flags for ambiguous terms

When the title / mix / filename contains a token from the ambiguous set, emit a `disamb:<term>` feature so the LR can learn which contexts it belongs in:

- `disamb:deep` (deep can mean deep house, deep techno, deep tech, deep dnb)
- `disamb:dark` (mood across many families)
- `disamb:melodic` (melodic house, melodic techno, melodic dubstep)
- `disamb:progressive` (progressive house, progressive trance, prog rock)
- `disamb:garage` (UK Garage, Garage House, Garage Rock)
- `disamb:tribal` (tribal house, tribal techno, tribal afro — but eligibility rule §6.2 still applies)
- `disamb:dub` (dub techno, dub deep tech, reggae dub)
- `disamb:organic` (organic house, organic chant)

### 6.7 Optional artist and label priors

New files (start empty, opt-in additions over time):
- `src/dj_registry/taxonomy/data/artist_priors.json`
- `src/dj_registry/taxonomy/data/label_priors.json`

Format:
```json
{
  "<Artist Name>": [
    {"category_id": "<v2_id>", "weight": <0.0-1.0>}
  ]
}
```

Loading: read at training time; emit `prior:artist:<id>` features per matching prior with float value = weight.

The eligibility rule (§6.2) consults these to decide if afro/tribal categories are allowed.

---

## 7. Confidence Calibration and Null Fallback

### 7.1 Confidence bands (used downstream for UX hinting)

Map calibrated model probability to a level:

| Probability | Level |
|---|---|
| ≥ 0.85 | `high` |
| ≥ 0.70 | `medium-high` |
| ≥ 0.55 | `medium` |
| ≥ 0.35 | `low` |
| < 0.35 | `unknown` |

The existing `dj_model.py:_confidence_from_model_score()` already returns a 0-1 confidence. Add a level mapper helper and store it on `LogicalTrack.dj_taxonomy_confidence_level`.

### 7.2 Null fallback

When the top candidate's calibrated confidence is `< 0.35`:
- Set `dj_taxonomy_id = "unclassified"`
- Set `dj_taxonomy_label = ""`
- Set `dj_taxonomy_confidence` to the raw value (don't zero it)
- Set `dj_taxonomy_confidence_level = "unknown"`
- Populate `dj_taxonomy_warnings` with one or more of:
  - `"low_top_score"` (best candidate < 0.35)
  - `"small_margin"` (top - second < 0.05)
  - `"no_provider_genre"` (no usable provider genre and weak tagger evidence)
  - `"provider_contradiction"` (multiple providers disagree on family)

Tracks marked `unclassified` are excluded from `outputs/playlists/by_subgenre/` (no playlist entry created for them) and don't get a `CATEGORY` segment in their COMMENT tag.

### 7.3 Surfacing in exports

`registry_overview.csv` must include a new column `dj_taxonomy_confidence_level` after `dj_taxonomy_confidence`. Update `src/dj_registry/sync/export.py:OVERVIEW_COLUMNS` accordingly.

---

## 8. Ground-Truth Expansion Workflow

The 247-row `outputs/dj_taxonomy_ground_truth.csv` is re-generated against the v2 taxonomy.

### 8.1 GPT re-labeling

Use existing `src/dj_registry/taxonomy/dj_ground_truth.py`:

```bash
dj-registry dj-taxonomy generate-ground-truth \
  --files D:/Music \
  --out outputs/dj_taxonomy_ground_truth.csv \
  --taxonomy src/dj_registry/taxonomy/dj_taxonomy.json \
  --force \
  --keep-going
```

GPT-5 sees:
- The full v2 taxonomy JSON (every category's full definition)
- Per-track context: artist, title, mix, label, provider genres, Songstats audio features, tagger fields, BPM

Prompt template requirement (update in `dj_ground_truth.py`): the prompt must:
1. List every valid `id` and prohibit hallucinated IDs
2. Emphasize the §6.2 anti-collapse rule: "Do not assign a tribal_afro/afro_house/spiritual category unless the provider genres explicitly mention 'afro' or 'tribal'."
3. Allow the explicit string `"unclassified"` as a valid output for tracks with weak/contradictory evidence

### 8.2 Quality controls

After GPT writes the CSV, an audit pass enforces:
- Each row's `category_id` exists in the v2 taxonomy or equals `"unclassified"`
- BPM in the labeled category's `bpm_range` ± 8 BPM (warn, don't reject)
- Provider genre tokens match the category's `source_genres` or `keywords` (warn if zero overlap)

A new helper `validate_ground_truth_csv(taxonomy, csv_path) -> AuditReport` lives in `dj_ground_truth.py`. The audit report goes to `outputs/dj_taxonomy_ground_truth_audit.csv` alongside the labels.

### 8.3 Reviewer columns

The CSV format already supports reviewer override columns (verify in `dj_ground_truth.py`). Required columns:
- `reviewer_label` — empty by default; if set, used instead of GPT's pick during training
- `reviewer_note` — free text
- `resolved_at` — timestamp when human approved

### 8.4 ≥3 examples per category

After labeling, a coverage check enforces every taxonomy ID has ≥ 3 labeled rows. Categories below the threshold:
- Are listed in `outputs/dj_taxonomy_coverage_report.csv`
- Are excluded from the LR training set (the model can't output them anyway)
- Are flagged `provisional: true` in their taxonomy entry (a new optional field; loaders default to false)

The training command refuses to ship a model whose holdout-eligible category count is < 80% of the taxonomy.

---

## 9. Training and Evaluation Requirements

### 9.1 Training command

```bash
dj-registry dj-taxonomy train-models \
  --labels outputs/dj_taxonomy_ground_truth.csv \
  --taxonomy src/dj_registry/taxonomy/dj_taxonomy.json \
  --model-dir outputs/registry/dj_taxonomy_model \
  --validation-split 0.2 \
  --seed 42
```

Inside `dj_model.py:train_dj_taxonomy_models()`:

- Use **all** rows where `reviewer_label` is set, otherwise `category_id` (GPT label). No 60-row arbitrary truncation.
- `class_weight="balanced"` per spec §19
- Add per-category sample weights for categories with 3-5 examples (boost ×1.5) so the LR doesn't ignore them entirely
- Train internal + external models per existing dual-model design
- Random seed configurable, default 42, for reproducibility

### 9.2 Training report

`outputs/registry/dj_taxonomy_model/{internal,external}/training_report.json` must include:

```json
{
  "version": "dj-taxonomy-v2.0",
  "trained_at": "<ISO8601>",
  "examples_total": <int>,
  "examples_used": <int>,
  "categories_in_training": <int>,
  "categories_in_taxonomy": <int>,
  "categories_below_min_examples": ["id1", "id2", ...],
  "validation": {
    "top_1_accuracy": <float>,
    "top_3_accuracy": <float>,
    "macro_f1": <float>,
    "weighted_f1": <float>,
    "per_category": {
      "<id>": {"precision": ..., "recall": ..., "f1": ..., "support": ...}
    },
    "confusion_top_20": [...]
  },
  "library_distribution_check": {
    "total_tracks": <int>,
    "predictions": {"<id>": <count>},
    "largest_bucket_share": <float>,
    "passes_20pct_cap": <bool>
  }
}
```

The `library_distribution_check` runs immediately after training: predict on every track in `tracks_master.csv`, count predicted labels, fail if the largest bucket exceeds 20%.

### 9.3 Anti-collapse training penalty

If `library_distribution_check.passes_20pct_cap` is false after a training pass:

1. Identify the offending bucket (largest share)
2. Add a class-weight penalty for that category in the next retrain: `class_weight[offending_id] *= 0.7`
3. Retrain (same code path)
4. If still failing after 3 consecutive retrains, the build fails with a diagnostic listing:
   - The persistently-offending category
   - Sample tracks predicted to it (top 20)
   - Suggestion: split further or tighten feature eligibility rules

This is implemented as a wrapper in `dj_model.py`:

```python
def train_with_collapse_guard(store, labels, ..., max_retrains=3):
    for attempt in range(max_retrains):
        result = train_dj_taxonomy_models(...)
        check = run_library_distribution_check(store, model_dir=result["model_dir"])
        if check["passes_20pct_cap"]:
            return result, check
        # Apply penalty for next pass
        ...
    raise CollapseGuardError(check)
```

### 9.4 Numeric targets (gates)

The build ships only when ALL of:

| Gate | Target |
|---|---|
| Top-1 accuracy (holdout) | ≥ 50% |
| Top-3 accuracy (holdout) | ≥ 75% |
| Macro F1 (holdout) | ≥ 0.30 |
| Largest bucket on library prediction | ≤ 20% |
| Categories with ≥ 3 training examples | 100% |

Each gate is a separate assertion in the training report; the CLI exits non-zero if any fail.

---

## 10. Implementation Phases (order of work)

After spec approval:

1. **Author `dj_taxonomy.json` v2** — encode every category from §5 with the schema in §5
2. **Regenerate ground truth** — `dj-registry dj-taxonomy generate-ground-truth --force`
3. **Run the audit pass** — confirm all rows validate; expand prompts for failures
4. **Update `features.py`** — implement §6 changes (provider weights, anti-collapse rule, mix-name parsing, BPM banding, disambiguation flags, optional priors loading)
5. **Update `dj_model.py`** — confidence-level mapping (§7.1), null fallback (§7.2), `train_with_collapse_guard` (§9.3), library distribution check (§9.2)
6. **Update `dj_schema.py`** — add `provisional` optional field to `DjCategory`
7. **Update `models.py`** — add `dj_taxonomy_confidence_level` field to `LogicalTrack`
8. **Update `sync/export.py`** — new CSV column for confidence level
9. **Update tag_writer + dj_tagger/cli** — already use `dj_taxonomy_label` and gracefully handle empty values; verify `unclassified` short-circuits the CATEGORY segment
10. **Train models** — `dj-registry dj-taxonomy train-models` with collapse guard
11. **Hit all §9.4 gates** — iterate on taxonomy splits / feature rules until gates pass
12. **Spot-check the 2 named tracks + 10 representative library tracks**
13. **Full `dj run`** against the user's library

If a gate cannot be passed within reasonable iterations:
- Re-examine §3 splitting boundaries (maybe a category needs to split further)
- Re-examine §6 eligibility rules (maybe another family needs a hard rule)
- Bring more ground-truth labels (GPT re-prompt with refined criteria)

---

## 11. Tests

Required tests in `tests/`:

### 11.1 Schema validation
- `tests/test_dj_taxonomy_schema.py` — load `dj_taxonomy.json`, assert every entry has all required fields, bpm_range / energy_range are integer pairs, vocal_profiles only from the allowed set, IDs unique, IDs are snake_case.

### 11.2 Anti-collapse regression
- `tests/test_dj_taxonomy_anti_collapse.py` — fixture set of 20 hand-built tracks with mid-BPM (118-128), percussion-heavy tagger (`TRIB` mood, `chant` vocal cue), but **no** "afro" / "tribal" / "spiritual" / "3-step" token in any provider field, no artist/label prior, no afro/tribal mix-name token. Assert that **zero** of these get classified into any afro/tribal/spiritual category — they must end up either in `tech_house` / `indie_dance` / `tribal_techno` (which is OK — tribal_techno does NOT require explicit afro signal) or `unclassified`. Specifically `tribal_afro_driver` count must be 0 across the fixture.

### 11.3 Confidence + null fallback
- `tests/test_dj_taxonomy_confidence.py` — tracks with weak/contradictory signals must return `unclassified` with confidence < 0.35 and appropriate warnings.

### 11.4 Named regression cases
- `tests/test_dj_taxonomy_regressions.py` — Nina Simone and Daval Robots tracks; assertions per §2.

### 11.5 Ground-truth regression
- `tests/test_dj_taxonomy_ground_truth.py` — load `outputs/dj_taxonomy_ground_truth.csv`, run classifier, assert top-1 ≥ 50%, top-3 ≥ 75%, macro F1 ≥ 0.30.

### 11.6 Library distribution
- `tests/test_dj_taxonomy_distribution.py` — synthetic library of 100 tracks across the taxonomy. After classification, no bucket > 20%, every category appears at least once.

---

## 12. Out of Scope

Stated explicitly so this isn't litigated again later:

- **3-level family → genre → subgenre hierarchy.** Rejected; flat labels stay.
- **Replacing the LR model with a rules engine or deep learning.** Rejected; LR + better features + better training.
- **LLM at inference time.** GPT is for ground-truth seeding only, never at the per-track classification path.
- **New top-level families beyond what's in v1.** All additions are sub-types or modern variants of existing electronic / non-electronic families.
- **Changing the COMMENT tag format.** Format stays `KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]`. Only the `dj_taxonomy_label` value source-of-truth changes.
- **Retroactively re-tagging existing files** without re-running `dj run`. Migration is automatic on the next `dj run`.
- **Migration of `tracks_master.csv` rows in place.** The retired `dark_indie_tech_house` ID just stops appearing after the next `dj run`; rows simply get re-classified by the new model.

---

## Appendix A — Vocabulary (reproduced from prior spec §15 for self-containment)

### Moods

```
ACID — acidic, 303, squelchy
ATM — atmospheric, ambient, spacious
CIN — cinematic, dramatic, soundtrack
DEEP — deep, late-night, subdued, dubby
DRK — dark, tense, nocturnal, gothic
EMO — emotional, melancholic, romantic
EUP — euphoric, uplifting, trance-like release
GRIT — gritty, rough, distorted
HYPN — hypnotic, repetitive, rolling, meditative
MEL — melodic, harmony-forward
MIN — minimal, sparse, reduced
ORG — organic, earthy, tribal, desert
PLAY — playful, funky, bouncy
PSY — psychedelic, trippy, acid, mental
RAW — raw, warehouse, industrial
SOUL — soulful, gospel, human warmth
SUB — subby, bass-heavy
SUN — sunlit, sunset, balearic
TENS — tense, suspenseful
TRIB — tribal, percussive, ritual
WARM — warm, rounded, smooth, inviting
WHSE — warehouse, rave-room, peak industrial
```

### Vocal profiles

```
INST  — instrumental / no vocal
VOC   — vocal-led
FVOC  — featured vocal / vocal hook
SPK   — spoken word
CHANT — chant / ritual vocal
DUB   — dub mix / reduced vocal / echo-heavy
TOOL  — DJ tool / percussive track
```

### Structure / groove

```
16H     — rolling, steady, hypnotic
16D     — driving, linear, forward-motion
32H     — extended hypnotic phrasing
32D     — long driving progression
BREAKS  — broken rhythm, breakbeat, garage, jungle
ROLLING — rolling groove
LINEAR  — functional, driving, tool-like
```

### Set roles

```
warmup — intro, opener, low-intensity
opener — first track of a set
builder — build energy from warmup to peak
driver — main-set groove, mid-high energy
peak — peak-time, highest intensity
bridge — transition between sections
reset — bring energy down before re-building
tool — DJ utility, percussive, intro-style
```

---

## Appendix B — Open Questions for Implementation

These can be resolved during the implementation phase (§10), not blocking spec approval:

1. **`provisional` field on taxonomy entries**: should this be inferred from training coverage at load time, or stored in the JSON manually? Recommendation: stored in JSON, set during the audit pass (§8).

2. **`unclassified` in `dj_taxonomy_label`**: empty string vs literal `"unclassified"` string? Downstream playlist code skips on empty, so empty string is safer.

3. **Per-category BPM tolerance for the §8.2 audit**: ±8 BPM proposed; may need tightening for genres with narrow BPM bands (e.g., DnB) and loosening for crossover styles. Tune during ground-truth audit.

4. **GPT prompt for re-labeling**: the existing prompt may need restructuring to handle 99 categories. If the prompt blows token limits, batch into family-sized chunks with explicit family-level pre-classification first. Defer to implementation.

5. **Should retired `dark_indie_tech_house` rows in older `tracks_master.csv` files be re-mapped at load time, or just left empty until next `dj run`?** Recommendation: leave empty; the next `dj run` overwrites them.
