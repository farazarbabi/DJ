# Update LLM Ground-Truth Prompt for Conservative Tribal/Afro Subgenre Classification

## Task

Edit the LLM ground-truth prompt used by the DJ music-library classification system so that it becomes more accurate, conservative, and musically correct when selecting modern subgenres.

The main problem to fix is that the LLM is currently too biased toward classifying tracks as:

- Afro House
- Afro Tech
- Tribal House
- Tribal Techno
- Tribal Organic House
- Afro / Tribal Driving
- Tribal / Afro variants

when the track only has weak cues such as:

- dark mood
- tense mood
- driving groove
- percussion
- organic atmosphere
- chant-like vocal
- female vocal
- desert / Tulum / ritual words
- internal tags like `TRIB`, `AFRO`, or `DRV`

This produces false positives.

The updated prompt must make the LLM distinguish between:

1. Actual genre/subgenre classification
2. Secondary texture/influence/mood tags

A track should not be classified as Tribal or Afro unless the evidence is strong.

---

## Required Prompt Change

Find and edit the existing LLM prompt used for ground-truth genre/subgenre classification.

Add a new section called:

```text
Modern Subgenre Selection Rules
```

If a similar section already exists, expand it instead of duplicating it.

---

## 1. Separate Genre From Descriptor Tags

The LLM must distinguish between taxonomy labels and descriptor tags.

### Taxonomy Labels

These may be used as `family`, `genre`, or `subgenre`:

```text
Organic House
Afro House
Tribal House
Melodic Techno
Dark Melodic Techno
Driving Techno
Progressive House
Indie Dance
Dark Disco
Downtempo
```

### Descriptor Tags

These should usually be secondary tags only:

```text
tribal
afro
mayan
ritual
ceremonial
shamanic
organic
percussive
ethnic
desert
tulum
chant
driving
hypnotic
dark
tense
female vocal
spoken vocal
```

Descriptor tags must not automatically become the main genre or subgenre.

---

## 2. Be Conservative With Afro and Tribal

The LLM must be conservative when assigning any Afro or Tribal genre/subgenre.

A track should only be classified as Afro House, Afro Tech, Tribal House, Tribal Techno, or Tribal Organic House if there is clear evidence from at least one strong source or multiple converging weak sources.

### Strong Evidence Examples

Use Afro or Tribal as a genre/subgenre when one or more of these is true:

```text
- Trusted source genre explicitly says Afro House, Afro Tech, Tribal House, Tribal Techno, or Tribal Organic House.
- Manual curated genre explicitly says Afro House, Afro Tech, Tribal House, or Tribal Organic House.
- DJ store category explicitly says Afro House, Afro Tech, Tribal House, or Organic House with tribal context.
- The arrangement is clearly built around dominant organic/hand percussion, ritual vocals, ceremonial rhythm, or tribal drum structure.
- Artist/label context strongly belongs to Afro House, Afro Tech, Tulum Organic House, ritual organic house, or tribal house scenes.
```

### Weak Evidence Examples

The following are not enough by themselves:

```text
- dark mood
- tense mood
- driving groove
- hypnotic groove
- percussion exists
- organic texture
- female vocal
- chant-like vocal
- low vocal
- Tulum/desert/ritual word appears in title/comment
- internal tag contains TRIB
- internal tag contains AFRO
- internal tag contains DRV
```

Weak evidence should usually become:

```json
{
  "secondary_tags": ["percussive", "organic", "driving", "ritual-influenced"]
}
```

not:

```json
{
  "genre": "Afro House",
  "subgenre": "Tribal House"
}
```

---

## 3. Distinguish Afro From Tribal From Organic

The LLM must not collapse Afro, Tribal, and Organic into the same concept.

### Afro House / Afro Tech

Use only when evidence points specifically to African or Afro-diasporic house/techno lineage.

Valid signals:

```text
Afro House source genre
Afro Tech source genre
African percussion context
African vocal/chant context
known Afro House artist or label
South African / West African / Afro-diasporic scene metadata
explicit Afro House or Afro Tech from a trusted source
```

Do not infer Afro from:

```text
tribal
mayan
ritual
ceremonial
organic
desert
Tulum
shamanic
ethnic
chant
percussion
```

### Tribal House / Tribal Organic House

Use when the arrangement is clearly built around ritualistic, tribal, ceremonial, or organic percussion.

Valid signals:

```text
dominant hand percussion
ritual or ceremonial vocal structure
tribal drum pattern is central to the track
organic percussion is foregrounded over synth melody
trusted metadata says Tribal House or Tribal Organic House
Tulum/Mayan/ceremonial context is supported by the sound and arrangement
```

### Organic House

Use for earthy, natural, melodic, ethnic, acoustic, or desert/Tulum-influenced house music, especially at slower house tempos.

Organic House can have tribal influence without being Tribal House.

### Melodic Techno / Dark Melodic Techno

Use for tracks with:

```text
minor-key melody
emotional synth line
progressive arrangement
tense/dark mood
techno or melodic techno metadata
driving but not percussion-dominant groove
vocal fragments or female vocal
```

Do not classify these as Tribal/Afro unless the tribal/afro evidence is strong.

---

## 4. Add Explicit Negative and Positive Examples

Add these examples directly to the LLM prompt.

---

### Negative Example: Erly Tepshi – Virgo

The LLM must not classify this type of track as Tribal/Afro.

Expected classification:

```json
{
  "family": "Techno",
  "genre": "Melodic Techno",
  "subgenre": "Dark Melodic Techno",
  "secondary_tags": [
    "driving",
    "tense",
    "female vocal",
    "progressive-influenced"
  ],
  "rejected_labels": [
    "Afro Techno",
    "Tribal Techno",
    "Tribal / Afro Driving Techno"
  ],
  "reason": "The track has a driving and tense melodic techno character, but does not have strong Afro or Tribal genre evidence. Internal tags such as TRIB.AFRO.DRV are weak descriptor cues, not taxonomy labels."
}
```

Rules illustrated:

```text
- Driving does not mean Tribal.
- Tense does not mean Tribal.
- Female vocal does not mean Tribal.
- Internal TRIB/AFRO shorthand does not override musical evidence.
- If percussion is not structurally dominant, avoid Tribal.
- If African/Afro-diasporic context is absent, avoid Afro.
```

---

### Positive Example: PAAX Tulum – Crisol (MIICHII Remix)

The LLM should still allow this type of track to be classified as tribal/organic.

Expected classification:

```json
{
  "family": "House",
  "genre": "Organic House",
  "subgenre": "Tribal Organic House",
  "secondary_tags": [
    "mayan-influenced",
    "ritual",
    "ceremonial",
    "organic percussion",
    "tulum sound"
  ],
  "influence_tags": [
    "mayan",
    "tribal"
  ],
  "rejected_labels": [
    "Afro House"
  ],
  "reason": "The track has clear ritual/ceremonial/organic tribal characteristics, but this does not automatically make it Afro House. Afro requires specific Afro/African/Afro-diasporic evidence."
}
```

Rules illustrated:

```text
- PAAX Tulum-style tracks may be Tribal Organic House.
- Tribal does not automatically mean Afro.
- Mayan/Tulum/ceremonial influence should map to tribal/organic, not Afro, unless explicit Afro evidence exists.
```

---

## 5. Required Output Behavior

Update the LLM prompt so that the LLM returns separate fields for taxonomy classification and descriptors.

The output format should support:

```json
{
  "family": "...",
  "genre": "...",
  "subgenre": "...",
  "confidence": 0.0,
  "secondary_tags": [],
  "mood_tags": [],
  "groove_tags": [],
  "texture_tags": [],
  "influence_tags": [],
  "rejected_labels": [],
  "reason": "..."
}
```

The LLM must explain rejected Afro/Tribal candidates when those candidates were plausible but rejected.

Example:

```json
{
  "rejected_labels": [
    {
      "label": "Afro House",
      "reason": "No explicit Afro House source, no African/Afro-diasporic context, and percussion evidence is not sufficient."
    },
    {
      "label": "Tribal Techno",
      "reason": "The track is driving and dark, but the arrangement is not percussion-dominant or ritualistic enough to justify Tribal Techno."
    }
  ]
}
```

---

## 6. Add Decision Gate to the LLM Prompt

The LLM prompt must include this decision gate:

```text
Before assigning any Afro or Tribal genre/subgenre, ask:

1. Is there an explicit trusted source genre saying Afro House, Afro Tech, Tribal House, Tribal Techno, or Tribal Organic House?
2. If not, are there at least three independent weak signals that all point to Afro/Tribal?
3. Is the percussion/ritual/ceremonial element structurally central to the arrangement, not merely present?
4. For Afro specifically, is there evidence of African or Afro-diasporic lineage/context?

If the answer is no, do not assign Afro or Tribal as the genre/subgenre. Put those words into secondary/influence/texture tags instead.
```

---

## 7. Prompt-Level Scoring Guidance

Add the following scoring guidance:

```text
- Explicit trusted genre source should dominate.
- Internal generated tags are weak evidence.
- Mood words are not genre evidence.
- Texture words are not genre evidence unless supported by arrangement/source metadata.
- Driving/rolling/hypnotic are groove descriptors, not genres.
- Dark/tense/romantic are mood descriptors, not genres.
- Organic can be a genre when supported by source metadata or sound profile.
- Tribal requires percussion/ritual structure to be central.
- Afro requires Afro/African/Afro-diasporic evidence.
```

---

## 8. Files to Modify

Search the repository for the current LLM genre prompt.

Likely locations include:

```text
prompts/
src/prompts/
genre_classifier/
classification/
ground_truth/
llm_ground_truth*
genre_prompt*
```

Use search terms:

```text
ground truth
genre classification
subgenre
family
taxonomy
Afro
Tribal
Organic House
```

Modify the prompt file in-place.

If the prompt is embedded in Python, TypeScript, YAML, or JSON, preserve the existing structure and escaping.

---

## 9. Add Tests or Fixtures

Add or update tests/fixtures so that the prompt behavior is validated.

---

### Virgo-Style Case

A track with:

```json
{
  "artist": "Erly Tepshi",
  "title": "Virgo",
  "mix": "Original Mix",
  "bpm": 120,
  "source_genre": "Techno",
  "internal_tags": ["TRIB", "AFRO", "DRV"],
  "mood": "tense",
  "vocal": "female vocal"
}
```

must not output Afro or Tribal as genre/subgenre.

Expected:

```json
{
  "family": "Techno",
  "genre": "Melodic Techno",
  "subgenre": "Dark Melodic Techno"
}
```

---

### PAAX-Style Case

A track with:

```json
{
  "artist": "PAAX Tulum",
  "title": "Crisol",
  "mix": "MIICHII Remix",
  "source_terms": ["organic", "ritual", "mayan", "ceremonial", "percussion", "tulum"],
  "arrangement": "organic percussion and ritual/ceremonial texture are central"
}
```

may output:

```json
{
  "family": "House",
  "genre": "Organic House",
  "subgenre": "Tribal Organic House"
}
```

but should not output Afro House unless explicit Afro evidence exists.

---

## 10. Acceptance Criteria

The updated prompt is successful if:

1. Tracks like Erly Tepshi – Virgo are no longer labeled Tribal/Afro.
2. Tracks like PAAX Tulum – Crisol can still be labeled Tribal Organic House.
3. Afro is no longer inferred from Tulum, Mayan, tribal, ritual, organic, or percussion cues alone.
4. The LLM clearly separates:
   - genre
   - subgenre
   - mood tags
   - groove tags
   - texture tags
   - influence tags
5. The LLM includes rejected labels and reasons when Afro/Tribal was considered but rejected.
6. The LLM remains constrained to the reference taxonomy.
7. The LLM is conservative when evidence is weak.
8. The prompt explicitly says:

```text
Tribal and Afro must be earned by evidence, not triggered by mood, percussion, or internal shorthand tags.
```

---

## 11. Implementation Command

After creating this file, run your coding agent with:

```bash
codex "Read update_llm_ground_truth_prompt.md and implement the requested prompt changes in the repository. Search for the existing LLM ground-truth genre classification prompt, update it in-place, and add or update fixtures/tests for the Virgo negative case and PAAX Tulum positive case."
```
