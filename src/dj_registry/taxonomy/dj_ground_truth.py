"""GPT-assisted ground-truth labeling for the DJ subgenre taxonomy.

Used to label tracks (aiff / mp3) against ``dj_taxonomy.json`` v2.2 to produce
a ground-truth CSV for the LR baseline and XGBoost subgenre classifiers. The
system prompt encodes:

- hard rules (only pick from taxonomy, JSON-only output)
- a 5-step decision process (BPM gate → mood/groove → keywords → provider → confidence)
- anti-bias rules that block the previous afro over-tagging
- evidence weighting (STRONG / MEDIUM / WEAK / NOT-evidence)
- confidence calibration bands
- few-shot examples covering the high-confusion clusters
- a fully-specified JSON output schema with ``rejected_afro`` flag

The taxonomy is sent inline ahead of the per-track evidence so the LLM has
access to ``allowed_dj_taxonomy.categories`` referenced by the prompt.

CSV note: list-valued LLM fields (``alternatives``, ``evidence_used``) are
written as semicolon-joined strings so they round-trip cleanly through
``csv.DictReader`` for downstream training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import RegistryConfig
from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from ..pipelines.orchestrator import run_full_pipeline
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .dj_schema import DjTaxonomy, external_evidence_available, load_dj_taxonomy
from .ground_truth import extract_chat_completion_json, extract_response_json

DJ_GROUND_TRUTH_COLUMNS = [
    "file_name",
    "track_id",
    "artist",
    "title",
    "mix",
    "category_id",
    "category_label",
    "family",
    "moods",
    "grooves",
    "set_roles",
    "bpm_range",
    "energy_range",
    "vocal_profiles",
    "source_genres",
    "keywords",
    "tagger_energy",
    "tagger_mood",
    "tagger_vocal",
    "tagger_structure",
    "bpm_hint",
    "confidence",
    "alternate_category_ids",
    "rationale",
    "warnings",
    "internal_evidence_available",
    "external_evidence_available",
    "external_sources",
    "label_source",
    "model",
    "taxonomy_version",
    "created_at",
]

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".aiff", ".aif", ".wav", ".flac", ".m4a"}
MAX_LABEL_OUTPUT_TOKENS = 8192


@dataclass
class DjGroundTruthStats:
    files_seen: int
    rows_written: int
    generated: int
    reused: int
    errors: int
    output_path: str
    collection_summary: dict[str, Any] | None = None
    error_message: str = ""


@dataclass
class DjApiConnectionTestResult:
    ok: bool
    provider: str
    model: str
    error_message: str = ""


class OpenAIDjGroundTruthClient:
    """Responses API client for constrained DJ taxonomy category labels."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = "gpt-5",
        base_url: str = "https://api.openai.com/v1",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        azure_api_version: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        self.azure_deployment = (
            azure_deployment
            or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        )
        self.azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "")
        self.use_azure = not (api_key or openai_key) and bool(self.azure_endpoint and azure_key)
        self.api_key = api_key or openai_key or azure_key
        self.model = model or os.environ.get("OPENAI_MODEL", "") or self.azure_deployment or "gpt-5"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required to generate DJ taxonomy ground truth"
            )
        if self.use_azure:
            if not self.azure_deployment:
                raise RuntimeError("AZURE_OPENAI_CHAT_DEPLOYMENT is required for Azure OpenAI ground truth generation")
            if not self.azure_api_version:
                raise RuntimeError("AZURE_OPENAI_API_VERSION is required for Azure OpenAI ground truth generation")

    @property
    def provider(self) -> str:
        return "azure" if self.use_azure else "openai"

    def label_track(
        self,
        context: dict[str, Any],
        taxonomy_json: dict[str, Any],
        *,
        validation_error: str | None = None,
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for OpenAI API calls") from exc

        if self.use_azure:
            return self._label_track_azure(httpx, context, taxonomy_json, validation_error=validation_error)

        payload = {
            "model": self.model,
            "instructions": _system_prompt(validation_error),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _build_input_text(context, taxonomy_json),
                        }
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "dj_functional_taxonomy_ground_truth",
                    "strict": True,
                    "schema": _response_schema(),
                }
            },
            "reasoning": {"effort": "low"},
            "max_output_tokens": MAX_LABEL_OUTPUT_TOKENS,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return extract_response_json(response.json())

    def _label_track_azure(
        self,
        httpx: Any,
        context: dict[str, Any],
        taxonomy_json: dict[str, Any],
        *,
        validation_error: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "messages": [
                {"role": "system", "content": _system_prompt(validation_error)},
                {
                    "role": "user",
                    "content": _build_input_text(context, taxonomy_json),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "dj_functional_taxonomy_ground_truth",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
            "max_completion_tokens": MAX_LABEL_OUTPUT_TOKENS,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._azure_chat_completions_url(),
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return extract_chat_completion_json(response.json())

    def _azure_chat_completions_url(self) -> str:
        endpoint = self.azure_endpoint.rstrip("/")
        deployment = self.azure_deployment.strip().strip("/")
        deployment = deployment.removeprefix("openai/deployments/").strip("/")
        if endpoint.endswith("/chat/completions"):
            base = endpoint
        elif endpoint.endswith(f"/openai/deployments/{deployment}"):
            base = f"{endpoint}/chat/completions"
        elif endpoint.endswith("/openai/deployments"):
            base = f"{endpoint}/{deployment}/chat/completions"
        else:
            base = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}api-version={self.azure_api_version}"


def generate_dj_ground_truth_csv(
    store: CsvStore,
    *,
    config: RegistryConfig | None = None,
    files_dir: str = "./files",
    output_path: str | None = None,
    taxonomy_path: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    force: bool = False,
    fail_fast: bool = True,
    show_progress: bool = False,
    cache_dir: str | None = None,
    client: Any | None = None,
    collect: bool = True,
    include_external: bool = True,
    include_songstats: bool = True,
    songstats_limit: int | None = None,
    rekordbox_xml: str | None = None,
    no_essentia: bool = False,
    workers: int | None = None,
) -> DjGroundTruthStats:
    taxonomy = load_dj_taxonomy(taxonomy_path)
    taxonomy_json = taxonomy.as_prompt_json()
    resolved_output_path = _resolve_output_path(output_path, store.output_dir)
    collection_summary = None
    if collect:
        registry_config = config or RegistryConfig()
        registry_config.output_dir = store.output_dir
        registry_config.library_roots = [files_dir]
        if rekordbox_xml:
            registry_config.rekordbox_xml_path = rekordbox_xml
        registry_config.load_env()
        collection_summary = run_full_pipeline(
            registry_config,
            dry_run=True,
            write_tags=False,
            include_rekordbox=include_external and bool(registry_config.rekordbox_xml_path),
            include_songstats=include_external and include_songstats and bool(registry_config.songstats_api_key),
            songstats_limit=songstats_limit,
            no_essentia=no_essentia,
            analysis_workers=workers,
            show_progress=show_progress,
        )

    audio_files = _audio_files(files_dir)
    if limit is not None:
        audio_files = audio_files[:limit]

    existing = _load_existing_rows(resolved_output_path)
    cache_root = Path(cache_dir or Path(store.output_dir) / "dj_taxonomy_ground_truth_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    api_client = client or OpenAIDjGroundTruthClient(model=model or os.environ.get("OPENAI_MODEL", "gpt-5"))
    model_name = getattr(api_client, "model", None) or model or "gpt-5"

    files = store.load_files()
    tracks = store.load_tracks()
    observations = store.load_observations()
    file_by_name = {f.file_name.lower(): f for f in files if f.file_name}
    track_by_id = {t.track_id: t for t in tracks if t.track_id}
    obs_by_track: dict[str, list[SourceObservation]] = {}
    for obs in observations:
        if obs.track_id:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

    rows: list[dict[str, Any]] = []
    generated = 0
    reused = 0
    errors = 0
    progress = ProgressBar(len(audio_files), label="DJ taxonomy ground truth", enabled=show_progress)
    for index, audio_path in enumerate(audio_files, start=1):
        file_name = audio_path.name
        existing_row = existing.get(file_name)
        if not force and existing_row and not _is_error_row(existing_row):
            rows.append(existing_row)
            reused += 1
            progress.update(index, f"reused {file_name}", new=generated, reused=reused, errors=errors)
            continue

        file_record = file_by_name.get(file_name.lower())
        track = track_by_id.get(file_record.track_id) if file_record else None
        if track is None:
            track = LogicalTrack(
                track_id=file_record.track_id if file_record else file_name,
                artist_canonical=file_record.embedded_artist if file_record else "",
                title_canonical=file_record.embedded_title if file_record else audio_path.stem,
                album_canonical=file_record.embedded_album if file_record else "",
            )
        track_obs = obs_by_track.get(track.track_id, [])
        context = _track_context(audio_path, track, file_record, track_obs)
        cache_path = cache_root / f"{_cache_key(model_name, taxonomy_json, context)}.json"

        try:
            if cache_path.exists() and not force:
                label = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                label = api_client.label_track(context, taxonomy_json)
                error = validate_dj_label(label, taxonomy)
                if error:
                    label = api_client.label_track(context, taxonomy_json, validation_error=error)
                cache_path.write_text(json.dumps(label, indent=2, sort_keys=True), encoding="utf-8")

            error = validate_dj_label(label, taxonomy)
            if error:
                errors += 1
                error_message = f"{file_name}: {error}"
                if fail_fast:
                    progress.finish(f"error {file_name}")
                    return DjGroundTruthStats(
                        files_seen=len(audio_files),
                        rows_written=0,
                        generated=generated,
                        reused=reused,
                        errors=errors,
                        output_path=resolved_output_path,
                        collection_summary=collection_summary,
                        error_message=error_message,
                    )
                row = _error_row(file_name, track, model_name, taxonomy, error)
            else:
                generated += 1
                row = _label_to_row(file_name, track, track_obs, label, model_name, taxonomy)
        except Exception as exc:
            errors += 1
            error_message = f"{file_name}: {_safe_error_message(exc)}"
            if fail_fast:
                progress.finish(f"error {file_name}")
                return DjGroundTruthStats(
                    files_seen=len(audio_files),
                        rows_written=0,
                        generated=generated,
                        reused=reused,
                        errors=errors,
                        output_path=resolved_output_path,
                        collection_summary=collection_summary,
                        error_message=error_message,
                )
            row = _error_row(file_name, track, model_name, taxonomy, error_message)
        rows.append(row)
        progress.update(index, f"processed {file_name}", new=generated, reused=reused, errors=errors)

    progress.finish("complete")
    _write_rows(resolved_output_path, rows)
    return DjGroundTruthStats(
        files_seen=len(audio_files),
        rows_written=len(rows),
        generated=generated,
        reused=reused,
        errors=errors,
        output_path=resolved_output_path,
        collection_summary=collection_summary,
    )


def test_dj_api_connection(*, model: str | None = None, client: Any | None = None) -> DjApiConnectionTestResult:
    api_client = client or OpenAIDjGroundTruthClient(model=model or os.environ.get("OPENAI_MODEL", "gpt-5"))
    provider = getattr(api_client, "provider", "unknown")
    model_name = getattr(api_client, "model", model or "unknown")
    taxonomy = load_dj_taxonomy()
    context = {
        "file": {"file_name": "connection_test.mp3"},
        "track": {
            "artist": "Connection Test",
            "title": "Dark Rolling Club Tool",
            "tagger_energy": "E4",
            "tagger_mood": "dark",
            "tagger_structure": "16H",
            "tagger_bpm": "126",
        },
        "observations": [],
    }
    try:
        label = api_client.label_track(context, taxonomy.as_prompt_json())
        error = validate_dj_label(label, taxonomy)
        if error:
            return DjApiConnectionTestResult(False, provider, model_name, error)
        return DjApiConnectionTestResult(True, provider, model_name)
    except Exception as exc:
        return DjApiConnectionTestResult(False, provider, model_name, _safe_error_message(exc))


def validate_dj_label(label: dict[str, Any], taxonomy: DjTaxonomy) -> str | None:
    category_id = str(label.get("category_id") or "").strip()
    if not taxonomy.validate_category_id(category_id):
        return f"Invalid DJ taxonomy category_id: {category_id}"
    alternatives = label.get("alternate_category_ids", [])
    if not isinstance(alternatives, list):
        return "alternate_category_ids must be a list"
    invalid_alts = [str(item) for item in alternatives if not taxonomy.validate_category_id(str(item))]
    if invalid_alts:
        return f"Invalid alternate_category_ids: {', '.join(invalid_alts)}"
    return None


SYSTEM_PROMPT = """\
You are a DJ-music subgenre classifier producing ground-truth labels for an
XGBoost training set. For each track, output exactly one category_id from
`allowed_dj_taxonomy.categories` plus structured metadata.

# Hard rules
1. Pick exactly one category_id that exists in the taxonomy. NEVER invent ids,
   labels, moods, grooves, or any taxonomy field.
2. Provider genres (Rekordbox, Spotify, Songstats, embedded ID3 tags) are
   HINTS ONLY — they are often wrong, outdated, or generic ("world", "house",
   "electronic"). Trust audio features and keyword fingerprints over provider
   strings when they conflict.
3. If evidence is ambiguous, still pick the single best category, lower the
   confidence, and list 1-3 allowed alternatives in `alternate_category_ids`.
4. Output JSON only. No prose outside the JSON object.

# Decision process (apply in order)
Step 1 — BPM gate: keep only categories whose `bpm_range` covers the track BPM
  (±2 BPM tolerance at the edges). This usually cuts candidates to 5-15.
Step 2 — Mood / groove / vocal profile / energy match: rank remaining
  candidates by overlap with detected features.
Step 3 — Keyword fingerprint: scan title, filename, artist, label, mix name,
  and provider tags for taxonomy `keywords`. A strong keyword hit (e.g. "saz",
  "oud", "tulum", "schranz", "amapiano", "drill", "anyma", "tim reaper")
  OUTRANKS provider genre strings.
Step 4 — Provider genre as tie-breaker only.
Step 5 — Set confidence per the calibration bands below.

# Anti-bias rules (CRITICAL — afro was previously over-predicted)
"Tribal" mood + percussion + chant vocal is NOT sufficient for any afro_*
category. Check regional / scene keywords FIRST, in this order:

  tulum / mayan / jungle / cenote / sunrise          -> tulum_tribal_*
  saz / baglama / altin gun / anatolian / turkish    -> anatolian_psych_house
  oud / qanun / darbuka / arabic / maqam             -> oriental_arabic_house
  duduk / ney / persian / indian / desert            -> desert_mystic_driver
  balkan / gypsy / brass / klezmer / romani          -> balkan_gypsy_groove
  bouzouki / greek / italian / mediterranean         -> mediterranean_folk_house
  baile / samba / candombe / brazilian / favela      -> latin_tribal_percussion
  slavic / russian / siberian / post-soviet          -> slavic_folk_chug
  playa / burning man / robot heart / wild west      -> burner_desert_house
  icaros / didgeridoo / ayahuasca / shamanic         -> ritual_shamanic_house

Only assign an afro_* category when at least ONE of these holds:
  - artist origin South Africa / Angola / Nigeria / Mozambique / Senegal
  - log drums audible (then strongly prefer amapiano_groove)
  - Yoruba / Zulu / Xhosa / Swahili / Wolof vocal
  - label such as MoBlack, Get Physical Afro, Stoney Boy, Realm Of
    Consciousness, Innervisions Afro, Cuttin' Headz, Madorasindahouse

A generic "world music" or "ethnic" provider tag is NEVER enough to pick afro.

# Evidence weighting
Score each evidence source according to its strength. Stronger evidence
overrides weaker evidence when they conflict.

STRONG (treat as decisive):
  - Trusted curated provider genre (Beatport sublabel, Bandcamp artist-set
    genre, label catalogue page).
  - Specific artist / label match against the named rosters above.
  - Audible distinctive instrumentation (saz, log drum, oud, amen break,
    303 line, reverse-bass kick).
  - Vocal language identification (Zulu, Turkish, Portuguese-BR, etc.).

MEDIUM (supports a candidate but rarely decides alone):
  - BPM in a tight range with low overlap.
  - Mix-name / version cues ("Sunset Edit", "Sped Up", "Dub Mix").
  - Energy level and structural arrangement.

WEAK (corroborating only — never decisive):
  - Internally-generated tagger tokens (TRIB / AFRO / DRV / HYP / ORG, etc.).
    These reflect a prior model's guess and frequently inherit its bias.
  - Generic provider strings ("electronic", "world", "house", "club").

NOT genre evidence on their own (these belong to other taxonomy fields):
  - Mood words (dark, tense, romantic, melodic, warm) -> moods, not lineage.
  - Texture words (organic, hand-percussion, acoustic, dusty) -> only count
    when backed by source metadata or a specific arrangement cue.
  - Groove descriptors (driving, rolling, hypnotic, swinging, chugging) ->
    grooves, not lineage.

Two hard lineage requirements:
  - `tribal_*` and tribal-tagged slots require percussion or ritual structure
    to be CENTRAL to the arrangement — not merely present in a fill or break.
  - `afro_*` requires explicit Afro / African / Afro-diasporic lineage or
    context (see the afro-allow conditions above). Percussion alone never
    satisfies this. Tribal and Afro must be EARNED by evidence, not triggered
    by mood, percussion, or internal shorthand tags.

# Other disambiguation rules
- UK 4x4 wobble bass: `bassline_niche` (Sheffield, 132-142) or
  `speed_garage_revival` (UKG-leaning, 128-138), NOT the older `garage_house`.
- Modern jungle producers (Tim Reaper, Coco Bryce, Sully, Tapes label, Future
  Retro London) -> `jungle_revival_modern`, not generic `jungle_breaks`.
- Sliding 808s + UK rap accent -> `uk_drill`, not `trap_bass_bridge`.
- Memphis cowbell + drift / car aesthetic -> `drift_phonk`.
- Anyma, Afterlife, Massano-style euphoric arpeggios at 128-138 BPM ->
  `trance_revival_modern`, not `melodic_techno_driver`, when the build is
  clearly trance-shaped (long sweep, supersaw lead, climactic drop).
- Pitched-up / TikTok-edited versions -> `sped_up_edit` ONLY if the title
  or tag explicitly says "sped up" / "nightcore". Otherwise classify the
  underlying genre.
- Amapiano log drum is distinctive — if heard, pick `amapiano_groove` even
  when the provider tag says "afro house".

# Confidence calibration
- 0.85-1.00: BPM in range + 3+ keyword/feature hits + clear mood/groove match.
- 0.60-0.85: BPM in range + 1-2 keyword hits + plausible mood/groove.
- 0.40-0.60: BPM in range, mood plausible, no strong keywords, provider hint
  agrees. Always list 2-3 alternatives at this band.
- <0.40: weak evidence. Pick best candidate, list 3+ alternatives, and add a
  short note in `rationale` flagging the ambiguity.

# Output schema (return EXACTLY this JSON shape, no extra keys)
{
  "category_id": "<one id from taxonomy>",
  "confidence": <float 0..1>,
  "rationale": "<one or two sentences: which evidence drove the choice, which obvious-looking category you rejected (especially if you considered any afro_* category and rejected it), and the short evidence tags that supported the call (e.g. 'keyword:saz, bpm:118, vocal:zulu')>",
  "alternate_category_ids": ["<id>", "<id>"],
  "warnings": "<empty string unless evidence is thin or ambiguous; otherwise a one-line flag>"
}

When you considered any afro_* category and rejected it (required true for
any tribal / percussive / chant-vocal track that is NOT genuinely afro),
state that explicitly inside `rationale`, e.g. "Rejected Afro House: no
African / Afro-diasporic lineage despite the provider tag."

Short evidence tags use the form `<source>:<value>`, e.g.: "bpm:122",
"mood:dark", "keyword:saz", "artist:Tim Reaper", "label:MoBlack",
"provider:tech_house", "vocal:zulu". Inline them in `rationale`.

# Few-shot examples

Example 1 — Indie tech, dark hypnotic
Input: "Hate (Original Mix) - Bedouin", BPM 124, mood dark/hypnotic, indie dance + tech house provider hints, instrumental.
Output: {"category_id":"driving_dark_indie_tech","confidence":0.9,"rationale":"124 BPM dark hypnotic with indie + tech house hints; classic Bedouin driver. Evidence: bpm:124, mood:dark, mood:hypnotic, provider:indie_dance.","alternate_category_ids":["rolling_dark_indie_tech","hypnotic_dark_indie_tech"],"warnings":""}

Example 2 — World/tribal that is NOT afro (the key case)
Input: "Üsküdara - Dönüş Edit", BPM 118, organic+tribal mood, female chant vocal, provider tag "world / afro house", saz audible.
Output: {"category_id":"anatolian_psych_house","confidence":0.88,"rationale":"Saz keyword + Turkish title outranks the generic afro provider tag; clearly Anatolian. Rejected Afro House: no African / Afro-diasporic lineage. Evidence: keyword:saz, title:turkish, bpm:118, mood:tribal.","alternate_category_ids":["oriental_arabic_house","desert_mystic_driver"],"warnings":""}

Example 3 — Tulum vs afro disambiguation
Input: "Yucatán Sunrise - Bona Fide", BPM 120, tribal/cinematic, chant vocal, provider "afro house, organic house".
Output: {"category_id":"tulum_tribal_driver","confidence":0.82,"rationale":"Yucatán/sunrise keywords + cinematic-tribal at 120 BPM fit the Tulum scene over generic Afro House Peak. Rejected Afro House: no African / Afro-diasporic context. Evidence: keyword:yucatan, keyword:sunrise, bpm:120, mood:cinematic.","alternate_category_ids":["tulum_tribal_sunrise","cinematic_tribal_builder","afro_house_peak"],"warnings":""}

Example 4 — Genuine afro (so the model knows the floor)
Input: "Umqombothi - Caiiro Remix", BPM 122, tribal/euphoric, Zulu vocal, label MoBlack.
Output: {"category_id":"afro_house_peak","confidence":0.93,"rationale":"Zulu vocal + MoBlack label + 122 BPM tribal-euphoric = clear afro house peak; no other regional cues. Evidence: label:MoBlack, vocal:zulu, artist:Caiiro, bpm:122.","alternate_category_ids":["deep_afro_house","afro_tech_driver"],"warnings":""}

Example 5 — UK bass disambiguation
Input: "Move (Skepsis Remix)", BPM 138, raw/playful, wobble bass, vocal hook, provider "uk garage / bassline".
Output: {"category_id":"bassline_niche","confidence":0.9,"rationale":"Skepsis + 138 BPM + wobble bass is core bassline; faster and rawer than garage_house, slightly above speed_garage_revival range. Evidence: artist:Skepsis, bpm:138, keyword:wobble, provider:bassline.","alternate_category_ids":["speed_garage_revival","garage_house"],"warnings":""}

Example 6 — Modern jungle revival
Input: "Tribute - Tim Reaper", BPM 168, amen breaks, modern production, provider "jungle / drum and bass".
Output: {"category_id":"jungle_revival_modern","confidence":0.94,"rationale":"Tim Reaper is the canonical modern jungle revival artist; cleaner production than 90s jungle_breaks. Evidence: artist:Tim Reaper, bpm:168, keyword:amen.","alternate_category_ids":["jungle_breaks","liquid_drum_and_bass"],"warnings":""}

Example 7 — Amapiano vs afro
Input: "Asibe Happy - Kabza De Small", BPM 113, warm/soulful, vocal, provider "afro house".
Output: {"category_id":"amapiano_groove","confidence":0.95,"rationale":"Kabza De Small + 113 BPM + log-drum bass is core amapiano, not afro_house_peak despite the provider tag. Rejected Afro House Peak: amapiano lineage is the right call when log drums are audible. Evidence: artist:Kabza De Small, bpm:113, keyword:log_drum, provider:afro_house.","alternate_category_ids":["deep_afro_house","afro_house_peak"],"warnings":""}

Example 8 — Ambiguous, low confidence
Input: "Untitled 04", BPM 121, vocal "spoken", mood "warm", no keywords, provider "house".
Output: {"category_id":"warm_deep_house","confidence":0.45,"rationale":"Warm + 121 BPM + spoken vocal best fits warm_deep_house but evidence is thin; multiple deep house slots plausible. Rejected any afro_*: no afro/tribal cues. Evidence: bpm:121, mood:warm, vocal:spoken.","alternate_category_ids":["lo_fi_deep_house","melodic_house_builder","organic_house_builder"],"warnings":"thin evidence"}

Example 9 — Minimal techno (Cocoon-style driver)
Input: "Steady Roller - Sven Väth", BPM 128, mood minimal/hypnotic, instrumental, provider "minimal techno / techno".
Output: {"category_id":"minimal_techno_tool","confidence":0.88,"rationale":"Sven Väth + 128 BPM + minimal/hypnotic instrumental fits minimal_techno_tool; mood is minimal not deep, ruling out hypnotic_deep_techno. Evidence: artist:Sven Väth, bpm:128, mood:minimal, provider:minimal_techno.","alternate_category_ids":["hypnotic_deep_techno","minimal_deep_tech"],"warnings":""}

Example 10 — Dub techno (Basic Channel)
Input: "Mantle - Maurizio", BPM 122, mood deep/dub/atmospheric, dub chord stab, provider "dub techno".
Output: {"category_id":"dub_techno","confidence":0.95,"rationale":"Maurizio + Basic Channel chord stab + 122 BPM + deep/dub/atmospheric is canonical dub_techno. Evidence: artist:Maurizio, label:Basic Channel, bpm:122, mood:dub.","alternate_category_ids":["hypnotic_deep_techno","minimal_dub_tool"],"warnings":""}

Example 11 — Micro house (Perlon)
Input: "Easy Lee - Ricardo Villalobos", BPM 124, mood minimal/warm/playful, clicks and textures, provider "minimal / micro house".
Output: {"category_id":"micro_house","confidence":0.92,"rationale":"Villalobos + Perlon-style clicks + 124 BPM + warm/playful minimal is core micro_house, warmer than minimal_techno_tool. Evidence: artist:Ricardo Villalobos, label:Perlon, bpm:124, keyword:clicks.","alternate_category_ids":["minimal_techno_tool","minimal_deep_tech"],"warnings":""}

Example 12 — Warm deep house (Larry Heard)
Input: "Can You Feel It - Mr. Fingers", BPM 118, mood warm/deep/soulful, vocal pads, provider "deep house / classic deep house".
Output: {"category_id":"warm_deep_house","confidence":0.94,"rationale":"Mr. Fingers is canonical warm deep house; 118 BPM + warm/deep/soulful with vocal pads fits cleanly. Evidence: artist:Mr. Fingers, bpm:118, mood:warm, mood:soulful.","alternate_category_ids":["soulful_vocal_house","melodic_house_builder"],"warnings":""}

Example 13 — Lo-fi deep house (dusty)
Input: "Don't You Want My Love - Moodymann", BPM 116, mood warm/gritty/deep, dusty texture, sampled vocal, provider "deep house / lo-fi house".
Output: {"category_id":"lo_fi_deep_house","confidence":0.91,"rationale":"Moodymann + dusty texture + sampled vocal at 116 BPM is core lo_fi_deep_house, distinct from cleaner warm_deep_house. Evidence: artist:Moodymann, bpm:116, keyword:dusty, keyword:lofi.","alternate_category_ids":["warm_deep_house","dub_deep_house"],"warnings":""}

Example 14 — Sunset Balearic (sub-110 BPM)
Input: "Sirius (Sunset Edit) - DJ Tennis", BPM 102, mood sunlit/warm/atmospheric, vocal, provider "balearic / chillout".
Output: {"category_id":"sunset_balearic_house","confidence":0.86,"rationale":"102 BPM + sunlit/atmospheric + 'Sunset Edit' mix cue fits sunset_balearic_house; below balearic_deep_house groove range. Evidence: bpm:102, mood:sunlit, mix_name:sunset_edit.","alternate_category_ids":["balearic_deep_house","balearic_organic_house"],"warnings":""}

Example 15 — Classic house (Frankie Knuckles piano)
Input: "Your Love - Frankie Knuckles", BPM 122, mood warm/euphoric/playful, vocal, piano hook, provider "house / classic house".
Output: {"category_id":"classic_house","confidence":0.96,"rationale":"Frankie Knuckles + piano hook + warm/euphoric vocal at 122 BPM is definitional classic_house. Evidence: artist:Frankie Knuckles, bpm:122, keyword:piano, mood:euphoric.","alternate_category_ids":["soulful_vocal_house","funky_disco_house"],"warnings":""}

Example 16 — Raw acid house (TB-303)
Input: "Acid Tracks - Phuture", BPM 124, mood acidic/raw/warehouse, instrumental, provider "acid house".
Output: {"category_id":"raw_acid_house","confidence":0.97,"rationale":"Phuture + 303 acid line + 124 BPM warehouse is foundational raw_acid_house; slower and looser than acid_tech_house_peak. Evidence: artist:Phuture, bpm:124, keyword:303, mood:acidic.","alternate_category_ids":["acid_tech_house_peak","acid_techno"],"warnings":""}

Example 17 — Funky disco house (French touch)
Input: "Music Sounds Better With You - Stardust", BPM 124, mood euphoric/playful/warm, vocal disco loop, provider "french house / disco".
Output: {"category_id":"funky_disco_house","confidence":0.92,"rationale":"Stardust + filtered disco loop + euphoric vocal is funky_disco_house origin material; nu_disco_house refers to the modern revival, not the original wave. Evidence: artist:Stardust, bpm:124, mood:euphoric, provider:french_house.","alternate_category_ids":["nu_disco_house","classic_house"],"warnings":""}
"""


def _system_prompt(validation_error: str | None = None) -> str:
    """Return the system prompt, with an optional retry suffix appended."""
    if not validation_error:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\nPrevious response was invalid: {validation_error}\n"


# Kept as a thin alias so external callers that reach in for the prompt by
# its historical name continue to work.
def _instructions(validation_error: str | None = None) -> str:
    return _system_prompt(validation_error)


def build_user_message(track: dict) -> str:
    """Format a single track's evidence into a per-track user message.

    Pass any subset of the fields below; missing or empty values are skipped
    so the LLM only sees signal, not placeholders.

    Recommended keys: title, artist, label, mix_name, filename, bpm,
    musical_key, energy, moods, grooves, vocal_profile, structure,
    provider_genres, provider_tags, audio_features.

    Returns a string ending with the explicit JSON-only instruction so the
    model doesn't drift into prose.
    """
    if not track:
        raise ValueError("track evidence is empty")

    lines = ["TRACK EVIDENCE:"]
    for key, val in track.items():
        if val in (None, "", [], {}):
            continue
        lines.append(f"  {key}: {val}")
    lines.append("")
    lines.append("Return the JSON object only.")
    return "\n".join(lines)


def _flatten_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """Project the rich `_track_context` dict into the flat evidence form
    expected by `build_user_message`.

    Empty / placeholder values are kept here so the build_user_message filter
    can drop them — keeps the projection rule in one place.
    """
    file_info = context.get("file") or {}
    track_info = context.get("track") or {}
    observations = context.get("observations") or []

    provider_genres = sorted({
        (obs.get("genre") or "").strip()
        for obs in observations
        if isinstance(obs, dict) and obs.get("genre")
    })
    audio_feature_keys = (
        "energy", "valence", "danceability",
        "instrumentalness", "acousticness", "speechiness", "liveness",
    )
    audio_features: dict[str, Any] = {}
    for key in audio_feature_keys:
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            val = obs.get(key)
            if val not in (None, "", []):
                audio_features[key] = val
                break

    bpm = (
        track_info.get("canonical_bpm")
        or track_info.get("tagger_bpm")
        or file_info.get("embedded_bpm")
        or ""
    )

    return {
        "filename": file_info.get("file_name") or "",
        "title": track_info.get("title") or file_info.get("embedded_title") or "",
        "artist": track_info.get("artist") or file_info.get("embedded_artist") or "",
        "mix_name": track_info.get("mix") or "",
        "label": track_info.get("label") or "",
        "bpm": bpm,
        "musical_key": track_info.get("canonical_key") or "",
        "energy": track_info.get("tagger_energy") or "",
        "moods": track_info.get("tagger_mood") or "",
        "vocal_profile": track_info.get("tagger_vocal") or "",
        "structure": track_info.get("tagger_structure") or "",
        "provider_genres": ", ".join(g for g in provider_genres if g),
        "provider_tags": file_info.get("embedded_genre") or "",
        "audio_features": (
            ", ".join(f"{k}={v}" for k, v in audio_features.items())
            if audio_features else ""
        ),
    }


def _build_input_text(context: dict[str, Any], taxonomy_json: dict[str, Any]) -> str:
    """Build the single text payload sent as the user-message content.

    The new prompt assumes `allowed_dj_taxonomy.categories` is accessible to
    the LLM. We inline the taxonomy JSON ahead of the track evidence so it
    arrives in the same turn as the per-track context.
    """
    taxonomy_blob = json.dumps(
        {"allowed_dj_taxonomy": taxonomy_json},
        ensure_ascii=True,
        sort_keys=True,
    )
    track_block = build_user_message(_flatten_for_prompt(context))
    return f"{taxonomy_blob}\n\n{track_block}"


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "category_id",
            "confidence",
            "alternate_category_ids",
            "rationale",
            "warnings",
        ],
        "properties": {
            "category_id": {"type": "string"},
            "confidence": {"type": "number"},
            "alternate_category_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
            "warnings": {"type": "string"},
        },
    }


def _track_context(
    audio_path: Path,
    track: LogicalTrack,
    file_record: FileRecord | None,
    observations: list[SourceObservation],
) -> dict[str, Any]:
    external_sources = sorted(
        {
            obs.source_system
            for obs in observations
            if obs.source_system and obs.source_system not in {"tag", "analysis_librosa", "analysis_essentia", "manual"}
        }
    )
    return {
        "file": {
            "file_name": audio_path.name,
            "path": str(audio_path),
            "embedded_title": file_record.embedded_title if file_record else "",
            "embedded_artist": file_record.embedded_artist if file_record else "",
            "embedded_album": file_record.embedded_album if file_record else "",
            "embedded_genre": file_record.embedded_genre if file_record else "",
            "embedded_bpm": file_record.embedded_bpm if file_record else "",
            "embedded_comment": file_record.embedded_comment if file_record else "",
        },
        "track": {
            "track_id": track.track_id,
            "artist": track.artist_canonical,
            "title": track.title_canonical,
            "mix": track.mix_canonical,
            "album": track.album_canonical,
            "label": track.label_canonical,
            "canonical_bpm": track.canonical_bpm,
            "canonical_key": track.canonical_key_camelot,
            "tagger_energy": track.tagger_energy,
            "tagger_mood": track.tagger_vibe,
            "tagger_mood_scores": track.tagger_vibe_scores,
            "tagger_vocal": track.tagger_vocal,
            "tagger_vocal_scores": track.tagger_vocal_scores,
            "tagger_structure": track.tagger_structure,
            "tagger_bpm": track.tagger_bpm,
            "tagger_confidences": track.tagger_confidences,
        },
        "evidence_availability": {
            "internal": bool(file_record or track.tagger_energy or track.tagger_vibe or track.tagger_vocal),
            "external": external_evidence_available(observations),
            "external_sources": external_sources,
        },
        "observations": [
            {
                "source": obs.source_system,
                "artist": obs.artist,
                "title": obs.title,
                "genre": obs.genre,
                "genres_all": obs.genres_all,
                "label": obs.label,
                "release_date": obs.release_date,
                "bpm": obs.bpm,
                "comments": obs.comments,
                "acousticness": obs.acousticness,
                "danceability": obs.danceability,
                "energy": obs.energy,
                "instrumentalness": obs.instrumentalness,
                "liveness": obs.liveness,
                "speechiness": obs.speechiness,
                "valence": obs.valence,
                "tagger_energy": obs.tagger_energy,
                "tagger_mood": obs.tagger_vibe,
                "tagger_vocal": obs.tagger_vocal,
                "tagger_structure": obs.tagger_structure,
            }
            for obs in observations
        ],
    }


def _label_to_row(
    file_name: str,
    track: LogicalTrack,
    observations: list[SourceObservation],
    label: dict[str, Any],
    model: str,
    taxonomy: DjTaxonomy,
) -> dict[str, Any]:
    category = taxonomy.category(str(label.get("category_id") or "").strip())
    row = {column: "" for column in DJ_GROUND_TRUTH_COLUMNS}
    row.update(category.metadata_row())
    external_sources = sorted(
        {
            obs.source_system
            for obs in observations
            if obs.source_system and obs.source_system not in {"tag", "analysis_librosa", "analysis_essentia", "manual"}
        }
    )
    row.update(
        {
            "file_name": file_name,
            "track_id": track.track_id,
            "artist": track.artist_canonical,
            "title": track.title_canonical,
            "mix": track.mix_canonical,
            "tagger_energy": track.tagger_energy,
            "tagger_mood": track.tagger_vibe,
            "tagger_vocal": track.tagger_vocal,
            "tagger_structure": track.tagger_structure,
            "bpm_hint": track.canonical_bpm or track.tagger_bpm,
            "confidence": label.get("confidence", ""),
            "alternate_category_ids": ";".join(str(item) for item in label.get("alternate_category_ids", [])),
            "rationale": label.get("rationale", ""),
            "warnings": label.get("warnings", ""),
            "internal_evidence_available": "YES" if track.tagger_energy or track.tagger_vibe or track.tagger_vocal else "NO",
            "external_evidence_available": "YES" if external_evidence_available(observations) else "NO",
            "external_sources": ";".join(external_sources),
            "label_source": "gpt5_seed",
            "model": model,
            "taxonomy_version": taxonomy.version,
            "created_at": now_iso(),
        }
    )
    return row


def _error_row(
    file_name: str,
    track: LogicalTrack,
    model: str,
    taxonomy: DjTaxonomy,
    error: str,
) -> dict[str, Any]:
    row = {column: "" for column in DJ_GROUND_TRUTH_COLUMNS}
    row.update(
        {
            "file_name": file_name,
            "track_id": track.track_id,
            "artist": track.artist_canonical,
            "title": track.title_canonical,
            "mix": track.mix_canonical,
            "warnings": error,
            "label_source": "gpt5_error",
            "model": model,
            "taxonomy_version": taxonomy.version,
            "created_at": now_iso(),
        }
    )
    return row


def _audio_files(files_dir: str) -> list[Path]:
    root = Path(files_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def _load_existing_rows(path: str) -> dict[str, dict[str, str]]:
    out = Path(path)
    if not out.exists():
        return {}
    with out.open(newline="", encoding="utf-8-sig") as f:
        return {
            row.get("file_name", ""): row
            for row in csv.DictReader(f)
            if row.get("file_name")
        }


def _is_error_row(row: dict[str, str]) -> bool:
    return (row.get("label_source") or "").strip().endswith("_error") or not (row.get("category_id") or "").strip()


def _write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DJ_GROUND_TRUTH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_output_path(output_path: str | None, output_dir: str) -> str:
    if not output_path:
        return str(_default_output_root(output_dir) / "dj_taxonomy_ground_truth.csv")
    normalized = Path(output_path).as_posix().lower().strip("./")
    if normalized in {"files/dj_taxonomy_ground_truth.csv", "outputs/registry/dj_taxonomy_ground_truth.csv"}:
        return str(_default_output_root(output_dir) / "dj_taxonomy_ground_truth.csv")
    return output_path


def _default_output_root(output_dir: str) -> Path:
    path = Path(output_dir)
    return path.parent if path.name == "registry" else path


def _cache_key(model: str, taxonomy_json: dict[str, Any], context: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "taxonomy": taxonomy_json, "context": context},
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_error_message(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__
