"""GPT-assisted ground truth for flat DJ-functional taxonomy categories."""

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
            "instructions": _instructions(validation_error),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "allowed_dj_taxonomy": taxonomy_json,
                                    "track_context": context,
                                },
                                ensure_ascii=True,
                                sort_keys=True,
                            ),
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
                {"role": "system", "content": _instructions(validation_error)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_dj_taxonomy": taxonomy_json,
                            "track_context": context,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
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


def _instructions(validation_error: str | None = None) -> str:
    retry = f"\nPrevious response was invalid: {validation_error}\n" if validation_error else ""
    return (
        "Create one ground-truth label for a DJ-functional music taxonomy classifier. "
        "Choose exactly one category_id from allowed_dj_taxonomy.categories. "
        "Never invent category IDs, labels, moods, grooves, or metadata. "
        "Provider genres from Rekordbox, Spotify, Songstats, or embedded tags are hints only; "
        "they may be wrong. Prefer the full evidence pattern: local tagger energy/mood/vocal/structure/BPM, "
        "filename/title/mix cues, provider audio features, provider genres, and label/artist context. "
        "If evidence is ambiguous, still choose the best allowed category but lower confidence and list allowed alternatives. "
        "Return JSON only. Few-shot guidance: "
        "Example A: dark/hypnotic E4, 124-126 BPM, indie dance + tech house hints -> driving_dark_indie_tech. "
        "Example B: tech house hints with featured vocal or strong hook -> vocal_hook_tech_house. "
        "Example C: organic/tribal mood + chant vocal alone is NOT enough for tribal_afro_driver — see Modern Subgenre Selection Rules. "
        "Example D: raw/warehouse mood, E5, 130+ BPM techno clues -> raw_warehouse_techno or peak_time_techno. "
        "Example E: minimal/deep/dub mood, sparse vocal, steady rolling low-mid energy -> minimal_deep_tech or dub_techno. "
        "\n\n"
        + _MODERN_SUBGENRE_SELECTION_RULES
        + retry
    )


# Spec: specs/update_llm_ground_truth_prompt.md — conservative tribal/afro
# subgenre classification. Tribal and Afro must be earned by evidence, not
# triggered by mood, percussion, or internal shorthand tags.
_MODERN_SUBGENRE_SELECTION_RULES = """\
Modern Subgenre Selection Rules
================================

Tribal and Afro must be EARNED by evidence, not triggered by mood, percussion,
or internal shorthand tags.

1. Separate genre from descriptor tags
--------------------------------------
The following are TAXONOMY LABELS and may be used as the chosen genre/subgenre:
  Organic House, Afro House, Tribal House, Melodic Techno, Dark Melodic Techno,
  Driving Techno, Progressive House, Indie Dance, Dark Disco, Downtempo.

The following are DESCRIPTOR TAGS and should NOT by themselves become the
chosen genre/subgenre:
  tribal, afro, mayan, ritual, ceremonial, shamanic, organic, percussive,
  ethnic, desert, tulum, chant, driving, hypnotic, dark, tense,
  female vocal, spoken vocal.

A descriptor tag belongs in the rationale field as context — not as the
chosen category.

2. Be conservative with Afro and Tribal
---------------------------------------
Only choose an Afro House / Afro Tech / Tribal House / Tribal Techno /
Tribal Organic House category if at least one STRONG evidence holds:
  - Trusted provider genre explicitly says "Afro House", "Afro Tech",
    "Tribal House", "Tribal Techno", or "Tribal Organic House".
  - Manually-curated rekordbox/embedded genre explicitly says the same.
  - Arrangement is dominated by organic hand-percussion, ritual vocals,
    ceremonial rhythm, or tribal drum structure.
  - Artist/label is clearly Afro House / Tulum-organic-house / ritual-house
    / tribal-house scene.

The following are WEAK evidence and are NOT sufficient on their own:
  - dark mood, tense mood, driving groove, hypnotic groove,
  - percussion exists, organic texture, female vocal, chant-like vocal,
  - low vocal, Tulum/desert/ritual words in title/comment,
  - internal tag containing TRIB / AFRO / DRV.

If only weak evidence is present, put those words in the rationale as
"secondary descriptors" and choose a non-afro/non-tribal category that
actually matches the dominant musical character (e.g., Dark Melodic Techno,
Driving Tech-House, Progressive House).

3. Distinguish Afro from Tribal from Organic
--------------------------------------------
Afro House / Afro-Tech requires evidence of African or Afro-diasporic
lineage — not just percussion, ritual, or organic atmosphere. Do NOT
infer Afro from tribal, mayan, ritual, ceremonial, organic, desert,
Tulum, shamanic, ethnic, chant, or percussion alone.

Tribal House / Tribal Organic House requires that ritual/ceremonial/tribal
percussion be STRUCTURALLY CENTRAL to the arrangement — not merely present.

Organic House is for earthy/natural/melodic/ethnic/acoustic music
(including desert/Tulum-style). Organic House may have tribal influence
without becoming Tribal House.

Melodic Techno / Dark Melodic Techno is the right call for tracks with a
minor-key melody, emotional synth, progressive arrangement, tense/dark
mood, and a driving-but-not-percussion-dominant groove — even if internal
tags include TRIB/AFRO/DRV.

4. Negative example: Erly Tepshi - Virgo
----------------------------------------
A track at 120 BPM with internal tags TRIB.AFRO.DRV, mood "tense", female
vocal, source_genre "Techno", and no explicit Afro/Tribal provider hint
must NOT be classified Afro/Tribal. The correct call is Dark Melodic
Techno. Driving + tense + female vocal + TRIB shorthand are descriptor
cues, not taxonomy labels.

5. Positive example: PAAX Tulum - Crisol (MIICHII Remix)
--------------------------------------------------------
A track whose arrangement is built around organic/ritual/ceremonial
percussion with Mayan/Tulum context CAN be Tribal Organic House. But it
should NOT be Afro House unless explicit Afro/African/Afro-diasporic
evidence exists.

6. Decision gate before choosing any Afro/Tribal category
---------------------------------------------------------
Before assigning an Afro or Tribal category, the answer to all of these
must be yes (or at least three independent weak signals must converge):

  (a) Does a trusted provider genre explicitly say Afro/Tribal?
  (b) If not, are there 3+ independent weak signals pointing to Afro/Tribal?
  (c) Is percussion/ritual/ceremonial element STRUCTURALLY central?
  (d) For Afro specifically: is there African or Afro-diasporic context?

If "no", put those concepts into the rationale as secondary descriptors
and pick a category that actually matches the music.

7. Rejected-label reasoning
---------------------------
When you considered an Afro or Tribal category but rejected it, briefly
note that in the rationale field (e.g., "Considered Tribal Techno but
percussion is not structurally dominant; chose Dark Melodic Techno").

8. Scoring guidance
-------------------
  - Explicit trusted provider genre dominates.
  - Internal generated tags (TRIB / AFRO / DRV / etc.) are WEAK evidence.
  - Mood words (dark, tense, romantic) are NOT genre evidence.
  - Texture words (organic, hand-percussion) are NOT genre evidence
    unless backed by source metadata or arrangement.
  - Driving / rolling / hypnotic are groove descriptors, not genres.
  - Tribal requires percussion/ritual structure to be CENTRAL.
  - Afro requires Afro/African/Afro-diasporic lineage/context.

"""


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
