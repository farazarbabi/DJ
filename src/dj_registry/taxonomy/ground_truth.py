"""GPT-assisted taxonomy ground-truth CSV generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import FileRecord, LogicalTrack, SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .classifier import GenreTaxonomy, load_taxonomy

GROUND_TRUTH_COLUMNS = [
    "file_name",
    "track_id",
    "artist",
    "title",
    "mix",
    "family",
    "genre",
    "subgenre",
    "energy",
    "vibe",
    "vocal",
    "structure",
    "set_role",
    "bpm_hint",
    "confidence",
    "alternate_family",
    "alternate_genre",
    "alternate_subgenre",
    "rationale",
    "warnings",
    "label_source",
    "model",
    "taxonomy_version",
    "created_at",
]

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".aiff", ".aif", ".wav", ".flac", ".m4a"}
MAX_LABEL_OUTPUT_TOKENS = 4096


@dataclass
class GroundTruthStats:
    files_seen: int
    rows_written: int
    generated: int
    reused: int
    errors: int
    output_path: str
    error_message: str = ""


@dataclass
class ApiConnectionTestResult:
    ok: bool
    provider: str
    model: str
    error_message: str = ""


class OpenAIGroundTruthClient:
    """Small Responses API client for structured taxonomy labels."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = "gpt-5",
        base_url: str = "https://api.openai.com/v1",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        azure_api_version: str | None = None,
        timeout: float = 90.0,
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
                "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required to generate taxonomy ground truth"
            )
        if self.use_azure:
            if not self.azure_deployment:
                raise RuntimeError("AZURE_OPENAI_CHAT_DEPLOYMENT is required for Azure OpenAI ground truth generation")
            if not self.azure_api_version:
                raise RuntimeError("AZURE_OPENAI_API_VERSION is required for Azure OpenAI ground truth generation")

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
                                    "allowed_taxonomy": taxonomy_json,
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
                    "name": "dj_taxonomy_ground_truth",
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
                            "allowed_taxonomy": taxonomy_json,
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
                    "name": "dj_taxonomy_ground_truth",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
            "max_completion_tokens": MAX_LABEL_OUTPUT_TOKENS,
        }
        url = self._azure_chat_completions_url()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
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

    @property
    def provider(self) -> str:
        return "azure" if self.use_azure else "openai"


def generate_ground_truth_csv(
    store: CsvStore,
    *,
    files_dir: str = "./files",
    output_path: str = "files/taxonomy_ground_truth.csv",
    taxonomy_path: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    force: bool = False,
    fail_fast: bool = True,
    show_progress: bool = False,
    cache_dir: str | None = None,
    client: Any | None = None,
) -> GroundTruthStats:
    """Generate a GPT-seeded taxonomy label CSV for audio files."""
    taxonomy = load_taxonomy(taxonomy_path)
    taxonomy_json = _load_taxonomy_json(taxonomy_path)
    audio_files = _audio_files(files_dir)
    if limit is not None:
        audio_files = audio_files[:limit]

    existing = _load_existing_rows(output_path)
    cache_root = Path(cache_dir or Path(store.output_dir) / "taxonomy_ground_truth_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    api_client = client or OpenAIGroundTruthClient(model=model or os.environ.get("OPENAI_MODEL", "gpt-5"))
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
    progress = ProgressBar(len(audio_files), label="Ground truth", enabled=show_progress)

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
                error = validate_label(label, taxonomy)
                if error:
                    label = api_client.label_track(context, taxonomy_json, validation_error=error)
                cache_path.write_text(json.dumps(label, indent=2, sort_keys=True), encoding="utf-8")

            error = validate_label(label, taxonomy)
            if error:
                errors += 1
                error_message = f"{file_name}: {error}"
                if fail_fast:
                    progress.finish(f"error {file_name}")
                    return GroundTruthStats(
                        files_seen=len(audio_files),
                        rows_written=0,
                        generated=generated,
                        reused=reused,
                        errors=errors,
                        output_path=output_path,
                        error_message=error_message,
                    )
                row = _error_row(file_name, track, model_name, taxonomy_path, error)
            else:
                generated += 1
                row = _label_to_row(file_name, track, label, model_name, taxonomy_path)
        except Exception as exc:
            errors += 1
            error_message = f"{file_name}: {_safe_error_message(exc)}"
            if fail_fast:
                progress.finish(f"error {file_name}")
                return GroundTruthStats(
                    files_seen=len(audio_files),
                    rows_written=0,
                    generated=generated,
                    reused=reused,
                    errors=errors,
                    output_path=output_path,
                    error_message=error_message,
                )
            row = _error_row(file_name, track, model_name, taxonomy_path, error_message)
        rows.append(row)
        progress.update(index, f"processed {file_name}", new=generated, reused=reused, errors=errors)

    progress.finish("complete")
    _write_rows(output_path, rows)
    return GroundTruthStats(
        files_seen=len(audio_files),
        rows_written=len(rows),
        generated=generated,
        reused=reused,
        errors=errors,
        output_path=output_path,
    )


def test_api_connection(
    *,
    model: str | None = None,
    client: Any | None = None,
) -> ApiConnectionTestResult:
    """Make one minimal taxonomy-label call to validate API connectivity."""
    api_client = client or OpenAIGroundTruthClient(model=model or os.environ.get("OPENAI_MODEL", "gpt-5"))
    provider = getattr(api_client, "provider", "unknown")
    model_name = getattr(api_client, "model", model or "unknown")
    taxonomy_json = {"House": {"Tech House": ["Rolling Tech House"]}}
    taxonomy = GenreTaxonomy(taxonomy_json)
    context = {
        "file": {"file_name": "connection_test.mp3"},
        "track": {
            "artist": "Connection Test",
            "title": "Rolling Club Tool",
            "tagger_energy": "E4",
            "tagger_structure": "16H",
            "tagger_bpm": "126",
        },
        "observations": [],
    }
    try:
        label = api_client.label_track(context, taxonomy_json)
        error = validate_label(label, taxonomy)
        if error:
            return ApiConnectionTestResult(False, provider, model_name, error)
        return ApiConnectionTestResult(True, provider, model_name)
    except Exception as exc:
        return ApiConnectionTestResult(False, provider, model_name, _safe_error_message(exc))


def extract_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON text from a Responses API response payload."""
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    if not texts:
        raise ValueError("OpenAI response did not contain output text")
    return json.loads("\n".join(texts))


def extract_chat_completion_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON text from an Azure/OpenAI Chat Completions payload."""
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Chat completion response did not contain choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content_value = message.get("content")
    if isinstance(content_value, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content_value
        ).strip()
    else:
        content = str(content_value or "").strip()
    if not content:
        usage = payload.get("usage") or {}
        raise ValueError(
            "Chat completion response did not contain message content "
            f"(finish_reason={choice.get('finish_reason')}, usage={usage})"
        )
    return json.loads(content)


def validate_label(label: dict[str, Any], taxonomy: GenreTaxonomy) -> str | None:
    family = str(label.get("family") or "").strip()
    genre = str(label.get("genre") or "").strip()
    subgenre = str(label.get("subgenre") or "").strip()
    if not taxonomy.validate_path(family, genre, subgenre):
        return f"Invalid taxonomy path: {family} > {genre} > {subgenre}"
    return None


def _instructions(validation_error: str | None = None) -> str:
    retry = f"\nPrevious response was invalid: {validation_error}\n" if validation_error else ""
    return (
        "Create one training label for a DJ music taxonomy classifier. "
        "Choose family, genre, and subgenre exactly from the allowed taxonomy. "
        "Use all supplied metadata as evidence, but do not assume Rekordbox, Spotify, "
        "Songstats, or embedded genre values are correct. They are only hints. "
        "Do not upload or request raw audio. Return only the required JSON object. "
        "Use concise rationale and warnings. "
        + retry
    )


def _response_schema() -> dict[str, Any]:
    string = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "family",
            "genre",
            "subgenre",
            "energy",
            "vibe",
            "vocal",
            "structure",
            "set_role",
            "bpm_hint",
            "confidence",
            "alternate_family",
            "alternate_genre",
            "alternate_subgenre",
            "rationale",
            "warnings",
        ],
        "properties": {
            "family": string,
            "genre": string,
            "subgenre": string,
            "energy": string,
            "vibe": string,
            "vocal": string,
            "structure": string,
            "set_role": string,
            "bpm_hint": string,
            "confidence": {"type": "number"},
            "alternate_family": string,
            "alternate_genre": string,
            "alternate_subgenre": string,
            "rationale": string,
            "warnings": string,
        },
    }


def _track_context(
    audio_path: Path,
    track: LogicalTrack,
    file_record: FileRecord | None,
    observations: list[SourceObservation],
) -> dict[str, Any]:
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
            "tagger_energy": track.tagger_energy,
            "tagger_vibe": track.tagger_vibe,
            "tagger_vocal": track.tagger_vocal,
            "tagger_vocal_scores": getattr(track, "tagger_vocal_scores", ""),
            "tagger_structure": track.tagger_structure,
            "tagger_bpm": track.tagger_bpm,
        },
        "observations": [
            {
                "source": obs.source_system,
                "artist": obs.artist,
                "title": obs.title,
                "genre": obs.genre,
                "genres_all": obs.genres_all,
                "label": obs.label,
                "bpm": obs.bpm,
                "comments": obs.comments,
                "acousticness": obs.acousticness,
                "danceability": obs.danceability,
                "energy": obs.energy,
                "instrumentalness": obs.instrumentalness,
                "valence": obs.valence,
                "tagger_energy": obs.tagger_energy,
                "tagger_vibe": obs.tagger_vibe,
                "tagger_vocal": obs.tagger_vocal,
                "tagger_vocal_scores": getattr(obs, "tagger_vocal_scores", ""),
                "tagger_structure": obs.tagger_structure,
            }
            for obs in observations
        ],
    }


def _label_to_row(
    file_name: str,
    track: LogicalTrack,
    label: dict[str, Any],
    model: str,
    taxonomy_path: str | None,
) -> dict[str, Any]:
    row = {column: "" for column in GROUND_TRUTH_COLUMNS}
    row.update(
        {
            "file_name": file_name,
            "track_id": track.track_id,
            "artist": track.artist_canonical,
            "title": track.title_canonical,
            "mix": track.mix_canonical,
            "family": label.get("family", ""),
            "genre": label.get("genre", ""),
            "subgenre": label.get("subgenre", ""),
            "energy": label.get("energy", ""),
            "vibe": label.get("vibe", ""),
            "vocal": label.get("vocal", ""),
            "structure": label.get("structure", ""),
            "set_role": label.get("set_role", ""),
            "bpm_hint": label.get("bpm_hint", ""),
            "confidence": label.get("confidence", ""),
            "alternate_family": label.get("alternate_family", ""),
            "alternate_genre": label.get("alternate_genre", ""),
            "alternate_subgenre": label.get("alternate_subgenre", ""),
            "rationale": label.get("rationale", ""),
            "warnings": label.get("warnings", ""),
            "label_source": "gpt5_seed",
            "model": model,
            "taxonomy_version": _taxonomy_version(taxonomy_path),
            "created_at": now_iso(),
        }
    )
    return row


def _error_row(
    file_name: str,
    track: LogicalTrack,
    model: str,
    taxonomy_path: str | None,
    error: str,
) -> dict[str, Any]:
    row = {column: "" for column in GROUND_TRUTH_COLUMNS}
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
            "taxonomy_version": _taxonomy_version(taxonomy_path),
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


def _load_existing_rows(output_path: str) -> dict[str, dict[str, str]]:
    path = Path(output_path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row.get("file_name", ""): row for row in csv.DictReader(f) if row.get("file_name")}


def _is_error_row(row: dict[str, str]) -> bool:
    return (row.get("label_source") or "").strip().lower().endswith("_error")


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message or exc.__class__.__name__


def _write_rows(output_path: str, rows: list[dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _load_taxonomy_json(taxonomy_path: str | None) -> dict[str, Any]:
    path = _taxonomy_file_path(taxonomy_path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _cache_key(model: str, taxonomy_json: dict[str, Any], context: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "taxonomy": taxonomy_json, "context": context},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _taxonomy_version(taxonomy_path: str | None) -> str:
    return _taxonomy_file_path(taxonomy_path).name


def _taxonomy_file_path(taxonomy_path: str | None) -> Path:
    if taxonomy_path:
        return Path(taxonomy_path)
    candidates = [
        Path.cwd() / "music_genre_taxonomy_3_level.json",
        Path(__file__).resolve().parents[3] / "music_genre_taxonomy_3_level.json",
        Path(__file__).with_name("genre_taxonomy_3_level.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
