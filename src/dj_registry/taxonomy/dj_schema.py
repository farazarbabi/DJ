"""Flat DJ-functional taxonomy schema backed by dj_taxonomy.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..category_codes import compact_category_label


@dataclass(frozen=True)
class DjTaxonomyCategory:
    id: str
    label: str
    family: str
    source_genres: tuple[str, ...]
    moods: tuple[str, ...]
    grooves: tuple[str, ...]
    set_roles: tuple[str, ...]
    bpm_range: tuple[int, int] | tuple[()]
    energy_range: tuple[int, int] | tuple[()]
    vocal_profiles: tuple[str, ...]
    keywords: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DjTaxonomyCategory":
        category_id = str(data.get("id") or "").strip()
        label = str(data.get("label") or "").strip()
        family = str(data.get("family") or "").strip()
        if not category_id or not label or not family:
            raise ValueError("Each DJ taxonomy category must include id, label, and family")
        return cls(
            id=category_id,
            label=label,
            family=family,
            source_genres=tuple(_strings(data.get("source_genres"))),
            moods=tuple(_strings(data.get("moods"))),
            grooves=tuple(_strings(data.get("grooves"))),
            set_roles=tuple(_strings(data.get("set_roles"))),
            bpm_range=_int_pair(data.get("bpm_range")),
            energy_range=_int_pair(data.get("energy_range")),
            vocal_profiles=tuple(_strings(data.get("vocal_profiles"))),
            keywords=tuple(_strings(data.get("keywords"))),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "source_genres": list(self.source_genres),
            "moods": list(self.moods),
            "grooves": list(self.grooves),
            "set_roles": list(self.set_roles),
            "bpm_range": list(self.bpm_range),
            "energy_range": list(self.energy_range),
            "vocal_profiles": list(self.vocal_profiles),
            "keywords": list(self.keywords),
        }

    def metadata_row(self) -> dict[str, str]:
        return {
            "category_id": self.id,
            "category_label": self.label,
            "family": self.family,
            "moods": ";".join(self.moods),
            "grooves": ";".join(self.grooves),
            "set_roles": ";".join(self.set_roles),
            "bpm_range": _range_text(self.bpm_range),
            "energy_range": _range_text(self.energy_range),
            "vocal_profiles": ";".join(self.vocal_profiles),
            "source_genres": ";".join(self.source_genres),
            "keywords": ";".join(self.keywords),
        }


class DjTaxonomy:
    """Validated flat category taxonomy for DJ-functional classification."""

    def __init__(self, version: str, categories: list[DjTaxonomyCategory], path: Path | None = None) -> None:
        if not categories:
            raise ValueError("DJ taxonomy must contain at least one category")
        by_id: dict[str, DjTaxonomyCategory] = {}
        for category in categories:
            if category.id in by_id:
                raise ValueError(f"Duplicate DJ taxonomy category id: {category.id}")
            by_id[category.id] = category
        self.version = version
        self.categories = categories
        self.by_id = by_id
        self.path = path

    def validate_category_id(self, category_id: str | None) -> bool:
        return bool(category_id and category_id in self.by_id)

    def category(self, category_id: str) -> DjTaxonomyCategory:
        try:
            return self.by_id[category_id]
        except KeyError as exc:
            raise ValueError(f"Unknown DJ taxonomy category_id: {category_id}") from exc

    def as_prompt_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "categories": [category.to_prompt_dict() for category in self.categories],
        }

    def hash(self) -> str:
        if self.path and self.path.exists():
            return file_hash(self.path)
        payload = json.dumps(self.as_prompt_json(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_dj_taxonomy(path: str | Path | None = None) -> DjTaxonomy:
    taxonomy_path = Path(path) if path else default_dj_taxonomy_path()
    with taxonomy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        raise ValueError("Expected dj_taxonomy.json shape with a top-level categories list")
    categories = [DjTaxonomyCategory.from_dict(item) for item in data["categories"]]
    version = str(data.get("version") or taxonomy_path.name)
    return DjTaxonomy(version, categories, taxonomy_path)


def default_dj_taxonomy_path() -> Path:
    candidates = [
        Path.cwd() / "src" / "dj_registry" / "taxonomy" / "dj_taxonomy.json",
        Path(__file__).with_name("dj_taxonomy.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("dj_taxonomy.json was not found")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def external_evidence_available(observations: list[Any]) -> bool:
    for obs in observations:
        source = str(getattr(obs, "source_system", "") or "").lower()
        if source and source not in {"tag", "analysis_librosa", "analysis_essentia", "manual"}:
            if any(
                str(getattr(obs, field, "") or "").strip()
                for field in (
                    "genre",
                    "genres_all",
                    "label",
                    "release_date",
                    "spotify_id",
                    "beatport_id",
                    "acousticness",
                    "danceability",
                    "energy",
                    "instrumentalness",
                    "valence",
                )
            ):
                return True
    return False


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_pair(value: Any) -> tuple[int, int] | tuple[()]:
    if not isinstance(value, list) or len(value) != 2:
        return ()
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return ()


def _range_text(value: tuple[int, int] | tuple[()]) -> str:
    if len(value) != 2:
        return ""
    return f"{value[0]}-{value[1]}"
