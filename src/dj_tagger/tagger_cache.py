"""Shared cache metadata for tagger results."""

from __future__ import annotations

import json

from .settings import derived_version, key_version, raw_version, tagger_version


def audio_features_signature(audio_features: dict[str, float] | None) -> str:
    """Stable fingerprint for Songstats audio features that affect derived scoring."""
    if not audio_features:
        return ""
    normalized = {k: float(v) for k, v in sorted(audio_features.items())}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def current_tagger_metadata(audio_features: dict[str, float] | None = None) -> dict[str, str]:
    """Current signature set for tagger caches."""
    return {
        "_tagger_version": tagger_version(),
        "_tagger_raw_sig": raw_version(),
        "_tagger_derived_sig": derived_version(),
        "_tagger_key_sig": key_version(),
        "_tagger_audio_features_sig": audio_features_signature(audio_features),
    }


def hydrate_tagger_result(result: dict, audio_features: dict[str, float] | None = None) -> dict:
    """Attach current cache metadata to a tagger result."""
    hydrated = dict(result)
    hydrated.update(current_tagger_metadata(audio_features))
    return hydrated


def merge_rederived_tagger(existing: dict | None, derived: dict, audio_features: dict[str, float] | None = None) -> dict:
    """Merge re-derived fields into an existing tagger record and restamp metadata."""
    merged = dict(existing or {})
    merged.update(derived)
    merged.update(current_tagger_metadata(audio_features))
    return merged


def key_signature_matches(result: dict | None) -> bool:
    """Return whether a cached tagger record contains collected key data.

    Key/librosa analysis is a data-collection result. Its stored signature is
    audit metadata only and must not force audio re-analysis when the filename
    + duration cache identity is present.
    """
    return bool(result) and bool(result.get("camelot") or result.get("key"))


def tagger_core_metadata_matches(result: dict | None) -> bool:
    """Check current derived signature, ignoring raw/key collection metadata."""
    if not result:
        return False
    return result.get("_tagger_derived_sig") == derived_version()


def derived_signature_matches(result: dict | None, audio_features: dict[str, float] | None = None) -> bool:
    if not result:
        return False
    return (
        result.get("_tagger_derived_sig") == derived_version()
        and result.get("_tagger_audio_features_sig", "") == audio_features_signature(audio_features)
    )


def tagger_metadata_matches(result: dict | None, audio_features: dict[str, float] | None = None) -> bool:
    if not result:
        return False
    return derived_signature_matches(result, audio_features)
