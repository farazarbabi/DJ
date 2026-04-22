"""Resolution reason codes and evidence summary formatting."""

from __future__ import annotations


# Reason codes for canonical_key_resolution_reason
AGREED_ALL_SOURCES = "agreed_all_sources"
AGREED_ANALYSES = "agreed_both_analyses"
AGREED_ANALYSIS_REKORDBOX = "agreed_analysis_and_rekordbox"
AGREED_ANALYSIS_SONGSTATS = "agreed_analysis_and_songstats"
ANALYSIS_OVERRODE_TAG = "analysis_overrode_conflicting_tag"
SINGLE_HIGH_CONFIDENCE = "single_high_confidence_source"
MANUAL_OVERRIDE = "manual_override"
CONFLICT_REVIEW = "metadata_conflict_requires_review"
LOW_CONFIDENCE_REVIEW = "low_confidence_requires_review"
NO_ANALYSIS = "no_analysis_available"
NO_EVIDENCE = "no_key_evidence"


# Review reason codes
REVIEW_CONFLICT = "source_conflict"
REVIEW_LOW_CONFIDENCE = "low_confidence"
REVIEW_NO_ANALYSIS = "no_analysis"
REVIEW_AMBIGUOUS_IDENTITY = "ambiguous_identity"


def classify_resolution(
    top_sources: list[str],
    has_conflict: bool,
) -> str:
    """Determine the resolution reason code from contributing sources."""
    if "manual" in top_sources:
        return MANUAL_OVERRIDE

    analysis_sources = [s for s in top_sources if s.startswith("analysis_")]
    external_sources = [s for s in top_sources if s in ("rekordbox", "songstats")]

    if not has_conflict:
        return AGREED_ALL_SOURCES
    if len(analysis_sources) == 2:
        return AGREED_ANALYSES
    if analysis_sources and "rekordbox" in external_sources:
        return AGREED_ANALYSIS_REKORDBOX
    if analysis_sources and "songstats" in external_sources:
        return AGREED_ANALYSIS_SONGSTATS
    if has_conflict and analysis_sources:
        return ANALYSIS_OVERRODE_TAG
    if len(top_sources) == 1:
        return SINGLE_HIGH_CONFIDENCE

    # Multiple non-analysis sources agree (e.g., tag+rekordbox)
    return "+".join(sorted(set(top_sources)))


def format_evidence_summary(
    source_keys: dict[str, str],
    canonical_camelot: str | None,
) -> str:
    """Format human-readable evidence summary.

    source_keys: {source_system: camelot_code} for key observations.
    """
    # Order sources consistently
    order = ["tag", "songstats", "rekordbox", "analysis_librosa", "analysis_essentia", "manual"]
    short_names = {
        "tag": "tag",
        "songstats": "songstats",
        "rekordbox": "rekordbox",
        "analysis_librosa": "librosa",
        "analysis_essentia": "essentia",
        "manual": "manual",
    }

    parts = []
    for src in order:
        name = short_names.get(src, src)
        val = source_keys.get(src, "null")
        parts.append(f"{name}={val}")

    result = " | ".join(parts)
    if canonical_camelot:
        result += f" -> canonical={canonical_camelot}"
    else:
        result += " -> review"

    return result
