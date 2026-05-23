"""Spec §11.2 — anti-collapse regression.

Hand-built mid-BPM percussion-heavy tracks WITHOUT any afro/tribal provider
genre token must NOT be eligible for afro/tribal classification. This tests
the feature-level rule (spec §6.2), not a trained model.
"""

from __future__ import annotations

from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.taxonomy.features import (
    is_afro_tribal_family,
    is_eligible_for_afro_tribal,
)


# ── Family membership ───────────────────────────────────────────────────────


def test_afro_house_family_gated():
    assert is_afro_tribal_family("Afro House")


def test_afro_tech_family_gated():
    assert is_afro_tribal_family("Afro-Tech")


def test_tribal_house_family_gated():
    assert is_afro_tribal_family("Tribal House")


def test_organic_tribal_family_gated():
    assert is_afro_tribal_family("Organic / Tribal")


def test_afro_cinematic_family_gated():
    assert is_afro_tribal_family("Afro House / Cinematic")


def test_techno_tribal_family_NOT_gated():
    """Tribal Techno does not require explicit afro signal (spec §11.2)."""
    assert not is_afro_tribal_family("Techno / Tribal")


def test_house_family_not_gated():
    assert not is_afro_tribal_family("House")


def test_indie_dance_family_not_gated():
    assert not is_afro_tribal_family("Indie Dance")


def test_latin_tech_house_not_gated():
    """Latin Tech-House has 'tribal' mood but Latin lineage; not gated."""
    assert not is_afro_tribal_family("Latin Tech-House")


# ── Eligibility decisions ───────────────────────────────────────────────────


def _percussive_techy_track(genres_all: str = "") -> tuple[LogicalTrack, list, FileRecord]:
    """A mid-BPM track with percussion+tagger signals but only the provided genre string."""
    track = LogicalTrack(
        artist_canonical="Some Artist",
        title_canonical="Some Track",
        mix_canonical="Original Mix",
        canonical_bpm="122",
        tagger_vibe="TRIB,HYPN",
        tagger_vocal="CHANT",
        tagger_energy="E4",
        tagger_structure="ROLLING",
    )
    obs = [
        SourceObservation(
            track_id=track.track_id,
            source_system="songstats",
            genre=genres_all.split(";")[0].strip() if ";" in genres_all else genres_all,
            genres_all=genres_all,
            energy="0.75",
            danceability="0.80",
        )
    ]
    frec = FileRecord(file_name="Some Artist - Some Track.aiff", embedded_genre=genres_all)
    return track, obs, frec


def test_no_afro_signal_means_ineligible_despite_percussion():
    """The key anti-collapse case: percussion + chant + mid-BPM without afro/tribal genre."""
    track, obs, frec = _percussive_techy_track(genres_all="Tech House; Indie Dance")
    assert not is_eligible_for_afro_tribal(track, obs, frec)


def test_no_signal_at_all_ineligible():
    track, obs, frec = _percussive_techy_track(genres_all="")
    # Empty genres + no mix-name afro token → still ineligible
    assert not is_eligible_for_afro_tribal(track, obs, frec)


def test_explicit_afro_in_genres_makes_eligible():
    track, obs, frec = _percussive_techy_track(genres_all="Afro House; Tribal House")
    assert is_eligible_for_afro_tribal(track, obs, frec)


def test_explicit_tribal_token_makes_eligible():
    track, obs, frec = _percussive_techy_track(genres_all="Tribal House; House")
    assert is_eligible_for_afro_tribal(track, obs, frec)


def test_spiritual_house_token_makes_eligible():
    track, obs, frec = _percussive_techy_track(genres_all="Spiritual House; Deep House")
    assert is_eligible_for_afro_tribal(track, obs, frec)


def test_3_step_token_makes_eligible():
    track, obs, frec = _percussive_techy_track(genres_all="3-Step; Tech House")
    assert is_eligible_for_afro_tribal(track, obs, frec)


def test_afro_in_mix_name_makes_eligible():
    track = LogicalTrack(
        artist_canonical="X",
        title_canonical="Y",
        mix_canonical="Afro Mix",
        canonical_bpm="122",
    )
    obs = [SourceObservation(track_id="", source_system="songstats", genre="Tech House")]
    assert is_eligible_for_afro_tribal(track, obs, None)


def test_tribal_in_mix_name_makes_eligible():
    track = LogicalTrack(mix_canonical="Tribal Remix", canonical_bpm="122")
    assert is_eligible_for_afro_tribal(track, [], None)


def test_shamanic_in_title_makes_eligible():
    track = LogicalTrack(title_canonical="Shamanic Ritual", canonical_bpm="120")
    assert is_eligible_for_afro_tribal(track, [], None)


def test_artist_prior_makes_eligible():
    track = LogicalTrack(artist_canonical="Black Coffee", canonical_bpm="122")
    artist_priors = {"Black Coffee": [{"category_id": "tribal_afro_driver", "weight": 0.8, "tag": "afro"}]}
    assert is_eligible_for_afro_tribal(track, [], None, artist_priors=artist_priors)


def test_label_prior_makes_eligible():
    track = LogicalTrack(label_canonical="MoBlack Records", canonical_bpm="122")
    label_priors = {"MoBlack Records": [{"category_id": "afro_house_peak", "tag": "afro"}]}
    assert is_eligible_for_afro_tribal(track, [], None, label_priors=label_priors)


# ── Named regression cases ──────────────────────────────────────────────────


def test_daval_robots_NOT_eligible_for_afro_tribal():
    """Daval — Robots (Original Mix) — Songstats says Organic House, NOT afro/tribal."""
    track = LogicalTrack(
        artist_canonical="Daval",
        title_canonical="Robots",
        mix_canonical="Original Mix",
        canonical_bpm="114",
        tagger_vibe="TENS",
        tagger_vocal="VOC",
        tagger_energy="E3",
    )
    obs = [
        SourceObservation(
            track_id="T-daval",
            source_system="songstats",
            genre="Organic House",
            genres_all="Dance; House; Melodic / Progressive House; Organic House",
        )
    ]
    assert not is_eligible_for_afro_tribal(track, obs, None)


def test_nina_simone_NOT_eligible_for_afro_tribal():
    """Nina Simone — I Put a Spell on You — Jazz; nothing afro/tribal."""
    track = LogicalTrack(
        artist_canonical="Nina Simone",
        title_canonical="I Put a Spell on You",
        canonical_bpm="89",
        tagger_vibe="ATM",
        tagger_vocal="VOC",
    )
    obs = [
        SourceObservation(
            track_id="T-nina",
            source_system="songstats",
            genre="Vocal Jazz",
            genres_all="Jazz; Vocal Jazz; Vocal; Soundtrack; Dance",
        )
    ]
    assert not is_eligible_for_afro_tribal(track, obs, None)
