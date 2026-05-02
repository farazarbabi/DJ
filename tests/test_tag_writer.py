from dj_registry.models import LogicalTrack
from dj_registry.sync.tag_writer import _build_comment_tag
from dj_tagger.formats import parse_tag


def test_build_comment_tag_uses_internal_category_label_code_without_structure():
    track = LogicalTrack(
        canonical_key_camelot="9A",
        canonical_bpm="126",
        tagger_energy="E4",
        tagger_vibe="HYPN",
        tagger_vocal="INST",
        tagger_structure="64H",
        dj_taxonomy_internal_label="Dark Tech-House Driver",
    )

    tag = _build_comment_tag(track)

    assert tag == "9A_126_E4_HYPN_INST_DRK.TECH.HOUS.DRV"
    parsed = parse_tag(tag or "")
    assert parsed is not None
    assert parsed["category"] == "DRK.TECH.HOUS.DRV"
    assert "structure" not in parsed


def test_build_comment_tag_can_write_category_only_tag():
    track = LogicalTrack(dj_taxonomy_internal_label="Organic Chant House")

    assert _build_comment_tag(track) == "??_???_E?_??_??_ORG.CHNT.HOUS"
