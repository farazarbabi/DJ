from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore
from dj_registry.store.obs_cache import ObsCache
from dj_registry.sync import tag_writer
from dj_registry.sync.tag_writer import _build_comment_tag, sync_tags
from dj_tagger.universal_cache import reset_cache
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


def test_build_comment_tag_normalizes_cached_binary_vocal_values():
    vocal_track = LogicalTrack(
        canonical_key_camelot="9A",
        canonical_bpm="126",
        tagger_energy="E4",
        tagger_vibe="HYPN",
        tagger_vocal="V",
    )
    instrumental_track = LogicalTrack(
        canonical_key_camelot="7A",
        canonical_bpm="124",
        tagger_energy="E3",
        tagger_vibe="DRK",
        tagger_vocal="NV",
    )

    assert _build_comment_tag(vocal_track) == "9A_126_E4_HYPN_VOC"
    assert _build_comment_tag(instrumental_track) == "7A_124_E3_DRK_INST"


def test_build_comment_tag_can_include_group_id_after_category():
    track = LogicalTrack(
        canonical_key_camelot="9A",
        canonical_bpm="126",
        tagger_energy="E4",
        tagger_vibe="HYPN",
        tagger_vocal="INST",
        dj_taxonomy_internal_label="Dark Tech-House Driver",
    )

    assert _build_comment_tag(track, group_id="G017") == "9A_126_E4_HYPN_INST_DRK.TECH.HOUS.DRV_G017"


def test_sync_tags_updates_file_record_and_raw_tag_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(
            track_id="T1",
            artist_canonical="Artist",
            title_canonical="Title",
            canonical_key_camelot="9A",
            canonical_bpm="126",
            tagger_energy="E4",
            tagger_vibe="HYPN",
            tagger_vocal="INST",
            dj_taxonomy_internal_label="Dark Tech-House Driver",
            primary_file_id="F1",
        )
    ])
    store.save_files([
        FileRecord(
            file_id="F1",
            track_id="T1",
            path_abs=str(tmp_path / "Track.mp3"),
            file_name="Track.mp3",
            audio_duration_sec=180.0,
            embedded_comment="old",
        )
    ])

    monkeypatch.setattr(tag_writer, "_write_full_tag", lambda path, track, tag_string=None: True)

    written, skipped, errors = sync_tags(store, dry_run=False)

    expected = "9A_126_E4_HYPN_INST_DRK.TECH.HOUS.DRV"
    assert (written, skipped, errors) == (1, 0, 0)
    refreshed_file = store.load_files()[0]
    assert refreshed_file.embedded_comment == expected
    assert refreshed_file.tag_write_status == "ok"

    cached_obs = ObsCache("cache/registry_cache.pkl").get_by_file(
        refreshed_file.path_abs,
        refreshed_file.audio_duration_sec,
        "tag",
    )
    assert cached_obs is not None
    assert cached_obs.comments == expected


def test_sync_tags_uses_group_ids_by_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    path = str(tmp_path / "Track.mp3")
    store.save_tracks([
        LogicalTrack(
            track_id="T1",
            canonical_key_camelot="9A",
            canonical_bpm="126",
            tagger_energy="E4",
            tagger_vibe="HYPN",
            tagger_vocal="INST",
            dj_taxonomy_internal_label="Dark Tech-House Driver",
            primary_file_id="F1",
        )
    ])
    store.save_files([
        FileRecord(
            file_id="F1",
            track_id="T1",
            path_abs=path,
            file_name="Track.mp3",
            audio_duration_sec=180.0,
        )
    ])

    written_tags = []
    monkeypatch.setattr(
        tag_writer,
        "_write_full_tag",
        lambda path, track, tag_string=None: written_tags.append(tag_string) or True,
    )

    written, skipped, errors = sync_tags(
        store,
        dry_run=False,
        group_ids_by_file={"Track.mp3": "G017"},
    )

    expected = "9A_126_E4_HYPN_INST_DRK.TECH.HOUS.DRV_G017"
    assert (written, skipped, errors) == (1, 0, 0)
    assert written_tags == [expected]
    assert store.load_files()[0].embedded_comment == expected
