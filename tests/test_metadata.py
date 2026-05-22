"""Tests for metadata read/write."""

from dj_tagger.metadata import read_existing_tag, write_tag


def test_mp3_roundtrip(mp3_file):
    """Write a tag to an MP3 file and read it back."""
    tag = "9A|E3|HYPN|INST"
    write_tag(mp3_file, tag, dry_run=False)
    result = read_existing_tag(mp3_file)
    assert result == tag


def test_flac_roundtrip(flac_file):
    """Write a tag to a FLAC file and read it back."""
    tag = "5A|E4|RAW|VOC"
    write_tag(flac_file, tag, dry_run=False)
    result = read_existing_tag(flac_file)
    assert result == tag


def test_dry_run_no_write(flac_file):
    """Dry run should not modify the file."""
    tag = "8A|E2|DEEP|INST"
    write_tag(flac_file, tag, dry_run=True)
    result = read_existing_tag(flac_file)
    assert result is None


def test_overwrite_tag(flac_file):
    """Writing a second tag should overwrite the first."""
    tag1 = "9A|E3|HYPN|INST"
    tag2 = "5A|E4|RAW|VOC"
    write_tag(flac_file, tag1, dry_run=False)
    write_tag(flac_file, tag2, dry_run=False)
    result = read_existing_tag(flac_file)
    assert result == tag2
