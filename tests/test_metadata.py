"""Tests for metadata read/write."""

from dj_tagger.metadata import read_existing_tag, write_tag


def test_wav_roundtrip(mp3_file):
    """Write a tag to a WAV file and read it back."""
    tag = "E3 | 9A | 126 | 64H | HYPN | NV"
    write_tag(mp3_file, tag, dry_run=False)
    result = read_existing_tag(mp3_file)
    assert result == tag


def test_flac_roundtrip(flac_file):
    """Write a tag to a FLAC file and read it back."""
    tag = "E4 | 5A | 130 | 32D | RAW | V"
    write_tag(flac_file, tag, dry_run=False)
    result = read_existing_tag(flac_file)
    assert result == tag


def test_dry_run_no_write(flac_file):
    """Dry run should not modify the file."""
    tag = "E2 | 8A | 122 | 64L | DEEP | NV"
    write_tag(flac_file, tag, dry_run=True)
    result = read_existing_tag(flac_file)
    assert result is None


def test_overwrite_tag(flac_file):
    """Writing a second tag should overwrite the first."""
    tag1 = "E3 | 9A | 126 | 64H | HYPN | NV"
    tag2 = "E4 | 5A | 130 | 32D | RAW | V"
    write_tag(flac_file, tag1, dry_run=False)
    write_tag(flac_file, tag2, dry_run=False)
    result = read_existing_tag(flac_file)
    assert result == tag2


def test_tag_with_group_id(flac_file):
    """Tag with group ID should roundtrip correctly."""
    tag = "E3 | 9A | 126 | 64H | HYPN | NV | G017"
    write_tag(flac_file, tag, dry_run=False)
    result = read_existing_tag(flac_file)
    assert result == tag
