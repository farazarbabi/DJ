"""Find and download tracks from Spotify playlist CSV exports.

Given one or more Exportify-style Spotify playlist CSVs, this:

  1. parses the track list (name + artists + exact duration),
  2. fuzzy-matches each track against the audio files already in a library
     directory, requiring artist agreement to avoid same-title collisions,
  3. downloads anything missing via ``yt-dlp`` YouTube search, extracting to
     WAV and converting losslessly to AIFF (configurable),
  4. embeds descriptive metadata (Title/Artist/Album/Genre/Year/Label) from the
     playlist row into each download, and marks fallback filenames with source
     quality markers so tool-downloaded files can be told apart from original files,
  5. flags downloads whose duration differs sharply from Spotify's — a strong
     signal that the search returned the wrong video, and
  6. prunes any previously-downloaded marked fallback file once the user has
     added a properly-named, unmarked curated original of that track to the library.

The matcher works off files on disk, so it does not require a populated
registry. ``yt-dlp`` and ``ffmpeg`` must be on PATH for the download step.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".aiff", ".aif", ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

# Words that carry no identity signal when comparing a title to a filename.
# NOTE: "remix" and "remixed" are intentionally NOT stopwords because remixes are
# distinct tracks by different artists, not just generic versions (like "mix", "mixed",
# "original", "extended", "radio edit"). If someone requests a remix, they should get
# the remix, not the original or extended version.
STOPWORDS = {
    "the", "a", "an", "of", "and", "feat", "featuring", "ft", "with",
    "original", "mix", "mixed", "extended", "version", "radio", "edit", "rework",
    "dub", "vip", "instrumental", "club",
}

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Filename marker for tool-downloaded (YouTube-sourced, converted) files, so they
# can be told apart from higher-quality, originally-AIFF library tracks
# (e.g. "Artist - Title[U].aiff"). It is filename-only — embedded Title/Artist
# tags stay clean — and the matcher's tokenizer ignores brackets, so it does not
# affect missing-track detection.
SOURCE_MARKER = "[U]"  # YouTube fallback
SOUNDEO_WAV_MARKER = "[W]"
SOUNDEO_MP3_MARKER = "[M]"
TOOL_SOURCE_MARKERS = (SOURCE_MARKER, SOUNDEO_WAV_MARKER, SOUNDEO_MP3_MARKER)

# Trailing version descriptors that denote the *same* track as an untagged
# Spotify title and should be ignored when matching. Spotify usually omits
# "Original ...", while a lossless/AIFF source typically tags it; and an
# "Extended ..." cut, though a longer edit, is the preferred DJ version of the
# same song. A true remix (e.g. "(Henry Saiz Remix)") is a distinct track and
# is deliberately NOT covered here. Only the parenthesized/bracketed form is
# stripped, so the "Artist - Title" separator dash is never touched.
_EQUIV_VERSION_RE = re.compile(
    r"\s*[\(\[]\s*(?:original|extended)"
    r"(?:\s+(?:mix|version|edit|re-?edit|cut|remix))?\s*[\)\]]\s*$",
    re.IGNORECASE,
)




def marker_for_soundeo_format(source_format: str) -> str:
    if source_format == "wav":
        return SOUNDEO_WAV_MARKER
    if source_format == "mp3":
        return SOUNDEO_MP3_MARKER
    return ""


def strip_tool_marker(stem: str) -> tuple[str, str]:
    for marker in TOOL_SOURCE_MARKERS:
        if stem.endswith(marker):
            return stem[: -len(marker)], marker
    return stem, ""

def version_key(stem: str) -> str:
    """Normalize a filename stem so Original/Extended variants share a key.

    Lowercases, drops a trailing equivalent-version descriptor (see
    ``_EQUIV_VERSION_RE``), and collapses whitespace, so ``"Artist - Title"``,
    ``"Artist - Title (Original Mix)"``, and ``"Artist - Title (Extended
    Version)"`` all map to the same key while a remix keeps its own identity.
    The artist prefix is preserved, so same-title different-artist tracks do
    not collide.
    """
    key = _EQUIV_VERSION_RE.sub("", stem.strip())
    return re.sub(r"\s+", " ", key).strip().lower()


def _prune_text_key(text: str) -> str:
    """Accent-insensitive normalized text for conservative duplicate pruning."""
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _artist_names_key(artists: str) -> set[str]:
    names: set[str] = set()
    for part in re.split(r"\s*(?:,|;|&|\band\b)\s*", artists):
        key = _prune_text_key(part)
        if key:
            names.add(key)
    return names


def _remixer_credit_prune_key(stem: str) -> tuple[frozenset[str], str, frozenset[str]]:
    """Return artist/title/remixer-credit identity for marked-download pruning.

    Some sources disagree on whether the remixer is a primary artist:
    ``A, Remixer - Track (Remixer Remix)`` vs ``A - Track (Remixer Remix)``.
    Keep true remixes distinct by requiring the full title/version text to
    match, and only allow extra marked artists when their name appears in the
    title/version text.
    """
    artists, title = _parse_stem_artist_title(stem)
    artist_set = _artist_names_key(artists)
    title_key = _prune_text_key(title)
    title_tokens = frozenset(tokens(title, drop_stop=False))
    return frozenset(artist_set), title_key, title_tokens


def _marked_is_superseded_by_unmarked(marked_stem: str, unmarked_stem: str) -> bool:
    marked_artists, marked_title, marked_title_tokens = _remixer_credit_prune_key(marked_stem)
    unmarked_artists, unmarked_title, _ = _remixer_credit_prune_key(unmarked_stem)
    if not marked_artists or not unmarked_artists:
        return False
    if marked_title != unmarked_title:
        return False
    if not unmarked_artists.issubset(marked_artists):
        return False
    extra_marked = marked_artists - unmarked_artists
    return all(set(name.split()).issubset(marked_title_tokens) for name in extra_marked)


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def tokens(s: str, drop_stop: bool = True) -> set[str]:
    """Lowercase, de-accent, and split into a set of significant word tokens."""
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return set(toks)


def clean_track_name(name: str) -> str:
    """Convert Spotify 'Title - Remixer Remix' to 'Title (Remixer Remix)'.

    Spotify exports put the version descriptor after ' - '; the library
    convention parenthesizes it. Names without ' - ' pass through unchanged.
    """
    if " - " in name:
        head, tail = name.split(" - ", 1)
        return f"{head.strip()} ({tail.strip()})"
    return name.strip()


def sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in Windows filenames."""
    name = _ILLEGAL.sub("", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return name


# --------------------------------------------------------------------------- #
# Playlist parsing
# --------------------------------------------------------------------------- #
@dataclass
class PlaylistTrack:
    name: str
    artists: list[str] = field(default_factory=list)
    duration_sec: float | None = None
    uri: str = ""
    playlist: str = ""
    album: str = ""
    year: str = ""
    genres: list[str] = field(default_factory=list)
    label: str = ""
    added_at: str = ""

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""

    @property
    def artist_display(self) -> str:
        return ", ".join(self.artists)

    @property
    def primary_genre(self) -> str:
        return self.genres[0] if self.genres else ""

    def title_tag(self) -> str:
        """Display title for the Title tag (library-style, parenthesized remix)."""
        return clean_track_name(self.name)

    def target_basename(self, marker: str = "") -> str:
        """Library-style 'Artist - Title' base filename (no extension).

        ``marker`` is appended verbatim after the sanitized stem (e.g. ``[U]``)
        to flag tool-downloaded files; it is filename-safe and ignored by the
        matcher's tokenizer, so it does not affect matching.
        """
        title = clean_track_name(self.name)
        base = f"{self.artist_display} - {title}" if self.artists else title
        return sanitize_filename(base) + marker

    def search_query(self) -> str:
        """YouTube search string: primary artist + raw track name."""
        return f"{self.primary_artist} {self.name}".strip()


def _split_artists(field_value: str) -> list[str]:
    return [a.strip() for a in field_value.split(";") if a.strip()]


def parse_playlist_csv(path: str | os.PathLike) -> list[PlaylistTrack]:
    """Parse an Exportify-style Spotify playlist CSV."""
    tracks: list[PlaylistTrack] = []
    name = os.path.basename(str(path))
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            track_name = (row.get("Track Name") or "").strip()
            if not track_name:
                continue
            try:
                ms = int(row.get("Duration (ms)") or 0)
            except (TypeError, ValueError):
                ms = 0
            release = (row.get("Release Date") or row.get("Album Release Date") or "").strip()
            genres = [g.strip() for g in (row.get("Genres") or "").split(",") if g.strip()]
            tracks.append(
                PlaylistTrack(
                    name=track_name,
                    artists=_split_artists(row.get("Artist Name(s)") or ""),
                    duration_sec=(ms / 1000.0) if ms else None,
                    uri=(row.get("Track URI") or "").strip(),
                    playlist=name,
                    album=(row.get("Album Name") or "").strip(),
                    year=release[:4],
                    genres=genres,
                    label=(row.get("Record Label") or "").strip(),
                    added_at=(row.get("Added At") or "").strip(),
                )
            )
    return tracks


def collect_playlists(inputs: list[str]) -> list[str]:
    """Expand directories to the *.csv files they contain; keep file inputs."""
    csv_paths: list[str] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            csv_paths.extend(sorted(str(c) for c in p.glob("*.csv")))
        elif p.is_file():
            csv_paths.append(str(p))
        else:
            logger.warning("fetch-missing: input not found: %s", inp)
    return csv_paths


def load_unique_tracks(csv_paths: list[str]) -> list[PlaylistTrack]:
    """Parse all CSVs and dedupe by Spotify URI (falling back to name+artist).

    Returned **newest-added first**: sorted by the Exportify ``Added At`` column
    descending (ISO-8601 sorts lexically as chronologically). This orders the
    fetch work set so the most recently added tracks download first (and get
    quota priority), and flows through to the match reports. The sort is stable,
    so equal timestamps keep first-seen CSV order and undated rows sort last.
    """
    seen: set[str] = set()
    out: list[PlaylistTrack] = []
    for cp in csv_paths:
        for t in parse_playlist_csv(cp):
            key = t.uri or f"{t.name}\x00{t.artist_display}".lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    out.sort(key=lambda t: t.added_at, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Library matching
# --------------------------------------------------------------------------- #
@dataclass
class FileEntry:
    path: str
    all_tokens: set[str]
    title_tokens: set[str]


def scan_library(music_dir: str | os.PathLike) -> list[FileEntry]:
    """Index audio files in a directory (non-recursive) for matching."""
    index: list[FileEntry] = []
    for entry in os.scandir(music_dir):
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in AUDIO_EXTS:
            continue
        # Title region = portion after the first ' - ' (after the artist).
        title = stem.split(" - ", 1)[1] if " - " in stem else stem
        index.append(
            FileEntry(entry.path, tokens(stem), tokens(title))
        )
    return index


def best_match(track: PlaylistTrack, index: list[FileEntry]) -> tuple[str | None, float]:
    """Return (best_path, score in [0,1]) for a track against the library.

    A match needs the playlist title present in the file AND either an artist
    match or a distinctive (>=3-token) title — this rejects same-title
    different-artist collisions while still allowing artist-less filenames.
    Missing remixer tokens lower coverage, so an alternate remix of a track you
    only own the original of scores below threshold.
    """
    t_title = tokens(track.name)
    artist_tok: set[str] = set()
    for a in track.artists:
        artist_tok |= tokens(a)

    best_path: str | None = None
    best_score = 0.0
    for fe in index:
        if not t_title:
            break
        title_cov = len(t_title & fe.title_tokens) / len(t_title)
        title_cov_any = len(t_title & fe.all_tokens) / len(t_title)
        title_score = max(title_cov, 0.85 * title_cov_any)
        has_artist = bool(artist_tok) and bool(artist_tok & fe.all_tokens)
        distinctive = len(t_title) >= 3

        if title_cov >= 0.999:
            if has_artist:
                score = 0.95
            elif distinctive:
                score = 0.9
            else:
                score = 0.45  # short title, no artist agreement -> reject
        else:
            score = 0.7 * title_score
            if has_artist:
                score += 0.2
            else:
                score *= 0.5

        if score > best_score:
            best_path, best_score = fe.path, score
    return best_path, best_score


@dataclass
class MatchResult:
    track: PlaylistTrack
    score: float
    closest: str | None


def classify_tracks(
    tracks: list[PlaylistTrack],
    index: list[FileEntry],
    threshold: float = 0.62,
) -> tuple[list[MatchResult], list[MatchResult]]:
    """Split tracks into (present, missing) by best-match score vs threshold."""
    present: list[MatchResult] = []
    missing: list[MatchResult] = []
    for t in tracks:
        path, score = best_match(t, index)
        res = MatchResult(t, score, path)
        (present if score >= threshold else missing).append(res)
    return present, missing


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
DEFAULT_TOLERANCE_SEC = 3.0   # accept a video within +-this of the Spotify length
DEFAULT_MAX_ATTEMPTS = 3      # one try + two retries before giving up
# Slack on the post-download ffprobe sanity check: selection already enforced
# the tolerance against YouTube's reported duration, so this only guards
# against grossly wrong downloads, not container rounding.
_SANITY_SLACK_SEC = 5.0


def build_candidate_command(
    query: str,
    *,
    count: int = 5,
    min_duration: int = 30,
    max_duration: int = 900,
) -> list[str]:
    """yt-dlp argv that prints 'id<TAB>duration<TAB>title' for search hits.

    Uses ``--skip-download`` so only metadata is fetched; the match-filter
    drops obvious non-tracks (hour-long mixes, tiny clips) before printing.
    """
    return [
        "yt-dlp", "--no-warnings", "--skip-download",
        "--match-filter", f"duration < {max_duration} & duration > {min_duration}",
        "--print", "%(id)s\t%(duration)s\t%(title)s",
        f"ytsearch{count}:{query}",
    ]


def build_download_command(video_id: str, out_template: str) -> list[str]:
    """yt-dlp argv to extract one specific video's audio to WAV."""
    return [
        "yt-dlp", "--no-playlist", "--no-warnings",
        "-x", "--audio-format", "wav",
        "-o", out_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]


def parse_candidate_lines(text: str) -> list[tuple[str, float | None, str]]:
    """Parse 'id<TAB>duration<TAB>title' lines into (id, duration, title)."""
    out: list[tuple[str, float | None, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        vid = parts[0]
        dur: float | None = None
        if len(parts) > 1:
            try:
                dur = float(parts[1])
            except ValueError:
                dur = None
        title = parts[2] if len(parts) > 2 else ""
        out.append((vid, dur, title))
    return out


def select_candidates(
    candidates: list[tuple[str, float | None, str]],
    expected: float | None,
    tolerance: float = DEFAULT_TOLERANCE_SEC,
) -> list[str]:
    """Return candidate video ids worth trying, best first.

    When the Spotify duration is known, keep only videos within ``tolerance``
    seconds of it, ordered by closeness. When it is unknown, keep the original
    search order (duration cannot be verified).
    """
    if expected is None:
        return [c[0] for c in candidates]
    scored = [
        (abs(dur - expected), vid)
        for vid, dur, _title in candidates
        if dur is not None and abs(dur - expected) <= tolerance
    ]
    scored.sort()
    return [vid for _diff, vid in scored]


def tools_available() -> tuple[bool, bool]:
    """Return (yt_dlp_present, ffmpeg_present)."""
    return shutil.which("yt-dlp") is not None, shutil.which("ffmpeg") is not None


def probe_duration(path: str | os.PathLike) -> float | None:
    """Return the audio duration in seconds via ffprobe, or None."""
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nokey=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        return float(out)
    except (ValueError, OSError):
        return None


def duration_mismatch(
    expected: float | None,
    actual: float | None,
    *,
    abs_sec: float = 20.0,
    pct: float = 12.0,
) -> str | None:
    """Return a human reason string if durations differ enough to be suspect.

    Requires BOTH an absolute gap over ``abs_sec`` and a relative gap over
    ``pct`` percent, so short tracks aren't flagged for small offsets.
    """
    if expected is None or actual is None:
        return None
    diff = actual - expected
    if abs(diff) > abs_sec and (abs(diff) / expected * 100.0) > pct:
        return f"{actual:.0f}s vs Spotify {expected:.0f}s ({diff:+.0f}s)"
    return None


def _audio_to_aiff(src_path: str, aiff_path: str) -> bool:
    """Convert an audio file to AIFF PCM. Returns True on success."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
         "-c:a", "pcm_s16be", aiff_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return os.path.exists(aiff_path) and os.path.getsize(aiff_path) > 0


def _wav_to_aiff(wav_path: str, aiff_path: str) -> bool:
    """Losslessly convert WAV -> AIFF (PCM). Returns True on success."""
    return _audio_to_aiff(wav_path, aiff_path)


@dataclass
class DownloadOutcome:
    # "ok" | "ok_wav" | "skip" | "no_match" | "fail" | "quota_skip"
    status: str
    outfile: str = ""
    detail: str = ""
    source: str = "youtube"  # "soundeo" | "youtube"
    # True when Soundeo *had* a matching result for this track (even if the
    # download then failed / hit quota). Prevents caching an on-Soundeo track as
    # "not found" just because YouTube also missed it.
    soundeo_listed: bool = False


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _finalize_wav(wav_path: str, aiff_path: str, audio_format: str) -> DownloadOutcome:
    """Convert the downloaded WAV to the requested format and report outcome."""
    if audio_format == "wav":
        return DownloadOutcome("ok", wav_path)
    if _wav_to_aiff(wav_path, aiff_path):
        os.remove(wav_path)
        return DownloadOutcome("ok", aiff_path)
    return DownloadOutcome("ok_wav", wav_path, "aiff conversion failed; kept wav")


def _embed_metadata(path: str, track: PlaylistTrack) -> None:
    """Embed descriptive tags from the playlist row so Rekordbox shows real
    metadata instead of the filename. Failures are logged, never fatal — a
    tagging hiccup must not discard a good download.
    """
    try:
        from dj_tagger.metadata import write_track_metadata

        write_track_metadata(
            path,
            title=track.title_tag(),
            artist=track.artist_display,
            album=track.album,
            album_artist=track.primary_artist,
            genre=track.primary_genre,
            year=track.year,
            label=track.label,
        )
    except Exception:
        logger.warning("fetch-missing: could not embed metadata into %s",
                       os.path.basename(path), exc_info=True)


def download_track(
    track: PlaylistTrack,
    dest_dir: str,
    *,
    audio_format: str = "aiff",
    tolerance: float = DEFAULT_TOLERANCE_SEC,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_duration: int = 30,
    max_duration: int = 900,
) -> DownloadOutcome:
    """Search YouTube and download one track, verifying its duration.

    Only YouTube results within ``tolerance`` seconds of the Spotify track
    length are considered; the closest is tried first and up to
    ``max_attempts`` candidates are attempted before giving up with
    ``no_match``. An existing file at the target path is kept if its duration
    is already within tolerance, otherwise it is re-downloaded. When the
    Spotify duration is unknown the top search hit is taken unverified.
    """
    base = track.target_basename(SOURCE_MARKER)
    aiff_path = os.path.join(dest_dir, base + ".aiff")
    wav_path = os.path.join(dest_dir, base + ".wav")
    final_path = aiff_path if audio_format == "aiff" else wav_path
    expected = track.duration_sec

    # Skip or re-verify an existing (tool-downloaded) file. If it is out of
    # bounds, keep it until a verified replacement is ready; a failed lookup
    # must not delete the user's only copy.
    replacing_existing = False
    if os.path.exists(final_path):
        if expected is None:
            return DownloadOutcome("skip", final_path,
                                   "already exists (no Spotify duration to verify)")
        actual = probe_duration(final_path)
        if actual is not None and abs(actual - expected) <= tolerance:
            return DownloadOutcome("skip", final_path,
                                   f"already exists, duration ok ({actual:.0f}s)")
        # Out of bounds — drop it and try to fetch a correct version.
        replacing_existing = True
        logger.info("fetch-missing: re-downloading out-of-bounds file %s (%s vs %ss)",
                    base, f"{actual:.0f}" if actual else "?", f"{expected:.0f}")

    # List candidates and keep those within tolerance, closest first.
    list_cmd = build_candidate_command(
        track.search_query(),
        count=max(max_attempts, 5),
        min_duration=min_duration, max_duration=max_duration,
    )
    proc = subprocess.run(list_cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    candidates = parse_candidate_lines(proc.stdout or "")
    if not candidates:
        lines = (proc.stderr or "").strip().splitlines()
        return DownloadOutcome("fail", "", lines[-1] if lines else "no search results")

    order = select_candidates(candidates, expected, tolerance)
    if expected is not None and not order:
        return DownloadOutcome(
            "no_match", "",
            f"no result within ±{tolerance:.0f}s of {expected:.0f}s "
            f"among {len(candidates)} candidate(s)",
        )
    if expected is None:
        order = order[:1]  # unverifiable: take the top hit only
    attempts = order[:max_attempts]

    download_base = base + ".__candidate__" if replacing_existing and audio_format == "wav" else base
    download_wav_path = os.path.join(dest_dir, download_base + ".wav")
    out_template = os.path.join(dest_dir, download_base + ".%(ext)s")
    last_detail = ""
    last_was_download_error = False
    for vid in attempts:
        _remove_quiet(download_wav_path)
        dl = subprocess.run(build_download_command(vid, out_template),
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        if not os.path.exists(download_wav_path):
            lines = (dl.stderr or dl.stdout or "").strip().splitlines()
            last_detail = lines[-1] if lines else "download produced no file"
            last_was_download_error = True  # transient (e.g. HTTP 403), not "absent"
            continue
        if expected is not None:
            actual = probe_duration(download_wav_path)
            if actual is None or abs(actual - expected) > tolerance + _SANITY_SLACK_SEC:
                last_detail = (f"got {actual:.0f}s vs {expected:.0f}s"
                               if actual is not None else "could not probe download")
                last_was_download_error = False  # a real (wrong-duration) result
                _remove_quiet(download_wav_path)
                continue
        if audio_format == "wav" and download_wav_path != wav_path:
            os.replace(download_wav_path, wav_path)
            outcome = DownloadOutcome("ok", wav_path)
        else:
            outcome = _finalize_wav(download_wav_path, aiff_path, audio_format)
        if outcome.status in ("ok", "ok_wav") and outcome.outfile:
            _embed_metadata(outcome.outfile, track)
        return outcome

    # A download error (403/network) is transient -> "fail" (retried next run,
    # not cached as not-found). Only a genuine duration miss is "no_match".
    if last_was_download_error:
        return DownloadOutcome("fail", "", last_detail or "download failed")
    return DownloadOutcome(
        "no_match", "",
        last_detail or f"no match within ±{tolerance:.0f}s after {len(attempts)} attempt(s)",
    )


# --------------------------------------------------------------------------- #
# Source routing: Soundeo (AIFF > WAV > MP3) -> YouTube fallback
# --------------------------------------------------------------------------- #
class _QuotaState:
    """Tracks whether Soundeo's daily download quota has been hit this run."""

    def __init__(self) -> None:
        self.exhausted = False


def acquire_track(
    track: PlaylistTrack,
    dest_dir: str,
    *,
    soundeo=None,
    quota: _QuotaState | None = None,
    audio_format: str = "aiff",
    tolerance: float = DEFAULT_TOLERANCE_SEC,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_duration: int = 30,
    max_duration: int = 900,
) -> DownloadOutcome:
    """Acquire one track, preferring Soundeo sources over YouTube.

    Per-track routing (search never costs quota, only a download does):

    * on Soundeo + quota available -> download unmarked, preferring AIFF > WAV > MP3;
    * on Soundeo + quota exhausted  -> ``quota_skip`` (retry after midnight CET),
      *not* downloaded from YouTube;
    * not on Soundeo (or Soundeo errored/disabled) -> fall back to
      :func:`download_track` (YouTube, ``[U]``-marked).

    A successful Soundeo AIFF lands at the unmarked ``Artist - Title`` path;
    Soundeo WAV/MP3 land as converted ``[W]``/``[M]`` AIFFs so they can be
    upgraded later when the real AIFF appears.
    """
    soundeo_listed = False
    if soundeo is not None:
        from .soundeo import SoundeoError, SoundeoQuotaExceeded, preferred_download_format

        try:
            results = soundeo.search(track)
            pick = soundeo.pick(track, results) if results else None
            if results and not pick:
                logger.debug("soundeo: found %d result(s) but no priority match for %s; trying YouTube",
                           len(results), track.name)
        except SoundeoError as exc:
            logger.warning("soundeo: search error for %s — %s; trying YouTube",
                           track.name, exc)
            pick = None

        if pick is not None:
            soundeo_listed = True  # on Soundeo — never cache as "not found"
            if quota is not None and quota.exhausted:
                return DownloadOutcome(
                    "quota_skip", "", "on Soundeo; daily quota exhausted", "soundeo",
                    soundeo_listed=True)
            # Unmarked path: treat a Soundeo original like a curated original,
            # and keep Soundeo's own "Artist - Title" naming (which carries the
            # real cut, e.g. "(Original Mix)") rather than the Spotify title,
            # which may name a different cut ("- Radio Edit"). Only YouTube
            # fallbacks use the Spotify name (with the [U] marker). The
            # stopword-tolerant matcher still sees this as the same track.
            source_format = preferred_download_format(pick, audio_format) or audio_format
            stem = sanitize_filename(pick.label) + marker_for_soundeo_format(source_format)
            dest = os.path.join(dest_dir, stem + "." + audio_format)
            source_path = dest if source_format == audio_format else os.path.join(
                dest_dir, stem + ".__soundeo__." + source_format
            )
            try:
                try:
                    soundeo.download(pick, source_path, audio_format=source_format)
                except TypeError:
                    soundeo.download(pick, source_path)
                if source_path != dest:
                    if audio_format == "aiff":
                        if not _audio_to_aiff(source_path, dest):
                            raise SoundeoError(f"could not convert Soundeo {source_format} to AIFF")
                        _remove_quiet(source_path)
                    else:
                        os.replace(source_path, dest)
                return DownloadOutcome("ok", dest, f"soundeo:{pick.id}:{source_format}", "soundeo",
                                       soundeo_listed=True)
            except SoundeoQuotaExceeded as exc:
                if quota is not None:
                    quota.exhausted = True
                return DownloadOutcome("quota_skip", "", str(exc), "soundeo",
                                       soundeo_listed=True)
            except SoundeoError as exc:
                logger.warning("soundeo: download error for %s - %s; trying YouTube",
                               track.name, exc)
                _remove_quiet(source_path)
                # fall through to YouTube

    outcome = download_track(
        track, dest_dir, audio_format=audio_format, tolerance=tolerance,
        max_attempts=max_attempts, min_duration=min_duration, max_duration=max_duration,
    )
    outcome.soundeo_listed = soundeo_listed
    return outcome


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _write_match_report(path: str, results: list[MatchResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["playlist", "name", "artists", "score", "closest", "query"])
        for r in results:
            w.writerow([
                r.track.playlist, r.track.name, r.track.artist_display,
                f"{r.score:.3f}",
                os.path.basename(r.closest) if r.closest else "",
                r.track.search_query(),
            ])


# --------------------------------------------------------------------------- #
# Representative per-playlist M3U8s
# --------------------------------------------------------------------------- #
def _playlist_stem(playlist_filename: str) -> str:
    """CSV basename -> sanitized playlist name (drops the .csv extension)."""
    stem = os.path.splitext(playlist_filename)[0]
    return sanitize_filename(stem) or "playlist"


def generate_spotify_playlists(
    csv_paths: list[str],
    library: str,
    *,
    threshold: float = 0.62,
    playlists_dir: str | None = None,
) -> dict[str, int]:
    """Write one Rekordbox-ready ``.m3u8`` per Spotify CSV, sorted by BPM+key.

    Each playlist mirrors its CSV: the library tracks (already-present, an
    Original/Extended variant, or just downloaded as ``[U]``) that resolve to
    that CSV's tracks. Tracks are sorted by BPM (ascending) with key as
    tiebreaker (Camelot wheel order). Tracks with no library match are skipped.
    Unlike :func:`load_unique_tracks`, this parses each CSV separately so
    per-playlist membership is preserved.

    Re-scans ``library`` so files downloaded earlier in this run are included.
    Returns counts: ``playlists_written``, ``tracks_added``, ``tracks_skipped``.
    """
    from dj_registry.adapters.tag_extractor import extract_tags
    from .playlist_sorting import bpm_sort_key, camelot_sort_key

    out_dir = Path(playlists_dir or os.path.join(library, "outputs", "playlists")) / "spotify"
    index = scan_library(library)

    written = added = skipped = 0
    for csv_path in csv_paths:
        playlist_tracks = parse_playlist_csv(csv_path)
        if not playlist_tracks:
            continue

        # Match playlist tracks to library files, extracting BPM+key for sorting
        resolved_entries: list[tuple[str, str, float, int]] = []  # (label, abs_path, bpm, key_pos)
        seen: set[str] = set()
        duplicates = 0

        for t in playlist_tracks:
            path, score = best_match(t, index)
            if not (path and score >= threshold):
                continue
            abs_path = os.path.abspath(path)
            norm_key = os.path.normcase(abs_path)
            if norm_key in seen:
                duplicates += 1
                continue
            seen.add(norm_key)

            # Extract BPM and key from file tags for sorting
            tags = extract_tags(path)
            bpm = _parse_bpm(tags.get("bpm", ""))
            key = tags.get("key_camelot", "")

            label = Path(path).stem
            resolved_entries.append((label, abs_path, bpm, camelot_sort_key(key)))

        # Sort by BPM (ascending), then by Camelot key position
        resolved_entries.sort(key=lambda e: (bpm_sort_key(e[2]), e[3]))

        lines = ["#EXTM3U"]
        for label, abs_path, _, _ in resolved_entries:
            lines.append(f"#EXTINF:-1,{label}")
            lines.append(abs_path)

        name = _playlist_stem(playlist_tracks[0].playlist or os.path.basename(csv_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        # utf-8-sig: Rekordbox requires a UTF-8 BOM on .m3u8 files, else entries
        # with non-ASCII path characters fail to match and the playlist imports
        # empty.
        (out_dir / f"{name}.m3u8").write_text("\n".join(lines), encoding="utf-8-sig")
        missed = len(playlist_tracks) - len(resolved_entries) - duplicates
        added += len(resolved_entries)
        skipped += missed
        written += 1
        notes = []
        if missed:
            notes.append(f"{missed} skipped")
        if duplicates:
            notes.append(f"{duplicates} dup")
        suffix = f" ({', '.join(notes)})" if notes else ""
        logger.info(
            "fetch-missing: playlist %s.m3u8 -> %d/%d track(s) resolved%s",
            name, len(resolved_entries), len(playlist_tracks), suffix,
        )

    if written:
        logger.info(
            "fetch-missing: wrote %d representative playlist(s) to %s",
            written, out_dir,
        )
    return {"playlists_written": written, "tracks_added": added, "tracks_skipped": skipped}


def _parse_bpm(bpm_str: str) -> float | None:
    """Parse BPM string to float, returning None if invalid."""
    if not bpm_str or not isinstance(bpm_str, str):
        return None
    try:
        bpm = float(bpm_str)
        if math.isfinite(bpm) and bpm > 0:
            return bpm
    except (ValueError, TypeError):
        pass
    return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
# Conventional location for a library's Spotify playlist CSV exports, used as
# the default when no explicit playlists path is given on the command line.
DEFAULT_PLAYLISTS_SUBDIR = "spotify-playlists"


def resolve_library_dir(path: str) -> str:
    """The directory to match against and download into.

    Accepts a directory (used as-is) or a file (its parent is used), so it
    works whether ``dj run`` is pointed at a library folder or a single track.
    """
    return path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))


def default_playlists_dir(library: str) -> str:
    """Conventional playlists dir for a library: ``<library>/spotify-playlists``."""
    return os.path.join(library, DEFAULT_PLAYLISTS_SUBDIR)


def tool_file_for(track: PlaylistTrack, library: str) -> str | None:
    """Return an existing marked tool-download for ``track``.

    Marked downloads are YouTube ``[U]`` plus non-AIFF Soundeo ``[W]``/``[M]``
    files. Unmarked files are treated as curated originals or real Soundeo AIFFs
    and are not part of the recheck/upgrade queue.
    """
    for marker in TOOL_SOURCE_MARKERS:
        base = track.target_basename(marker)
        for ext in (".aiff", ".wav"):
            cand = os.path.join(library, base + ext)
            if os.path.exists(cand):
                return cand
    return None


def unmarked_version_keys(library: str) -> set[str]:
    """Version keys (see :func:`version_key`) of every *unmarked* audio file.

    These represent the user's curated originals — including Original/Extended
    variants of an untagged Spotify title — and are what both the download-skip
    and the prune use to decide a track is already present.
    """
    keys: set[str] = set()
    for entry in os.scandir(library):
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in AUDIO_EXTS:
            continue
        if not strip_tool_marker(stem)[1]:
            keys.add(version_key(stem))
    return keys


def prune_superseded_downloads(library: str, *, dry_run: bool = False) -> list[str]:
    """Delete marked tool downloads whose curated original now exists.

    Once the user adds a properly-named, unmarked original for a track
    previously fetched with ``[U]``, ``[W]``, or ``[M]``, the marked copy is a
    redundant, lower-quality duplicate. The original counts whether it is named
    exactly ``Artist - Title`` or carries an equivalent-version suffix the AIFF
    source adds (``(Original Mix)``, ``(Extended Mix)``, …) — matched via
    :func:`version_key`, in any audio format. Returns the paths removed — or,
    under ``dry_run``, the paths that would be removed.
    """
    unmarked_keys: set[str] = set()
    unmarked_stems: list[str] = []
    marked: list[tuple[str, str, str]] = []  # (path, clean stem, version-key)
    for entry in os.scandir(library):
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in AUDIO_EXTS:
            continue
        clean_stem, marker = strip_tool_marker(stem)
        if marker:
            marked.append((entry.path, clean_stem, version_key(clean_stem)))
        else:
            unmarked_keys.add(version_key(stem))
            unmarked_stems.append(stem)

    removed: list[str] = []
    for path, stem, key in marked:
        superseded = key in unmarked_keys or any(
            _marked_is_superseded_by_unmarked(stem, original)
            for original in unmarked_stems
        )
        if superseded:
            removed.append(path)
            if not dry_run:
                _remove_quiet(path)
    return removed


_NOT_FOUND_CACHE_NAME = "not_found_cache.json"


def _track_identity(track: PlaylistTrack) -> str:
    """Stable per-track key for the not-found cache (URI, else name+artist)."""
    return track.uri or f"{track.name}\x00{track.artist_display}".lower()


def _track_text_identity(track: PlaylistTrack) -> str:
    """Artist/title cache key used when Spotify URI changes between exports."""
    return f"{track.name}\x00{track.artist_display}".lower()


def _not_found_cache_keys(track: PlaylistTrack) -> set[str]:
    keys = {_track_text_identity(track)}
    if track.uri:
        keys.add(track.uri)
    return keys


def _cache_entry_matches_track(entry: object, track: PlaylistTrack) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        f"{entry.get('name', '')}\x00{entry.get('artists', '')}".lower()
        == _track_text_identity(track)
    )


def _is_cached_not_found(cache: dict, track: PlaylistTrack) -> bool:
    keys = _not_found_cache_keys(track)
    if any(key in cache for key in keys):
        return True
    return any(_cache_entry_matches_track(entry, track) for entry in cache.values())


def _forget_cached_not_found(cache: dict, track: PlaylistTrack) -> None:
    for key in _not_found_cache_keys(track):
        cache.pop(key, None)
    for key, entry in list(cache.items()):
        if _cache_entry_matches_track(entry, track):
            cache.pop(key, None)


def _remember_cached_not_found(cache: dict, track: PlaylistTrack, detail: str) -> None:
    cache[_track_identity(track)] = {
        "artists": track.artist_display,
        "name": track.name,
        "detail": detail,
    }


def load_not_found_cache(report_dir: str) -> dict:
    """Load the set of tracks previously not found on Soundeo *or* YouTube."""
    path = os.path.join(report_dir, _NOT_FOUND_CACHE_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_not_found_cache(report_dir: str, cache: dict) -> None:
    path = os.path.join(report_dir, _NOT_FOUND_CACHE_NAME)
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except OSError:
        logger.warning("fetch-missing: could not write %s", path)


def forget_not_found(
    library: str, queries: list[str], *, dry_run: bool = False,
    report_dir: str | None = None,
) -> list[str]:
    """Drop entries from the not-found cache so they're re-searched next run.

    Each query is matched case-insensitively as a substring of a cached entry's
    ``Artist - Title`` label (so ``"Malevolence"`` or the full label both work),
    or the literal ``all`` clears every entry. Returns the labels removed.
    """
    report_dir = report_dir or os.path.join(library, "outputs", "fetch")
    cache = load_not_found_cache(report_dir)
    wants = [q.strip().lower() for q in queries if q.strip()]
    clear_all = "all" in wants
    removed: list[str] = []
    for key in list(cache.keys()):
        v = cache[key]
        label = f"{v.get('artists', '')} - {v.get('name', '')}".strip(" -")
        if clear_all or any(q in label.lower() for q in wants):
            removed.append(label)
            if not dry_run:
                del cache[key]
    if removed and not dry_run:
        save_not_found_cache(report_dir, cache)
    return removed


# --------------------------------------------------------------------------- #
# Persistent Soundeo-tags log
# --------------------------------------------------------------------------- #
# Records which library files came from Soundeo and are stored with their
# genuine release tags (never overwritten with Spotify-row values). Keyed by the
# file's normcased absolute path. Its purpose is twofold: it's the durable
# record of Soundeo provenance (download_log.csv is overwritten every run), and
# it lets the tag-repair pass skip files already carrying native Soundeo tags.
_SOUNDEO_TAGS_LOG_NAME = "soundeo_tags.json"


def _soundeo_log_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def load_soundeo_tags_log(report_dir: str) -> dict:
    path = os.path.join(report_dir, _SOUNDEO_TAGS_LOG_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_soundeo_tags_log(report_dir: str, log: dict) -> None:
    path = os.path.join(report_dir, _SOUNDEO_TAGS_LOG_NAME)
    try:
        os.makedirs(report_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except OSError:
        logger.warning("fetch-missing: could not write %s", path)


def _record_soundeo_tags(log: dict, path: str, soundeo_id: str) -> None:
    """Read a Soundeo file's native tags off disk and record them in ``log``."""
    tags = {}
    try:
        from dj_tagger.metadata import read_descriptive_tags
        tags = read_descriptive_tags(path)
    except Exception:
        logger.debug("could not read Soundeo tags from %s", path, exc_info=True)
    log[_soundeo_log_key(path)] = {"soundeo_id": soundeo_id, "basename": os.path.basename(path),
                                   "tags": tags}


def _parse_stem_artist_title(stem: str) -> tuple[str, str]:
    """'Artist - Title (Mix)' -> ('Artist', 'Title (Mix)'); no dash -> ('', stem)."""
    if " - " in stem:
        head, tail = stem.split(" - ", 1)
        return head.strip(), tail.strip()
    return "", stem.strip()


def refix_soundeo_tags(
    library: str, *, dry_run: bool = False, report_dir: str | None = None,
    audio_format: str = "aiff", limit: int | None = None,
) -> dict:
    """Restore genuine Soundeo tags on library files that came from Soundeo.

    Older downloads had Spotify-row tags written over their native ones (and the
    Soundeo AIFF is now stored *as is*). This pass identifies Soundeo-sourced
    files with a **quota-free** signal — Soundeo flags a search result the
    account already owns as ``downloaded`` — and re-downloads the matching cut to
    recover its native tags. Selection matches the on-disk file's **duration**,
    so the same cut is restored. Curated originals (never downloaded from
    Soundeo) don't carry the flag and are left untouched. Files already in the
    Soundeo-tags log are skipped.

    An AIFF file is overwritten in place (same name + same cut => same duration
    => same analysis cache key => no re-analysis). A non-AIFF owned file (e.g. a
    curated ``.mp3``) is **upgraded** — the AIFF is written to the ``.aiff``
    sibling and the inferior original removed — since AIFF beats any lossy
    format; that changes the cache key, but the audio genuinely changed format,
    so re-analysis is warranted.

    Returns a summary dict: ``candidates/fixed/upgraded/not_owned/errors``.
    """
    if not os.path.isdir(library):
        raise FileNotFoundError(f"library dir not found: {library}")
    report_dir = report_dir or os.path.join(library, "outputs", "fetch")
    os.makedirs(report_dir, exist_ok=True)
    log = load_soundeo_tags_log(report_dir)

    # Candidate = any unmarked audio file (Soundeo files and curated originals
    # both look like this; [U] YouTube files are excluded — they keep Spotify
    # tags). A non-AIFF owned file (e.g. a curated .mp3) is *upgraded* to the
    # genuine Soundeo AIFF, since AIFF is superior to any lossy format.
    fmt_exts = {".aiff", ".aif"} if audio_format == "aiff" else {"." + audio_format}
    candidates: list[os.DirEntry] = []
    for entry in os.scandir(library):
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in AUDIO_EXTS or strip_tool_marker(stem)[1]:
            continue
        if _soundeo_log_key(entry.path) in log:
            continue  # already carries native Soundeo tags
        candidates.append(entry)

    summary = {"candidates": len(candidates), "fixed": 0, "upgraded": 0,
               "not_owned": 0, "errors": 0, "report_dir": report_dir}
    logger.info("refix-soundeo-tags: %d unmarked candidate file(s) (not yet logged)",
                len(candidates))
    if not candidates:
        return summary

    from .soundeo import SoundeoAuthError, SoundeoClient, SoundeoError

    soundeo = SoundeoClient.from_env(audio_format=audio_format)
    if soundeo is None:
        raise RuntimeError("Soundeo credentials not set (SOUNDEO_USER/SOUNDEO_PASS)")
    try:
        soundeo.login()
    except SoundeoAuthError as exc:
        soundeo.close()
        raise RuntimeError(f"Soundeo login failed: {exc}") from exc

    # An AIFF file is overwritten **in place** (same name + same cut => same
    # duration => same tagger cache key => no re-analysis). A non-AIFF owned
    # file is upgraded: the AIFF is written to the ``.aiff`` sibling path and the
    # inferior original removed (this one does change the cache key, but the
    # audio genuinely changed format, so re-analysis is warranted and the set is
    # tiny). ``summary["upgraded"]`` counts the format upgrades.
    processed = 0
    try:
        for entry in candidates:
            if limit is not None and processed >= limit:
                break
            processed += 1
            stem, ext = os.path.splitext(entry.name)
            artist, title = _parse_stem_artist_title(stem)
            track = PlaylistTrack(name=title, artists=[artist] if artist else [])
            on_disk = probe_duration(entry.path)
            try:
                results = soundeo.search(track)
                owned = soundeo.pick_owned(track, results, target_duration=on_disk)
            except SoundeoError as exc:
                logger.warning("refix: search error for %s — %s", entry.name, exc)
                summary["errors"] += 1
                continue
            if owned is None:
                summary["not_owned"] += 1
                continue
            is_upgrade = ext.lower() not in fmt_exts
            dest = (os.path.join(library, stem + "." + audio_format)
                    if is_upgrade else entry.path)
            if dry_run:
                verb = "WOULD upgrade" if is_upgrade else "WOULD fix"
                logger.info("refix: %s %s  (soundeo:%s '%s')",
                            verb, entry.name, owned.id, owned.label)
                summary["fixed"] += 1
                summary["upgraded"] += int(is_upgrade)
                continue
            try:
                soundeo.download(owned, dest, assume_free=True)
            except SoundeoError as exc:
                logger.warning("refix: re-download failed for %s — %s", entry.name, exc)
                summary["errors"] += 1
                continue
            if is_upgrade and os.path.normcase(dest) != os.path.normcase(entry.path):
                _remove_quiet(entry.path)  # drop the inferior lossy original
            _record_soundeo_tags(log, dest, owned.id)
            save_soundeo_tags_log(report_dir, log)  # persist incrementally
            summary["fixed"] += 1
            summary["upgraded"] += int(is_upgrade)
            logger.info("refix: %s %s  (soundeo:%s)",
                        "upgraded" if is_upgrade else "fixed", entry.name, owned.id)
    finally:
        soundeo.close()
        if not dry_run:
            save_soundeo_tags_log(report_dir, log)

    logger.info(
        "refix-soundeo-tags: done — fixed=%d (upgraded=%d) not_owned=%d errors=%d "
        "(of %d candidate(s))", summary["fixed"], summary["upgraded"],
        summary["not_owned"], summary["errors"], len(candidates),
    )
    return summary


def fetch_missing(
    playlists: list[str],
    library: str,
    *,
    audio_format: str = "aiff",
    threshold: float = 0.62,
    tolerance: float = DEFAULT_TOLERANCE_SEC,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_duration: int = 30,
    max_duration: int = 900,
    dry_run: bool = False,
    report_dir: str | None = None,
    use_soundeo: bool = True,
    force_lookup: bool = False,
    check_marked_upgrades: bool = False,
) -> dict:
    """Match playlist CSVs against ``library`` and download what's missing.

    When Soundeo credentials are present (``SOUNDEO_USER``/``SOUNDEO_PASS`` in
    ``.env``) and ``use_soundeo`` is True, each track is sourced from Soundeo's
    source first (AIFF > WAV > MP3) and from YouTube only as a fallback; pass
    ``use_soundeo=False`` to force YouTube-only.

    Downloads are duration-verified: only YouTube results within ``tolerance``
    seconds of the Spotify track are accepted, retrying up to ``max_attempts``
    times before reporting the track as unmatched. Previously tool-downloaded
    files are only rechecked for Soundeo upgrades/duration when
    ``check_marked_upgrades`` is enabled.

    Returns a summary dict with counts, the missing :class:`MatchResult` list,
    a list of unmatched ``(track, detail)`` tuples, and the report directory.
    Raises ``FileNotFoundError`` if the library is missing and ``RuntimeError``
    if required external tools are unavailable.
    """
    if not os.path.isdir(library):
        raise FileNotFoundError(f"library dir not found: {library}")

    # Drop any marked fallback download the user has since replaced with a curated original.
    pruned = prune_superseded_downloads(library, dry_run=dry_run)
    if pruned:
        verb = "would remove" if dry_run else "removed"
        logger.info(
            "fetch-missing: %s %d superseded marked download(s) replaced by curated originals",
            verb, len(pruned),
        )
        for p in pruned:
            logger.info("  %s %s", verb, os.path.basename(p))

    csv_paths = collect_playlists(playlists)
    if not csv_paths:
        logger.warning("fetch-missing: no playlist CSVs found in %s", playlists)

    tracks = load_unique_tracks(csv_paths)
    index = scan_library(library)
    present, missing = classify_tracks(tracks, index, threshold=threshold)

    # An untagged Spotify title is satisfied by a curated Original/Extended
    # variant on disk (the AIFF source tags "Original", and an "Extended" cut is
    # the preferred DJ version). Treat any such variant as present so we never
    # download a marked fallback copy of a track the user already owns in a preferred form.
    lib_keys = unmarked_version_keys(library)
    variant_present = [r for r in missing if version_key(r.track.target_basename()) in lib_keys]
    if variant_present:
        seen = {id(r) for r in variant_present}
        missing = [r for r in missing if id(r) not in seen]
        present.extend(variant_present)
        logger.info(
            "fetch-missing: %d track(s) present via Original/Extended variant — not downloading",
            len(variant_present),
        )

    report_dir = report_dir or os.path.join(library, "outputs", "fetch")
    os.makedirs(report_dir, exist_ok=True)
    _write_match_report(os.path.join(report_dir, "matched_report.csv"), present)
    _write_match_report(os.path.join(report_dir, "missing_report.csv"), missing)

    # Skip tracks already known to be on neither Soundeo nor YouTube, so repeat
    # runs don't re-search them every time. --force-lookup retries them. `missing`
    # itself (and missing_report.csv) stays complete; only the work set is trimmed.
    not_found = load_not_found_cache(report_dir)
    soundeo_log = load_soundeo_tags_log(report_dir)  # provenance of as-is Soundeo files
    for r in present:  # a now-present track is no longer "not found"
        _forget_cached_not_found(not_found, r.track)
    to_download = [r for r in missing
                   if force_lookup or not _is_cached_not_found(not_found, r.track)]
    cached_skipped = len(missing) - len(to_download)
    if cached_skipped:
        logger.info(
            "fetch-missing: skipping %d track(s) previously not found on Soundeo "
            "or YouTube (use --force-lookup to retry)", cached_skipped,
        )

    # Work set: genuinely-missing tracks (download) plus, when explicitly
    # requested, already tool-downloaded tracks (check for Soundeo upgrade and
    # re-verify duration).
    work = list(to_download)
    marked_present = [r for r in present if tool_file_for(r.track, library)]
    reverify = marked_present if check_marked_upgrades else []
    work.extend(reverify)
    fully_present = len(present) - len(marked_present)

    logger.info(
        "fetch-missing: fetch queue: %d to fetch, %d existing marked "
        "download(s)%s, %d cached not-found skipped, %d already present as "
        "originals/curated",
        len(to_download), len(marked_present),
        " queued for Soundeo upgrade/duration check"
        if check_marked_upgrades else
        " not checked (use --check-marked-upgrades)",
        cached_skipped, fully_present,
    )

    summary = {
        "total": len(tracks), "playlists": len(csv_paths),
        "present": len(present), "missing": len(missing), "reverify": len(reverify),
        "pruned": len(pruned),
        "downloaded": 0, "skipped": 0, "failed": 0, "unmatched": 0,
        "soundeo": 0, "youtube": 0, "quota_skipped": 0, "deferred": 0,
        "cached_skipped": cached_skipped,
        "missing_results": missing, "unmatched_results": [], "report_dir": report_dir,
    }
    logger.info(
        "fetch-missing: %d unique track(s) across %d playlist(s) — %d present, %d missing",
        len(tracks), len(csv_paths), len(present), len(missing),
    )
    logger.info("fetch-missing: reports written to %s", report_dir)

    if dry_run or not work:
        if dry_run:
            for r in missing:
                logger.info("  [%.2f] %s - %s", r.score, r.track.artist_display, r.track.name)
            logger.info("fetch-missing: playlists reflect the current library (no downloads in dry-run)")
        summary.update(generate_spotify_playlists(csv_paths, library, threshold=threshold))
        return summary

    # Soundeo (primary source) — built once and reused; absent creds leaves it
    # disabled (YouTube-only). A login *failure* aborts the run rather than
    # silently downloading everything from YouTube — the user must fix the
    # credentials or pass --no-soundeo to opt into YouTube explicitly. quota
    # tracks the daily limit so we stop (not YouTube) once it is hit.
    soundeo = None
    quota = _QuotaState()
    if use_soundeo and audio_format == "aiff":
        from .soundeo import SoundeoAuthError, SoundeoClient

        soundeo = SoundeoClient.from_env(audio_format=audio_format)
        if soundeo is not None:
            try:
                soundeo.login()
            except SoundeoAuthError as exc:
                soundeo.close()
                raise RuntimeError(
                    f"Soundeo login failed: {exc}. Aborting — not falling back to "
                    f"YouTube automatically. Fix SOUNDEO_USER/SOUNDEO_PASS, or pass "
                    f"--no-soundeo to download from YouTube only."
                ) from exc
    elif use_soundeo and audio_format != "aiff":
        logger.info("Soundeo: skipped (only used for --format aiff) — using YouTube only")

    if soundeo is not None:
        logger.info("Source order: Soundeo (AIFF > WAV > MP3) first, YouTube fallback")
    elif not use_soundeo:
        logger.info("Source: YouTube only (--no-soundeo)")
    elif audio_format == "aiff":
        logger.info("Source: YouTube only (no SOUNDEO_USER/SOUNDEO_PASS configured)")

    has_ytdlp, has_ffmpeg = tools_available()
    if not has_ytdlp:
        # YouTube is the fallback; required outright only when Soundeo is off.
        if soundeo is None:
            raise RuntimeError("yt-dlp not found on PATH (required to download)")
        logger.warning("yt-dlp not found on PATH; tracks absent from Soundeo cannot be fetched")
    if audio_format == "aiff" and not has_ffmpeg and soundeo is None:
        raise RuntimeError("ffmpeg not found on PATH (required for AIFF conversion)")

    log_path = os.path.join(report_dir, "download_log.csv")
    try:
        with open(log_path, "w", encoding="utf-8", newline="") as lf:
            log = csv.writer(lf)
            log.writerow(["status", "source", "artists", "name", "query", "outfile", "detail"])
            for i, r in enumerate(work, 1):
                t = r.track
                # Soundeo quota is spent: stop the whole run (do NOT fall back to
                # YouTube for the remainder). Re-run after the midnight CET reset.
                if quota.exhausted:
                    deferred = work[i - 1:]
                    summary["deferred"] = len(deferred)
                    for dr in deferred:
                        log.writerow(["deferred", "soundeo", dr.track.artist_display,
                                      dr.track.name, dr.track.search_query(), "",
                                      "soundeo daily limit reached"])
                    logger.warning(
                        "soundeo: daily download limit reached — STOPPING. "
                        "%d track(s) deferred; re-run after midnight CET to continue.",
                        len(deferred),
                    )
                    break
                prefix = f"[{i}/{len(work)}]"
                label = f"{t.artist_display} - {t.name}"
                is_reverify = i > len(to_download)

                outcome = acquire_track(
                    t, library, soundeo=soundeo, quota=quota,
                    audio_format=audio_format, tolerance=tolerance,
                    max_attempts=max_attempts,
                    min_duration=min_duration, max_duration=max_duration,
                )
                log.writerow([outcome.status, outcome.source, t.artist_display, t.name,
                              t.search_query(), outcome.outfile, outcome.detail])
                lf.flush()

                # Single consolidated log line per track with status
                if outcome.status == "skip":
                    summary["skipped"] += 1
                    _forget_cached_not_found(not_found, t)
                    status_msg = (
                        "present (Soundeo checked, duration ok)"
                        if soundeo is not None else
                        "present (duration ok)"
                    )
                    logger.info("%s %s | %s", prefix, label, status_msg)
                elif outcome.status == "quota_skip":
                    summary["quota_skipped"] += 1
                    logger.info("%s %s | deferred (quota limit, retry after midnight CET)", prefix, label)
                elif outcome.status == "no_match":
                    summary["unmatched"] += 1
                    summary["unmatched_results"].append((t, outcome.detail))
                    if not outcome.soundeo_listed:
                        _remember_cached_not_found(not_found, t, outcome.detail)
                        save_not_found_cache(report_dir, not_found)
                    logger.warning("%s %s | not found (%s)", prefix, label, outcome.detail)
                elif outcome.status == "fail":
                    summary["failed"] += 1
                    logger.warning("%s %s | failed (%s: %s)", prefix, label, outcome.source, outcome.detail)
                else:
                    summary["downloaded"] += 1
                    summary[outcome.source] = summary.get(outcome.source, 0) + 1
                    _forget_cached_not_found(not_found, t)  # found now - clear any stale mark
                    if (outcome.source == "soundeo" and outcome.outfile
                            and not strip_tool_marker(os.path.splitext(os.path.basename(outcome.outfile))[0])[1]):
                        sid = outcome.detail.split("soundeo:", 1)[-1] if outcome.detail else ""
                        _record_soundeo_tags(soundeo_log, outcome.outfile, sid)
                    src = "Soundeo" if outcome.source == "soundeo" else "YouTube"
                    logger.info("%s %s | downloaded from %s (%s)", prefix, label, src,
                                os.path.basename(outcome.outfile))
    finally:
        if soundeo is not None:
            soundeo.close()

    save_not_found_cache(report_dir, not_found)
    save_soundeo_tags_log(report_dir, soundeo_log)

    # A Soundeo original lands unmarked, so it may now supersede a [U] YouTube
    # copy fetched on an earlier run — prune those before regenerating playlists.
    superseded = prune_superseded_downloads(library)
    if superseded:
        logger.info("fetch-missing: pruned %d marked copy(ies) superseded by Soundeo originals",
                    len(superseded))

    if summary["unmatched_results"]:
        with open(os.path.join(report_dir, "unmatched_report.csv"),
                  "w", encoding="utf-8", newline="") as mf:
            w = csv.writer(mf)
            w.writerow(["artists", "name", "query", "duration_sec", "detail"])
            for t, detail in summary["unmatched_results"]:
                w.writerow([t.artist_display, t.name, t.search_query(),
                            f"{t.duration_sec:.0f}" if t.duration_sec else "", detail])

    logger.info(
        "fetch-missing: done — downloaded=%d (soundeo=%d youtube=%d) "
        "skipped=%d quota_skipped=%d deferred=%d unmatched=%d failed=%d cached_skipped=%d",
        summary["downloaded"], summary["soundeo"], summary["youtube"],
        summary["skipped"], summary["quota_skipped"], summary["deferred"],
        summary["unmatched"], summary["failed"], summary["cached_skipped"],
    )
    if summary["unmatched_results"]:
        logger.info(
            "fetch-missing: %d track(s) had no result within ±%.0fs (see %s):",
            summary["unmatched"], tolerance, os.path.join(report_dir, "unmatched_report.csv"),
        )
        for t, detail in summary["unmatched_results"]:
            logger.info("  %s - %s  (%s)", t.artist_display, t.name, detail)

    summary.update(generate_spotify_playlists(csv_paths, library, threshold=threshold))
    return summary
