"""Find and download tracks from Spotify playlist CSV exports.

Given one or more Exportify-style Spotify playlist CSVs, this:

  1. parses the track list (name + artists + exact duration),
  2. fuzzy-matches each track against the audio files already in a library
     directory, requiring artist agreement to avoid same-title collisions,
  3. downloads anything missing via ``yt-dlp`` YouTube search, extracting to
     WAV and converting losslessly to AIFF (configurable),
  4. flags downloads whose duration differs sharply from Spotify's — a strong
     signal that the search returned the wrong video.

The matcher works off files on disk, so it does not require a populated
registry. ``yt-dlp`` and ``ffmpeg`` must be on PATH for the download step.
"""

from __future__ import annotations

import csv
import logging
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
STOPWORDS = {
    "the", "a", "an", "of", "and", "feat", "featuring", "ft", "with",
    "original", "mix", "extended", "version", "radio", "edit", "remix",
    "remixes", "rework", "dub", "vip", "instrumental", "club",
}

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""

    @property
    def artist_display(self) -> str:
        return ", ".join(self.artists)

    def target_basename(self) -> str:
        """Library-style 'Artist - Title' base filename (no extension)."""
        title = clean_track_name(self.name)
        base = f"{self.artist_display} - {title}" if self.artists else title
        return sanitize_filename(base)

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
            tracks.append(
                PlaylistTrack(
                    name=track_name,
                    artists=_split_artists(row.get("Artist Name(s)") or ""),
                    duration_sec=(ms / 1000.0) if ms else None,
                    uri=(row.get("Track URI") or "").strip(),
                    playlist=name,
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
    """Parse all CSVs and dedupe by Spotify URI (falling back to name+artist)."""
    seen: set[str] = set()
    out: list[PlaylistTrack] = []
    for cp in csv_paths:
        for t in parse_playlist_csv(cp):
            key = t.uri or f"{t.name}\x00{t.artist_display}".lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
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
def build_ytdlp_command(
    query: str,
    out_template: str,
    *,
    min_duration: int = 30,
    max_duration: int = 900,
    search_count: int = 5,
) -> list[str]:
    """Construct the yt-dlp argv for a duration-bounded YouTube search."""
    return [
        "yt-dlp", "--no-playlist", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--match-filter", f"duration < {max_duration} & duration > {min_duration}",
        "-o", out_template,
        f"ytsearch{search_count}:{query}",
    ]


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


def _wav_to_aiff(wav_path: str, aiff_path: str) -> bool:
    """Losslessly convert WAV -> AIFF (PCM). Returns True on success."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-c:a", "pcm_s16be", aiff_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return os.path.exists(aiff_path) and os.path.getsize(aiff_path) > 0


@dataclass
class DownloadOutcome:
    status: str          # "ok" | "ok_wav" | "skip" | "fail"
    outfile: str = ""
    detail: str = ""


def download_track(
    track: PlaylistTrack,
    dest_dir: str,
    *,
    audio_format: str = "aiff",
    min_duration: int = 30,
    max_duration: int = 900,
) -> DownloadOutcome:
    """Search YouTube for one track and download it into ``dest_dir``.

    With ``audio_format='aiff'`` the audio is extracted to WAV then converted;
    if conversion fails the WAV is kept as a fallback.
    """
    base = track.target_basename()
    aiff_path = os.path.join(dest_dir, base + ".aiff")
    wav_path = os.path.join(dest_dir, base + ".wav")
    final_path = aiff_path if audio_format == "aiff" else wav_path

    if os.path.exists(final_path):
        return DownloadOutcome("skip", final_path, "already exists")

    cmd = build_ytdlp_command(
        track.search_query(),
        os.path.join(dest_dir, base + ".%(ext)s"),
        min_duration=min_duration,
        max_duration=max_duration,
    )
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if not os.path.exists(wav_path):
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        return DownloadOutcome("fail", "", lines[-1] if lines else "no output file")

    if audio_format == "wav":
        return DownloadOutcome("ok", wav_path)

    if _wav_to_aiff(wav_path, aiff_path):
        os.remove(wav_path)
        return DownloadOutcome("ok", aiff_path)
    return DownloadOutcome("ok_wav", wav_path, "aiff conversion failed; kept wav")


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
# Orchestration
# --------------------------------------------------------------------------- #
def resolve_library_dir(path: str) -> str:
    """The directory to match against and download into.

    Accepts a directory (used as-is) or a file (its parent is used), so it
    works whether ``dj run`` is pointed at a library folder or a single track.
    """
    return path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))


def fetch_missing(
    playlists: list[str],
    library: str,
    *,
    audio_format: str = "aiff",
    threshold: float = 0.62,
    min_duration: int = 30,
    max_duration: int = 900,
    dry_run: bool = False,
    report_dir: str | None = None,
) -> dict:
    """Match playlist CSVs against ``library`` and download what's missing.

    Returns a summary dict with counts, the missing :class:`MatchResult` list,
    a list of probable-mismatch ``(track, outfile, reason)`` tuples, and the
    report directory. Raises ``FileNotFoundError`` if the library is missing
    and ``RuntimeError`` if required external tools are unavailable.
    """
    if not os.path.isdir(library):
        raise FileNotFoundError(f"library dir not found: {library}")

    csv_paths = collect_playlists(playlists)
    if not csv_paths:
        logger.warning("fetch-missing: no playlist CSVs found in %s", playlists)

    tracks = load_unique_tracks(csv_paths)
    index = scan_library(library)
    present, missing = classify_tracks(tracks, index, threshold=threshold)

    report_dir = report_dir or os.path.join(library, "outputs", "fetch")
    os.makedirs(report_dir, exist_ok=True)
    _write_match_report(os.path.join(report_dir, "matched_report.csv"), present)
    _write_match_report(os.path.join(report_dir, "missing_report.csv"), missing)

    summary = {
        "total": len(tracks), "playlists": len(csv_paths),
        "present": len(present), "missing": len(missing),
        "downloaded": 0, "skipped": 0, "failed": 0,
        "missing_results": missing, "mismatches": [], "report_dir": report_dir,
    }
    logger.info(
        "fetch-missing: %d unique track(s) across %d playlist(s) — %d present, %d missing",
        len(tracks), len(csv_paths), len(present), len(missing),
    )
    logger.info("fetch-missing: reports written to %s", report_dir)

    if dry_run or not missing:
        if dry_run:
            for r in missing:
                logger.info("  [%.2f] %s - %s", r.score, r.track.artist_display, r.track.name)
        return summary

    has_ytdlp, has_ffmpeg = tools_available()
    if not has_ytdlp:
        raise RuntimeError("yt-dlp not found on PATH (required to download)")
    if audio_format == "aiff" and not has_ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (required for AIFF conversion)")

    log_path = os.path.join(report_dir, "download_log.csv")
    with open(log_path, "w", encoding="utf-8", newline="") as lf:
        log = csv.writer(lf)
        log.writerow(["status", "artists", "name", "query", "outfile", "detail"])
        for i, r in enumerate(missing, 1):
            t = r.track
            outcome = download_track(
                t, library, audio_format=audio_format,
                min_duration=min_duration, max_duration=max_duration,
            )
            log.writerow([outcome.status, t.artist_display, t.name,
                          t.search_query(), outcome.outfile, outcome.detail])
            lf.flush()
            prefix = f"[{i}/{len(missing)}]"
            if outcome.status == "skip":
                summary["skipped"] += 1
                logger.info("%s skip (exists): %s", prefix, t.target_basename())
            elif outcome.status == "fail":
                summary["failed"] += 1
                logger.warning("%s FAILED: %s — %s", prefix, t.name, outcome.detail)
            else:
                summary["downloaded"] += 1
                logger.info("%s ok: %s", prefix, os.path.basename(outcome.outfile))
                reason = duration_mismatch(t.duration_sec, probe_duration(outcome.outfile))
                if reason:
                    summary["mismatches"].append((t, outcome.outfile, reason))

    if summary["mismatches"]:
        with open(os.path.join(report_dir, "probable_mismatches.csv"),
                  "w", encoding="utf-8", newline="") as mf:
            w = csv.writer(mf)
            w.writerow(["artists", "name", "outfile", "detail"])
            for t, outfile, reason in summary["mismatches"]:
                w.writerow([t.artist_display, t.name, outfile, reason])

    logger.info(
        "fetch-missing: done — downloaded=%d skipped=%d failed=%d",
        summary["downloaded"], summary["skipped"], summary["failed"],
    )
    if summary["mismatches"]:
        logger.info(
            "fetch-missing: %d download(s) are probably NOT an exact match (see %s):",
            len(summary["mismatches"]), os.path.join(report_dir, "probable_mismatches.csv"),
        )
        for t, _outfile, reason in summary["mismatches"]:
            logger.info("  %s - %s  (%s)", t.artist_display, t.name, reason)
    return summary
