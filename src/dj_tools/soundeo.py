"""Soundeo download source — authenticated music-pool client (primary source).

Soundeo (https://soundeo.com) is a DJ music pool the user subscribes to. It
serves music-pool downloads that are superior to a YouTube rip, so
``fetch-missing`` prefers Soundeo formats in AIFF > WAV > MP3 order and falls
back to YouTube only when a usable Soundeo source is not available. This
automates the user's own paid account.

Soundeo has **no public API**; this client logs in and drives the site's
AJAX-JSON endpoints with ``httpx`` (one persistent session for cookie
continuity) + ``lxml.html`` to parse the embedded result markup — no new heavy
dependencies. Every endpoint returns a JSON envelope
(``{content, header, jsActions, success, flash, ...}``) where ``content`` /
``header`` carry HTML fragments.

Flow (pinned from a real HAR capture):

* **login**  — ``GET`` then ``POST /account/logoreg`` with CakePHP form fields
  ``_method`` / ``data[User][login]`` / ``data[User][password]`` /
  ``data[remember]`` (no CSRF token); success is JSON ``success: true``.
* **search** — ``GET /search?q=<query>``; ``content`` holds
  ``div.trackitem[data-track-id]`` rows with ``<strong><a>Artist - Title</a>``,
  ``<time>m:ss</time>`` and ``a.track-download-lnk[data-track-format]`` links
  (format codes ``1``=MP3, ``2``=WAV, ``3``=AIFF).
* **download** — ``GET /download/<id>/3`` returns ``jsActions.redirect.url``
  (a tokenized CDN link on ``dl*.sndstatic.com``); a second ``GET`` of that URL
  streams the AIFF. Quota does not block search, only the download step.

Downloads are duration- and quota-limited; the daily quota resets midnight CET.
A download that cannot return a CDN URL because the limit is spent raises
:class:`SoundeoQuotaExceeded` so the orchestrator can defer the rest to
tomorrow rather than YouTube them.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # avoid a runtime import cycle with spotify_fetch
    from .spotify_fetch import PlaylistTrack

logger = logging.getLogger(__name__)

BASE_URL = "https://soundeo.com"
LOGIN_PATH = "/account/logoreg"          # GET (form) then POST (credentials)
SEARCH_PATH = "/search"
SEARCH_QUERY_PARAM = "q"
DOWNLOAD_INFO_TEMPLATE = "/download/{id}/{fmt}"   # JSON -> jsActions.redirect.url
# Soundeo numeric format codes (data-track-format on the download links).
FORMAT_CODE = {"mp3": "1", "wav": "2", "aiff": "3"}
_CODE_TO_FORMAT = {v: k for k, v in FORMAT_CODE.items()}
SOUNDEO_FORMAT_PRIORITY = ("aiff", "wav", "mp3")
# Substrings in a failed download's flash message that mean "out of quota"
# rather than a one-off error (so we defer vs. fall back to YouTube). Refine if
# a real exhausted-quota response surfaces a different wording.
_QUOTA_HINTS = ("limit", "quota", "reset", "premium", "no downloads",
                "downloads left", "reached")

_TIMEOUT = 60.0
_DEFAULT_RATE = 1.0  # max requests/sec (be a polite scraper)

# A trailing *generic* version descriptor on a Spotify title (e.g. "Diclofél -
# Radio Edit", "Title (Original Mix)"). Stripped from the **search query** only:
# Soundeo may list the very same song under a different generic cut (usually the
# Original/Extended Mix), so searching the raw "… Radio Edit" would miss it. A
# remixer name is NOT generic and is deliberately kept, so a requested remix
# still narrows the search. `pick()` does the final version selection.
_SEARCH_VERSION_RE = re.compile(
    r"\s*[-(\[]\s*(?:original|extended|radio)"
    r"(?:\s+(?:mix|version|edit|re-?edit|cut))?\s*[)\]]?\s*$",
    re.IGNORECASE,
)


def _search_title(name: str) -> str:
    """Drop a trailing generic version descriptor for the Soundeo search query.

    ``"Diclofél - Radio Edit"`` / ``"Title (Original Mix)"`` -> ``"Diclofél"`` /
    ``"Title"`` so the search finds the track whichever generic cut Soundeo
    lists; a remixer name (``"- Mateo! Remix"``) is left intact.
    """
    return _SEARCH_VERSION_RE.sub("", name).strip() or name.strip()


def _version_rank(title: str) -> int:
    """Preference rank for a Soundeo candidate's cut: lower is better.

    Extended Mix (0) > Original Mix (1) > plain / other (2) > Radio Edit (3).
    So every track prefers the Extended cut, then the Original, then a
    suffix-less listing, and picks a Radio Edit — the short broadcast cut, least
    useful for DJing — only when nothing better matches.
    """
    t = title.lower()
    if "extended" in t:
        return 0
    if "radio" in t:
        return 3
    if "original" in t:
        return 1
    return 2


def acceptable_source_formats(target_format: str) -> tuple[str, ...]:
    """Soundeo source formats to accept for a requested output format."""
    if target_format == "aiff":
        return SOUNDEO_FORMAT_PRIORITY
    if target_format == "wav":
        return ("wav", "mp3")
    return (target_format,)


def preferred_download_format(result: "SoundeoResult", target_format: str = "aiff") -> str | None:
    """Best Soundeo source format for ``result`` and requested output format."""
    available = set(result.formats)
    for fmt in acceptable_source_formats(target_format):
        if fmt in available:
            return fmt
    return None


def _format_rank(result: "SoundeoResult", target_format: str) -> int:
    fmt = preferred_download_format(result, target_format)
    formats = acceptable_source_formats(target_format)
    return formats.index(fmt) if fmt in formats else len(formats) + 1


class SoundeoError(Exception):
    """Base class for Soundeo client failures."""


class SoundeoAuthError(SoundeoError):
    """Login failed (bad credentials, changed form, captcha, etc.)."""


class SoundeoQuotaExceeded(SoundeoError):
    """The daily download quota is used up; resets at midnight CET."""


@dataclass
class SoundeoResult:
    """One search hit on Soundeo."""

    id: str
    artist: str
    title: str
    duration_sec: float | None = None
    key: str = ""
    bpm: str = ""
    formats: list[str] = field(default_factory=list)
    # True when Soundeo marks this track as already downloaded by this account
    # (the result's download container carries a ``downloaded`` class). Such a
    # re-download does not spend the daily quota, and the flag is the reliable
    # signal that a library file originated from Soundeo (vs. a curated original).
    downloaded: bool = False

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title}".strip(" -")


def _score(track: PlaylistTrack, result: SoundeoResult) -> float:
    """Title/artist token-coverage score in [0, 1], source-agnostic.

    Mirrors the library matcher: how many of the Spotify track's title tokens
    appear in the candidate, gated by artist agreement so a same-title different
    -artist hit (e.g. two unrelated "Kryptonite"s) does not win. Reuses the
    shared tokenizer so Soundeo matching behaves like the on-disk matcher.
    """
    from .spotify_fetch import tokens  # lazy: avoid import cycle

    want_title = tokens(track.name)
    want_artist = tokens(track.artist_display)
    have = tokens(result.label)
    have_title = tokens(result.title)
    have_artist = tokens(result.artist)
    if not want_title:
        return 0.0
    title_cov = len(want_title & (have_title | have)) / len(want_title)
    artist_ok = bool(want_artist & (have_artist | have)) if want_artist else True
    distinctive = len(want_title) >= 3
    if not artist_ok and not distinctive:
        return min(title_cov, 0.45)  # same-title collision guard
    return title_cov if artist_ok else title_cov * 0.7


def _parse_mmss(text: str) -> float | None:
    """'8:25' -> 505.0; '1:02:03' -> 3723.0; None if unparseable."""
    parts = text.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    secs = 0.0
    for n in nums:
        secs = secs * 60 + n
    return secs if nums else None


class SoundeoClient:
    """Authenticated Soundeo session: login, search, download.

    One instance == one logged-in session, reused across a fetch run (the
    persistent ``httpx.Client`` keeps the auth cookies). Construct via
    :meth:`from_env` to read ``SOUNDEO_USER`` / ``SOUNDEO_PASS`` from the
    environment (loaded from ``.env`` like the Spotify/Songstats credentials).
    """

    def __init__(
        self,
        user: str,
        password: str,
        *,
        base_url: str = BASE_URL,
        audio_format: str = "aiff",
        rate_limit: float = _DEFAULT_RATE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.user = user
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.audio_format = audio_format
        self.rate_limit = rate_limit
        self._client = httpx.Client(
            base_url=self.base_url, timeout=_TIMEOUT, follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                # Both are required for the server to return its JSON envelope
                # instead of the full HTML page (content negotiation).
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": self.base_url + "/",
            },
            transport=transport,  # tests inject httpx.MockTransport
        )
        self._logged_in = False
        self._last_request = 0.0
        # Remaining daily downloads, parsed from each response header's
        # `span-downloads` counter; 0 means out of credit. None = unknown yet.
        self._remaining: int | None = None

    # -- lifecycle ---------------------------------------------------------- #
    @classmethod
    def from_env(cls, *, audio_format: str = "aiff") -> SoundeoClient | None:
        """Build from ``SOUNDEO_USER`` / ``SOUNDEO_PASS``; None if unset.

        Loads ``.env`` via the shared registry loader so the same file that
        holds the Spotify/Songstats secrets supplies the Soundeo credentials.
        Missing credentials disable Soundeo entirely (YouTube-only), keeping the
        feature opt-in and fully backward compatible.
        """
        try:
            from dj_registry.config import RegistryConfig

            RegistryConfig().load_env()
        except Exception:  # registry extra not installed; env may still be set
            pass
        user = os.environ.get("SOUNDEO_USER", "").strip()
        password = os.environ.get("SOUNDEO_PASS", "").strip()
        if not user or not password:
            return None
        return cls(user, password, audio_format=audio_format)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SoundeoClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        if self.rate_limit <= 0:
            return
        interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request = time.time()

    def _get_json(self, path: str, **kw) -> dict:
        self._throttle()
        resp = self._client.get(path, **kw)
        resp.raise_for_status()
        env = _as_envelope(resp)
        self._note_remaining(env)
        return env

    def _note_remaining(self, env: dict) -> None:
        n = _remaining_downloads(env)
        if n is not None:
            self._remaining = n

    # -- auth --------------------------------------------------------------- #
    def login(self) -> None:
        """Authenticate and retain the session cookies.

        GET the login form (sets the initial session cookie), then POST the
        CakePHP-style credential fields. Raises :class:`SoundeoAuthError` on any
        failure so the orchestrator can degrade to YouTube-only.
        """
        self._throttle()
        try:
            self._client.get(LOGIN_PATH).raise_for_status()
            self._throttle()
            resp = self._client.post(LOGIN_PATH, data={
                "_method": "POST",
                "data[User][login]": self.user,
                "data[User][password]": self.password,
                "data[remember]": "1",
            })
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SoundeoAuthError(f"login request failed: {exc}") from exc

        ctype = resp.headers.get("content-type", "")
        env = _as_envelope(resp)
        if not env:
            raise SoundeoAuthError(
                f"login response was not the expected JSON envelope "
                f"(status={resp.status_code}, content-type={ctype!r}); the site "
                f"may have changed or blocked the request")
        # `success` is the canonical signal; a logged-in header confirms it.
        authed = "/account/logout" in (env.get("header") or "")
        if not env.get("success") and not authed:
            raise SoundeoAuthError(
                f"login rejected: {_flash_text(env) or 'invalid credentials or changed form'}")
        self._logged_in = True
        self._note_remaining(env)
        if self._remaining is not None:
            logger.info("soundeo: logged in as %s (%d download(s) left today)",
                        self.user, self._remaining)
        else:
            logger.info("soundeo: logged in as %s", self.user)

    # -- search ------------------------------------------------------------- #
    def search(self, track: PlaylistTrack) -> list[SoundeoResult]:
        """Search Soundeo for ``track`` (does not consume download quota)."""
        if not self._logged_in:
            self.login()
        query = f"{track.primary_artist} {_search_title(track.name)}".strip()
        try:
            env = self._get_json(SEARCH_PATH, params={SEARCH_QUERY_PARAM: query})
        except httpx.HTTPError as exc:
            logger.warning("soundeo: search failed for %r: %s", query, exc)
            return []
        return _parse_results(env.get("content") or "")

    def pick(
        self, track: PlaylistTrack, results: list[SoundeoResult],
    ) -> SoundeoResult | None:
        """Best Soundeo result for ``track``, using a priority fallback chain.

        Fallback order (each checked only if previous found nothing):
        1. Extended versions (exact title/artist, no duration check)
        2. DJ range 6:30-8:00 min (exact title/artist, extended versions)
        3. Strict ±3s tolerance (original cut duration match)
        4. Best overall (Extended > Original > plain > Radio Edit)

        This prefers high-quality extended mixes from Soundeo over YouTube fallback,
        upgrading the original Spotify cut when available.
        """
        matches = self._title_artist_matches(track, results)
        if not matches:
            return None

        _DJ_MIN_SEC = 390.0  # 6:30
        _DJ_MAX_SEC = 480.0  # 8:00

        # 1. Extended versions (exact title/artist match, no duration check)
        extended_versions = [r for r in matches if _version_rank(r.title) == 0]  # rank 0 = Extended Mix
        if extended_versions:
            extended_versions.sort(key=lambda r: (
                _format_rank(r, self.audio_format),
                -(r.duration_sec or 0.0),
            ))
            return extended_versions[0]

        # 2. DJ range 6:30-8:00 min (extended versions, same track)
        dj_range_matches = [
            r for r in matches
            if r.duration_sec is not None and _DJ_MIN_SEC <= r.duration_sec <= _DJ_MAX_SEC
        ]
        if dj_range_matches:
            dj_range_matches.sort(key=lambda r: (
                _format_rank(r, self.audio_format), _version_rank(r.title),
                -(r.duration_sec or 0.0),
            ))
            return dj_range_matches[0]

        # 3. Strict ±3s tolerance (original cut or close match)
        expected_sec = track.duration_sec
        if expected_sec is not None:
            close_matches = [
                r for r in matches
                if r.duration_sec is not None and abs(r.duration_sec - expected_sec) <= 3.0
            ]
            if close_matches:
                close_matches.sort(key=lambda r: (
                    _format_rank(r, self.audio_format), _version_rank(r.title),
                    -(r.duration_sec or 0.0),
                ))
                return close_matches[0]

        # 4. Best overall (format > version rank > duration)
        matches.sort(key=lambda r: (
            _format_rank(r, self.audio_format), _version_rank(r.title),
            -(r.duration_sec or 0.0),
        ))
        return matches[0]

    def pick_owned(
        self, track: PlaylistTrack, results: list[SoundeoResult],
        *, target_duration: float | None = None,
    ) -> SoundeoResult | None:
        """Best **already-downloaded** AIFF result for ``track`` (repair mode).

        Only results Soundeo flags as ``downloaded`` (re-download is free and
        proves the track came from Soundeo) are considered. When
        ``target_duration`` is given (the on-disk file's length), the cut whose
        duration matches it wins, so a re-download restores native tags on the
        **same cut** the library already holds — same duration → same analysis
        cache key, no re-analysis. Returns None if no owned AIFF result matches.
        """
        owned = [
            r for r in self._title_artist_matches(track, results)
            if r.downloaded and self.audio_format in r.formats
        ]
        if not owned:
            return None
        if target_duration is not None:
            owned.sort(key=lambda r: abs((r.duration_sec or 0.0) - target_duration))
        else:
            owned.sort(key=lambda r: (_version_rank(r.title), -(r.duration_sec or 0.0)))
        return owned[0]

    def _title_artist_matches(
        self, track: PlaylistTrack, results: list[SoundeoResult],
    ) -> list[SoundeoResult]:
        """Results whose title/artist tokens match ``track`` and offer a usable format.

        Shared by :meth:`pick` and :meth:`pick_owned`: all title tokens present,
        no extra remixer tokens (unless the query itself names the remix), and
        artist agreement (or a distinctive >=3-token title).
        """
        from .spotify_fetch import tokens  # lazy: avoid import cycle

        want = tokens(track.name)
        want_artist = tokens(track.artist_display)
        if not want:
            return []
        fmt = self.audio_format
        matches: list[SoundeoResult] = []
        for r in results:
            # Require a usable Soundeo source format. For AIFF output,
            # WAV/MP3 Soundeo sources are still preferred over YouTube and
            # converted by the caller.
            if preferred_download_format(r, fmt) is None:
                continue
            have_title = tokens(r.title)
            have_artist = tokens(r.artist)
            if not want <= (have_title | have_artist):
                continue  # not all Spotify title tokens present
            # Extra identifying tokens (a remixer name) mean a different version
            # than this suffix-less Spotify track wants — Original/Extended carry
            # none (stopwords), a remix does.
            if have_title - want - want_artist:
                continue
            artist_ok = bool(want_artist & (have_artist | have_title)) if want_artist else True
            if not artist_ok and len(want) < 3:  # allow artistless match only for distinctive titles
                continue
            matches.append(r)
        return matches

    # -- download ----------------------------------------------------------- #
    def download(self, result: SoundeoResult, dest_path: str,
                 *, assume_free: bool = False, audio_format: str | None = None) -> None:
        """Download ``result`` to ``dest_path`` in the client's audio format.

        Two-step: ``GET /download/<id>/<fmt>`` yields a tokenized CDN URL, then
        that URL streams the file. Raises :class:`SoundeoQuotaExceeded` when the
        daily quota is spent (no CDN URL + a limit message) and
        :class:`SoundeoError` on other failures, leaving no partial file.

        ``assume_free`` skips the pre-flight "0 credits left" guard: a track the
        account already owns (``result.downloaded``) re-downloads without
        spending quota, so a repair pass can run even when today's credit is 0.
        A genuinely metered download with no credit still fails safely (Soundeo
        returns no URL -> :class:`SoundeoQuotaExceeded`).
        """
        if not self._logged_in:
            self.login()
        # Out of credit (counter known to be 0): stop before spending a request,
        # so the caller defers rather than falling back to YouTube. An
        # already-owned re-download (assume_free) bypasses this — it costs no
        # credit — but still fails safely below if Soundeo withholds the URL.
        if not assume_free and self._remaining is not None and self._remaining <= 0:
            raise SoundeoQuotaExceeded(
                "Soundeo daily download limit reached (0 downloads left)")
        requested_format = audio_format or self.audio_format
        if requested_format not in result.formats:
            raise SoundeoError(f"format {requested_format!r} not available on Soundeo")
        fmt = FORMAT_CODE.get(requested_format, requested_format)
        try:
            info = self._get_json(DOWNLOAD_INFO_TEMPLATE.format(id=result.id, fmt=fmt))
        except httpx.HTTPStatusError as exc:
            # e.g. 404: this track isn't downloadable on Soundeo (upcoming /
            # vote-required / format not offered). Treat as "not available here"
            # so the caller backfalls to YouTube for this one track.
            raise SoundeoError(
                f"not available on Soundeo (HTTP {exc.response.status_code})") from exc
        except httpx.HTTPError as exc:
            raise SoundeoError(f"download info request failed: {exc}") from exc
        url = (info.get("jsActions") or {}).get("redirect", {}).get("url", "")
        if not url:
            msg = _flash_text(info) or "no download URL returned"
            # No URL because the daily limit is spent (counter hit 0, or a limit
            # message) -> quota, which stops the run. Otherwise a genuine
            # per-track failure -> SoundeoError, which backfalls to YouTube.
            if (self._remaining is not None and self._remaining <= 0) or _looks_like_quota(msg):
                raise SoundeoQuotaExceeded(f"Soundeo download limit reached: {msg}")
            raise SoundeoError(f"Soundeo download failed: {msg}")

        tmp = dest_path + ".part"
        try:
            self._throttle()
            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                if "audio" not in ctype and "octet-stream" not in ctype:
                    raise SoundeoError(f"unexpected content-type {ctype!r} from CDN")
                os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
        except httpx.HTTPError as exc:
            _unlink_quiet(tmp)
            raise SoundeoError(f"download failed: {exc}") from exc
        except SoundeoError:
            _unlink_quiet(tmp)
            raise
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            _unlink_quiet(tmp)
            raise SoundeoError("download produced no data")
        os.replace(tmp, dest_path)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _as_envelope(resp: httpx.Response) -> dict:
    """Soundeo replies with a JSON envelope; tolerate a non-JSON body."""
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _flash_text(env: dict) -> str:
    """Human-readable text from the envelope's flash/alerts, for error messages."""
    bits = []
    for key in ("flash", "alerts"):
        v = env.get(key)
        if v:
            bits.append(v if isinstance(v, str) else str(v))
    return " ".join(bits).strip()


def _looks_like_quota(message: str) -> bool:
    m = message.lower()
    return any(h in m for h in _QUOTA_HINTS)


def _remaining_downloads(env: dict) -> int | None:
    """Remaining daily downloads from a response header's `span-downloads`.

    Every Soundeo JSON envelope carries the top-menu header, which includes
    ``<span id='span-downloads'>…N…</span>`` (N = downloads left in the Main
    pool). Returns N, or None if the header is absent/unparseable.
    """
    header = env.get("header") or ""
    if "span-downloads" not in header:
        return None
    try:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(header)
        nodes = doc.xpath('//span[@id="span-downloads"]')
        if not nodes:
            return None
        m = re.search(r"\d+", nodes[0].text_content())
        return int(m.group()) if m else None
    except Exception:
        return None


def _parse_results(content_html: str) -> list[SoundeoResult]:
    """Parse the search ``content`` HTML into :class:`SoundeoResult` rows."""
    if not content_html.strip():
        return []
    try:
        from lxml import html as lxml_html
    except Exception:
        logger.warning("soundeo: lxml not available; cannot parse search results")
        return []
    doc = lxml_html.fromstring(content_html)
    out: list[SoundeoResult] = []
    for item in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " trackitem ")]'):
        tid = item.get("data-track-id") or ""
        if not tid:
            continue
        label_nodes = item.xpath('.//strong//a/text()')
        label = (label_nodes[0] if label_nodes else "").strip()
        artist, _, title = label.partition(" - ")
        if not title:
            artist, title = "", label
        time_nodes = item.xpath('.//time/text()')
        duration = _parse_mmss(time_nodes[0]) if time_nodes else None
        codes = item.xpath('.//a[contains(@class,"track-download-lnk")]/@data-track-format')
        formats = [_CODE_TO_FORMAT.get(c, c) for c in codes]
        # The download container gets a `downloaded` class once this account has
        # grabbed the track (a free re-download, and proof it came from Soundeo).
        dl_div = item.xpath(
            './/div[contains(concat(" ", normalize-space(@class), " "), " download ")]')
        downloaded = bool(dl_div) and "downloaded" in (dl_div[0].get("class") or "")
        out.append(SoundeoResult(
            id=tid, artist=artist.strip(), title=title.strip(),
            duration_sec=duration, formats=formats, downloaded=downloaded,
        ))
    return out


def _unlink_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
