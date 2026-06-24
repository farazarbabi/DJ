"""Soundeo download source — authenticated music-pool client (primary source).

Soundeo (https://soundeo.com) is a DJ music pool the user subscribes to. It
serves the genuine original **AIFF** releases, far superior to a YouTube rip, so
``fetch-missing`` prefers it and falls back to YouTube only when a track is not
on Soundeo. This automates the user's own paid account.

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
# Substrings in a failed download's flash message that mean "out of quota"
# rather than a one-off error (so we defer vs. fall back to YouTube). Refine if
# a real exhausted-quota response surfaces a different wording.
_QUOTA_HINTS = ("limit", "quota", "reset", "premium", "no downloads",
                "downloads left", "reached")

_TIMEOUT = 60.0
_DEFAULT_RATE = 1.0  # max requests/sec (be a polite scraper)


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
        return _as_envelope(resp)

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
        logger.info("soundeo: logged in as %s", self.user)

    # -- search ------------------------------------------------------------- #
    def search(self, track: PlaylistTrack) -> list[SoundeoResult]:
        """Search Soundeo for ``track`` (does not consume download quota)."""
        if not self._logged_in:
            self.login()
        query = f"{track.primary_artist} {track.name}".strip()
        try:
            env = self._get_json(SEARCH_PATH, params={SEARCH_QUERY_PARAM: query})
        except httpx.HTTPError as exc:
            logger.warning("soundeo: search failed for %r: %s", query, exc)
            return []
        return _parse_results(env.get("content") or "")

    def pick(
        self, track: PlaylistTrack, results: list[SoundeoResult],
        *, threshold: float = 0.62, tolerance: float = 6.0,
    ) -> SoundeoResult | None:
        """Best Soundeo result for ``track``, or None if none is good enough.

        Requires the chosen format to be available and (when both durations are
        known) the lengths to agree within ``tolerance`` seconds, so a wrong
        remix/edit is rejected. Ties broken by closest duration.
        """
        fmt = self.audio_format
        best: tuple[float, float, SoundeoResult] | None = None
        for r in results:
            # Require the format to be explicitly offered. Tracks without an
            # AIFF download link (mp3-only, or vote-required/upcoming entries
            # that parse to no formats) are skipped so the caller backfalls to
            # YouTube instead of hitting a 404 on the download endpoint.
            if fmt not in r.formats:
                continue
            score = _score(track, r)
            if score < threshold:
                continue
            if (track.duration_sec and r.duration_sec
                    and abs(track.duration_sec - r.duration_sec) > tolerance):
                continue
            dur_gap = (abs(track.duration_sec - r.duration_sec)
                       if track.duration_sec and r.duration_sec else 0.0)
            cand = (-score, dur_gap, r)
            if best is None or cand[:2] < best[:2]:
                best = cand
        return best[2] if best else None

    # -- download ----------------------------------------------------------- #
    def download(self, result: SoundeoResult, dest_path: str) -> None:
        """Download ``result`` to ``dest_path`` in the client's audio format.

        Two-step: ``GET /download/<id>/<fmt>`` yields a tokenized CDN URL, then
        that URL streams the file. Raises :class:`SoundeoQuotaExceeded` when the
        daily quota is spent (no CDN URL + a limit message) and
        :class:`SoundeoError` on other failures, leaving no partial file.
        """
        if not self._logged_in:
            self.login()
        fmt = FORMAT_CODE.get(self.audio_format, self.audio_format)
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
            if _looks_like_quota(msg):
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
        out.append(SoundeoResult(
            id=tid, artist=artist.strip(), title=title.strip(),
            duration_sec=duration, formats=formats,
        ))
    return out


def _unlink_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
