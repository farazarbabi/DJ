"""Soundeo download source — authenticated music-pool client (primary source).

Soundeo (https://soundeo.com) is a DJ music pool the user subscribes to. It
serves the genuine original **AIFF** releases, far superior to a YouTube rip, so
``fetch-missing`` prefers it and falls back to YouTube only when a track is not
on Soundeo. This automates the user's own paid account.

Soundeo has **no public API**; this client logs in and scrapes the
server-rendered site with ``httpx`` (a persistent session for cookie
continuity) + ``lxml.html`` for parsing — no new heavy dependencies.

Downloads are duration- and quota-limited:

  * search does **not** consume quota; only a download does,
  * the daily quota resets at midnight CET,
  * a download that hits the quota raises :class:`SoundeoQuotaExceeded` so the
    orchestrator can skip the rest for tomorrow rather than YouTube them.

The concrete HTTP details (login URL + form fields + CSRF token name, the search
endpoint and its result markup, the per-format download URL, and the
quota-exhausted response shape) are pinned from a HAR capture of a real session.
Every such value is centralized in the ``# HAR:`` constants below so wiring the
real endpoints is a localized edit; the matching (:meth:`SoundeoClient.pick`),
credential loading, and orchestration around them are source-agnostic and
already complete.
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

# --------------------------------------------------------------------------- #
# HAR-pinned constants — fill/confirm these from the captured session.
# --------------------------------------------------------------------------- #
BASE_URL = "https://soundeo.com"
# HAR: GET page that carries the login form + CSRF token, and the POST target.
LOGIN_PAGE_PATH = "/account/login"
LOGIN_POST_PATH = "/account/login"
# HAR: form field names on the login POST (confirm exact keys).
LOGIN_FIELD_USER = "email"
LOGIN_FIELD_PASS = "password"
# HAR: CSRF token — the <input name=...> on the login page and the POST key it
# maps to. Leave CSRF_INPUT_NAME = "" if the site uses no CSRF token.
CSRF_INPUT_NAME = "_token"
CSRF_FIELD = "_token"
# HAR: search endpoint + query param, and the per-format download URL template.
SEARCH_PATH = "/search"
SEARCH_QUERY_PARAM = "q"
# {fmt} in {"aiff","wav","mp3"}; {id} is SoundeoResult.id.
DOWNLOAD_PATH_TEMPLATE = "/download/{id}/{fmt}"
FORMAT_CODE = {"aiff": "aiff", "wav": "wav", "mp3": "mp3"}

_TIMEOUT = 30.0
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
            headers={"User-Agent": "Mozilla/5.0 (dj-tools soundeo client)"},
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
        Missing credentials disable Soundeo entirely (YouTube-only), which keeps
        the feature opt-in and fully backward compatible.
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

    # -- auth --------------------------------------------------------------- #
    def login(self) -> None:
        """Authenticate and retain the session cookies.

        GET the login page, lift the CSRF token (if any), POST credentials.
        Raises :class:`SoundeoAuthError` on any failure so the orchestrator can
        degrade to YouTube-only.
        """
        self._throttle()
        try:
            page = self._client.get(LOGIN_PAGE_PATH)
            page.raise_for_status()
            data = {LOGIN_FIELD_USER: self.user, LOGIN_FIELD_PASS: self.password}
            token = self._extract_csrf(page.text)
            if CSRF_INPUT_NAME and token:
                data[CSRF_FIELD] = token
            self._throttle()
            resp = self._client.post(LOGIN_POST_PATH, data=data)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SoundeoAuthError(f"login request failed: {exc}") from exc
        if not self._login_succeeded(resp):
            raise SoundeoAuthError("login rejected (check SOUNDEO_USER/SOUNDEO_PASS)")
        self._logged_in = True
        logger.info("soundeo: logged in as %s", self.user)

    @staticmethod
    def _extract_csrf(html: str) -> str:
        """Read the CSRF token value from the login page, or '' if none."""
        if not CSRF_INPUT_NAME:
            return ""
        try:
            from lxml import html as lxml_html

            doc = lxml_html.fromstring(html)
            nodes = doc.xpath(f'//input[@name="{CSRF_INPUT_NAME}"]/@value')
            return nodes[0] if nodes else ""
        except Exception:
            return ""

    def _login_succeeded(self, resp: httpx.Response) -> bool:
        """Whether the POST landed in an authenticated state.

        HAR: refine against the real response (a redirect to the dashboard, an
        auth cookie being set, or absence of an error banner). The cookie check
        is a sane default for a form-login site.
        """
        if any("session" in c.lower() or "auth" in c.lower()
               for c in self._client.cookies.keys()):
            return True
        return resp.status_code == 200 and "logout" in resp.text.lower()

    # -- search ------------------------------------------------------------- #
    def search(self, track: PlaylistTrack) -> list[SoundeoResult]:
        """Search Soundeo for ``track`` (does not consume download quota)."""
        if not self._logged_in:
            self.login()
        query = f"{track.primary_artist} {track.name}".strip()
        self._throttle()
        try:
            resp = self._client.get(SEARCH_PATH, params={SEARCH_QUERY_PARAM: query})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("soundeo: search failed for %r: %s", query, exc)
            return []
        return self._parse_results(resp.text)

    @staticmethod
    def _parse_results(html: str) -> list[SoundeoResult]:
        """Parse the search-results markup into :class:`SoundeoResult` rows.

        HAR: implement against the real result-row structure (the container
        selector and the per-row id / artist / title / duration / format
        attributes). Returns [] when nothing parses.
        """
        # Placeholder until the HAR pins the markup; kept import-safe and tested
        # via SoundeoClient.pick with constructed SoundeoResult lists.
        return []

    def pick(
        self, track: PlaylistTrack, results: list[SoundeoResult],
        *, threshold: float = 0.62, tolerance: float = 6.0,
    ) -> SoundeoResult | None:
        """Best Soundeo result for ``track``, or None if none is good enough.

        Requires the chosen format to be available and (when both durations are
        known) the lengths to agree within ``tolerance`` seconds, so a wrong
        remix/edit is rejected. Ties broken by closest duration.
        """
        fmt = FORMAT_CODE.get(self.audio_format, self.audio_format)
        best: tuple[float, float, SoundeoResult] | None = None
        for r in results:
            if r.formats and fmt not in r.formats:
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

        Raises :class:`SoundeoQuotaExceeded` when the daily quota is spent and
        :class:`SoundeoError` on other failures, leaving no partial file.
        """
        if not self._logged_in:
            self.login()
        fmt = FORMAT_CODE.get(self.audio_format, self.audio_format)
        url = DOWNLOAD_PATH_TEMPLATE.format(id=result.id, fmt=fmt)
        self._throttle()
        tmp = dest_path + ".part"
        try:
            with self._client.stream("GET", url) as resp:
                if self._is_quota_response(resp):
                    raise SoundeoQuotaExceeded(
                        "daily Soundeo download quota exhausted (resets midnight CET)")
                resp.raise_for_status()
                os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
        except SoundeoQuotaExceeded:
            _unlink_quiet(tmp)
            raise
        except httpx.HTTPError as exc:
            _unlink_quiet(tmp)
            raise SoundeoError(f"download failed: {exc}") from exc
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            _unlink_quiet(tmp)
            raise SoundeoError("download produced no data")
        os.replace(tmp, dest_path)

    @staticmethod
    def _is_quota_response(resp: httpx.Response) -> bool:
        """Detect a quota-exhausted response before reading the body.

        HAR: refine against the real signal (a 402/403, a redirect to an
        upgrade page, or a JSON/HTML error). The content-type heuristic below —
        a non-audio body where an audio download was expected — is a safe
        default that also catches "you hit your limit" HTML pages.
        """
        ctype = resp.headers.get("content-type", "").lower()
        if resp.status_code in (402, 403, 429):
            return True
        if resp.status_code == 200 and ("text/html" in ctype or "application/json" in ctype):
            return True
        return False


def _unlink_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
