"""Tests for the Soundeo download-source client.

The HTTP layer is exercised with ``httpx.MockTransport`` (no network); the
match/selection logic is deterministic. The HAR-pinned login/search/download
endpoints are still provisional, so these tests target the *mechanism* (CSRF
lift, cookie-based login success, quota detection, streamed write, and the
source-agnostic ``pick`` scoring), not the exact live markup.
"""

import os

import httpx
import pytest

from dj_tools.soundeo import (
    SoundeoAuthError,
    SoundeoClient,
    SoundeoQuotaExceeded,
    SoundeoResult,
    _score,
)
from dj_tools.spotify_fetch import PlaylistTrack


def _track(name, artists, duration=None):
    return PlaylistTrack(name=name, artists=artists, duration_sec=duration)


def _client(handler, **kw):
    return SoundeoClient("u@e.com", "pw", transport=httpx.MockTransport(handler), **kw)


# --------------------------------------------------------------------------- #
# from_env
# --------------------------------------------------------------------------- #
def test_from_env_returns_none_without_credentials(monkeypatch):
    monkeypatch.delenv("SOUNDEO_USER", raising=False)
    monkeypatch.delenv("SOUNDEO_PASS", raising=False)
    assert SoundeoClient.from_env() is None


def test_from_env_builds_client_with_credentials(monkeypatch):
    monkeypatch.setenv("SOUNDEO_USER", "u@e.com")
    monkeypatch.setenv("SOUNDEO_PASS", "secret")
    c = SoundeoClient.from_env()
    assert c is not None and c.user == "u@e.com"
    c.close()


# --------------------------------------------------------------------------- #
# pick / scoring
# --------------------------------------------------------------------------- #
def test_pick_selects_best_artist_title_match():
    track = _track("Kryptonite", ["3 Doors Down"], duration=234)
    results = [
        SoundeoResult("1", "Cristhian Valencia", "Kryptonite", 360, formats=["aiff"]),
        SoundeoResult("2", "3 Doors Down", "Kryptonite", 233, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200))
    pick = c.pick(track, results)
    c.close()
    assert pick is not None and pick.id == "2"  # artist agreement wins


def test_pick_rejects_when_duration_off():
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune", 280, formats=["aiff"])]
    c = _client(lambda r: httpx.Response(200))
    assert c.pick(track, results, tolerance=6.0) is None
    c.close()


def test_pick_skips_when_format_unavailable():
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune", 200, formats=["mp3", "wav"])]
    c = _client(lambda r: httpx.Response(200))  # default audio_format aiff
    assert c.pick(track, results) is None
    c.close()


def test_score_guards_same_title_different_artist():
    track = _track("Kryptonite", ["3 Doors Down"])
    wrong = SoundeoResult("1", "Cristhian Valencia", "Kryptonite")
    # Single shared title token, no artist agreement -> capped below threshold.
    assert _score(track, wrong) <= 0.45


# --------------------------------------------------------------------------- #
# login
# --------------------------------------------------------------------------- #
def test_login_extracts_csrf_and_succeeds_on_auth_cookie():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200, html='<input name="_token" value="abc123">')
        seen["body"] = request.content.decode()
        # Set an auth cookie to signal a successful login.
        return httpx.Response(200, headers={"set-cookie": "session=xyz; Path=/"})

    c = _client(handler)
    c.login()
    c.close()
    assert "abc123" in seen["body"]  # CSRF token round-tripped
    assert c._logged_in


def test_login_raises_on_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, html="<input name='_token' value='t'>")
        return httpx.Response(200, html="<div>Invalid credentials</div>")

    c = _client(handler)
    with pytest.raises(SoundeoAuthError):
        c.login()
    c.close()


# --------------------------------------------------------------------------- #
# download / quota
# --------------------------------------------------------------------------- #
def test_download_writes_audio_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "login" in str(request.url):
            return httpx.Response(200, headers={"set-cookie": "session=x"},
                                  html="<input name='_token' value='t'>")
        return httpx.Response(200, headers={"content-type": "audio/aiff"},
                              content=b"FORM\x00\x00AIFF-bytes")

    c = _client(handler)
    c.login()
    dest = tmp_path / "Artist - Tune.aiff"
    c.download(SoundeoResult("42", "Artist", "Tune", formats=["aiff"]), str(dest))
    c.close()
    assert dest.exists() and dest.read_bytes().startswith(b"FORM")
    assert not (tmp_path / "Artist - Tune.aiff.part").exists()


def test_download_raises_quota_on_html_response(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "login" in str(request.url):
            return httpx.Response(200, headers={"set-cookie": "session=x"},
                                  html="<input name='_token' value='t'>")
        # Quota page: HTML where audio was expected.
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<h1>Daily limit reached</h1>")

    c = _client(handler)
    c.login()
    dest = tmp_path / "Artist - Tune.aiff"
    with pytest.raises(SoundeoQuotaExceeded):
        c.download(SoundeoResult("42", "Artist", "Tune", formats=["aiff"]), str(dest))
    c.close()
    assert not dest.exists()
    assert not (tmp_path / "Artist - Tune.aiff.part").exists()


def test_is_quota_response_codes():
    c = _client(lambda r: httpx.Response(200))
    R = httpx.Response
    assert c._is_quota_response(R(402))
    assert c._is_quota_response(R(429))
    assert c._is_quota_response(R(200, headers={"content-type": "text/html"}))
    assert not c._is_quota_response(R(200, headers={"content-type": "audio/aiff"}))
    c.close()
