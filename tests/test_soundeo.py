"""Tests for the Soundeo download-source client.

The HTTP layer is exercised with ``httpx.MockTransport`` (no network) against
the JSON-envelope flow captured from a real session: login -> search (HTML in
``content``) -> two-step download (JSON ``jsActions.redirect.url`` then the CDN
file). Parsing and match selection are deterministic.
"""

import httpx
import pytest

from dj_tools.soundeo import (
    SoundeoAuthError,
    SoundeoClient,
    SoundeoError,
    SoundeoQuotaExceeded,
    SoundeoResult,
    _flash_text,
    _looks_like_quota,
    _parse_mmss,
    _parse_results,
    _score,
)
from dj_tools.spotify_fetch import PlaylistTrack

# A structurally faithful slice of a real /search `content` fragment.
SEARCH_HTML = """
<div class="folder">
  <div class="trackitem" data-track-id="11858310">
    <div class="info">
      <strong><a href="/track/township-rebellion-magna-terram-original-mix-11858310.html">
        Township Rebellion - Magna Terram (Original Mix)</a></strong>
      <time>8:25</time>
    </div>
    <div class="download">
      <a href="javascript:void(0);" class="track-download-lnk"
         data-track-id="11858310" data-track-format="1"><span>MP3</span></a>
      <a href="javascript:void(0);" class="track-download-lnk"
         data-track-id="11858310" data-track-format="3"><span>AIFF</span></a>
    </div>
  </div>
</div>
"""

_LOGGED_IN_HEADER = '<ul class="top-menu"><a href="/account/logout">logout</a></ul>'
_CDN_URL = "https://dl1.sndstatic.com/cdownload/2019-08-24/11858310.aiff?downloadToken=tok"


def _track(name, artists, duration=None):
    return PlaylistTrack(name=name, artists=artists, duration_sec=duration)


def _client(handler, **kw):
    return SoundeoClient("u@e.com", "pw", transport=httpx.MockTransport(handler), **kw)


def _full_flow_handler(*, download_json=None, cdn_ctype="audio/x-aiff"):
    """A handler implementing login + search + two-step download."""
    dl = download_json or {"success": True,
                           "jsActions": {"redirect": {"url": _CDN_URL}}}

    def handler(request: httpx.Request) -> httpx.Response:
        url = request.url
        if "sndstatic.com" in url.host:
            return httpx.Response(200, headers={"content-type": cdn_ctype},
                                  content=b"FORM\x00\x00AIFF-bytes")
        path = url.path
        if path == "/account/logoreg":
            if request.method == "GET":
                return httpx.Response(200, json={"success": False, "content": "<form/>"})
            return httpx.Response(200, json={"success": True, "header": _LOGGED_IN_HEADER})
        if path == "/search":
            return httpx.Response(200, json={"success": True, "content": SEARCH_HTML})
        if path.startswith("/download/"):
            return httpx.Response(200, json=dl)
        return httpx.Response(404, json={"success": False})

    return handler


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_parse_mmss():
    assert _parse_mmss("8:25") == 505
    assert _parse_mmss("1:02:03") == 3723
    assert _parse_mmss("nope") is None


def test_parse_results_extracts_row():
    rows = _parse_results(SEARCH_HTML)
    assert len(rows) == 1
    r = rows[0]
    assert r.id == "11858310"
    assert r.artist == "Township Rebellion"
    assert r.title == "Magna Terram (Original Mix)"
    assert r.duration_sec == 505
    assert r.formats == ["mp3", "aiff"]  # codes 1 and 3


def test_parse_results_empty():
    assert _parse_results("") == []


def test_looks_like_quota_and_flash_text():
    assert _looks_like_quota("Your daily download limit has been reached")
    assert not _looks_like_quota("Track temporarily unavailable")
    assert _flash_text({"flash": "msg one", "alerts": "msg two"}) == "msg one msg two"


# --------------------------------------------------------------------------- #
# from_env
# --------------------------------------------------------------------------- #
def test_from_env_returns_none_without_credentials(monkeypatch):
    monkeypatch.setenv("SOUNDEO_USER", "")
    monkeypatch.setenv("SOUNDEO_PASS", "")
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
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "2"  # artist agreement + duration
    c.close()


def test_pick_keeps_extended_despite_longer_duration():
    # The Extended Mix runs far longer than Spotify's cut; must NOT be rejected.
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune (Extended Mix)", 360,
                             formats=["aiff"])]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "9"
    c.close()


def test_pick_prefers_extended_over_original():
    track = _track("Magna Terram", ["Township Rebellion"], duration=200)
    results = [
        SoundeoResult("orig", "Township Rebellion", "Magna Terram (Original Mix)",
                      200, formats=["aiff"]),
        SoundeoResult("ext", "Township Rebellion", "Magna Terram (Extended Mix)",
                      505, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "ext"
    c.close()


def test_pick_falls_back_to_original_without_extended():
    track = _track("Magna Terram", ["Township Rebellion"])
    results = [SoundeoResult("orig", "Township Rebellion",
                             "Magna Terram (Original Mix)", 200, formats=["aiff"])]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "orig"
    c.close()


def test_pick_excludes_true_remix_for_untagged_track():
    # Spotify lists the original (no suffix); a real remix must not be chosen.
    track = _track("Magna Terram", ["Township Rebellion"])
    results = [
        SoundeoResult("rmx", "Township Rebellion",
                      "Magna Terram (Henry Saiz Remix)", 400, formats=["aiff"]),
        SoundeoResult("orig", "Township Rebellion",
                      "Magna Terram (Original Mix)", 200, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "orig"
    c.close()


def test_pick_matches_requested_remix():
    # When the Spotify title names a remix, that remix should match.
    track = _track("Kryptonite - Mateo! Remix", ["Cristhian Valencia"])
    results = [
        SoundeoResult("orig", "Cristhian Valencia", "Kryptonite (Original Mix)",
                      200, formats=["aiff"]),
        SoundeoResult("rmx", "Cristhian Valencia", "Kryptonite (Mateo! Remix)",
                      240, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "rmx"
    c.close()


def test_pick_skips_when_aiff_unavailable():
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune", 200, formats=["mp3", "wav"])]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results) is None
    c.close()


def test_score_guards_same_title_different_artist():
    track = _track("Kryptonite", ["3 Doors Down"])
    wrong = SoundeoResult("1", "Cristhian Valencia", "Kryptonite")
    assert _score(track, wrong) <= 0.45


# --------------------------------------------------------------------------- #
# login
# --------------------------------------------------------------------------- #
def test_login_succeeds_on_success_flag_and_logout_header():
    c = _client(_full_flow_handler())
    c.login()
    assert c._logged_in
    c.close()


def test_login_raises_when_success_false():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"success": False})
        return httpx.Response(200, json={"success": False, "header": "<a>login</a>"})

    c = _client(handler)
    with pytest.raises(SoundeoAuthError):
        c.login()
    c.close()


# --------------------------------------------------------------------------- #
# search + download (end to end via MockTransport)
# --------------------------------------------------------------------------- #
def test_search_parses_results():
    c = _client(_full_flow_handler())
    results = c.search(_track("Magna Terram", ["Township Rebellion"]))
    c.close()
    assert len(results) == 1 and results[0].id == "11858310"


def test_download_two_step_writes_file(tmp_path):
    c = _client(_full_flow_handler())
    dest = tmp_path / "Township Rebellion - Magna Terram.aiff"
    c.download(SoundeoResult("11858310", "Township Rebellion", "Magna Terram",
                             505, formats=["aiff"]), str(dest))
    c.close()
    assert dest.exists() and dest.read_bytes().startswith(b"FORM")
    assert not (tmp_path / (dest.name + ".part")).exists()


def test_download_raises_quota_when_limit_message(tmp_path):
    handler = _full_flow_handler(download_json={
        "success": False, "jsActions": {},
        "flash": "Your daily download limit has been reached",
    })
    c = _client(handler)
    dest = tmp_path / "x.aiff"
    with pytest.raises(SoundeoQuotaExceeded):
        c.download(SoundeoResult("1", "A", "B", formats=["aiff"]), str(dest))
    c.close()
    assert not dest.exists()


def test_download_raises_error_on_other_failure(tmp_path):
    handler = _full_flow_handler(download_json={
        "success": False, "jsActions": {}, "flash": "Track not available",
    })
    c = _client(handler)
    with pytest.raises(SoundeoError) as ei:
        c.download(SoundeoResult("1", "A", "B", formats=["aiff"]), str(tmp_path / "x.aiff"))
    assert not isinstance(ei.value, SoundeoQuotaExceeded)
    c.close()


def test_download_rejects_non_audio_cdn_body(tmp_path):
    handler = _full_flow_handler(cdn_ctype="text/html")  # CDN served an error page
    c = _client(handler)
    with pytest.raises(SoundeoError):
        c.download(SoundeoResult("1", "A", "B", formats=["aiff"]), str(tmp_path / "x.aiff"))
    c.close()
    assert not (tmp_path / "x.aiff").exists()


def test_download_404_raises_soundeo_error_not_quota(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "sndstatic.com" in request.url.host:
            return httpx.Response(200, headers={"content-type": "audio/x-aiff"}, content=b"x")
        path = request.url.path
        if path == "/account/logoreg":
            return (httpx.Response(200, json={"success": True, "header": _LOGGED_IN_HEADER})
                    if request.method == "POST" else httpx.Response(200, json={"success": False}))
        if path.startswith("/download/"):
            return httpx.Response(404, json={"success": False})
        return httpx.Response(404)

    c = _client(handler)
    c.login()
    with pytest.raises(SoundeoError) as ei:
        c.download(SoundeoResult("14272211", "A", "B", formats=["aiff"]), str(tmp_path / "x.aiff"))
    assert not isinstance(ei.value, SoundeoQuotaExceeded)  # 404 != quota
    c.close()
    assert not (tmp_path / "x.aiff").exists()


def test_pick_skips_when_formats_unknown():
    # No parsed download links (e.g. vote-required/upcoming) -> not picked.
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune", 200, formats=[])]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results) is None
    c.close()
