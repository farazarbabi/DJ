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
    preferred_download_format,
    _search_title,
    _version_rank,
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

# Same shape, but the account already owns this track: `download downloaded`.
SEARCH_HTML_OWNED = """
<div class="folder">
  <div class="trackitem" data-track-id="10217876">
    <div class="info">
      <strong><a href="/track/gab-rhome-madama-firefly-original-mix-10217876.html">
        Gab Rhome - Madama Firefly (Original Mix)</a></strong>
      <time>7:13</time>
    </div>
    <div class="download downloaded">
      <a href="javascript:void(0);" class="track-download-lnk"
         data-track-id="10217876" data-track-format="1"><span>MP3</span></a>
      <a href="javascript:void(0);" class="track-download-lnk"
         data-track-id="10217876" data-track-format="3"><span>AIFF</span></a>
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


def test_parse_results_sets_downloaded_flag():
    assert _parse_results(SEARCH_HTML)[0].downloaded is False
    owned = _parse_results(SEARCH_HTML_OWNED)[0]
    assert owned.downloaded is True and owned.id == "10217876"


def test_pick_owned_requires_downloaded_flag():
    track = _track("Magna Terram", ["Township Rebellion"])
    not_owned = SoundeoResult("1", "Township Rebellion", "Magna Terram (Original Mix)",
                              505, formats=["aiff"], downloaded=False)
    owned = SoundeoResult("2", "Township Rebellion", "Magna Terram (Original Mix)",
                          505, formats=["aiff"], downloaded=True)
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick_owned(track, [not_owned]) is None      # not owned -> skip
    assert c.pick_owned(track, [not_owned, owned]).id == "2"
    c.close()


def test_pick_owned_matches_on_disk_duration():
    # Two owned cuts; pick_owned restores the one matching the on-disk length.
    track = _track("Magna Terram", ["Township Rebellion"])
    radio = SoundeoResult("r", "Township Rebellion", "Magna Terram (Radio Edit)",
                          190, formats=["aiff"], downloaded=True)
    ext = SoundeoResult("e", "Township Rebellion", "Magna Terram (Extended Mix)",
                        505, formats=["aiff"], downloaded=True)
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick_owned(track, [radio, ext], target_duration=500).id == "e"
    assert c.pick_owned(track, [radio, ext], target_duration=195).id == "r"
    c.close()


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


def test_search_title_strips_generic_version_keeps_remix():
    # Generic cuts are stripped so the search finds whichever cut Soundeo lists.
    assert _search_title("Diclofél - Radio Edit") == "Diclofél"
    assert _search_title("Title (Original Mix)") == "Title"
    assert _search_title("Title - Extended Mix") == "Title"
    # A remixer name is identifying and must stay in the query.
    assert _search_title("Kryptonite - Mateo! Remix") == "Kryptonite - Mateo! Remix"
    # No version -> unchanged.
    assert _search_title("Magna Terram") == "Magna Terram"


def test_search_query_omits_radio_edit(monkeypatch):
    # The failing case: Spotify says "Radio Edit" but Soundeo lists the Original
    # Mix. The search query must drop "Radio Edit" so the Original Mix is found.
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/account/logoreg":
            return (httpx.Response(200, json={"success": True, "header": _LOGGED_IN_HEADER})
                    if request.method == "POST" else httpx.Response(200, json={"success": False}))
        if path == "/search":
            seen["q"] = request.url.params.get("q")
            return httpx.Response(200, json={"success": True, "content": SEARCH_HTML})
        return httpx.Response(404)

    c = _client(handler)
    c.search(_track("Diclofél - Radio Edit", ["Julian Schraven"]))
    c.close()
    assert seen["q"] == "Julian Schraven Diclofél"


def test_version_rank_orders_extended_original_plain_radio():
    assert _version_rank("Title (Extended Mix)") == 0
    assert _version_rank("Title (Original Mix)") == 1
    assert _version_rank("Title") == 2
    assert _version_rank("Title (Radio Edit)") == 3


def test_pick_prefers_original_over_radio_edit():
    # Spotify title is a Radio Edit, but the Original Mix must win when both list.
    track = _track("Diclofél - Radio Edit", ["Julian Schraven"])
    results = [
        SoundeoResult("radio", "Julian Schraven", "Diclofél (Radio Edit)",
                      190, formats=["aiff"]),
        SoundeoResult("orig", "Julian Schraven", "Diclofél (Original Mix)",
                      420, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "orig"
    c.close()


def test_pick_prefers_plain_over_radio_edit():
    track = _track("Diclofél - Radio Edit", ["Julian Schraven"])
    results = [
        SoundeoResult("radio", "Julian Schraven", "Diclofél (Radio Edit)",
                      420, formats=["aiff"]),  # longer, but still the radio cut
        SoundeoResult("plain", "Julian Schraven", "Diclofél", 190, formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "plain"
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


def test_pick_accepts_wav_or_mp3_when_aiff_unavailable():
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [SoundeoResult("9", "Artist", "Some Tune", 200, formats=["mp3", "wav"])]
    c = _client(lambda r: httpx.Response(200, json={}))
    picked = c.pick(track, results)
    assert picked is not None and picked.id == "9"
    assert preferred_download_format(picked, "aiff") == "wav"
    c.close()


def test_pick_prioritizes_soundeo_format_before_version():
    track = _track("Some Tune", ["Artist"], duration=200)
    results = [
        SoundeoResult("mp3_ext", "Artist", "Some Tune (Extended Mix)", 500,
                      formats=["mp3"]),
        SoundeoResult("wav_orig", "Artist", "Some Tune (Original Mix)", 300,
                      formats=["wav"]),
        SoundeoResult("aiff_plain", "Artist", "Some Tune", 200,
                      formats=["aiff"]),
    ]
    c = _client(lambda r: httpx.Response(200, json={}))
    assert c.pick(track, results).id == "aiff_plain"
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


def test_download_uses_requested_source_format(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "sndstatic.com" in request.url.host:
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFFwav")
        path = request.url.path
        if path == "/account/logoreg":
            return (httpx.Response(200, json={"success": True, "header": _LOGGED_IN_HEADER})
                    if request.method == "POST" else httpx.Response(200, json={"success": False}))
        if path.startswith("/download/"):
            seen["path"] = path
            return httpx.Response(200, json={"success": True,
                                             "jsActions": {"redirect": {"url": _CDN_URL}}})
        return httpx.Response(404)

    c = _client(handler)
    dest = tmp_path / "source.wav"
    c.download(SoundeoResult("11858310", "A", "B", formats=["mp3", "wav"]),
               str(dest), audio_format="wav")
    c.close()
    assert seen["path"].endswith("/download/11858310/2")
    assert dest.exists() and dest.read_bytes().startswith(b"RIFF")


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


# --------------------------------------------------------------------------- #
# out-of-credit detection via the downloads counter
# --------------------------------------------------------------------------- #
_HEADER_N = ('<ul class="top-menu"><li id="top-menu-downloads"><a href="/account/logout">'
             '<span id=\'span-downloads\'><span title="Main (will be reset in 5h)">{n}</span>'
             '</span></a></li></ul>')


def test_remaining_downloads_parses_header():
    from dj_tools.soundeo import _remaining_downloads
    assert _remaining_downloads({"header": _HEADER_N.format(n=19)}) == 19
    assert _remaining_downloads({"header": _HEADER_N.format(n=0)}) == 0
    assert _remaining_downloads({"header": "<ul></ul>"}) is None
    assert _remaining_downloads({}) is None


def test_download_no_url_with_zero_credit_is_quota(tmp_path):
    # Login reports 0 left; the download endpoint returns no URL -> QUOTA, not error.
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/account/logoreg":
            hdr = _HEADER_N.format(n=0)
            return (httpx.Response(200, json={"success": True, "header": hdr})
                    if request.method == "POST" else httpx.Response(200, json={"success": False}))
        if path.startswith("/download/"):
            return httpx.Response(200, json={"success": False, "jsActions": {},
                                             "header": _HEADER_N.format(n=0),
                                             "flash": "no download URL returned"})
        return httpx.Response(404)

    c = _client(handler)
    c.login()
    with pytest.raises(SoundeoQuotaExceeded):
        c.download(SoundeoResult("1", "A", "B", formats=["aiff"]), str(tmp_path / "x.aiff"))
    c.close()


def test_download_short_circuits_when_credit_zero(tmp_path):
    # With the counter already at 0, download() must raise quota WITHOUT a request.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"success": True, "header": _HEADER_N.format(n=0)})

    c = _client(handler)
    c.login()
    calls.clear()
    with pytest.raises(SoundeoQuotaExceeded):
        c.download(SoundeoResult("1", "A", "B", formats=["aiff"]), str(tmp_path / "x.aiff"))
    assert calls == []  # no download request was attempted
    c.close()


def test_download_assume_free_bypasses_zero_credit(tmp_path):
    # An owned re-download proceeds even at 0 credit (it costs nothing).
    def handler(request: httpx.Request) -> httpx.Response:
        if "sndstatic.com" in request.url.host:
            return httpx.Response(200, headers={"content-type": "audio/x-aiff"},
                                  content=b"FORM\x00\x00AIFF")
        path = request.url.path
        if path == "/account/logoreg":
            hdr = _HEADER_N.format(n=0)
            return (httpx.Response(200, json={"success": True, "header": hdr})
                    if request.method == "POST" else httpx.Response(200, json={"success": False}))
        if path.startswith("/download/"):
            return httpx.Response(200, json={"success": True,
                                             "jsActions": {"redirect": {"url": _CDN_URL}}})
        return httpx.Response(404)

    c = _client(handler)
    c.login()
    assert c._remaining == 0
    dest = tmp_path / "owned.aiff"
    c.download(SoundeoResult("1", "A", "B", formats=["aiff"], downloaded=True),
               str(dest), assume_free=True)
    assert dest.exists() and dest.read_bytes().startswith(b"FORM")
    c.close()
