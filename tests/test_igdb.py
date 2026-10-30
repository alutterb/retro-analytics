import pytest
import requests

from retro_analytics import igdb

SECRET = "s3cr3t-value"
TOKEN = "t0ken-value"


class Reply:
    def __init__(self, status, body=None):
        self.status_code = status
        self.body = body

    def json(self):
        return self.body


class FakeSession:
    """Answers a token request, then the queued API replies in order; records every post."""

    def __init__(self, *replies, token_reply=None):
        self.replies = list(replies)
        self.token_reply = token_reply or Reply(200, {"access_token": TOKEN, "expires_in": 5000000})
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url == igdb.TOKEN_URL:
            return self.token_reply
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def client(session, clock=None):
    clock = clock or Clock()
    return igdb.Client("client-id", SECRET, session=session, sleep=clock.sleep, clock=clock)


def test_credentials_need_both_variables():
    assert igdb.credentials({"TWITCH_CLIENT_ID": "a", "TWITCH_CLIENT_SECRET": "b"}) == ("a", "b")
    assert igdb.credentials({"TWITCH_CLIENT_ID": "a"}) is None
    assert igdb.credentials({"TWITCH_CLIENT_ID": " ", "TWITCH_CLIENT_SECRET": "b"}) is None
    assert igdb.credentials({}) is None


def test_secret_goes_in_the_token_request_body_not_the_url():
    session = FakeSession(Reply(200, [{"id": 1}]))
    assert client(session).query("games", "fields name;") == [{"id": 1}]
    (token_url, token_args), (api_url, api_args) = session.posts
    assert SECRET not in token_url and token_args["data"]["client_secret"] == SECRET
    assert api_url == "https://api.igdb.com/v4/games"
    assert api_args["headers"] == {"Client-ID": "client-id", "Authorization": f"Bearer {TOKEN}"}
    assert api_args["data"] == b"fields name;"


def test_requests_stay_under_four_per_second():
    clock = Clock()
    session = FakeSession(Reply(200, []), Reply(200, []))
    api = client(session, clock)
    api.query("games", "fields name;")
    api.query("games", "fields name;")
    # Token, then two queries: each waits out the interval since the one before.
    assert clock.sleeps == [igdb.MIN_INTERVAL, igdb.MIN_INTERVAL]


def test_rate_limit_and_network_errors_are_retried():
    clock = Clock()
    session = FakeSession(Reply(429), requests.ConnectionError("reset"), Reply(200, [{"id": 7}]))
    assert client(session, clock).query("games", "fields name;") == [{"id": 7}]
    assert len(session.posts) == 4


def test_a_rejected_request_is_not_retried_and_leaks_no_credential():
    session = FakeSession(Reply(401, {"message": "Authorization Failure"}))
    with pytest.raises(igdb.IgdbError) as raised:
        client(session).query("games", "fields name;")
    assert str(raised.value) == "IGDB games query failed: HTTP 401"
    assert len(session.posts) == 2
    assert SECRET not in str(raised.value) and TOKEN not in str(raised.value)


def test_a_failed_login_says_so_without_the_secret():
    session = FakeSession(token_reply=Reply(400, {"message": "invalid client secret"}))
    with pytest.raises(igdb.IgdbError, match="IGDB login failed: HTTP 400") as raised:
        client(session).query("games", "fields name;")
    assert SECRET not in str(raised.value)


def test_persistent_server_errors_give_up():
    session = FakeSession(*[Reply(503)] * 5)
    with pytest.raises(igdb.IgdbError, match="HTTP 503"):
        client(session).query("games", "fields name;")


def test_query_all_pages_until_a_short_page():
    full = [{"id": i} for i in range(igdb.PAGE)]
    session = FakeSession(Reply(200, full), Reply(200, [{"id": -1}]))
    rows = client(session).query_all("games", "fields name; where platforms = (18);")
    assert len(rows) == igdb.PAGE + 1
    bodies = [args["data"].decode() for url, args in session.posts if url != igdb.TOKEN_URL]
    assert bodies == [
        "fields name; where platforms = (18); limit 500; offset 0;",
        "fields name; where platforms = (18); limit 500; offset 500;",
    ]


def test_a_network_error_message_never_reaches_the_error():
    session = FakeSession(*[requests.ConnectionError(f"connection to host with {SECRET} reset")] * 5)
    with pytest.raises(igdb.IgdbError, match="IGDB games query failed: ConnectionError") as raised:
        client(session).query("games", "fields name;")
    assert SECRET not in str(raised.value)
