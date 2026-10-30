"""Query IGDB (the Internet Game Database) for user ratings.

IGDB is free for non-commercial use through a Twitch application. Its
client id and secret come from the ``TWITCH_CLIENT_ID`` and
``TWITCH_CLIENT_SECRET`` environment variables and are only ever sent in
request bodies and headers: never in a URL, a log line or an error message,
and the access token lives in memory for one run. Requests are spaced to stay
under IGDB's limit of 4 per second.
"""

import os
import time

import requests

from retro_analytics import PipelineError

TOKEN_URL = 'https://id.twitch.tv/oauth2/token'
API_URL = 'https://api.igdb.com/v4/'
PAGE = 500  # IGDB's largest page
MIN_INTERVAL = 0.26  # seconds between requests: just under 4 per second
RETRY_STATUS = (429, 500, 502, 503, 504)


class IgdbError(PipelineError):
    """IGDB could not be queried; the message names the request, never a credential."""


def credentials(environ=os.environ):
    """(client id, secret) from the environment, or None when either is missing."""
    client_id = environ.get('TWITCH_CLIENT_ID', '').strip()
    secret = environ.get('TWITCH_CLIENT_SECRET', '').strip()
    return (client_id, secret) if client_id and secret else None


class Client:
    """Throttled IGDB queries, with retries and backoff on rate limiting and server trouble."""

    def __init__(self, client_id, secret, session=None, timeout=(10, 30), retries=4,
                 sleep=time.sleep, clock=time.monotonic):
        self._client_id = client_id
        self._secret = secret
        self._token = None
        self.session = session or requests.Session()
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.clock = clock
        self._last_request = None

    def _post(self, url, what, **kwargs):
        for attempt in range(self.retries + 1):
            if self._last_request is not None:
                wait = MIN_INTERVAL - (self.clock() - self._last_request)
                if wait > 0:
                    self.sleep(wait)
            try:
                response = self.session.post(url, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                # The exception text is left out: only this module's own words reach the log.
                problem, response = type(e).__name__, None
            finally:
                self._last_request = self.clock()
            if response is not None:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError:
                        raise IgdbError(f"{what} failed: the reply is not JSON") from None
                problem = f"HTTP {response.status_code}"
                if response.status_code not in RETRY_STATUS:
                    break
            if attempt < self.retries:
                self.sleep(2 ** attempt)
        raise IgdbError(f"{what} failed: {problem}")

    def _authorization(self):
        if self._token is None:
            reply = self._post(TOKEN_URL, "IGDB login", data={
                'client_id': self._client_id,
                'client_secret': self._secret,
                'grant_type': 'client_credentials',
            })
            self._token = reply.get('access_token') if isinstance(reply, dict) else None
            if not self._token:
                raise IgdbError("IGDB login failed: no access token in the reply")
        return {'Client-ID': self._client_id, 'Authorization': f'Bearer {self._token}'}

    def query(self, endpoint, body):
        """One Apicalypse query, such as ('games', 'fields name; where id = (1, 2);')."""
        return self._post(API_URL + endpoint, f"IGDB {endpoint} query",
                          headers=self._authorization(), data=body.encode('utf-8'))

    def query_all(self, endpoint, body):
        """Every row of a query, a page at a time; `body` must not set limit or offset."""
        rows = []
        while True:
            page = self.query(endpoint, f"{body} limit {PAGE}; offset {len(rows)};")
            rows.extend(page)
            if len(page) < PAGE:
                return rows
