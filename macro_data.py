from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


# ============================================================
# Strategy 2 macro data
# ============================================================
# BLS series: LNS14000000
# Civilian unemployment rate, seasonally adjusted.
#
# Anti-lookahead rule:
# A month's unemployment value is usable only from its actual
# Employment Situation release date onward.
#
# Note:
# BLS API returns the currently published historical series, so older
# values may include later seasonal revisions. Release timing is still
# point-in-time safe. If we later want fully vintage-correct values,
# we can replace only this data loader with ALFRED vintages.
# ============================================================

BLS_SERIES_ID = "LNS14000000"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_ARCHIVE_URL = "https://www.bls.gov/bls/news-release/empsit.htm"
BLS_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"

DEFAULT_START_YEAR = 2008
HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
USER_AGENT = "quant-strategy-tracker/1.0 (public BLS data)"


# ============================================================
# HTTP
# ============================================================


def _request_bytes(request: Request) -> bytes:
    last_error = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                time.sleep(2 * attempt)
    raise RuntimeError(f"BLS download failed after {HTTP_RETRIES} attempts: {last_error}")


def _get_text(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        method="GET",
    )
    return _request_bytes(request).decode("utf-8", errors="replace")


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    return json.loads(_request_bytes(request).decode("utf-8", errors="replace"))


# ============================================================
# UNEMPLOYMENT VALUES
# ============================================================


def _year_chunks(start_year: int, end_year: int, size: int = 10):
    year = start_year
    while year <= end_year:
        chunk_end = min(year + size - 1, end_year)
        yield year, chunk_end
        year = chunk_end + 1

