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


def fetch_unemployment_values(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Download monthly BLS unemployment-rate observations."""

    if end_year is None:
        end_year = pd.Timestamp.today().year
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    rows = []

    for chunk_start, chunk_end in _year_chunks(start_year, end_year):
        payload = {
            "seriesid": [BLS_SERIES_ID],
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        response = _post_json(BLS_API_URL, payload)

        status = str(response.get("status", "")).upper()
        if status != "REQUEST_SUCCEEDED":
            raise RuntimeError(
                f"BLS API request failed: status={status}, message={response.get('message')}"
            )

        series_list = response.get("Results", {}).get("series", [])
        if not series_list:
            raise RuntimeError("BLS API returned no unemployment series data")

        series = series_list[0]
        if series.get("seriesID") != BLS_SERIES_ID:
            raise RuntimeError(f"Unexpected BLS series: {series.get('seriesID')}")

        for item in series.get("data", []):
            period = str(item.get("period", ""))
            if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                continue

            year = int(item["year"])
            month = int(period[1:])

            # BLS can include placeholder rows for periods whose value has not
            # been published yet (for example value="-").  Those rows are not
            # observations and must be ignored rather than converted to float.
            raw_value = str(item.get("value", "")).strip()
            try:
                unemployment_rate = float(raw_value)
            except (TypeError, ValueError):
                print(
                    "Skipping unpublished/non-numeric BLS observation: "
                    f"{year}-{month:02d} value={raw_value!r}"
                )
                continue

            rows.append(
                {
                    "Reference Month": pd.Period(year=year, month=month, freq="M"),
                    "Unemployment Rate": unemployment_rate,
                }
            )

    if not rows:
        raise RuntimeError("No unemployment observations downloaded from BLS")

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Reference Month"], keep="last")
        .sort_values("Reference Month")
        .reset_index(drop=True)
    )


# ============================================================
# HISTORICAL RELEASE-DATE PARSING
# ============================================================


class _ArchiveParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and re.search(r"/archives/empsit_\d{8}\.htm(?:l)?$", href, re.I):
            self.href = href
            self.text = []

    def handle_data(self, data):
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.href is not None:
            text = " ".join("".join(self.text).split())
            self.links.append((self.href, text))
            self.href = None
            self.text = []


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.cell_text = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_text = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self.in_cell and tag in {"td", "th"}:
            self.row.append(" ".join("".join(self.cell_text).split()))
            self.in_cell = False
            self.cell_text = []
        elif self.in_row and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
            self.row = []


def _parse_reference_month(text: str) -> pd.Period | None:
    match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
        text,
        flags=re.I,
    )
    if not match:
        return None
    month = datetime.strptime(match.group(1)[:3].title(), "%b").month
    return pd.Period(year=int(match.group(2)), month=month, freq="M")


def _parse_archive_date(href: str) -> pd.Timestamp | None:
    match = re.search(r"empsit_(\d{2})(\d{2})(\d{4})\.htm(?:l)?$", href, flags=re.I)
    if not match:
        return None
    return pd.Timestamp(
        year=int(match.group(3)),
        month=int(match.group(1)),
        day=int(match.group(2)),
    )


def _parse_schedule_date(text: str) -> pd.Timestamp | None:
    cleaned = text.replace(".", "").strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return pd.Timestamp(datetime.strptime(cleaned, fmt))
        except ValueError:
            pass
    return None


def fetch_employment_release_dates() -> pd.DataFrame:
    """Map each Employment Situation reference month to its actual release date."""

    rows = []

    # Historical archive. The archive filename contains the real release date.
    parser = _ArchiveParser()
    parser.feed(_get_text(BLS_ARCHIVE_URL))

    for href, text in parser.links:
        reference_month = _parse_reference_month(text)
        release_date = _parse_archive_date(href)
        if reference_month is None or release_date is None:
            continue
        rows.append(
            {
                "Reference Month": reference_month,
                "Release Date": release_date,
                "Release URL": urljoin(BLS_ARCHIVE_URL, href),
                "Release Date Source": "BLS archive",
            }
        )

    # Current schedule, in case the latest report has not yet moved to archive.
    table = _TableParser()
    table.feed(_get_text(BLS_SCHEDULE_URL))

    for cells in table.rows:
        if len(cells) < 2:
            continue
        reference_month = _parse_reference_month(cells[0])
        release_date = _parse_schedule_date(cells[1])
        if reference_month is None or release_date is None:
            continue
        rows.append(
            {
                "Reference Month": reference_month,
                "Release Date": release_date,
                "Release URL": BLS_SCHEDULE_URL,
                "Release Date Source": "BLS schedule",
            }
        )

    if not rows:
        raise RuntimeError("Could not parse any Employment Situation release dates")

    result = pd.DataFrame(rows)
    result["Priority"] = result["Release Date Source"].map(
        {"BLS archive": 0, "BLS schedule": 1}
    ).fillna(9)

    return (
        result
        .sort_values(["Reference Month", "Priority", "Release Date"])
        .drop_duplicates(subset=["Reference Month"], keep="first")
        .drop(columns=["Priority"])
        .sort_values("Reference Month")
        .reset_index(drop=True)
    )


# ============================================================
# COMBINED RELEASE TABLE
# ============================================================


def load_unemployment_releases(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Combine unemployment values with the date each value became available.

    The unemployment 12M MA uses the latest 12 released monthly observations.
    """

    values = fetch_unemployment_values(start_year=start_year, end_year=end_year)
    releases = fetch_employment_release_dates()

    data = values.merge(
        releases,
        on="Reference Month",
        how="inner",
        validate="one_to_one",
    ).sort_values("Release Date").reset_index(drop=True)

    if data.empty:
        raise RuntimeError("No unemployment values matched BLS release dates")

    reference_start = data["Reference Month"].dt.to_timestamp(how="start")
    invalid = data["Release Date"] < reference_start
    if invalid.any():
        bad = data.loc[invalid].iloc[0]
        raise RuntimeError(
            f"Invalid release mapping: {bad['Reference Month']} -> {bad['Release Date']}"
        )

    data["Unemployment 12M MA"] = (
        data["Unemployment Rate"].rolling(window=12, min_periods=12).mean()
    )
    data["Unemployment <= 12M MA"] = (
        data["Unemployment Rate"] <= data["Unemployment 12M MA"]
    )

    return data


# ============================================================
# DAILY AS-OF STATE
# ============================================================


def build_daily_unemployment_state(
    trading_dates,
    start_year: int = DEFAULT_START_YEAR,
) -> pd.DataFrame:
    """
    Map the latest already-published unemployment observation to each trading day.

    Because Strategy 2 runs after the close, a report released that morning is
    considered available for that day's close calculation.
    """

    trading_index = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
    if trading_index.empty:
        raise ValueError("trading_dates cannot be empty")

    releases = (
        load_unemployment_releases(
            start_year=start_year,
            end_year=int(trading_index.max().year),
        )
        .dropna(subset=["Unemployment 12M MA"])
        .copy()
    )

    if releases.empty:
        raise RuntimeError("Not enough unemployment history for a 12M average")

    releases["Release Date"] = pd.to_datetime(releases["Release Date"]).dt.normalize()

    left = pd.DataFrame({"Date": trading_index}).sort_values("Date")
    right = releases[
        [
            "Reference Month",
            "Release Date",
            "Unemployment Rate",
            "Unemployment 12M MA",
            "Unemployment <= 12M MA",
            "Release Date Source",
        ]
    ].sort_values("Release Date")

    mapped = pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="Release Date",
        direction="backward",
        allow_exact_matches=True,
    )

    return mapped.set_index("Date")
