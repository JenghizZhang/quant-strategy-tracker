from datetime import timedelta
import math

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


MARKET_CALENDAR = mcal.get_calendar("NYSE")

# 每次运行检查最近多少个交易日
REPAIR_LOOKBACK_SESSIONS = 10

DATA_PROVIDER = "Yahoo Finance via yfinance"


class LatestSessionUnavailable(RuntimeError):
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _valid_price(value):
    """
    Valid market price:
    - numeric
    - finite
    - > 0
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(value)
        and value > 0
    )


def _normalize_daily_index(df):
    """
    Normalize Yahoo daily timestamps to plain dates.
    """

    df = df.copy()

    idx = pd.DatetimeIndex(
        df.index
    )

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    df.index = idx.normalize()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df.sort_index()


def _daily_row_is_valid(
    df,
    date,
):
    """
    Date existing is NOT enough.

    Both Open and Close must be valid.
    """

    if date not in df.index:
        return False

    row = df.loc[date]

    return (
        _valid_price(
            row["Open"]
        )
        and
        _valid_price(
            row["Close"]
        )
    )


# ============================================================
# MARKET CALENDAR
# ============================================================

def get_latest_completed_session(
    now=None,
):
    """
    Return latest officially completed
    US stock market session.

    Handles:
    - weekends
    - holidays
    - early closes
    """

    if now is None:

        now_utc = pd.Timestamp.now(
            tz="UTC"
        )

    else:

        now_utc = pd.Timestamp(
            now
        )

        if now_utc.tzinfo is None:

            now_utc = (
                now_utc.tz_localize(
                    "UTC"
                )
            )

        else:

            now_utc = (
                now_utc.tz_convert(
                    "UTC"
                )
            )

    now_et = now_utc.tz_convert(
        "America/New_York"
    )

    start_date = (
        now_et.date()
        - timedelta(days=20)
    )

    end_date = (
        now_et.date()
        + timedelta(days=1)
    )

    schedule = (
        MARKET_CALENDAR.schedule(
            start_date=start_date,
            end_date=end_date,
        )
    )

    if schedule.empty:

        raise LatestSessionUnavailable(
            "No market sessions found."
        )

    completed = schedule[
        schedule["market_close"]
        <= now_utc
    ]

    if completed.empty:

        raise LatestSessionUnavailable(
            "No completed market "
            "session yet."
        )

    session_date = (
        pd.Timestamp(
            completed.index[-1]
        )
        .normalize()
    )

    row = completed.iloc[-1]

    return {
        "date": session_date,

        "market_open": (
            pd.Timestamp(
                row["market_open"]
            )
            .tz_convert("UTC")
        ),

        "market_close": (
            pd.Timestamp(
                row["market_close"]
            )
            .tz_convert("UTC")
        ),
    }


def _get_recent_completed_sessions(
    latest_session,
    count=REPAIR_LOOKBACK_SESSIONS,
):
    """
    Build official session information
    for the most recent completed sessions.

    These dates MUST exist in daily history.
    """

    target_date = (
        latest_session["date"]
    )

    start_date = (
        target_date
        - pd.Timedelta(days=35)
    )

    schedule = (
        MARKET_CALENDAR.schedule(
            start_date=start_date.date(),
            end_date=target_date.date(),
        )
    )

    if schedule.empty:

        raise LatestSessionUnavailable(
            "Cannot build recent "
            "market schedule."
        )

    schedule = schedule.tail(
        count
    )

    sessions = []

    for date, row in (
        schedule.iterrows()
    ):

        sessions.append(
            {
                "date": (
                    pd.Timestamp(date)
                    .normalize()
                ),

                "market_open": (
                    pd.Timestamp(
                        row["market_open"]
                    )
                    .tz_convert("UTC")
                ),

                "market_close": (
                    pd.Timestamp(
                        row["market_close"]
                    )
                    .tz_convert("UTC")
                ),
            }
        )

    return sessions


# ============================================================
# INTRADAY DOWNLOAD
# ============================================================

def _download_intraday(
    ticker,
    interval,
):
    """
    Download recent intraday history.

    5m:
        use 30 days so recent historical
        missing sessions can be repaired.

    1m:
        use 7 days.

    prepost=False is requested, AND
    official market hours are filtered again.
    """

    if interval == "5m":

        period = "30d"

    elif interval == "1m":

        period = "7d"

    else:

        raise ValueError(
            f"Unsupported interval: "
            f"{interval}"
        )

    df = (
        yf.Ticker(ticker)
        .history(
            period=period,
            interval=interval,
            auto_adjust=True,
            actions=False,
            prepost=False,
        )
    )

    if df.empty:
        return pd.DataFrame()

    if (
        "Open" not in df.columns
        or
        "Close" not in df.columns
    ):
        return pd.DataFrame()

    df = df[
        [
            "Open",
            "Close",
        ]
    ].copy()

    df.index = pd.to_datetime(
        df.index,
        utc=True,
    )

    df = df.sort_index()

    return df


# ============================================================
# BUILD ONE DAILY BAR FROM INTRADAY
# ============================================================

def _build_daily_from_intraday(
    intraday,
    session,
    interval,
):
    """
    Build Open / Close for ONE official session.

    Normal trading day:

    5m:
        Open  = 09:30 bar Open
        Close = 15:55 bar Close

    1m:
        Open  = 09:30 bar Open
        Close = 15:59 bar Close

    Early-close days are handled automatically
    using official market_close.

    Premarket and after-hours are NEVER used.
    """

    if (
        intraday is None
        or intraday.empty
    ):
        return None

    market_open = (
        session["market_open"]
    )

    market_close = (
        session["market_close"]
    )

    if interval == "5m":

        bar_length = (
            pd.Timedelta(
                minutes=5
            )
        )

    elif interval == "1m":

        bar_length = (
            pd.Timedelta(
                minutes=1
            )
        )

    else:

        raise ValueError(
            f"Unsupported interval: "
            f"{interval}"
        )

    expected_first_bar = (
        market_open
    )

    expected_last_bar = (
        market_close
        - bar_length
    )

    # --------------------------------------------------------
    # Strict regular-session filter
    # --------------------------------------------------------

    regular = intraday[
        (
            intraday.index
            >= market_open
        )
        &
        (
            intraday.index
            < market_close
        )
    ].copy()

    if regular.empty:
        return None

    # Must have official opening bar
    if (
        expected_first_bar
        not in regular.index
    ):
        return None

    # Must have official closing bar
    #
    # Do NOT use merely:
    # regular.iloc[-1]
    #
    # because incomplete data could otherwise
    # be mistaken for the close.
    if (
        expected_last_bar
        not in regular.index
    ):
        return None

    first_bar = regular.loc[
        expected_first_bar
    ]

    last_bar = regular.loc[
        expected_last_bar
    ]

    open_price = (
        first_bar["Open"]
    )

    close_price = (
        last_bar["Close"]
    )

    if not (
        _valid_price(open_price)
        and
        _valid_price(close_price)
    ):
        return None

    return {
        "Open": float(
            open_price
        ),

        "Close": float(
            close_price
        ),

        "source": (
            f"{interval}_fallback"
        ),
    }


# ============================================================
# MAIN DATA LOADER
# ============================================================

def load_daily_history(
    ticker,
    start_date,
    session=None,
):
    """
    Download daily history and automatically
    repair recent missing trading sessions.

    Workflow:

        Daily
          ↓
        Compare recent dates with
        official NYSE calendar
          ↓
        Missing / incomplete session
          ↓
        Try 5m regular-session data
          ↓
        Try 1m regular-session data
          ↓
        Still missing
          ↓
        FAIL

    Missing trading days are never silently skipped.
    """

    if session is None:

        session = (
            get_latest_completed_session()
        )

    target_date = (
        session["date"]
    )

    # Yahoo end date is exclusive
    end_date = (
        target_date
        + pd.Timedelta(days=1)
    )

    print(
        f"Downloading "
        f"{ticker} daily..."
    )

    daily = (
        yf.Ticker(ticker)
        .history(
            start=start_date,
            end=end_date.strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=True,
            actions=False,
            prepost=False,
        )
    )

    if daily.empty:

        raise RuntimeError(
            f"No daily data returned "
            f"for {ticker}"
        )

    if (
        "Open" not in daily.columns
        or
        "Close" not in daily.columns
    ):

        raise RuntimeError(
            f"Missing Open/Close "
            f"for {ticker}"
        )

    daily = daily[
        [
            "Open",
            "Close",
        ]
    ].copy()

    daily = (
        _normalize_daily_index(
            daily
        )
    )

    # ========================================================
    # CHECK RECENT OFFICIAL TRADING SESSIONS
    # ========================================================

    recent_sessions = (
        _get_recent_completed_sessions(
            session
        )
    )

    missing_sessions = []

    for recent_session in (
        recent_sessions
    ):

        date = (
            recent_session["date"]
        )

        if not _daily_row_is_valid(
            daily,
            date,
        ):

            missing_sessions.append(
                recent_session
            )

    # ========================================================
    # EVERYTHING COMPLETE
    # ========================================================

    if not missing_sessions:

        print(
            f"✅ {ticker}: "
            f"recent "
            f"{len(recent_sessions)} "
            f"sessions complete"
        )

        latest_source = "daily"

        daily.attrs[
            "provider"
        ] = DATA_PROVIDER

        daily.attrs[
            "latest_source"
        ] = latest_source

        daily.attrs[
            "repaired_sessions"
        ] = []

        return (
            daily,
            latest_source,
            session,
        )

    # ========================================================
    # REPORT GAPS
    # ========================================================

    missing_dates = [
        item["date"].strftime(
            "%Y-%m-%d"
        )
        for item in (
            missing_sessions
        )
    ]

    print(
        f"⚠️ {ticker}: "
        f"missing/incomplete sessions: "
        f"{', '.join(missing_dates)}"
    )

    # Remove invalid daily rows first
    for missing in (
        missing_sessions
    ):

        date = missing[
            "date"
        ]

        if date in daily.index:

            daily = daily.drop(
                index=date
            )

    # ========================================================
    # DOWNLOAD 5m ONCE
    # ========================================================

    print(
        f"⚠️ {ticker}: "
        f"loading 5m repair data..."
    )

    intraday_5m = (
        _download_intraday(
            ticker,
            "5m",
        )
    )

    repaired = []
    unresolved = []

    for missing in (
        missing_sessions
    ):

        date = missing[
            "date"
        ]

        fallback = (
            _build_daily_from_intraday(
                intraday_5m,
                missing,
                "5m",
            )
        )

        if fallback is None:

            unresolved.append(
                missing
            )

            continue

        daily.loc[
            date,
            "Open",
        ] = fallback["Open"]

        daily.loc[
            date,
            "Close",
        ] = fallback["Close"]

        repaired.append(
            {
                "date": (
                    date.strftime(
                        "%Y-%m-%d"
                    )
                ),

                "source": (
                    "5m_fallback"
                ),
            }
        )

        print(
            f"✅ {ticker}: "
            f"repaired "
            f"{date.date()} "
            f"with 5m"
        )

    # ========================================================
    # TRY 1m FOR ANYTHING 5m COULD NOT REPAIR
    # ========================================================

    if unresolved:

        print(
            f"⚠️ {ticker}: "
            f"some sessions could not "
            f"be repaired with 5m; "
            f"loading 1m..."
        )

        intraday_1m = (
            _download_intraday(
                ticker,
                "1m",
            )
        )

        still_unresolved = []

        for missing in unresolved:

            date = missing[
                "date"
            ]

            fallback = (
                _build_daily_from_intraday(
                    intraday_1m,
                    missing,
                    "1m",
                )
            )

            if fallback is None:

                still_unresolved.append(
                    missing
                )

                continue

            daily.loc[
                date,
                "Open",
            ] = fallback["Open"]

            daily.loc[
                date,
                "Close",
            ] = fallback["Close"]

            repaired.append(
                {
                    "date": (
                        date.strftime(
                            "%Y-%m-%d"
                        )
                    ),

                    "source": (
                        "1m_fallback"
                    ),
                }
            )

            print(
                f"✅ {ticker}: "
                f"repaired "
                f"{date.date()} "
                f"with 1m"
            )

        unresolved = (
            still_unresolved
        )

    # ========================================================
    # FAIL IF ANY SESSION IS STILL MISSING
    # ========================================================

    if unresolved:

        dates = [
            item["date"].strftime(
                "%Y-%m-%d"
            )
            for item in unresolved
        ]

        raise LatestSessionUnavailable(
            f"{ticker}: unable to repair "
            f"official trading session(s): "
            f"{', '.join(dates)}. "
            f"Daily, 5m, and 1m "
            f"were all unavailable."
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    daily = daily.sort_index()

    for recent_session in (
        recent_sessions
    ):

        date = (
            recent_session["date"]
        )

        if not _daily_row_is_valid(
            daily,
            date,
        ):

            raise LatestSessionUnavailable(
                f"{ticker}: "
                f"{date.date()} remains "
                f"invalid after repair."
            )

    # ========================================================
    # DETERMINE SOURCE FOR LATEST SESSION
    # ========================================================

    latest_source = "daily"

    for item in repaired:

        if (
            item["date"]
            == target_date.strftime(
                "%Y-%m-%d"
            )
        ):

            latest_source = (
                item["source"]
            )

            break

    # ========================================================
    # ATTACH AUDIT METADATA
    # ========================================================

    daily.attrs[
        "provider"
    ] = DATA_PROVIDER

    daily.attrs[
        "latest_source"
    ] = latest_source

    daily.attrs[
        "repaired_sessions"
    ] = repaired

    print(
        f"✅ {ticker}: "
        f"{target_date.date()} "
        f"latest source="
        f"{latest_source}"
    )

    return (
        daily,
        latest_source,
        session,
    )
