from datetime import timedelta
import math

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


MARKET_CALENDAR = mcal.get_calendar("NYSE")


class LatestSessionUnavailable(RuntimeError):
    pass


# ============================================================
# HELPERS
# ============================================================

def _valid_price(value):
    """
    A usable market price must be:
    - not NaN
    - finite
    - greater than zero
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
    Convert Yahoo daily timestamps to plain
    trading dates.
    """

    df = df.copy()

    idx = pd.DatetimeIndex(
        df.index
    )

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    df.index = idx.normalize()

    # Keep only one row per trading date.
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
    Yahoo can sometimes expose today's date
    before today's daily OHLC row is complete.

    Merely having the date is therefore NOT
    sufficient.

    Both Open and Close must contain valid
    positive finite numbers.
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
    Find the latest US stock-market session
    that has officially completed.

    Handles:
    - weekends
    - holidays
    - normal closes
    - early closes
    """

    if now is None:

        now_utc = pd.Timestamp.now(
            tz="UTC"
        )

    else:

        now_utc = pd.Timestamp(now)

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
        - timedelta(days=10)
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

    latest = (
        completed.iloc[-1]
    )

    market_open = (
        pd.Timestamp(
            latest["market_open"]
        )
        .tz_convert("UTC")
    )

    market_close = (
        pd.Timestamp(
            latest["market_close"]
        )
        .tz_convert("UTC")
    )

    return {
        "date": session_date,
        "market_open": market_open,
        "market_close": market_close,
    }


# ============================================================
# INTRADAY DATA
# ============================================================

def _download_intraday(
    ticker,
    interval,
):
    """
    Download recent Yahoo intraday data.

    prepost=False is requested.

    We ALSO manually restrict timestamps to
    the official regular trading session.
    """

    df = (
        yf.Ticker(ticker)
        .history(
            period="5d",
            interval=interval,
            auto_adjust=True,
            actions=False,
            prepost=False,
        )
    )

    if df.empty:
        return df

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

    # Convert timestamps to UTC.
    df.index = pd.to_datetime(
        df.index,
        utc=True,
    )

    df = df.sort_index()

    return df


def _build_daily_from_intraday(
    ticker,
    session,
    interval,
):
    """
    Reconstruct today's daily Open / Close
    from REGULAR SESSION data only.

    Normal trading day:

    5m:
        Open  = 09:30 bar Open
        Close = 15:55 bar Close

    1m:
        Open  = 09:30 bar Open
        Close = 15:59 bar Close

    Early-close days are handled using the
    official exchange calendar.

    Premarket and after-hours prices are
    never used.
    """

    intraday = (
        _download_intraday(
            ticker,
            interval,
        )
    )

    if intraday.empty:
        return None

    market_open = (
        session["market_open"]
    )

    market_close = (
        session["market_close"]
    )

    if interval == "5m":

        bar_length = pd.Timedelta(
            minutes=5
        )

    elif interval == "1m":

        bar_length = pd.Timedelta(
            minutes=1
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
    # Strictly filter to official regular trading hours.
    #
    # Anything before market open or at/after market close
    # is discarded.
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

    # --------------------------------------------------------
    # Opening bar must exist.
    # --------------------------------------------------------

    if (
        expected_first_bar
        not in regular.index
    ):
        return None

    # --------------------------------------------------------
    # Closing bar must exist.
    #
    # This is important:
    #
    # We never use "the last available bar".
    #
    # We specifically require the bar that ends
    # exactly at the official market close.
    # --------------------------------------------------------

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

    open_price = first_bar[
        "Open"
    ]

    close_price = last_bar[
        "Close"
    ]

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
# DAILY HISTORY
# ============================================================

def load_daily_history(
    ticker,
    start_date,
    session=None,
):
    """
    Download full daily history and guarantee
    that the latest completed market session
    has VALID Open and Close prices.

    Priority:

        1. Valid Yahoo Daily
        2. Complete regular-session 5m
        3. Complete regular-session 1m
        4. Fail safely

    IMPORTANT:

    Yahoo may publish today's date before
    today's Daily OHLC fields are finished.

    Therefore:

        date exists

    does NOT automatically mean:

        Daily data is ready.
    """

    if session is None:

        session = (
            get_latest_completed_session()
        )

    target_date = (
        session["date"]
    )

    # Yahoo `end` is exclusive.
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
            f"No daily data "
            f"returned for "
            f"{ticker}"
        )

    if (
        "Open" not in daily.columns
        or
        "Close" not in daily.columns
    ):

        raise RuntimeError(
            f"Missing Open/Close "
            f"columns for "
            f"{ticker}"
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
    # 1. VALID DAILY DATA
    # ========================================================

    if _daily_row_is_valid(
        daily,
        target_date,
    ):

        print(
            f"✅ {ticker}: "
            f"{target_date.date()} "
            f"source=daily"
        )

        return (
            daily,
            "daily",
            session,
        )

    # ========================================================
    # Daily date exists but row is incomplete
    # ========================================================

    if target_date in daily.index:

        print(
            f"⚠️ {ticker}: "
            f"daily row exists for "
            f"{target_date.date()} "
            f"but Open/Close is "
            f"incomplete."
        )

        # Remove the bad temporary Yahoo row.
        daily = daily.drop(
            index=target_date
        )

    else:

        print(
            f"⚠️ {ticker}: "
            f"daily missing "
            f"{target_date.date()}."
        )

    # ========================================================
    # 2. TRY 5-MINUTE REGULAR SESSION
    # ========================================================

    print(
        f"⚠️ {ticker}: "
        f"trying 5m "
        f"regular-session data..."
    )

    fallback = (
        _build_daily_from_intraday(
            ticker,
            session,
            "5m",
        )
    )

    # ========================================================
    # 3. TRY 1-MINUTE REGULAR SESSION
    # ========================================================

    if fallback is None:

        print(
            f"⚠️ {ticker}: "
            f"5m unavailable or "
            f"incomplete, "
            f"trying 1m..."
        )

        fallback = (
            _build_daily_from_intraday(
                ticker,
                session,
                "1m",
            )
        )

    # ========================================================
    # 4. FAIL SAFELY
    # ========================================================

    if fallback is None:

        raise LatestSessionUnavailable(
            f"{ticker}: cannot obtain "
            f"complete regular-session "
            f"data for "
            f"{target_date.date()} "
            f"from daily, 5m, or 1m."
        )

    # ========================================================
    # Append temporary reconstructed daily row.
    # ========================================================

    daily.loc[
        target_date,
        "Open",
    ] = fallback["Open"]

    daily.loc[
        target_date,
        "Close",
    ] = fallback["Close"]

    daily = (
        daily.sort_index()
    )

    # ========================================================
    # Final safety check
    # ========================================================

    if not _daily_row_is_valid(
        daily,
        target_date,
    ):

        raise LatestSessionUnavailable(
            f"{ticker}: reconstructed "
            f"{target_date.date()} "
            f"still has invalid "
            f"Open/Close data."
        )

    print(
        f"✅ {ticker}: "
        f"{target_date.date()} "
        f"source="
        f"{fallback['source']}"
    )

    return (
        daily,
        fallback["source"],
        session,
    )
