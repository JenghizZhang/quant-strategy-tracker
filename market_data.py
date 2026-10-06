from datetime import timedelta

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


MARKET_CALENDAR = mcal.get_calendar("NYSE")


class LatestSessionUnavailable(RuntimeError):
    pass


def _normalize_daily_index(df):
    """
    Convert Yahoo daily timestamps to plain trading dates.
    """
    df = df.copy()

    idx = pd.DatetimeIndex(df.index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    df.index = idx.normalize()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    return df.sort_index()


def get_latest_completed_session(now=None):
    """
    Find the latest US stock-market session that has
    already officially closed.

    This correctly handles:
    - weekends
    - market holidays
    - early-close days
    """

    if now is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    else:
        now_utc = pd.Timestamp(now)

        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")

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

    schedule = MARKET_CALENDAR.schedule(
        start_date=start_date,
        end_date=end_date,
    )

    if schedule.empty:
        raise LatestSessionUnavailable(
            "No market sessions found."
        )

    completed = schedule[
        schedule["market_close"] <= now_utc
    ]

    if completed.empty:
        raise LatestSessionUnavailable(
            "No completed market session yet."
        )

    session_date = pd.Timestamp(
        completed.index[-1]
    ).normalize()

    session = completed.iloc[-1]

    market_open = pd.Timestamp(
        session["market_open"]
    ).tz_convert("UTC")

    market_close = pd.Timestamp(
        session["market_close"]
    ).tz_convert("UTC")

    return {
        "date": session_date,
        "market_open": market_open,
        "market_close": market_close,
    }


def _download_intraday(ticker, interval):
    """
    Download recent intraday bars.

    prepost=False is intentional, but we ALSO manually
    filter to the official regular session below.
    """

    df = yf.Ticker(ticker).history(
        period="5d",
        interval=interval,
        auto_adjust=True,
        actions=False,
        prepost=False,
    )

    if df.empty:
        return df

    df = df[["Open", "Close"]].copy()

    # Force all intraday timestamps into UTC so that
    # comparisons with official market open/close are exact.
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
    Build today's daily Open/Close from regular-session
    intraday bars only.

    IMPORTANT:

    5m:
        Open  = 09:30 bar Open
        Close = 15:55-16:00 bar Close

    1m:
        Open  = 09:30 bar Open
        Close = 15:59-16:00 bar Close

    On early-close days the expected final bar is calculated
    from the official exchange close time automatically.

    After-hours bars are NEVER used.
    """

    intraday = _download_intraday(
        ticker,
        interval,
    )

    if intraday.empty:
        return None

    market_open = session["market_open"]
    market_close = session["market_close"]

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
            f"Unsupported interval: {interval}"
        )

    expected_first_bar = market_open

    expected_last_bar = (
        market_close - bar_length
    )

    # Double protection:
    # even though prepost=False was requested,
    # manually discard everything outside RTH.
    regular = intraday[
        (intraday.index >= market_open)
        & (intraday.index < market_close)
    ].copy()

    if regular.empty:
        return None

    # We require the actual opening bar.
    if expected_first_bar not in regular.index:
        return None

    # We require the actual closing-session bar.
    # This prevents a 15:30 price, for example,
    # from being mistaken for the closing price.
    if expected_last_bar not in regular.index:
        return None

    first_bar = regular.loc[
        expected_first_bar
    ]

    last_bar = regular.loc[
        expected_last_bar
    ]

    open_price = float(
        first_bar["Open"]
    )

    close_price = float(
        last_bar["Close"]
    )

    if (
        pd.isna(open_price)
        or pd.isna(close_price)
    ):
        return None

    return {
        "Open": open_price,
        "Close": close_price,
        "source": f"{interval}_fallback",
    }


def load_daily_history(
    ticker,
    start_date,
    session=None,
):
    """
    Load historical daily prices and guarantee that the
    latest completed trading session is included.

    Source priority:

        1. Yahoo Daily
        2. Regular-session 5m
        3. Regular-session 1m
        4. Fail safely

    Returns:
        daily_df
        latest_source
        session
    """

    if session is None:
        session = (
            get_latest_completed_session()
        )

    target_date = session["date"]

    # Request through the day AFTER the desired session
    # because Yahoo's `end` parameter is exclusive.
    end_date = (
        target_date
        + pd.Timedelta(days=1)
    )

    print(
        f"Downloading {ticker} daily..."
    )

    daily = yf.Ticker(ticker).history(
        start=start_date,
        end=end_date.strftime(
            "%Y-%m-%d"
        ),
        interval="1d",
        auto_adjust=True,
        actions=False,
        prepost=False,
    )

    if daily.empty:
        raise RuntimeError(
            f"No daily data returned for "
            f"{ticker}"
        )

    if (
        "Open" not in daily.columns
        or "Close" not in daily.columns
    ):
        raise RuntimeError(
            f"Missing Open/Close for "
            f"{ticker}"
        )

    daily = daily[
        ["Open", "Close"]
    ].copy()

    daily = _normalize_daily_index(
        daily
    )

    # --------------------------------------------------
    # 1. DAILY already has the latest completed session
    # --------------------------------------------------
    if target_date in daily.index:
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

    # --------------------------------------------------
    # 2. Try 5-minute RTH fallback
    # --------------------------------------------------
    print(
        f"⚠️ {ticker}: daily missing "
        f"{target_date.date()}, "
        f"trying 5m..."
    )

    fallback = (
        _build_daily_from_intraday(
            ticker,
            session,
            "5m",
        )
    )

    # --------------------------------------------------
    # 3. Try 1-minute RTH fallback
    # --------------------------------------------------
    if fallback is None:
        print(
            f"⚠️ {ticker}: 5m unavailable, "
            f"trying 1m..."
        )

        fallback = (
            _build_daily_from_intraday(
                ticker,
                session,
                "1m",
            )
        )

    # --------------------------------------------------
    # 4. Fail safely
    # --------------------------------------------------
    if fallback is None:
        raise LatestSessionUnavailable(
            f"{ticker}: cannot obtain "
            f"complete regular-session data "
            f"for {target_date.date()} "
            f"from daily, 5m, or 1m."
        )

    # Append a temporary daily row.
    #
    # On the next Action run, once Yahoo Daily contains
    # this trading day, this temporary row is naturally
    # replaced because the entire history is downloaded
    # and recalculated from scratch.
    daily.loc[
        target_date,
        "Open",
    ] = fallback["Open"]

    daily.loc[
        target_date,
        "Close",
    ] = fallback["Close"]

    daily = daily.sort_index()

    print(
        f"✅ {ticker}: "
        f"{target_date.date()} "
        f"source={fallback['source']}"
    )

    return (
        daily,
        fallback["source"],
        session,
    )
