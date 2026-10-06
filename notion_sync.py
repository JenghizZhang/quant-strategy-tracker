import json
import math
import os
import time
import urllib.error
import urllib.request

import pandas as pd
import pandas_market_calendars as mcal


# ============================================================
# CONFIG
# ============================================================

NOTION_TOKEN = os.environ["NOTION_TOKEN"]

STRATEGY1_STATUS_DATA_SOURCE_ID = os.environ[
    "STRATEGY1_STATUS_DATA_SOURCE_ID"
]

STRATEGY1_MONTHLY_DATA_SOURCE_ID = os.environ[
    "STRATEGY1_MONTHLY_DATA_SOURCE_ID"
]

NOTION_VERSION = "2025-09-03"

SUMMARY_FILE = "output/strategy1_summary.json"
MONTHLY_FILE = "output/strategy1_monthly.csv"

NYSE = mcal.get_calendar("NYSE")

WRITE_DELAY_SECONDS = 0.35


# ============================================================
# PERFORMANCE PRECISION
#
# Notion Percent stores decimal values:
#
# 0.0221 = 2.21%
#
# So keeping 4 decimal places in the stored decimal
# gives us 2 decimal places in displayed percentage.
# ============================================================

PERFORMANCE_STORAGE_DECIMALS = 6

NUMBER_TOLERANCE = 1e-12


# ============================================================
# NOTION API
# ============================================================

def notion_request(
    method,
    path,
    body=None,
):
    """
    Make a Notion API request.

    Never prints the token.
    """

    url = (
        "https://api.notion.com/v1/"
        + path
    )

    data = None

    if body is not None:
        data = json.dumps(
            body
        ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": (
                f"Bearer {NOTION_TOKEN}"
            ),
            "Notion-Version": (
                NOTION_VERSION
            ),
            "Content-Type": (
                "application/json"
            ),
        },
    )

    try:

        with urllib.request.urlopen(
            request
        ) as response:

            text = (
                response
                .read()
                .decode("utf-8")
            )

            if not text:
                return {}

            return json.loads(
                text
            )

    except urllib.error.HTTPError as e:

        error_body = (
            e.read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            f"Notion API failed: "
            f"HTTP {e.code} "
            f"{e.reason}\n"
            f"{error_body}"
        ) from e


def query_all_pages(
    data_source_id,
):
    """
    Query all rows from a Notion data source.

    Handles pagination automatically.
    """

    pages = []

    start_cursor = None

    while True:

        body = {
            "page_size": 100,
        }

        if (
            start_cursor
            is not None
        ):
            body[
                "start_cursor"
            ] = start_cursor

        result = notion_request(
            "POST",
            (
                "data_sources/"
                f"{data_source_id}"
                "/query"
            ),
            body,
        )

        pages.extend(
            result.get(
                "results",
                [],
            )
        )

        if not result.get(
            "has_more",
            False,
        ):
            break

        start_cursor = result.get(
            "next_cursor"
        )

        if not start_cursor:
            break

    return pages


# ============================================================
# PROPERTY BUILDERS
# ============================================================

def notion_title(
    value,
):
    return {
        "title": [
            {
                "type": "text",
                "text": {
                    "content": str(
                        value
                    )
                },
            }
        ]
    }


def notion_text(
    value,
):
    return {
        "rich_text": [
            {
                "type": "text",
                "text": {
                    "content": str(
                        value
                    )
                },
            }
        ]
    }


def notion_select(
    value,
):
    return {
        "select": {
            "name": str(
                value
            )
        }
    }


def notion_number(
    value,
):
    """
    General number.

    Keep full precision for:
    - NDX Close
    - MA30
    - MA50
    - MA200
    """

    return {
        "number": float(
            value
        )
    }


def normalize_performance(
    value,
):
    """
    Normalize a return before writing/comparing.

    Example:

        0.02214202312929
        ->
        0.0221

    Notion displays:

        2.21%
    """

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"Invalid performance value: "
            f"{value}"
        )

    return round(
        value,
        PERFORMANCE_STORAGE_DECIMALS,
    )


def notion_performance_number(
    value,
):
    return {
        "number": (
            normalize_performance(
                value
            )
        )
    }


def notion_checkbox(
    value,
):
    return {
        "checkbox": bool(
            value
        )
    }


def notion_date(
    value,
):
    return {
        "date": {
            "start": str(
                value
            )
        }
    }


# ============================================================
# PROPERTY READERS
# ============================================================

def get_title_value(
    page,
    property_name,
):
    prop = (
        page
        .get(
            "properties",
            {},
        )
        .get(
            property_name,
            {},
        )
    )

    title = prop.get(
        "title",
        [],
    )

    if not title:
        return None

    parts = []

    for item in title:

        plain_text = item.get(
            "plain_text"
        )

        if (
            plain_text
            is not None
        ):
            parts.append(
                plain_text
            )

    return "".join(
        parts
    )


def get_number_value(
    page,
    property_name,
):
    prop = (
        page
        .get(
            "properties",
            {},
        )
        .get(
            property_name,
            {},
        )
    )

    return prop.get(
        "number"
    )


def get_date_value(
    page,
    property_name,
):
    prop = (
        page
        .get(
            "properties",
            {},
        )
        .get(
            property_name,
            {},
        )
    )

    date = prop.get(
        "date"
    )

    if not date:
        return None

    return date.get(
        "start"
    )


# ============================================================
# VALUE COMPARISON
# ============================================================

def numbers_equal(
    current,
    expected,
):
    """
    expected is already normalized.

    This forces old long-decimal Notion values
    to be rewritten once.

    After that, repeated runs remain stable.
    """

    if current is None:
        return False

    try:

        current = float(
            current
        )

        expected = float(
            expected
        )

    except (
        TypeError,
        ValueError,
    ):
        return False

    if (
        not math.isfinite(
            current
        )
        or
        not math.isfinite(
            expected
        )
    ):
        return False

    return math.isclose(
        current,
        expected,
        rel_tol=0,
        abs_tol=NUMBER_TOLERANCE,
    )


def dates_equal(
    current,
    expected,
):
    if current is None:
        return False

    try:

        current_date = (
            pd.Timestamp(
                current
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        expected_date = (
            pd.Timestamp(
                expected
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        return (
            current_date
            == expected_date
        )

    except Exception:
        return False


# ============================================================
# CURRENT STATUS
# ============================================================

def find_current_status_page():
    """
    Find:

        Name = Strategy 1
    """

    result = notion_request(
        "POST",
        (
            "data_sources/"
            f"{STRATEGY1_STATUS_DATA_SOURCE_ID}"
            "/query"
        ),
        {
            "filter": {
                "property": "Name",
                "title": {
                    "equals": (
                        "Strategy 1"
                    )
                },
            },
            "page_size": 10,
        },
    )

    pages = result.get(
        "results",
        [],
    )

    if len(pages) == 0:

        raise RuntimeError(
            "Could not find "
            "'Strategy 1' row in "
            "Current Status."
        )

    if len(pages) > 1:

        raise RuntimeError(
            "More than one row named "
            "'Strategy 1' was found "
            "in Current Status."
        )

    return pages[0][
        "id"
    ]


# ============================================================
# NEXT TRADING DAY
# ============================================================

def get_next_trading_day(
    signal_date,
):
    """
    Signal:
        today's close

    Execution:
        next actual NYSE trading-day open
    """

    signal_date = (
        pd.Timestamp(
            signal_date
        )
        .normalize()
    )

    start_date = (
        signal_date
        + pd.Timedelta(
            days=1
        )
    )

    end_date = (
        signal_date
        + pd.Timedelta(
            days=14
        )
    )

    schedule = NYSE.schedule(
        start_date=(
            start_date.date()
        ),
        end_date=(
            end_date.date()
        ),
    )

    if schedule.empty:

        raise RuntimeError(
            "Could not determine "
            "next trading day."
        )

    return (
        pd.Timestamp(
            schedule.index[0]
        )
        .strftime(
            "%Y-%m-%d"
        )
    )


# ============================================================
# DATA SOURCE DISPLAY
# ============================================================

def short_source_name(
    source,
):
    mapping = {
        "daily": "daily",
        "5m_fallback": "5m",
        "1m_fallback": "1m",
    }

    return mapping.get(
        source,
        source,
    )


def build_data_source_text(
    data_sources,
):
    """
    Examples:

    Yahoo | NDX/SPX: daily | QQQ/SPY/AGG: 5m

    Yahoo | NDX/SPX/QQQ/SPY/AGG: daily
    """

    order = [
        "NDX",
        "SPX",
        "QQQ",
        "SPY",
        "AGG",
    ]

    grouped = {}

    for ticker in order:

        source = (
            short_source_name(
                data_sources[
                    ticker
                ]
            )
        )

        grouped.setdefault(
            source,
            [],
        ).append(
            ticker
        )

    parts = [
        "Yahoo"
    ]

    for (
        source,
        tickers,
    ) in grouped.items():

        names = "/".join(
            tickers
        )

        parts.append(
            f"{names}: {source}"
        )

    return " | ".join(
        parts
    )


# ============================================================
# LOAD STRATEGY OUTPUT
# ============================================================

def load_strategy_output():

    with open(
        SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(
            f
        )

    if (
        "current_status"
        not in payload
    ):

        raise RuntimeError(
            "strategy1_summary.json "
            "does not contain "
            "current_status."
        )

    return payload[
        "current_status"
    ]


# ============================================================
# SYNC CURRENT STATUS
# ============================================================

def sync_current_status():

    status = (
        load_strategy_output()
    )

    signal_date = status[
        "Signal Date"
    ]

    execute_date = (
        get_next_trading_day(
            signal_date
        )
    )

    data_source_text = (
        build_data_source_text(
            status[
                "Data Sources"
            ]
        )
    )

    page_id = (
        find_current_status_page()
    )

    last_updated = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    properties = {

        "Current Holding": (
            notion_select(
                status[
                    "Current Holding"
                ]
            )
        ),

        "Today's Signal": (
            notion_select(
                status[
                    "Today's Signal"
                ]
            )
        ),

        "Next Holding": (
            notion_select(
                status[
                    "Next Holding"
                ]
            )
        ),

        "Signal Date": (
            notion_date(
                signal_date
            )
        ),

        "Execute Date": (
            notion_date(
                execute_date
            )
        ),

        "NDX Close": (
            notion_number(
                status[
                    "NDX Close"
                ]
            )
        ),

        "NDX MA30": (
            notion_number(
                status[
                    "NDX MA30"
                ]
            )
        ),

        "NDX > MA30": (
            notion_checkbox(
                status[
                    "NDX > MA30"
                ]
            )
        ),

        "SPX MA50": (
            notion_number(
                status[
                    "SPX MA50"
                ]
            )
        ),

        "SPX MA200": (
            notion_number(
                status[
                    "SPX MA200"
                ]
            )
        ),

        "SPX MA50 > MA200": (
            notion_checkbox(
                status[
                    "SPX MA50 > MA200"
                ]
            )
        ),

        "Reason": (
            notion_text(
                status[
                    "Reason"
                ]
            )
        ),

        "Last Updated": (
            notion_date(
                last_updated
            )
        ),

        "Data Source": (
            notion_text(
                data_source_text
            )
        ),
    }

    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": (
                properties
            )
        },
    )

    print()
    print("=" * 70)
    print(
        "NOTION CURRENT STATUS"
    )
    print("=" * 70)

    print(
        "Name            :",
        "Strategy 1",
    )

    print(
        "Signal Date     :",
        signal_date,
    )

    print(
        "Execute Date    :",
        execute_date,
    )

    print(
        "Current Holding :",
        status[
            "Current Holding"
        ],
    )

    print(
        "Today's Signal  :",
        status[
            "Today's Signal"
        ],
    )

    print(
        "Next Holding    :",
        status[
            "Next Holding"
        ],
    )

    print(
        "Data Source     :",
        data_source_text,
    )

    print()

    print(
        "✅ Strategy 1 Current Status "
        "updated in Notion"
    )


# ============================================================
# LOAD MONTHLY PERFORMANCE
# ============================================================

def load_monthly_results():
    """
    Read:

        output/strategy1_monthly.csv

    Expected columns:

        Month
        Strategy 1
        QQQ
        SPY
    """

    monthly = pd.read_csv(
        MONTHLY_FILE,
        dtype={
            "Month": str,
        },
    )

    required_columns = {
        "Month",
        "Strategy 1",
        "QQQ",
        "SPY",
    }

    missing = (
        required_columns
        - set(
            monthly.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Monthly CSV missing "
            "required column(s): "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    monthly = monthly[
        [
            "Month",
            "Strategy 1",
            "QQQ",
            "SPY",
        ]
    ].copy()

    monthly = (
        monthly.sort_values(
            "Month"
        )
    )

    for column in [
        "Strategy 1",
        "QQQ",
        "SPY",
    ]:

        if monthly[
            column
        ].isna().any():

            bad_months = (
                monthly.loc[
                    monthly[
                        column
                    ].isna(),
                    "Month",
                ]
                .tolist()
            )

            raise RuntimeError(
                f"Monthly CSV contains "
                f"NaN in {column}: "
                f"{bad_months}"
            )

    # --------------------------------------------------------
    # Normalize values only for Notion.
    #
    # CSV remains full precision.
    # --------------------------------------------------------

    for column in [
        "Strategy 1",
        "QQQ",
        "SPY",
    ]:

        monthly[
            column
        ] = (
            monthly[
                column
            ]
            .map(
                normalize_performance
            )
        )

    return monthly


# ============================================================
# EXISTING MONTH ROWS
# ============================================================

def get_existing_month_rows():
    """
    Return:

        {
            "2026-10": page,
            "2026-09": page,
            ...
        }

    Duplicate month titles are treated
    as an error.
    """

    pages = (
        query_all_pages(
            STRATEGY1_MONTHLY_DATA_SOURCE_ID
        )
    )

    result = {}

    for page in pages:

        month = (
            get_title_value(
                page,
                "Month",
            )
        )

        if not month:
            continue

        if month in result:

            raise RuntimeError(
                "Duplicate Monthly "
                "Performance row found: "
                f"{month}"
            )

        result[
            month
        ] = page

    return result


# ============================================================
# CREATE MONTH ROW
# ============================================================

def create_month_row(
    month,
    month_start,
    strategy_return,
    qqq_return,
    spy_return,
):

    return notion_request(
        "POST",
        "pages",
        {
            "parent": {
                "type": (
                    "data_source_id"
                ),
                "data_source_id": (
                    STRATEGY1_MONTHLY_DATA_SOURCE_ID
                ),
            },

            "properties": {

                "Month": (
                    notion_title(
                        month
                    )
                ),

                "Month Start": (
                    notion_date(
                        month_start
                    )
                ),

                "Strategy 1": (
                    notion_performance_number(
                        strategy_return
                    )
                ),

                "QQQ": (
                    notion_performance_number(
                        qqq_return
                    )
                ),

                "SPY": (
                    notion_performance_number(
                        spy_return
                    )
                ),
            },
        },
    )


# ============================================================
# UPDATE MONTH ROW
# ============================================================

def update_month_row(
    page_id,
    month_start,
    strategy_return,
    qqq_return,
    spy_return,
):

    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": {

                "Month Start": (
                    notion_date(
                        month_start
                    )
                ),

                "Strategy 1": (
                    notion_performance_number(
                        strategy_return
                    )
                ),

                "QQQ": (
                    notion_performance_number(
                        qqq_return
                    )
                ),

                "SPY": (
                    notion_performance_number(
                        spy_return
                    )
                ),
            }
        },
    )


# ============================================================
# SYNC MONTHLY PERFORMANCE
# ============================================================

def sync_monthly_performance():

    print()
    print("=" * 70)
    print(
        "NOTION MONTHLY PERFORMANCE"
    )
    print("=" * 70)

    monthly = (
        load_monthly_results()
    )

    existing = (
        get_existing_month_rows()
    )

    updated = 0
    created = 0
    unchanged = 0

    for _, row in (
        monthly.iterrows()
    ):

        month = str(
            row["Month"]
        )

        month_start = (
            pd.Timestamp(
                f"{month}-01"
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        strategy_return = float(
            row["Strategy 1"]
        )

        qqq_return = float(
            row["QQQ"]
        )

        spy_return = float(
            row["SPY"]
        )

        # ====================================================
        # CREATE NEW MONTH
        # ====================================================

        if (
            month
            not in existing
        ):

            create_month_row(
                month=month,

                month_start=(
                    month_start
                ),

                strategy_return=(
                    strategy_return
                ),

                qqq_return=(
                    qqq_return
                ),

                spy_return=(
                    spy_return
                ),
            )

            created += 1

            print(
                f"➕ {month}: created"
            )

            time.sleep(
                WRITE_DELAY_SECONDS
            )

            continue

        # ====================================================
        # EXISTING MONTH
        # ====================================================

        page = existing[
            month
        ]

        current_month_start = (
            get_date_value(
                page,
                "Month Start",
            )
        )

        current_strategy = (
            get_number_value(
                page,
                "Strategy 1",
            )
        )

        current_qqq = (
            get_number_value(
                page,
                "QQQ",
            )
        )

        current_spy = (
            get_number_value(
                page,
                "SPY",
            )
        )

        is_same = (

            dates_equal(
                current_month_start,
                month_start,
            )

            and

            numbers_equal(
                current_strategy,
                strategy_return,
            )

            and

            numbers_equal(
                current_qqq,
                qqq_return,
            )

            and

            numbers_equal(
                current_spy,
                spy_return,
            )
        )

        # ====================================================
        # UNCHANGED
        # ====================================================

        if is_same:

            unchanged += 1

            continue

        # ====================================================
        # UPDATE
        # ====================================================

        update_month_row(
            page_id=(
                page["id"]
            ),

            month_start=(
                month_start
            ),

            strategy_return=(
                strategy_return
            ),

            qqq_return=(
                qqq_return
            ),

            spy_return=(
                spy_return
            ),
        )

        updated += 1

        print(
            f"✏️ {month}: updated"
        )

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    print()

    print(
        "Monthly rows in backtest :",
        len(
            monthly
        ),
    )

    print(
        "Updated                  :",
        updated,
    )

    print(
        "Created                  :",
        created,
    )

    print(
        "Unchanged                :",
        unchanged,
    )

    print()

    print(
        "✅ Strategy 1 Monthly "
        "Performance synced to Notion"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Syncing Strategy 1 "
        "to Notion..."
    )

    # 1. Current Status
    sync_current_status()

    # 2. Monthly Performance
    sync_monthly_performance()

    print()
    print("=" * 70)

    print(
        "✅ Strategy 1 "
        "Notion sync completed"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
