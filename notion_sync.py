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

STRATEGY1_ANNUAL_DATA_SOURCE_ID = os.environ[
    "STRATEGY1_ANNUAL_DATA_SOURCE_ID"
]

STRATEGY1_HISTORY_DATA_SOURCE_ID = os.environ[
    "STRATEGY1_HISTORY_DATA_SOURCE_ID"
]

CURRENT_POSITIONS_DATA_SOURCE_ID = os.environ[
    "CURRENT_POSITIONS_DATA_SOURCE_ID"
]

STRATEGY_OVERVIEW_DATA_SOURCE_ID = os.environ[
    "STRATEGY_OVERVIEW_DATA_SOURCE_ID"
]


NOTION_VERSION = "2025-09-03"

SUMMARY_FILE = "output/strategy1_summary.json"
MONTHLY_FILE = "output/strategy1_monthly.csv"
ANNUAL_FILE = "output/strategy1_annual.csv"
HISTORY_FILE = "output/strategy1_position_history.csv"

NYSE = mcal.get_calendar("NYSE")

WRITE_DELAY_SECONDS = 0.35

HISTORY_REPAIR_DAYS = 45


# ============================================================
# PERFORMANCE PRECISION
#
# Notion percent stores decimals:
#
# 0.022210 = 2.2210%
#
# 6 storage decimal places
# =
# 4 displayed percentage decimal places
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
    url = (
        "https://api.notion.com/v1/"
        + path
    )

    data = None

    if body is not None:
        data = json.dumps(
            body
        ).encode(
            "utf-8"
        )

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
                .decode(
                    "utf-8"
                )
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
    return {
        "number": float(
            value
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


def normalize_performance(
    value,
):
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

    return "".join(
        item.get(
            "plain_text",
            "",
        )
        for item in title
    )


def get_rich_text_value(
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

    items = prop.get(
        "rich_text",
        [],
    )

    return "".join(
        item.get(
            "plain_text",
            "",
        )
        for item in items
    )


def get_select_value(
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

    value = prop.get(
        "select"
    )

    if not value:
        return None

    return value.get(
        "name"
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
# COMPARISON
# ============================================================

def numbers_equal(
    current,
    expected,
):
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
        return (
            pd.Timestamp(
                current
            )
            .strftime(
                "%Y-%m-%d"
            )
            ==
            pd.Timestamp(
                expected
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

    except Exception:
        return False


# ============================================================
# LOAD STRATEGY JSON
# ============================================================

def load_strategy_payload():
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

    if (
        "summary"
        not in payload
    ):
        raise RuntimeError(
            "strategy1_summary.json "
            "does not contain summary."
        )

    return payload


def load_strategy_output():
    return (
        load_strategy_payload()[
            "current_status"
        ]
    )


def load_strategy_summary():
    return (
        load_strategy_payload()[
            "summary"
        ]
    )


# ============================================================
# NEXT TRADING DAY
# ============================================================

def get_next_trading_day(
    signal_date,
):
    signal_date = (
        pd.Timestamp(
            signal_date
        )
        .normalize()
    )

    schedule = NYSE.schedule(
        start_date=(
            signal_date
            + pd.Timedelta(
                days=1
            )
        ).date(),

        end_date=(
            signal_date
            + pd.Timedelta(
                days=14
            )
        ).date(),
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

        parts.append(
            f"{'/'.join(tickers)}: "
            f"{source}"
        )

    return " | ".join(
        parts
    )


# ============================================================
# STRATEGY 1 DETAIL — CURRENT STATUS
# ============================================================

def find_current_status_page():
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
            "Strategy 1 row in "
            "Current Status."
        )

    if len(pages) > 1:
        raise RuntimeError(
            "More than one "
            "Strategy 1 row found "
            "in Current Status."
        )

    return pages[0][
        "id"
    ]


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
            "properties": properties
        },
    )

    print()
    print("=" * 70)
    print("NOTION CURRENT STATUS")
    print("=" * 70)

    print(
        "Name            : "
        "Strategy 1"
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
# MAIN PAGE — CURRENT POSITIONS
# ============================================================

def find_main_current_position_page():
    result = notion_request(
        "POST",
        (
            "data_sources/"
            f"{CURRENT_POSITIONS_DATA_SOURCE_ID}"
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
            "Strategy 1 row in "
            "main Current Positions."
        )

    if len(pages) > 1:
        raise RuntimeError(
            "More than one "
            "Strategy 1 row found "
            "in main Current Positions."
        )

    return pages[0][
        "id"
    ]


def sync_main_current_positions():
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

    page_id = (
        find_main_current_position_page()
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

        "Last Updated": (
            notion_date(
                last_updated
            )
        ),
    }

    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": properties
        },
    )

    print()
    print("=" * 70)
    print(
        "NOTION MAIN CURRENT POSITIONS"
    )
    print("=" * 70)

    print(
        "Name            : "
        "Strategy 1"
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
        "Signal Date     :",
        signal_date,
    )

    print(
        "Execute Date    :",
        execute_date,
    )

    print()

    print(
        "✅ Main Current Positions "
        "Strategy 1 updated in Notion"
    )


# ============================================================
# MAIN PAGE — STRATEGY OVERVIEW
# ============================================================

def find_strategy_overview_page(
    name,
):
    result = notion_request(
        "POST",
        (
            "data_sources/"
            f"{STRATEGY_OVERVIEW_DATA_SOURCE_ID}"
            "/query"
        ),
        {
            "filter": {
                "property": "Name",
                "title": {
                    "equals": name
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
            f"Could not find "
            f"'{name}' row in "
            f"Strategy Overview."
        )

    if len(pages) > 1:
        raise RuntimeError(
            f"More than one "
            f"'{name}' row found in "
            f"Strategy Overview."
        )

    return pages[0]


def strategy_overview_matches(
    page,
    metrics,
):
    return (

        numbers_equal(
            get_number_value(
                page,
                "YTD Return",
            ),
            normalize_performance(
                metrics[
                    "YTD Return"
                ]
            ),
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "Since 2016",
            ),
            normalize_performance(
                metrics[
                    "Since 2016"
                ]
            ),
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "CAGR",
            ),
            normalize_performance(
                metrics[
                    "CAGR"
                ]
            ),
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "Max Drawdown",
            ),
            normalize_performance(
                metrics[
                    "Max Drawdown"
                ]
            ),
        )
    )


def update_strategy_overview_row(
    page_id,
    metrics,
):
    properties = {

        "YTD Return": (
            notion_performance_number(
                metrics[
                    "YTD Return"
                ]
            )
        ),

        "Since 2016": (
            notion_performance_number(
                metrics[
                    "Since 2016"
                ]
            )
        ),

        "CAGR": (
            notion_performance_number(
                metrics[
                    "CAGR"
                ]
            )
        ),

        "Max Drawdown": (
            notion_performance_number(
                metrics[
                    "Max Drawdown"
                ]
            )
        ),
    }

    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": properties
        },
    )


def sync_strategy_overview():
    print()
    print("=" * 70)
    print(
        "NOTION STRATEGY OVERVIEW"
    )
    print("=" * 70)

    summary = (
        load_strategy_summary()
    )

    names = [
        "Strategy 1",
        "QQQ",
        "SPY",
    ]

    updated = 0
    unchanged = 0

    for name in names:

        if name not in summary:
            raise RuntimeError(
                f"Summary does not contain "
                f"'{name}'."
            )

        metrics = summary[
            name
        ]

        page = (
            find_strategy_overview_page(
                name
            )
        )

        if strategy_overview_matches(
            page,
            metrics,
        ):

            print(
                f"⏭️ {name}: unchanged"
            )

            unchanged += 1

            continue

        update_strategy_overview_row(
            page[
                "id"
            ],
            metrics,
        )

        print(
            f"✏️ {name}: updated"
        )

        updated += 1

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    print()

    print(
        "Rows checked :",
        len(
            names
        ),
    )

    print(
        "Updated      :",
        updated,
    )

    print(
        "Unchanged    :",
        unchanged,
    )

    print()

    print(
        "✅ Strategy Overview synced "
        "to Notion"
    )


# ============================================================
# MONTHLY PERFORMANCE
# ============================================================

def load_monthly_results():
    monthly = pd.read_csv(
        MONTHLY_FILE,
        dtype={
            "Month": str,
        },
    )

    required = {
        "Month",
        "Strategy 1",
        "QQQ",
        "SPY",
    }

    missing = (
        required
        - set(
            monthly.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Monthly CSV missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    return (
        monthly[
            [
                "Month",
                "Strategy 1",
                "QQQ",
                "SPY",
            ]
        ]
        .sort_values(
            "Month"
        )
    )


def find_month_page(
    month,
):
    result = notion_request(
        "POST",
        (
            "data_sources/"
            f"{STRATEGY1_MONTHLY_DATA_SOURCE_ID}"
            "/query"
        ),
        {
            "filter": {
                "property": "Month",
                "title": {
                    "equals": month
                },
            },
            "page_size": 10,
        },
    )

    pages = result.get(
        "results",
        [],
    )

    if len(pages) > 1:
        raise RuntimeError(
            f"Duplicate Month row: "
            f"{month}"
        )

    if not pages:
        return None

    return pages[0]


def create_month_row(
    month,
    month_start,
    strategy_return,
    qqq_return,
    spy_return,
):
    notion_request(
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


def sync_one_month(
    row,
):
    month = str(
        row[
            "Month"
        ]
    )

    month_start = (
        pd.Timestamp(
            f"{month}-01"
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    strategy_return = (
        normalize_performance(
            row[
                "Strategy 1"
            ]
        )
    )

    qqq_return = (
        normalize_performance(
            row[
                "QQQ"
            ]
        )
    )

    spy_return = (
        normalize_performance(
            row[
                "SPY"
            ]
        )
    )

    page = (
        find_month_page(
            month
        )
    )

    if page is None:

        create_month_row(
            month,
            month_start,
            strategy_return,
            qqq_return,
            spy_return,
        )

        print(
            f"➕ {month}: created"
        )

        return "created"

    is_same = (

        dates_equal(
            get_date_value(
                page,
                "Month Start",
            ),
            month_start,
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "Strategy 1",
            ),
            strategy_return,
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "QQQ",
            ),
            qqq_return,
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "SPY",
            ),
            spy_return,
        )
    )

    if is_same:

        print(
            f"⏭️ {month}: unchanged"
        )

        return "unchanged"

    update_month_row(
        page[
            "id"
        ],
        month_start,
        strategy_return,
        qqq_return,
        spy_return,
    )

    print(
        f"✏️ {month}: updated"
    )

    return "updated"


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

    recent = (
        monthly
        .tail(2)
        .copy()
    )

    updated = 0
    created = 0
    unchanged = 0

    print(
        "Checking months:"
    )

    for month in recent[
        "Month"
    ]:

        print(
            f"  {month}"
        )

    print()

    for _, row in (
        recent.iterrows()
    ):

        result = (
            sync_one_month(
                row
            )
        )

        if result == "updated":
            updated += 1

        elif result == "created":
            created += 1

        else:
            unchanged += 1

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    print()

    print(
        "Months checked :",
        len(
            recent
        ),
    )

    print(
        "Updated        :",
        updated,
    )

    print(
        "Created        :",
        created,
    )

    print(
        "Unchanged      :",
        unchanged,
    )

    print()

    print(
        "✅ Recent Strategy 1 Monthly "
        "Performance synced to Notion"
    )


# ============================================================
# ANNUAL PERFORMANCE
# ============================================================

def load_annual_results():
    annual = pd.read_csv(
        ANNUAL_FILE
    )

    required = {
        "Year",
        "Strategy 1",
        "QQQ",
        "SPY",
    }

    missing = (
        required
        - set(
            annual.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Annual CSV missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    annual = annual[
        [
            "Year",
            "Strategy 1",
            "QQQ",
            "SPY",
        ]
    ].copy()

    annual[
        "Year"
    ] = (
        annual[
            "Year"
        ]
        .astype(
            int
        )
    )

    return annual.sort_values(
        "Year"
    )


def annual_period_name(
    year,
    latest_year,
):
    if year == latest_year:
        return (
            f"{year} YTD"
        )

    return str(
        year
    )


def get_existing_annual_rows():
    pages = (
        query_all_pages(
            STRATEGY1_ANNUAL_DATA_SOURCE_ID
        )
    )

    result = {}

    for page in pages:

        year = (
            get_number_value(
                page,
                "Year",
            )
        )

        if year is None:
            continue

        year = int(
            year
        )

        if year in result:
            raise RuntimeError(
                f"Duplicate Annual "
                f"year: {year}"
            )

        result[
            year
        ] = page

    return result


def create_annual_row(
    year,
    period,
    strategy_return,
    qqq_return,
    spy_return,
):
    notion_request(
        "POST",
        "pages",
        {
            "parent": {
                "type": (
                    "data_source_id"
                ),
                "data_source_id": (
                    STRATEGY1_ANNUAL_DATA_SOURCE_ID
                ),
            },

            "properties": {

                "Period": (
                    notion_title(
                        period
                    )
                ),

                "Year": (
                    notion_number(
                        year
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


def update_annual_row(
    page_id,
    period,
    year,
    strategy_return,
    qqq_return,
    spy_return,
):
    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": {

                "Period": (
                    notion_title(
                        period
                    )
                ),

                "Year": (
                    notion_number(
                        year
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


def annual_row_is_complete(
    page,
):
    return all(
        get_number_value(
            page,
            name,
        )
        is not None

        for name in [
            "Strategy 1",
            "QQQ",
            "SPY",
        ]
    )


def sync_one_year(
    row,
    existing,
    latest_year,
):
    year = int(
        row[
            "Year"
        ]
    )

    period = (
        annual_period_name(
            year,
            latest_year,
        )
    )

    strategy_return = (
        normalize_performance(
            row[
                "Strategy 1"
            ]
        )
    )

    qqq_return = (
        normalize_performance(
            row[
                "QQQ"
            ]
        )
    )

    spy_return = (
        normalize_performance(
            row[
                "SPY"
            ]
        )
    )

    page = existing.get(
        year
    )

    if page is None:

        create_annual_row(
            year,
            period,
            strategy_return,
            qqq_return,
            spy_return,
        )

        print(
            f"➕ {period}: created"
        )

        return "created"

    is_same = (

        get_title_value(
            page,
            "Period",
        )
        == period

        and

        numbers_equal(
            get_number_value(
                page,
                "Strategy 1",
            ),
            strategy_return,
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "QQQ",
            ),
            qqq_return,
        )

        and

        numbers_equal(
            get_number_value(
                page,
                "SPY",
            ),
            spy_return,
        )
    )

    if is_same:

        print(
            f"⏭️ {period}: unchanged"
        )

        return "unchanged"

    update_annual_row(
        page[
            "id"
        ],
        period,
        year,
        strategy_return,
        qqq_return,
        spy_return,
    )

    print(
        f"✏️ {period}: updated"
    )

    return "updated"


def sync_annual_performance():
    print()
    print("=" * 70)
    print(
        "NOTION ANNUAL PERFORMANCE"
    )
    print("=" * 70)

    annual = (
        load_annual_results()
    )

    latest_year = int(
        annual[
            "Year"
        ].max()
    )

    existing = (
        get_existing_annual_rows()
    )

    needs_full_sync = False

    for _, row in (
        annual.iterrows()
    ):

        year = int(
            row[
                "Year"
            ]
        )

        page = existing.get(
            year
        )

        if (
            page is None
            or
            not annual_row_is_complete(
                page
            )
        ):
            needs_full_sync = True
            break

    if needs_full_sync:

        years_to_sync = (
            annual.copy()
        )

        print(
            "Annual history is not "
            "fully initialized."
        )

        print(
            "Syncing complete history."
        )

    else:

        years_to_sync = (
            annual
            .tail(2)
            .copy()
        )

        print(
            "Annual history already initialized."
        )

        print(
            "Checking latest two years only."
        )

    print()

    print(
        "Checking periods:"
    )

    for _, row in (
        years_to_sync.iterrows()
    ):

        print(
            " ",
            annual_period_name(
                int(
                    row[
                        "Year"
                    ]
                ),
                latest_year,
            ),
        )

    print()

    updated = 0
    created = 0
    unchanged = 0

    for _, row in (
        years_to_sync.iterrows()
    ):

        result = (
            sync_one_year(
                row,
                existing,
                latest_year,
            )
        )

        if result == "updated":
            updated += 1

        elif result == "created":
            created += 1

        else:
            unchanged += 1

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    print()

    print(
        "Years checked :",
        len(
            years_to_sync
        ),
    )

    print(
        "Updated       :",
        updated,
    )

    print(
        "Created       :",
        created,
    )

    print(
        "Unchanged     :",
        unchanged,
    )

    print()

    print(
        "✅ Strategy 1 Annual "
        "Performance synced to Notion"
    )


# ============================================================
# POSITION HISTORY
# ============================================================

def load_position_history():
    history = pd.read_csv(
        HISTORY_FILE,
        dtype=str,
    )

    required = {
        "Change",
        "Signal Date",
        "Execute Date",
        "From",
        "To",
        "Reason",
    }

    missing = (
        required
        - set(
            history.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Position History CSV missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    history = history[
        [
            "Change",
            "Signal Date",
            "Execute Date",
            "From",
            "To",
            "Reason",
        ]
    ].copy()

    if (
        history
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Position History contains "
            "missing values."
        )

    history[
        "Signal Date"
    ] = (
        history[
            "Signal Date"
        ]
        .map(
            lambda value:
            pd.Timestamp(
                value
            )
            .strftime(
                "%Y-%m-%d"
            )
        )
    )

    history[
        "Execute Date"
    ] = (
        history[
            "Execute Date"
        ]
        .map(
            lambda value:
            pd.Timestamp(
                value
            )
            .strftime(
                "%Y-%m-%d"
            )
        )
    )

    return (
        history
        .sort_values(
            [
                "Signal Date",
                "Execute Date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def history_key(
    signal_date,
    execute_date,
):
    return (
        str(
            signal_date
        ),
        str(
            execute_date
        ),
    )


def get_existing_history_rows():
    pages = (
        query_all_pages(
            STRATEGY1_HISTORY_DATA_SOURCE_ID
        )
    )

    result = {}

    for page in pages:

        signal_date = (
            get_date_value(
                page,
                "Signal Date",
            )
        )

        execute_date = (
            get_date_value(
                page,
                "Execute Date",
            )
        )

        if (
            signal_date is None
            or
            execute_date is None
        ):
            continue

        signal_date = (
            pd.Timestamp(
                signal_date
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        execute_date = (
            pd.Timestamp(
                execute_date
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        key = (
            history_key(
                signal_date,
                execute_date,
            )
        )

        if key in result:
            raise RuntimeError(
                "Duplicate Position History "
                f"record found: {key}"
            )

        result[
            key
        ] = page

    return result


def create_history_row(
    row,
):
    notion_request(
        "POST",
        "pages",
        {
            "parent": {
                "type": (
                    "data_source_id"
                ),
                "data_source_id": (
                    STRATEGY1_HISTORY_DATA_SOURCE_ID
                ),
            },

            "properties": {

                "Change": (
                    notion_title(
                        row[
                            "Change"
                        ]
                    )
                ),

                "Signal Date": (
                    notion_date(
                        row[
                            "Signal Date"
                        ]
                    )
                ),

                "Execute Date": (
                    notion_date(
                        row[
                            "Execute Date"
                        ]
                    )
                ),

                "From": (
                    notion_select(
                        row[
                            "From"
                        ]
                    )
                ),

                "To": (
                    notion_select(
                        row[
                            "To"
                        ]
                    )
                ),

                "Reason": (
                    notion_text(
                        row[
                            "Reason"
                        ]
                    )
                ),
            },
        },
    )


def update_history_row(
    page_id,
    row,
):
    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "properties": {

                "Change": (
                    notion_title(
                        row[
                            "Change"
                        ]
                    )
                ),

                "Signal Date": (
                    notion_date(
                        row[
                            "Signal Date"
                        ]
                    )
                ),

                "Execute Date": (
                    notion_date(
                        row[
                            "Execute Date"
                        ]
                    )
                ),

                "From": (
                    notion_select(
                        row[
                            "From"
                        ]
                    )
                ),

                "To": (
                    notion_select(
                        row[
                            "To"
                        ]
                    )
                ),

                "Reason": (
                    notion_text(
                        row[
                            "Reason"
                        ]
                    )
                ),
            }
        },
    )


def archive_history_row(
    page_id,
):
    notion_request(
        "PATCH",
        f"pages/{page_id}",
        {
            "archived": True
        },
    )


def history_page_matches(
    page,
    row,
):
    return (

        get_title_value(
            page,
            "Change",
        )
        == row[
            "Change"
        ]

        and

        dates_equal(
            get_date_value(
                page,
                "Signal Date",
            ),
            row[
                "Signal Date"
            ],
        )

        and

        dates_equal(
            get_date_value(
                page,
                "Execute Date",
            ),
            row[
                "Execute Date"
            ],
        )

        and

        get_select_value(
            page,
            "From",
        )
        == row[
            "From"
        ]

        and

        get_select_value(
            page,
            "To",
        )
        == row[
            "To"
        ]

        and

        get_rich_text_value(
            page,
            "Reason",
        )
        == row[
            "Reason"
        ]
    )


def sync_position_history():
    print()
    print("=" * 70)
    print(
        "NOTION POSITION HISTORY"
    )
    print("=" * 70)

    history = (
        load_position_history()
    )

    if history.empty:
        raise RuntimeError(
            "Position History CSV is empty."
        )

    existing = (
        get_existing_history_rows()
    )

    latest_status = (
        load_strategy_output()
    )

    latest_date = (
        pd.Timestamp(
            latest_status[
                "Signal Date"
            ]
        )
        .normalize()
    )

    cutoff_date = (
        latest_date
        - pd.Timedelta(
            days=HISTORY_REPAIR_DAYS
        )
    )

    old_history = history[
        pd.to_datetime(
            history[
                "Signal Date"
            ]
        )
        < cutoff_date
    ]

    missing_old_history = False

    for _, row in (
        old_history.iterrows()
    ):

        key = (
            history_key(
                row[
                    "Signal Date"
                ],
                row[
                    "Execute Date"
                ],
            )
        )

        if key not in existing:
            missing_old_history = True
            break

    if (
        not existing
        or
        missing_old_history
    ):

        rows_to_sync = (
            history.copy()
        )

        full_sync = True

        print(
            "Position History is not "
            "fully initialized."
        )

        print(
            "Syncing complete history."
        )

    else:

        rows_to_sync = history[
            pd.to_datetime(
                history[
                    "Signal Date"
                ]
            )
            >= cutoff_date
        ].copy()

        full_sync = False

        print(
            "Position History already initialized."
        )

        print(
            f"Checking recent "
            f"{HISTORY_REPAIR_DAYS} "
            f"days only."
        )

    print()

    print(
        "Backtest position changes :",
        len(
            history
        ),
    )

    print(
        "Rows to check             :",
        len(
            rows_to_sync
        ),
    )

    print()

    created = 0
    updated = 0
    unchanged = 0
    archived = 0

    current_keys = set()

    for _, row in (
        rows_to_sync.iterrows()
    ):

        key = (
            history_key(
                row[
                    "Signal Date"
                ],
                row[
                    "Execute Date"
                ],
            )
        )

        current_keys.add(
            key
        )

        page = existing.get(
            key
        )

        if page is None:

            create_history_row(
                row
            )

            created += 1

            print(
                f"➕ "
                f"{row['Signal Date']} → "
                f"{row['Execute Date']} "
                f"{row['Change']}"
            )

            time.sleep(
                WRITE_DELAY_SECONDS
            )

            continue

        if history_page_matches(
            page,
            row,
        ):

            unchanged += 1

            continue

        update_history_row(
            page[
                "id"
            ],
            row,
        )

        updated += 1

        print(
            f"✏️ "
            f"{row['Signal Date']} → "
            f"{row['Execute Date']} "
            f"{row['Change']}"
        )

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    if not full_sync:

        for (
            key,
            page,
        ) in existing.items():

            signal_date = (
                pd.Timestamp(
                    key[
                        0
                    ]
                )
                .normalize()
            )

            if (
                signal_date
                < cutoff_date
            ):
                continue

            if key in current_keys:
                continue

            archive_history_row(
                page[
                    "id"
                ]
            )

            archived += 1

            print(
                f"🗑️ "
                f"{key[0]} → "
                f"{key[1]}: "
                f"stale record archived"
            )

            time.sleep(
                WRITE_DELAY_SECONDS
            )

    print()

    print(
        "Created   :",
        created,
    )

    print(
        "Updated   :",
        updated,
    )

    print(
        "Unchanged :",
        unchanged,
    )

    print(
        "Archived  :",
        archived,
    )

    print()

    print(
        "✅ Strategy 1 Position History "
        "synced to Notion"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Syncing Strategy 1 "
        "to Notion..."
    )

    # 1. Strategy 1 detail status
    sync_current_status()

    # 2. Main-page Current Positions
    sync_main_current_positions()

    # 3. Main-page Strategy Overview
    sync_strategy_overview()

    # 4. Current month + previous month
    sync_monthly_performance()

    # 5. Current year + previous year
    sync_annual_performance()

    # 6. Recent Position History
    sync_position_history()

    print()
    print("=" * 70)

    print(
        "✅ Strategy 1 "
        "Notion sync completed"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
