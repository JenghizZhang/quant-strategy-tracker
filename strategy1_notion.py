import json
import os
import time

import pandas as pd
import pandas_market_calendars as mcal

from notion_client import (
    NotionClient,
    dates_equal,
    get_date_value,
    get_number_value,
    get_rich_text_value,
    get_select_value,
    get_title_value,
    normalize_performance,
    notion_checkbox,
    notion_date,
    notion_number,
    notion_performance_number,
    notion_select,
    notion_text,
    notion_text_span,
    notion_title,
    notion_user_mention,
    numbers_equal,
)


# ============================================================
# CONFIG
# ============================================================

NOTION_TOKEN = os.environ[
    "NOTION_TOKEN"
]

NOTION_MAIN_PAGE_ID = os.environ[
    "NOTION_MAIN_PAGE_ID"
]

NOTION_MENTION_NAME = os.environ[
    "NOTION_MENTION_NAME"
]

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

ANNUAL_PERFORMANCE_DATA_SOURCE_ID = os.environ[
    "ANNUAL_PERFORMANCE_DATA_SOURCE_ID"
]


SUMMARY_FILE = (
    "output/strategy1_summary.json"
)

MONTHLY_FILE = (
    "output/strategy1_monthly.csv"
)

ANNUAL_FILE = (
    "output/strategy1_annual.csv"
)

HISTORY_FILE = (
    "output/strategy1_position_history.csv"
)


WRITE_DELAY_SECONDS = 0.35
HISTORY_REPAIR_DAYS = 45

NYSE = mcal.get_calendar(
    "NYSE"
)

notion = NotionClient(
    NOTION_TOKEN
)


# ============================================================
# LOAD STRATEGY OUTPUT
# ============================================================

def load_payload():
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


def load_status():
    return load_payload()[
        "current_status"
    ]


def load_summary():
    return load_payload()[
        "summary"
    ]


# ============================================================
# GENERIC HELPERS
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


def find_single_row(
    data_source_id,
    property_name,
    value,
):
    result = notion.request(
        "POST",
        (
            "data_sources/"
            f"{data_source_id}"
            "/query"
        ),
        {
            "filter": {
                "property": (
                    property_name
                ),
                "title": {
                    "equals": value
                },
            },
            "page_size": 10,
        },
    )

    rows = result.get(
        "results",
        [],
    )

    if len(rows) == 0:
        raise RuntimeError(
            f"Could not find "
            f"'{value}' in "
            f"{property_name}."
        )

    if len(rows) > 1:
        raise RuntimeError(
            f"Duplicate row found "
            f"for '{value}'."
        )

    return rows[0]


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
        source = short_source_name(
            data_sources[
                ticker
            ]
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

    for source, tickers in (
        grouped.items()
    ):
        parts.append(
            f"{'/'.join(tickers)}: "
            f"{source}"
        )

    return " | ".join(
        parts
    )


# ============================================================
# CURRENT STATUS — STRATEGY 1 PAGE
# ============================================================

def sync_current_status():
    status = load_status()

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

    page = find_single_row(
        STRATEGY1_STATUS_DATA_SOURCE_ID,
        "Name",
        "Strategy 1",
    )

    last_updated = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    notion.request(
        "PATCH",
        f"pages/{page['id']}",
        {
            "properties": {
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
        },
    )

    print()
    print("=" * 70)
    print(
        "NOTION CURRENT STATUS"
    )
    print("=" * 70)

    print(
        "Name            : Strategy 1"
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
# MAIN CURRENT POSITIONS
# ============================================================

def sync_main_current_positions():
    status = load_status()

    signal_date = status[
        "Signal Date"
    ]

    execute_date = (
        get_next_trading_day(
            signal_date
        )
    )

    page = find_single_row(
        CURRENT_POSITIONS_DATA_SOURCE_ID,
        "Name",
        "Strategy 1",
    )

    last_updated = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    notion.request(
        "PATCH",
        f"pages/{page['id']}",
        {
            "properties": {
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
        },
    )

    print()
    print("=" * 70)
    print(
        "NOTION MAIN CURRENT POSITIONS"
    )
    print("=" * 70)

    print(
        "Name            : Strategy 1"
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
# STRATEGY OVERVIEW
# ============================================================

def overview_row_matches(
    page,
    metrics,
):
    fields = [
        "YTD Return",
        "Since 2016",
        "CAGR",
        "Max Drawdown",
    ]

    for field in fields:
        if not numbers_equal(
            get_number_value(
                page,
                field,
            ),
            normalize_performance(
                metrics[
                    field
                ]
            ),
        ):
            return False

    return True


def sync_strategy_overview():
    print()
    print("=" * 70)
    print(
        "NOTION STRATEGY OVERVIEW"
    )
    print("=" * 70)

    summary = load_summary()

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
                f"Missing '{name}' "
                f"in strategy summary."
            )

        metrics = summary[
            name
        ]

        page = find_single_row(
            STRATEGY_OVERVIEW_DATA_SOURCE_ID,
            "Name",
            name,
        )

        if overview_row_matches(
            page,
            metrics,
        ):
            print(
                f"⏭️ {name}: unchanged"
            )

            unchanged += 1
            continue

        notion.request(
            "PATCH",
            f"pages/{page['id']}",
            {
                "properties": {
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
            },
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
            "Month": str
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
    result = notion.request(
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

    rows = result.get(
        "results",
        [],
    )

    if len(rows) > 1:
        raise RuntimeError(
            f"Duplicate Month row: "
            f"{month}"
        )

    if not rows:
        return None

    return rows[0]


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

    values = {
        "Strategy 1": (
            normalize_performance(
                row[
                    "Strategy 1"
                ]
            )
        ),
        "QQQ": (
            normalize_performance(
                row[
                    "QQQ"
                ]
            )
        ),
        "SPY": (
            normalize_performance(
                row[
                    "SPY"
                ]
            )
        ),
    }

    page = find_month_page(
        month
    )

    properties = {
        "Month Start": (
            notion_date(
                month_start
            )
        ),

        "Strategy 1": (
            notion_performance_number(
                values[
                    "Strategy 1"
                ]
            )
        ),

        "QQQ": (
            notion_performance_number(
                values[
                    "QQQ"
                ]
            )
        ),

        "SPY": (
            notion_performance_number(
                values[
                    "SPY"
                ]
            )
        ),
    }

    if page is None:
        properties[
            "Month"
        ] = notion_title(
            month
        )

        notion.request(
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
                "properties": (
                    properties
                ),
            },
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
        all(
            numbers_equal(
                get_number_value(
                    page,
                    field,
                ),
                value,
            )
            for field, value
            in values.items()
        )
    )

    if is_same:
        print(
            f"⏭️ {month}: unchanged"
        )

        return "unchanged"

    notion.request(
        "PATCH",
        f"pages/{page['id']}",
        {
            "properties": properties
        },
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

    counts = {
        "updated": 0,
        "created": 0,
        "unchanged": 0,
    }

    for _, row in (
        recent.iterrows()
    ):
        result = sync_one_month(
            row
        )

        counts[
            result
        ] += 1

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
        counts[
            "updated"
        ],
    )
    print(
        "Created        :",
        counts[
            "created"
        ],
    )
    print(
        "Unchanged      :",
        counts[
            "unchanged"
        ],
    )

    print()
    print(
        "✅ Recent Strategy 1 Monthly "
        "Performance synced to Notion"
    )


# ============================================================
# ANNUAL PERFORMANCE — REUSABLE
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


def existing_annual_rows(
    data_source_id,
):
    pages = (
        notion.query_all_pages(
            data_source_id
        )
    )

    result = {}

    for page in pages:
        year = get_number_value(
            page,
            "Year",
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


def annual_row_complete(
    page,
):
    return all(
        get_number_value(
            page,
            field,
        )
        is not None
        for field in [
            "Strategy 1",
            "QQQ",
            "SPY",
        ]
    )


def sync_annual_table(
    data_source_id,
    heading,
    success_message,
):
    print()
    print("=" * 70)
    print(
        heading
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
        existing_annual_rows(
            data_source_id
        )
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
            not annual_row_complete(
                page
            )
        ):
            needs_full_sync = True
            break

    if needs_full_sync:
        rows = annual.copy()

        print(
            "Annual history is not "
            "fully initialized."
        )
        print(
            "Syncing complete history."
        )

    else:
        rows = (
            annual
            .tail(2)
            .copy()
        )

        print(
            "Annual history already "
            "initialized."
        )
        print(
            "Checking latest two "
            "years only."
        )

    print()
    print(
        "Checking periods:"
    )

    for _, row in (
        rows.iterrows()
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

    counts = {
        "updated": 0,
        "created": 0,
        "unchanged": 0,
    }

    for _, row in (
        rows.iterrows()
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

        values = {
            "Strategy 1": (
                normalize_performance(
                    row[
                        "Strategy 1"
                    ]
                )
            ),
            "QQQ": (
                normalize_performance(
                    row[
                        "QQQ"
                    ]
                )
            ),
            "SPY": (
                normalize_performance(
                    row[
                        "SPY"
                    ]
                )
            ),
        }

        page = existing.get(
            year
        )

        properties = {
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
                    values[
                        "Strategy 1"
                    ]
                )
            ),
            "QQQ": (
                notion_performance_number(
                    values[
                        "QQQ"
                    ]
                )
            ),
            "SPY": (
                notion_performance_number(
                    values[
                        "SPY"
                    ]
                )
            ),
        }

        if page is None:
            notion.request(
                "POST",
                "pages",
                {
                    "parent": {
                        "type": (
                            "data_source_id"
                        ),
                        "data_source_id": (
                            data_source_id
                        ),
                    },
                    "properties": (
                        properties
                    ),
                },
            )

            print(
                f"➕ {period}: created"
            )

            counts[
                "created"
            ] += 1

        else:
            is_same = (
                get_title_value(
                    page,
                    "Period",
                )
                == period
                and
                all(
                    numbers_equal(
                        get_number_value(
                            page,
                            field,
                        ),
                        value,
                    )
                    for field, value
                    in values.items()
                )
            )

            if is_same:
                print(
                    f"⏭️ {period}: unchanged"
                )

                counts[
                    "unchanged"
                ] += 1

            else:
                notion.request(
                    "PATCH",
                    f"pages/{page['id']}",
                    {
                        "properties": (
                            properties
                        )
                    },
                )

                print(
                    f"✏️ {period}: updated"
                )

                counts[
                    "updated"
                ] += 1

        time.sleep(
            WRITE_DELAY_SECONDS
        )

    print()
    print(
        "Years checked :",
        len(
            rows
        ),
    )
    print(
        "Updated       :",
        counts[
            "updated"
        ],
    )
    print(
        "Created       :",
        counts[
            "created"
        ],
    )
    print(
        "Unchanged     :",
        counts[
            "unchanged"
        ],
    )

    print()
    print(
        success_message
    )


def sync_strategy1_annual():
    sync_annual_table(
        STRATEGY1_ANNUAL_DATA_SOURCE_ID,
        (
            "NOTION ANNUAL PERFORMANCE"
        ),
        (
            "✅ Strategy 1 Annual "
            "Performance synced to Notion"
        ),
    )


def sync_main_annual():
    sync_annual_table(
        ANNUAL_PERFORMANCE_DATA_SOURCE_ID,
        (
            "NOTION MAIN ANNUAL PERFORMANCE"
        ),
        (
            "✅ Main Annual Performance "
            "synced to Notion"
        ),
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

    for field in [
        "Signal Date",
        "Execute Date",
    ]:
        history[
            field
        ] = (
            history[
                field
            ]
            .map(
                lambda value:
                pd.Timestamp(
                    value
                ).strftime(
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


def existing_history_rows():
    pages = (
        notion.query_all_pages(
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
            ).strftime(
                "%Y-%m-%d"
            )
        )

        execute_date = (
            pd.Timestamp(
                execute_date
            ).strftime(
                "%Y-%m-%d"
            )
        )

        key = history_key(
            signal_date,
            execute_date,
        )

        if key in result:
            raise RuntimeError(
                "Duplicate Position History "
                f"record: {key}"
            )

        result[
            key
        ] = page

    return result


def history_matches(
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


def history_properties(
    row,
):
    return {
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
        existing_history_rows()
    )

    latest_date = (
        pd.Timestamp(
            load_status()[
                "Signal Date"
            ]
        )
        .normalize()
    )

    cutoff = (
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
        < cutoff
    ]

    missing_old = False

    for _, row in (
        old_history.iterrows()
    ):
        key = history_key(
            row[
                "Signal Date"
            ],
            row[
                "Execute Date"
            ],
        )

        if key not in existing:
            missing_old = True
            break

    if (
        not existing
        or
        missing_old
    ):
        rows = history.copy()
        full_sync = True

        print(
            "Position History is not "
            "fully initialized."
        )
        print(
            "Syncing complete history."
        )

    else:
        rows = history[
            pd.to_datetime(
                history[
                    "Signal Date"
                ]
            )
            >= cutoff
        ].copy()

        full_sync = False

        print(
            "Position History already "
            "initialized."
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
            rows
        ),
    )
    print()

    counts = {
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "archived": 0,
    }

    current_keys = set()

    for _, row in (
        rows.iterrows()
    ):
        key = history_key(
            row[
                "Signal Date"
            ],
            row[
                "Execute Date"
            ],
        )

        current_keys.add(
            key
        )

        page = existing.get(
            key
        )

        if page is None:
            notion.request(
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
                    "properties": (
                        history_properties(
                            row
                        )
                    ),
                },
            )

            counts[
                "created"
            ] += 1

            print(
                f"➕ "
                f"{row['Signal Date']} → "
                f"{row['Execute Date']} "
                f"{row['Change']}"
            )

        elif history_matches(
            page,
            row,
        ):
            counts[
                "unchanged"
            ] += 1

        else:
            notion.request(
                "PATCH",
                f"pages/{page['id']}",
                {
                    "properties": (
                        history_properties(
                            row
                        )
                    )
                },
            )

            counts[
                "updated"
            ] += 1

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
        for key, page in (
            existing.items()
        ):
            signal_date = (
                pd.Timestamp(
                    key[0]
                )
                .normalize()
            )

            if signal_date < cutoff:
                continue

            if key in current_keys:
                continue

            notion.request(
                "PATCH",
                f"pages/{page['id']}",
                {
                    "archived": True
                },
            )

            counts[
                "archived"
            ] += 1

            print(
                f"🗑️ "
                f"{key[0]} → "
                f"{key[1]} "
                f"archived"
            )

            time.sleep(
                WRITE_DELAY_SECONDS
            )

    print()
    print(
        "Created   :",
        counts[
            "created"
        ],
    )
    print(
        "Updated   :",
        counts[
            "updated"
        ],
    )
    print(
        "Unchanged :",
        counts[
            "unchanged"
        ],
    )
    print(
        "Archived  :",
        counts[
            "archived"
        ],
    )

    print()
    print(
        "✅ Strategy 1 Position History "
        "synced to Notion"
    )


# ============================================================
# REBALANCE NOTIFICATION
# ============================================================

def sync_rebalance_notification():
    print()
    print("=" * 70)
    print(
        "NOTION REBALANCE NOTIFICATION"
    )
    print("=" * 70)

    status = load_status()

    current_holding = status[
        "Current Holding"
    ]

    next_holding = status[
        "Next Holding"
    ]

    if (
        current_holding
        == next_holding
    ):
        print(
            f"No position change: "
            f"{current_holding} → "
            f"{next_holding}"
        )

        print(
            "⏭️ No Notion notification sent"
        )

        return

    execute_date = (
        get_next_trading_day(
            status[
                "Signal Date"
            ]
        )
    )

    user_id = (
        notion.find_user_id(
            NOTION_MENTION_NAME
        )
    )

    message = (
        "\n\n"
        "🔔 Strategy 1 换仓信号\n\n"
        f"{current_holding} → "
        f"{next_holding}\n\n"
        f"Signal Date: "
        f"{status['Signal Date']}\n"
        f"Execute Date: "
        f"{execute_date}\n\n"
        f"Reason:\n"
        f"{status['Reason']}"
    )

    notion.create_comment(
        NOTION_MAIN_PAGE_ID,
        [
            notion_user_mention(
                user_id
            ),
            notion_text_span(
                message
            ),
        ],
    )

    print(
        "Position change detected:"
    )
    print(
        f"{current_holding} → "
        f"{next_holding}"
    )

    print(
        "✅ Rebalance comment + "
        "@mention sent to Notion"
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def sync_strategy1():
    print(
        "Syncing Strategy 1 "
        "to Notion..."
    )

    sync_current_status()

    sync_main_current_positions()

    sync_strategy_overview()

    sync_monthly_performance()

    sync_strategy1_annual()

    sync_main_annual()

    sync_position_history()

    sync_rebalance_notification()

    print()
    print("=" * 70)

    print(
        "✅ Strategy 1 "
        "Notion sync completed"
    )

    print("=" * 70)
