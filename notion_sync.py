import json
import os
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

NOTION_VERSION = "2025-09-03"

SUMMARY_FILE = (
    "output/strategy1_summary.json"
)

NYSE = mcal.get_calendar("NYSE")


# ============================================================
# NOTION API
# ============================================================

def notion_request(
    method,
    path,
    body=None,
):
    """
    Make a Notion API request without ever
    printing the token.
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


# ============================================================
# FIND CURRENT STATUS ROW
# ============================================================

def find_current_status_page():
    """
    Find the existing row whose title is:

        Current

    We UPDATE that page.

    We do NOT create another Current row.
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
                    "equals": "Current"
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
            "Could not find the "
            "'Current' row in "
            "Strategy 1 Current Status."
        )

    if len(pages) > 1:

        raise RuntimeError(
            "More than one row named "
            "'Current' was found. "
            "Refusing to guess which "
            "one should be updated."
        )

    return pages[0]["id"]


# ============================================================
# NEXT TRADING DAY
# ============================================================

def get_next_trading_day(
    signal_date,
):
    """
    Signal is generated at today's close.

    Execute Date must therefore be the NEXT
    actual US trading session.

    This automatically handles:
    - weekends
    - holidays
    """

    signal_date = pd.Timestamp(
        signal_date
    ).normalize()

    start_date = (
        signal_date
        + pd.Timedelta(days=1)
    )

    end_date = (
        signal_date
        + pd.Timedelta(days=14)
    )

    schedule = NYSE.schedule(
        start_date=start_date.date(),
        end_date=end_date.date(),
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
        .strftime("%Y-%m-%d")
    )


# ============================================================
# DATA SOURCE DISPLAY
# ============================================================

def short_source_name(
    source,
):
    """
    Convert internal source names into
    cleaner Notion display names.
    """

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
    Example:

    Yahoo | NDX/SPX: daily | QQQ/SPY/AGG: 5m

    If everything is daily:

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
# PROPERTY HELPERS
# ============================================================

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


# ============================================================
# LOAD BACKTEST OUTPUT
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

    # ----------------------------------------
    # Signal Date
    # ----------------------------------------

    signal_date = status[
        "Signal Date"
    ]

    # ----------------------------------------
    # Execute Date
    #
    # Must be next ACTUAL trading day.
    # ----------------------------------------

    execute_date = (
        get_next_trading_day(
            signal_date
        )
    )

    # ----------------------------------------
    # Human-readable data source
    # ----------------------------------------

    data_source_text = (
        build_data_source_text(
            status[
                "Data Sources"
            ]
        )
    )

    # ----------------------------------------
    # Find existing Notion row
    # ----------------------------------------

    page_id = (
        find_current_status_page()
    )

    # ----------------------------------------
    # Last Updated
    #
    # Store a real timestamp rather than
    # only a date.
    # ----------------------------------------

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

    # ----------------------------------------
    # Update page
    # ----------------------------------------

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
# MAIN
# ============================================================

def main():

    print(
        "Syncing Strategy 1 "
        "Current Status to Notion..."
    )

    sync_current_status()


if __name__ == "__main__":
    main()
