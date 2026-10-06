from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from market_data import MARKET_CALENDAR


# ============================================================
# Strategy 3 — Notion payload builder
#
# IMPORTANT:
#   This file DOES NOT write to Notion.
#   It only reads Strategy 3 backtest output files and converts
#   them into clean records for the later strategy3_sync.py step.
# ============================================================


OUTPUT_DIR = Path("output")

SUMMARY_FILE = OUTPUT_DIR / "strategy3_summary.json"
MONTHLY_FILE = OUTPUT_DIR / "strategy3_monthly.csv"
ANNUAL_FILE = OUTPUT_DIR / "strategy3_annual.csv"
POSITION_HISTORY_FILE = OUTPUT_DIR / "strategy3_position_history.csv"

STRATEGY_NAME = "Strategy 3"

DATA_SOURCE_IDS = {
    "current_status": "a88a3950-4020-44b9-aabc-fd5c21d4fdd4",
    "monthly_performance": "28dc31b7-c9d2-43bf-a79d-3903b800f797",
    "annual_performance": "4d036f51-4bef-4b38-8cff-db9f86a9ffa6",
    "position_history": "7546cee8-11ab-4575-9806-d5872ba6a7dc",
}


# ============================================================
# HELPERS
# ============================================================

def _require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Strategy 3 output file: {path}. "
            "Run `python strategy3.py` first."
        )


def _to_date_string(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def _to_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def get_next_trading_session(signal_date) -> str:
    """
    Return the first NYSE session after signal_date.

    Strategy 3:
      signal at today's close
      -> execute at next NYSE trading day's open
    """
    signal_date = pd.Timestamp(signal_date).normalize()

    schedule = MARKET_CALENDAR.schedule(
        start_date=(signal_date + pd.Timedelta(days=1)).date(),
        end_date=(signal_date + pd.Timedelta(days=10)).date(),
    )

    if schedule.empty:
        raise RuntimeError(
            f"Could not find next NYSE session after {signal_date.date()}."
        )

    next_session = pd.Timestamp(schedule.index[0]).tz_localize(None).normalize()
    return next_session.date().isoformat()


# ============================================================
# LOAD OUTPUTS
# ============================================================

def load_outputs():
    for path in (
        SUMMARY_FILE,
        MONTHLY_FILE,
        ANNUAL_FILE,
        POSITION_HISTORY_FILE,
    ):
        _require_file(path)

    with SUMMARY_FILE.open("r", encoding="utf-8") as f:
        summary_json = json.load(f)

    monthly = pd.read_csv(MONTHLY_FILE)
    annual = pd.read_csv(ANNUAL_FILE)
    history = pd.read_csv(POSITION_HISTORY_FILE)

    return summary_json, monthly, annual, history


# ============================================================
# CURRENT STATUS
# ============================================================

def build_current_status_record(summary_json: dict) -> dict:
    status = summary_json["current_status"]

    signal_date = status["Signal Date"]
    execute_date = get_next_trading_session(signal_date)

    sources = status.get("Data Sources", {})
    source_text = (
        "Yahoo | "
        f"QQQ:{sources.get('QQQ', 'daily')} / "
        f"SPY:{sources.get('SPY', 'daily')} / "
        f"AGG:{sources.get('AGG', 'daily')}"
    )

    return {
        "Name": STRATEGY_NAME,
        "Current Holding": status["Current Holding"],
        "Today's Signal": status["Today's Signal"],
        "Next Holding": status["Next Holding"],
        "Signal Date": _to_date_string(signal_date),
        "Execute Date": execute_date,
        "QQQ Close": _to_float(status["QQQ Close"]),
        "SPY Close": _to_float(status["SPY Close"]),
        "N": _to_float(status["N"]),
        "V": _to_float(status["V"]),
        "N > V": _to_bool(status["N > V"]),
        "N Cross Up": _to_bool(status["N Cross Up"]),
        "N Cross Down": _to_bool(status["N Cross Down"]),
        "SPY 40W MA": _to_float(status["SPY 40W MA"]),
        "SPY > 40W MA": _to_bool(status["SPY > 40W MA"]),
        "SPY Cross Up": _to_bool(status["SPY Cross Up"]),
        "SPY Cross Down": _to_bool(status["SPY Cross Down"]),
        "Reason": status["Reason"],
        "Last Updated": datetime.now(timezone.utc).isoformat(),
        "Data Source": source_text,
    }


# ============================================================
# MONTHLY PERFORMANCE
# ============================================================

def build_monthly_records(monthly: pd.DataFrame) -> list[dict]:
    records = []

    required = {"Month", "Strategy 3", "QQQ", "SPY"}
    missing = required.difference(monthly.columns)
    if missing:
        raise RuntimeError(
            f"Strategy 3 monthly output missing columns: {sorted(missing)}"
        )

    for _, row in monthly.iterrows():
        month = str(row["Month"])

        records.append(
            {
                "Month": month,
                "Month Start": f"{month}-01",
                "Strategy 3": _to_float(row["Strategy 3"]),
                "QQQ": _to_float(row["QQQ"]),
                "SPY": _to_float(row["SPY"]),
            }
        )

    return records


# ============================================================
# ANNUAL PERFORMANCE
# ============================================================

def build_annual_records(annual: pd.DataFrame) -> list[dict]:
    records = []

    required = {"Year", "Strategy 3", "QQQ", "SPY"}
    missing = required.difference(annual.columns)
    if missing:
        raise RuntimeError(
            f"Strategy 3 annual output missing columns: {sorted(missing)}"
        )

    latest_year = int(pd.to_numeric(annual["Year"]).max())

    for _, row in annual.iterrows():
        year = int(row["Year"])
        period = f"{year} YTD" if year == latest_year else str(year)

        records.append(
            {
                "Period": period,
                "Year": year,
                "Strategy 3": _to_float(row["Strategy 3"]),
                "QQQ": _to_float(row["QQQ"]),
                "SPY": _to_float(row["SPY"]),
            }
        )

    return records


# ============================================================
# POSITION HISTORY
# ============================================================

def build_position_history_records(history: pd.DataFrame) -> list[dict]:
    required = {
        "Change",
        "Signal Date",
        "Execute Date",
        "From",
        "To",
        "Reason",
    }

    missing = required.difference(history.columns)
    if missing:
        raise RuntimeError(
            f"Strategy 3 position-history output missing columns: {sorted(missing)}"
        )

    records = []

    for _, row in history.iterrows():
        records.append(
            {
                "Change": str(row["Change"]),
                "Signal Date": _to_date_string(row["Signal Date"]),
                "Execute Date": _to_date_string(row["Execute Date"]),
                "From": str(row["From"]),
                "To": str(row["To"]),
                "Reason": str(row["Reason"]),
            }
        )

    return records


# ============================================================
# COMPLETE NOTION PAYLOAD
# ============================================================

def build_notion_payload() -> dict:
    summary_json, monthly, annual, history = load_outputs()

    payload = {
        "data_source_ids": DATA_SOURCE_IDS,
        "current_status": build_current_status_record(summary_json),
        "monthly_performance": build_monthly_records(monthly),
        "annual_performance": build_annual_records(annual),
        "position_history": build_position_history_records(history),
    }

    return payload


# ============================================================
# PREVIEW / VALIDATION ONLY
# ============================================================

def main():
    payload = build_notion_payload()

    current = payload["current_status"]
    monthly = payload["monthly_performance"]
    annual = payload["annual_performance"]
    history = payload["position_history"]

    print("=" * 70)
    print("Strategy 3 — Notion Payload Preview")
    print("=" * 70)

    print()
    print("CURRENT STATUS")
    print("-" * 70)
    for key, value in current.items():
        print(f"{key:<18}: {value}")

    print()
    print("MONTHLY PERFORMANCE")
    print("-" * 70)
    print("Rows:", len(monthly))
    if monthly:
        print("First:", monthly[0])
        print("Last :", monthly[-1])

    print()
    print("ANNUAL PERFORMANCE")
    print("-" * 70)
    print("Rows:", len(annual))
    if annual:
        print("First:", annual[0])
        print("Last :", annual[-1])

    print()
    print("POSITION HISTORY")
    print("-" * 70)
    print("Rows:", len(history))
    if history:
        print("First:", history[0])
        print("Last :", history[-1])

    print()
    print("✅ Payload built successfully.")
    print("ℹ️ No Notion write was performed.")


if __name__ == "__main__":
    main()
