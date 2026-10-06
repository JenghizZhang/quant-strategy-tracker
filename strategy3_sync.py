from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from strategy3_notion import build_notion_payload


# ============================================================
# Strategy 3 — Notion Sync
#
# Syncs Strategy 3's own four databases:
#   1) Current Status
#   2) Monthly Performance
#   3) Annual Performance
#   4) Position History
#
# ALSO syncs Strategy 3 into the three shared/main tables:
#   5) Main Current Positions
#   6) Strategy Overview
#   7) Main Annual Performance
#
# It only writes Strategy 3's row/column in shared tables.
# It does NOT overwrite Strategy 1 / Strategy 2 values.
#
# Required secret:
#   NOTION_TOKEN
#
# Optional env:
#   FULL_REBUILD=true|false
#
# Recommended normal setting:
#   FULL_REBUILD=false
# ============================================================


NOTION_VERSION = "2025-09-03"
NOTION_API_BASE = "https://api.notion.com/v1"

FULL_REBUILD = os.getenv("FULL_REBUILD", "false").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5

OUTPUT_DIR = Path("output")
SUMMARY_FILE = OUTPUT_DIR / "strategy3_summary.json"


# ------------------------------------------------------------
# Strategy 3 private data sources
# ------------------------------------------------------------

STRATEGY3_CURRENT_STATUS_DS = "a88a3950-4020-44b9-aabc-fd5c21d4fdd4"
STRATEGY3_MONTHLY_DS = "28dc31b7-c9d2-43bf-a79d-3903b800f797"
STRATEGY3_ANNUAL_DS = "4d036f51-4bef-4b38-8cff-db9f86a9ffa6"
STRATEGY3_HISTORY_DS = "7546cee8-11ab-4575-9806-d5872ba6a7dc"


# ------------------------------------------------------------
# Shared/main data sources
# ------------------------------------------------------------

MAIN_CURRENT_POSITIONS_DS = "ca90d024-d1b0-4679-b1d7-12973dea12b3"
MAIN_STRATEGY_OVERVIEW_DS = "c77336af-8286-4235-af16-b931725657f3"
MAIN_ANNUAL_PERFORMANCE_DS = "05cb4313-a653-4424-b6e4-08b1514dc783"


# ============================================================
# NOTION HTTP CLIENT
# ============================================================

def get_token() -> str:
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
    if not token:
        raise RuntimeError(
            "Missing NOTION_TOKEN / NOTION_API_KEY environment variable."
        )
    return token


def notion_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{NOTION_API_BASE}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        req = request.Request(
            url=url,
            data=body,
            method=method,
            headers=headers,
        )

        try:
            with request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}

        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")

            if exc.code == 429 or 500 <= exc.code <= 599:
                last_error = RuntimeError(
                    f"Notion API HTTP {exc.code}: {details}"
                )

                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_seconds = float(retry_after)
                    except ValueError:
                        sleep_seconds = min(2 ** attempt, 10)
                else:
                    sleep_seconds = min(2 ** attempt, 10)

                if attempt < MAX_RETRIES:
                    print(
                        f"⚠️ Notion temporary error {exc.code}; "
                        f"retrying in {sleep_seconds:.1f}s..."
                    )
                    time.sleep(sleep_seconds)
                    continue

            raise RuntimeError(
                f"Notion API error {exc.code} for {method} {path}:\n"
                f"{details}"
            ) from exc

        except error.URLError as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                sleep_seconds = min(2 ** attempt, 10)
                print(
                    "⚠️ Notion connection error; "
                    f"retrying in {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)
                continue

    raise RuntimeError(
        f"Notion request failed after {MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# PROPERTY BUILDERS
# ============================================================

def prop_title(value: Any) -> dict:
    text = "" if value is None else str(value)
    return {
        "title": [
            {
                "type": "text",
                "text": {"content": text},
            }
        ]
    }


def prop_rich_text(value: Any) -> dict:
    text = "" if value is None else str(value)
    chunks = [text[i:i + 1900] for i in range(0, len(text), 1900)] or [""]

    return {
        "rich_text": [
            {
                "type": "text",
                "text": {"content": chunk},
            }
            for chunk in chunks
        ]
    }


def prop_number(value: Any) -> dict:
    return {"number": None if value is None else float(value)}


def prop_checkbox(value: Any) -> dict:
    return {"checkbox": bool(value)}


def prop_select(value: Any) -> dict:
    if value is None or str(value).strip() == "":
        return {"select": None}
    return {"select": {"name": str(value)}}


def prop_status(value: Any) -> dict:
    if value is None or str(value).strip() == "":
        return {"status": None}
    return {"status": {"name": str(value)}}


def prop_date(value: Any) -> dict:
    if value is None or str(value).strip() == "":
        return {"date": None}
    return {"date": {"start": str(value)}}


def build_property_for_type(prop_type: str, value: Any) -> dict:
    if prop_type == "title":
        return prop_title(value)
    if prop_type == "rich_text":
        return prop_rich_text(value)
    if prop_type == "number":
        return prop_number(value)
    if prop_type == "checkbox":
        return prop_checkbox(value)
    if prop_type == "select":
        return prop_select(value)
    if prop_type == "status":
        return prop_status(value)
    if prop_type == "date":
        return prop_date(value)

    raise RuntimeError(
        f"Unsupported Notion property type for sync: {prop_type}"
    )


# ============================================================
# BASIC NOTION HELPERS
# ============================================================

def query_all_pages(data_source_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    start_cursor = None

    while True:
        payload: dict[str, Any] = {"page_size": 100}

        if start_cursor:
            payload["start_cursor"] = start_cursor

        response = notion_request(
            "POST",
            f"/data_sources/{data_source_id}/query",
            payload,
        )

        results.extend(response.get("results", []))

        if not response.get("has_more"):
            break

        start_cursor = response.get("next_cursor")
        if not start_cursor:
            break

    return results


def get_data_source(data_source_id: str) -> dict[str, Any]:
    return notion_request(
        "GET",
        f"/data_sources/{data_source_id}",
    )


def create_page(
    data_source_id: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return notion_request(
        "POST",
        "/pages",
        {
            "parent": {
                "type": "data_source_id",
                "data_source_id": data_source_id,
            },
            "properties": properties,
        },
    )


def update_page(
    page_id: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return notion_request(
        "PATCH",
        f"/pages/{page_id}",
        {
            "properties": properties,
        },
    )


def archive_page(page_id: str) -> None:
    notion_request(
        "PATCH",
        f"/pages/{page_id}",
        {"archived": True},
    )


# ============================================================
# READ PROPERTY VALUES
# ============================================================

def get_plain_text(prop: dict[str, Any] | None) -> str:
    if not prop:
        return ""

    prop_type = prop.get("type")

    if prop_type == "title":
        items = prop.get("title", [])
    elif prop_type == "rich_text":
        items = prop.get("rich_text", [])
    else:
        return ""

    return "".join(
        item.get("plain_text", "")
        for item in items
    )


def get_select_name(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None

    selected = prop.get("select")
    if not selected:
        return None

    return selected.get("name")


def get_status_name(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None

    selected = prop.get("status")
    if not selected:
        return None

    return selected.get("name")


def get_option_name(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None

    prop_type = prop.get("type")

    if prop_type == "select":
        return get_select_name(prop)

    if prop_type == "status":
        return get_status_name(prop)

    return None


def get_date_start(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None

    value = prop.get("date")
    if not value:
        return None

    return value.get("start")


def get_number(prop: dict[str, Any] | None):
    if not prop:
        return None
    return prop.get("number")


def get_checkbox(prop: dict[str, Any] | None) -> bool:
    if not prop:
        return False
    return bool(prop.get("checkbox", False))


# ============================================================
# SCHEMA HELPERS FOR SHARED TABLES
# ============================================================

def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def find_title_property_name(schema: dict[str, Any]) -> str:
    for name, prop in schema.get("properties", {}).items():
        if prop.get("type") == "title":
            return name

    raise RuntimeError("No title property found in Notion data source.")


def find_property_name(
    schema: dict[str, Any],
    candidates: list[str],
    allowed_types: set[str] | None = None,
) -> str | None:
    props = schema.get("properties", {})

    # Exact first.
    for candidate in candidates:
        prop = props.get(candidate)
        if prop and (
            allowed_types is None
            or prop.get("type") in allowed_types
        ):
            return candidate

    # Normalized fallback.
    by_normalized = {
        normalize_name(name): name
        for name in props
    }

    for candidate in candidates:
        name = by_normalized.get(normalize_name(candidate))
        if not name:
            continue

        prop = props[name]
        if (
            allowed_types is None
            or prop.get("type") in allowed_types
        ):
            return name

    return None


def add_if_exists(
    result: dict[str, Any],
    schema: dict[str, Any],
    candidates: list[str],
    value: Any,
    allowed_types: set[str],
) -> str | None:
    name = find_property_name(
        schema,
        candidates,
        allowed_types,
    )

    if not name:
        return None

    prop_type = schema["properties"][name]["type"]
    result[name] = build_property_for_type(prop_type, value)
    return name


def ensure_percent_number_property(
    data_source_id: str,
    property_name: str,
) -> None:
    schema = get_data_source(data_source_id)
    existing = schema.get("properties", {}).get(property_name)

    if existing:
        if existing.get("type") != "number":
            raise RuntimeError(
                f"Shared table property '{property_name}' exists "
                f"but is type '{existing.get('type')}', not number."
            )
        return

    print(
        f"➕ Shared Annual Performance is missing "
        f"'{property_name}' column; creating it as Percent."
    )

    notion_request(
        "PATCH",
        f"/data_sources/{data_source_id}",
        {
            "properties": {
                property_name: {
                    "number": {
                        "format": "percent"
                    }
                }
            }
        },
    )

    verify = get_data_source(data_source_id)
    prop = verify.get("properties", {}).get(property_name)

    if not prop or prop.get("type") != "number":
        raise RuntimeError(
            f"Failed to create shared annual column '{property_name}'."
        )

    print(f"✅ Created shared annual column: {property_name}")


# ============================================================
# COMPARISON
# ============================================================

def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def values_equal(a: Any, b: Any) -> bool:
    a = normalize_scalar(a)
    b = normalize_scalar(b)

    if isinstance(a, float) and isinstance(b, float):
        # Ignore tiny Yahoo historical adjusted-price drift.
        # 1e-5 in return space = 0.001 percentage point,
        # well below Notion's visible 0.01% precision.
        return abs(a - b) <= 1e-5

    return a == b


def record_equal(
    desired: dict[str, Any],
    actual: dict[str, Any],
    ignore_fields: set[str] | None = None,
) -> bool:
    ignore_fields = ignore_fields or set()

    for key, desired_value in desired.items():
        if key in ignore_fields:
            continue

        actual_value = actual.get(key)

        if key == "Year" and actual_value is not None:
            actual_value = int(actual_value)

        if not values_equal(desired_value, actual_value):
            return False

    return True


# ============================================================
# STRATEGY 3 PRIVATE TABLE PROPERTY MAPS
# ============================================================

def current_status_properties(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Name": prop_title(record["Name"]),
        "Current Holding": prop_select(record["Current Holding"]),
        "Today's Signal": prop_select(record["Today's Signal"]),
        "Next Holding": prop_select(record["Next Holding"]),
        "Signal Date": prop_date(record["Signal Date"]),
        "Execute Date": prop_date(record["Execute Date"]),
        "QQQ Close": prop_number(record["QQQ Close"]),
        "SPY Close": prop_number(record["SPY Close"]),
        "N": prop_number(record["N"]),
        "V": prop_number(record["V"]),
        "N > V": prop_checkbox(record["N > V"]),
        "N Cross Up": prop_checkbox(record["N Cross Up"]),
        "N Cross Down": prop_checkbox(record["N Cross Down"]),
        "SPY 40W MA": prop_number(record["SPY 40W MA"]),
        "SPY > 40W MA": prop_checkbox(record["SPY > 40W MA"]),
        "SPY Cross Up": prop_checkbox(record["SPY Cross Up"]),
        "SPY Cross Down": prop_checkbox(record["SPY Cross Down"]),
        "Reason": prop_rich_text(record["Reason"]),
        "Last Updated": prop_date(record["Last Updated"]),
        "Data Source": prop_rich_text(record["Data Source"]),
    }


def monthly_properties(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Month": prop_title(record["Month"]),
        "Month Start": prop_date(record["Month Start"]),
        "Strategy 3": prop_number(record["Strategy 3"]),
        "QQQ": prop_number(record["QQQ"]),
        "SPY": prop_number(record["SPY"]),
    }


def annual_properties(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Period": prop_title(record["Period"]),
        "Year": prop_number(record["Year"]),
        "Strategy 3": prop_number(record["Strategy 3"]),
        "QQQ": prop_number(record["QQQ"]),
        "SPY": prop_number(record["SPY"]),
    }


def history_properties(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Change": prop_title(record["Change"]),
        "Signal Date": prop_date(record["Signal Date"]),
        "Execute Date": prop_date(record["Execute Date"]),
        "From": prop_select(record["From"]),
        "To": prop_select(record["To"]),
        "Reason": prop_rich_text(record["Reason"]),
    }


def page_current_status(page: dict[str, Any]) -> dict[str, Any]:
    p = page.get("properties", {})
    return {
        "Name": get_plain_text(p.get("Name")),
        "Current Holding": get_option_name(p.get("Current Holding")),
        "Today's Signal": get_option_name(p.get("Today's Signal")),
        "Next Holding": get_option_name(p.get("Next Holding")),
        "Signal Date": get_date_start(p.get("Signal Date")),
        "Execute Date": get_date_start(p.get("Execute Date")),
        "QQQ Close": get_number(p.get("QQQ Close")),
        "SPY Close": get_number(p.get("SPY Close")),
        "N": get_number(p.get("N")),
        "V": get_number(p.get("V")),
        "N > V": get_checkbox(p.get("N > V")),
        "N Cross Up": get_checkbox(p.get("N Cross Up")),
        "N Cross Down": get_checkbox(p.get("N Cross Down")),
        "SPY 40W MA": get_number(p.get("SPY 40W MA")),
        "SPY > 40W MA": get_checkbox(p.get("SPY > 40W MA")),
        "SPY Cross Up": get_checkbox(p.get("SPY Cross Up")),
        "SPY Cross Down": get_checkbox(p.get("SPY Cross Down")),
        "Reason": get_plain_text(p.get("Reason")),
        "Last Updated": get_date_start(p.get("Last Updated")),
        "Data Source": get_plain_text(p.get("Data Source")),
    }


def page_monthly(page: dict[str, Any]) -> dict[str, Any]:
    p = page.get("properties", {})
    return {
        "Month": get_plain_text(p.get("Month")),
        "Month Start": get_date_start(p.get("Month Start")),
        "Strategy 3": get_number(p.get("Strategy 3")),
        "QQQ": get_number(p.get("QQQ")),
        "SPY": get_number(p.get("SPY")),
    }


def page_annual(page: dict[str, Any]) -> dict[str, Any]:
    p = page.get("properties", {})
    return {
        "Period": get_plain_text(p.get("Period")),
        "Year": get_number(p.get("Year")),
        "Strategy 3": get_number(p.get("Strategy 3")),
        "QQQ": get_number(p.get("QQQ")),
        "SPY": get_number(p.get("SPY")),
    }


def page_history(page: dict[str, Any]) -> dict[str, Any]:
    p = page.get("properties", {})
    return {
        "Change": get_plain_text(p.get("Change")),
        "Signal Date": get_date_start(p.get("Signal Date")),
        "Execute Date": get_date_start(p.get("Execute Date")),
        "From": get_option_name(p.get("From")),
        "To": get_option_name(p.get("To")),
        "Reason": get_plain_text(p.get("Reason")),
    }


# ============================================================
# GENERIC PRIVATE-TABLE UPSERT
# ============================================================

def sync_collection(
    *,
    label: str,
    data_source_id: str,
    desired_records: list[dict[str, Any]],
    key_fn,
    page_record_fn,
    property_fn,
    archive_stale: bool,
) -> dict[str, int]:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    existing_pages = query_all_pages(data_source_id)

    existing_by_key: dict[Any, dict[str, Any]] = {}
    duplicate_pages: list[dict[str, Any]] = []

    for page in existing_pages:
        actual_record = page_record_fn(page)
        key = key_fn(actual_record)

        if key in existing_by_key:
            duplicate_pages.append(page)
        else:
            existing_by_key[key] = page

    created = 0
    updated = 0
    unchanged = 0
    archived = 0

    desired_keys = set()

    for record in desired_records:
        key = key_fn(record)
        desired_keys.add(key)

        existing = existing_by_key.get(key)

        if existing is None:
            create_page(
                data_source_id,
                property_fn(record),
            )
            created += 1
            print(f"➕ {key}: created")
            continue

        actual = page_record_fn(existing)

        if record_equal(record, actual):
            unchanged += 1
            print(f"⏭️ {key}: unchanged")
            continue

        update_page(
            existing["id"],
            property_fn(record),
        )
        updated += 1
        print(f"✏️ {key}: updated")

    if archive_stale:
        for key, page in existing_by_key.items():
            if key not in desired_keys:
                archive_page(page["id"])
                archived += 1
                print(f"🗑️ {key}: archived")

        for page in duplicate_pages:
            archive_page(page["id"])
            archived += 1
            print(f"🗑️ duplicate row {page['id']}: archived")

    print()
    print(f"Rows checked : {len(desired_records)}")
    print(f"Created      : {created}")
    print(f"Updated      : {updated}")
    print(f"Unchanged    : {unchanged}")
    if archive_stale:
        print(f"Archived     : {archived}")

    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "archived": archived,
    }


# ============================================================
# STRATEGY 3 PRIVATE CURRENT STATUS
# ============================================================

def sync_current_status(
    data_source_id: str,
    record: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print("NOTION CURRENT STATUS")
    print("=" * 70)

    print(f"Name            : {record['Name']}")
    print(f"Signal Date     : {record['Signal Date']}")
    print(f"Execute Date    : {record['Execute Date']}")
    print(f"Current Holding : {record['Current Holding']}")
    print(f"Today's Signal  : {record['Today\'s Signal']}")
    print(f"Next Holding    : {record['Next Holding']}")
    print(f"N               : {record['N']:.6f}")
    print(f"V               : {record['V']:.6f}")
    print(f"SPY 40W MA      : {record['SPY 40W MA']:.4f}")

    pages = query_all_pages(data_source_id)

    matching = []
    for page in pages:
        actual = page_current_status(page)
        if actual["Name"] == record["Name"]:
            matching.append(page)

    if not matching:
        create_page(
            data_source_id,
            current_status_properties(record),
        )
        print()
        print("➕ Strategy 3 Current Status created in Notion")
        return

    canonical = matching[0]
    actual = page_current_status(canonical)

    if record_equal(
        record,
        actual,
        ignore_fields={"Last Updated"},
    ):
        # Keep Last Updated consistent with Strategy 1 / 2:
        # refresh the timestamp even when the market/status fields are unchanged.
        update_page(
            canonical["id"],
            {
                "Last Updated": prop_date(
                    record["Last Updated"]
                )
            },
        )
        print()
        print("⏭️ Strategy 3 status unchanged; Last Updated refreshed")
    else:
        update_page(
            canonical["id"],
            current_status_properties(record),
        )
        print()
        print("✅ Strategy 3 Current Status updated in Notion")

    if FULL_REBUILD and len(matching) > 1:
        for duplicate in matching[1:]:
            archive_page(duplicate["id"])
            print(
                f"🗑️ Duplicate Current Status row archived: "
                f"{duplicate['id']}"
            )


# ============================================================
# STRATEGY 3 PRIVATE MONTHLY / ANNUAL / HISTORY
# ============================================================

def history_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        record["Signal Date"],
        record["Execute Date"],
        record["From"],
        record["To"],
    )


def history_key_label(key: tuple[str, str, str, str]) -> str:
    signal_date, execute_date, old, new = key
    return f"{signal_date} → {execute_date} | {old} → {new}"


def sync_position_history(
    data_source_id: str,
    records: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 70)
    print("NOTION POSITION HISTORY")
    print("=" * 70)

    print(
        "Full rebuild mode enabled."
        if FULL_REBUILD
        else "Safe upsert mode."
    )

    existing_pages = query_all_pages(data_source_id)

    existing_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []

    for page in existing_pages:
        actual = page_history(page)
        key = history_key(actual)

        if key in existing_by_key:
            duplicates.append(page)
        else:
            existing_by_key[key] = page

    created = 0
    updated = 0
    unchanged = 0
    archived = 0

    desired_keys = set()

    print()
    print(f"Backtest position changes : {len(records)}")
    print(f"Rows to check             : {len(records)}")
    print()

    for record in records:
        key = history_key(record)
        desired_keys.add(key)

        existing = existing_by_key.get(key)

        if existing is None:
            create_page(
                data_source_id,
                history_properties(record),
            )
            created += 1
            print(f"➕ {history_key_label(key)} created")
            continue

        actual = page_history(existing)

        if record_equal(record, actual):
            unchanged += 1
            continue

        update_page(
            existing["id"],
            history_properties(record),
        )
        updated += 1
        print(f"✏️ {history_key_label(key)} updated")

    if FULL_REBUILD:
        for key, page in existing_by_key.items():
            if key not in desired_keys:
                archive_page(page["id"])
                archived += 1
                print(f"🗑️ {history_key_label(key)} archived")

        for page in duplicates:
            archive_page(page["id"])
            archived += 1
            print(f"🗑️ duplicate Position History row archived: {page['id']}")

    print()
    print(f"Created   : {created}")
    print(f"Updated   : {updated}")
    print(f"Unchanged : {unchanged}")
    if FULL_REBUILD:
        print(f"Archived  : {archived}")

    print()
    print("✅ Strategy 3 Position History synced to Notion")


# ============================================================
# SHARED TABLE: MAIN CURRENT POSITIONS
# ============================================================

def sync_main_current_positions(record: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("NOTION MAIN CURRENT POSITIONS")
    print("=" * 70)

    schema = get_data_source(MAIN_CURRENT_POSITIONS_DS)
    title_name = find_title_property_name(schema)

    desired: dict[str, Any] = {}
    desired[title_name] = prop_title("Strategy 3")

    add_if_exists(
        desired, schema,
        ["Current Holding"],
        record["Current Holding"],
        {"select", "status"},
    )
    add_if_exists(
        desired, schema,
        ["Today's Signal", "Today Signal"],
        record["Today's Signal"],
        {"select", "status"},
    )
    add_if_exists(
        desired, schema,
        ["Next Holding"],
        record["Next Holding"],
        {"select", "status"},
    )
    add_if_exists(
        desired, schema,
        ["Signal Date"],
        record["Signal Date"],
        {"date"},
    )
    add_if_exists(
        desired, schema,
        ["Execute Date"],
        record["Execute Date"],
        {"date"},
    )
    add_if_exists(
        desired, schema,
        ["Pending Change"],
        record["Current Holding"] != record["Next Holding"],
        {"checkbox"},
    )
    add_if_exists(
        desired, schema,
        ["Reason"],
        record["Reason"],
        {"rich_text"},
    )
    add_if_exists(
        desired, schema,
        ["Last Updated"],
        record["Last Updated"],
        {"date"},
    )
    add_if_exists(
        desired, schema,
        ["Data Source", "Data Sources"],
        record["Data Source"],
        {"rich_text"},
    )

    pages = query_all_pages(MAIN_CURRENT_POSITIONS_DS)
    matching = [
        page for page in pages
        if get_plain_text(
            page.get("properties", {}).get(title_name)
        ) == "Strategy 3"
    ]

    if not matching:
        create_page(
            MAIN_CURRENT_POSITIONS_DS,
            desired,
        )
        print("➕ Main Current Positions Strategy 3 created in Notion")
    else:
        update_page(
            matching[0]["id"],
            desired,
        )
        print("✅ Main Current Positions Strategy 3 updated in Notion")

    print(f"Current Holding : {record['Current Holding']}")
    print(f"Today's Signal  : {record['Today\'s Signal']}")
    print(f"Next Holding    : {record['Next Holding']}")
    print(f"Signal Date     : {record['Signal Date']}")
    print(f"Execute Date    : {record['Execute Date']}")


# ============================================================
# SHARED TABLE: STRATEGY OVERVIEW
# ============================================================

def load_summary_metrics() -> dict[str, float]:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_FILE}. Run strategy3.py first."
        )

    with SUMMARY_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["summary"]["Strategy 3"]


def sync_main_strategy_overview(
    metrics: dict[str, float],
    last_updated: str,
) -> None:
    print()
    print("=" * 70)
    print("NOTION STRATEGY OVERVIEW")
    print("=" * 70)

    schema = get_data_source(MAIN_STRATEGY_OVERVIEW_DS)
    title_name = find_title_property_name(schema)

    desired: dict[str, Any] = {
        title_name: prop_title("Strategy 3")
    }

    mapped = {}

    mapped["YTD Return"] = add_if_exists(
        desired, schema,
        ["YTD Return", "YTD"],
        metrics["YTD Return"],
        {"number"},
    )

    mapped["Since 2016"] = add_if_exists(
        desired, schema,
        ["Since 2016", "Since 2016 Return", "Total Return"],
        metrics["Since 2016"],
        {"number"},
    )

    mapped["CAGR"] = add_if_exists(
        desired, schema,
        ["CAGR"],
        metrics["CAGR"],
        {"number"},
    )

    mapped["Max Drawdown"] = add_if_exists(
        desired, schema,
        ["Max Drawdown", "MDD"],
        metrics["Max Drawdown"],
        {"number"},
    )

    add_if_exists(
        desired, schema,
        ["Last Updated"],
        last_updated,
        {"date"},
    )

    missing_metrics = [
        name for name, mapped_name in mapped.items()
        if mapped_name is None
    ]

    if missing_metrics:
        raise RuntimeError(
            "Strategy Overview is missing expected numeric columns: "
            + ", ".join(missing_metrics)
        )

    pages = query_all_pages(MAIN_STRATEGY_OVERVIEW_DS)
    matching = [
        page for page in pages
        if get_plain_text(
            page.get("properties", {}).get(title_name)
        ) == "Strategy 3"
    ]

    if not matching:
        create_page(
            MAIN_STRATEGY_OVERVIEW_DS,
            desired,
        )
        print("➕ Strategy 3 Overview row created")
    else:
        update_page(
            matching[0]["id"],
            desired,
        )
        print("✏️ Strategy 3 Overview row updated")

    print(f"YTD Return    : {metrics['YTD Return'] * 100:.2f}%")
    print(f"Since 2016    : {metrics['Since 2016'] * 100:.2f}%")
    print(f"CAGR          : {metrics['CAGR'] * 100:.2f}%")
    print(f"Max Drawdown  : {metrics['Max Drawdown'] * 100:.2f}%")
    print()
    print("✅ Strategy 3 Overview synced to Notion")


# ============================================================
# SHARED TABLE: MAIN ANNUAL PERFORMANCE
# ============================================================

def sync_main_annual_performance(
    annual_records: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 70)
    print("NOTION MAIN ANNUAL PERFORMANCE")
    print("=" * 70)

    # This shared table may not yet have a Strategy 3 column.
    # If missing, create exactly one Percent column named Strategy 3.
    ensure_percent_number_property(
        MAIN_ANNUAL_PERFORMANCE_DS,
        "Strategy 3",
    )

    schema = get_data_source(MAIN_ANNUAL_PERFORMANCE_DS)
    title_name = find_title_property_name(schema)

    strategy3_prop = find_property_name(
        schema,
        ["Strategy 3"],
        {"number"},
    )

    if not strategy3_prop:
        raise RuntimeError(
            "Could not resolve Strategy 3 column in main Annual Performance."
        )

    year_prop = find_property_name(
        schema,
        ["Year"],
        {"number"},
    )

    pages = query_all_pages(MAIN_ANNUAL_PERFORMANCE_DS)

    existing_by_period = {
        get_plain_text(
            page.get("properties", {}).get(title_name)
        ): page
        for page in pages
    }

    created = 0
    updated = 0
    unchanged = 0

    print("Checking complete Strategy 3 annual column.")
    print()

    for record in annual_records:
        period = record["Period"]
        desired_value = float(record["Strategy 3"])
        existing = existing_by_period.get(period)

        if existing is None:
            props = {
                title_name: prop_title(period),
                strategy3_prop: prop_number(desired_value),
            }

            if year_prop:
                props[year_prop] = prop_number(record["Year"])

            create_page(
                MAIN_ANNUAL_PERFORMANCE_DS,
                props,
            )

            created += 1
            print(f"➕ {period}: Strategy 3 created")
            continue

        actual_value = get_number(
            existing.get("properties", {}).get(strategy3_prop)
        )

        if values_equal(desired_value, actual_value):
            unchanged += 1
            print(f"⏭️ {period}: unchanged")
            continue

        update_page(
            existing["id"],
            {
                strategy3_prop: prop_number(desired_value)
            },
        )

        updated += 1
        print(f"✏️ {period}: Strategy 3 updated")

    print()
    print(f"Years checked : {len(annual_records)}")
    print(f"Updated       : {updated}")
    print(f"Created       : {created}")
    print(f"Unchanged     : {unchanged}")
    print()
    print("✅ Main Annual Performance Strategy 3 column synced to Notion")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Syncing Strategy 3 to Notion...")
    print(
        "Sync mode       :",
        "FULL REBUILD" if FULL_REBUILD else "SAFE UPSERT",
    )

    payload = build_notion_payload()

    # ========================================================
    # Strategy 3 private tables
    # ========================================================

    sync_current_status(
        STRATEGY3_CURRENT_STATUS_DS,
        payload["current_status"],
    )

    sync_collection(
        label="NOTION MONTHLY PERFORMANCE",
        data_source_id=STRATEGY3_MONTHLY_DS,
        desired_records=payload["monthly_performance"],
        key_fn=lambda r: r["Month"],
        page_record_fn=page_monthly,
        property_fn=monthly_properties,
        archive_stale=FULL_REBUILD,
    )

    print()
    print("✅ Strategy 3 Monthly Performance synced to Notion")

    sync_collection(
        label="NOTION ANNUAL PERFORMANCE",
        data_source_id=STRATEGY3_ANNUAL_DS,
        desired_records=payload["annual_performance"],
        key_fn=lambda r: r["Period"],
        page_record_fn=page_annual,
        property_fn=annual_properties,
        archive_stale=FULL_REBUILD,
    )

    print()
    print("✅ Strategy 3 Annual Performance synced to Notion")

    sync_position_history(
        STRATEGY3_HISTORY_DS,
        payload["position_history"],
    )

    # ========================================================
    # Shared/main tables
    # ========================================================

    sync_main_current_positions(
        payload["current_status"],
    )

    metrics = load_summary_metrics()

    sync_main_strategy_overview(
        metrics,
        payload["current_status"]["Last Updated"],
    )

    sync_main_annual_performance(
        payload["annual_performance"],
    )

    print()
    print("=" * 70)
    print("✅ Strategy 3 Notion sync completed")
    print("✅ Strategy 3 is now included in the shared/main tables")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 70)
        print("❌ Strategy 3 Notion sync failed")
        print("=" * 70)
        print(str(exc))
        sys.exit(1)
