from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
from urllib import error, request

from strategy3_notion import build_notion_payload


# ============================================================
# Strategy 3 — Notion Sync
#
# Syncs ONLY Strategy 3's own four databases:
#   1) Current Status
#   2) Monthly Performance
#   3) Annual Performance
#   4) Position History
#
# It does NOT touch:
#   - Main Current Positions
#   - Main Strategy Overview
#   - Main Annual Performance
#   - Strategy 1 / Strategy 2 pages
#
# Required secret:
#   NOTION_TOKEN
#
# Optional env:
#   FULL_REBUILD=true
#
# FULL_REBUILD=true:
#   - checks the complete backtest history
#   - creates missing rows
#   - updates changed rows
#   - archives stale rows that no longer exist in the new backtest
#
# FULL_REBUILD=false:
#   - still performs safe upserts
#   - does NOT archive stale historical rows
# ============================================================


NOTION_VERSION = "2025-09-03"
NOTION_API_BASE = "https://api.notion.com/v1"

FULL_REBUILD = os.getenv("FULL_REBUILD", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5


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

            # Rate limit / temporary server errors: retry.
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
# NOTION PROPERTY BUILDERS
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

    # Notion rich-text items have a per-item character limit.
    # Reasons/Data Source are short, but split defensively.
    chunks = [text[i:i + 1900] for i in range(0, len(text), 1900)]
    if not chunks:
        chunks = [""]

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


def prop_date(value: Any) -> dict:
    if value is None or str(value).strip() == "":
        return {"date": None}
    return {"date": {"start": str(value)}}


# ============================================================
# RECORD -> NOTION PROPERTY MAPS
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


# ============================================================
# QUERY HELPERS
# ============================================================

def query_all_pages(data_source_id: str) -> list[dict[str, Any]]:
    """
    Query every live row from a Notion data source.
    """
    results: list[dict[str, Any]] = []
    start_cursor = None

    while True:
        payload: dict[str, Any] = {
            "page_size": 100,
        }

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
# NORMALIZATION / COMPARISON
# ============================================================

def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def values_equal(a: Any, b: Any) -> bool:
    a = normalize_scalar(a)
    b = normalize_scalar(b)

    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-10

    return a == b


def page_current_status(page: dict[str, Any]) -> dict[str, Any]:
    p = page.get("properties", {})
    return {
        "Name": get_plain_text(p.get("Name")),
        "Current Holding": get_select_name(p.get("Current Holding")),
        "Today's Signal": get_select_name(p.get("Today's Signal")),
        "Next Holding": get_select_name(p.get("Next Holding")),
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
        "From": get_select_name(p.get("From")),
        "To": get_select_name(p.get("To")),
        "Reason": get_plain_text(p.get("Reason")),
    }


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

        # Year may come back from Notion as float.
        if key == "Year" and actual_value is not None:
            actual_value = int(actual_value)

        if not values_equal(desired_value, actual_value):
            return False

    return True


# ============================================================
# CREATE / UPDATE / ARCHIVE
# ============================================================

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
        {
            "archived": True,
        },
    )


# ============================================================
# GENERIC UPSERT
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
    ignore_compare_fields: set[str] | None = None,
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

        if record_equal(
            record,
            actual,
            ignore_fields=ignore_compare_fields,
        ):
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

        # Duplicate live rows are also stale by definition.
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
# CURRENT STATUS
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
    print(f"Data Source     : {record['Data Source']}")

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

    # Last Updated intentionally changes every run. Ignore it when deciding
    # whether market/status content changed, but still write it if we update.
    if record_equal(
        record,
        actual,
        ignore_fields={"Last Updated"},
    ):
        print()
        print("⏭️ Strategy 3 Current Status unchanged")
    else:
        update_page(
            canonical["id"],
            current_status_properties(record),
        )
        print()
        print("✅ Strategy 3 Current Status updated in Notion")

    # Clean duplicates only in full rebuild.
    if FULL_REBUILD and len(matching) > 1:
        for duplicate in matching[1:]:
            archive_page(duplicate["id"])
            print(
                f"🗑️ Duplicate Current Status row archived: "
                f"{duplicate['id']}"
            )


# ============================================================
# MONTHLY / ANNUAL / HISTORY
# ============================================================

def sync_monthly(
    data_source_id: str,
    records: list[dict[str, Any]],
) -> None:
    print()
    print("Sync mode       :", "FULL REBUILD" if FULL_REBUILD else "SAFE UPSERT")

    if FULL_REBUILD:
        print("Checking complete monthly history.")

    sync_collection(
        label="NOTION MONTHLY PERFORMANCE",
        data_source_id=data_source_id,
        desired_records=records,
        key_fn=lambda r: r["Month"],
        page_record_fn=page_monthly,
        property_fn=monthly_properties,
        archive_stale=FULL_REBUILD,
    )

    if FULL_REBUILD:
        print()
        print("✅ Complete Strategy 3 Monthly Performance rebuilt in Notion")
    else:
        print()
        print("✅ Strategy 3 Monthly Performance synced to Notion")


def sync_annual(
    data_source_id: str,
    records: list[dict[str, Any]],
) -> None:
    sync_collection(
        label="NOTION ANNUAL PERFORMANCE",
        data_source_id=data_source_id,
        desired_records=records,
        key_fn=lambda r: r["Period"],
        page_record_fn=page_annual,
        property_fn=annual_properties,
        archive_stale=FULL_REBUILD,
    )

    print()
    print("✅ Strategy 3 Annual Performance synced to Notion")


def history_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    Stable identity for one historical position transition.

    Change alone is not unique, so use:
      Signal Date + Execute Date + From + To
    """
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

    if FULL_REBUILD:
        print("Full rebuild mode enabled.")
        print("Checking complete Position History.")
        print("Old rows not present in the new backtest will be archived.")
    else:
        print("Safe upsert mode.")
        print("Stale historical rows will NOT be archived.")

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
# MAIN
# ============================================================

def main() -> None:
    print("Syncing Strategy 3 to Notion...")
    print(
        "Sync mode       :",
        "FULL REBUILD" if FULL_REBUILD else "SAFE UPSERT",
    )

    payload = build_notion_payload()

    ids = payload["data_source_ids"]

    sync_current_status(
        ids["current_status"],
        payload["current_status"],
    )

    sync_monthly(
        ids["monthly_performance"],
        payload["monthly_performance"],
    )

    sync_annual(
        ids["annual_performance"],
        payload["annual_performance"],
    )

    sync_position_history(
        ids["position_history"],
        payload["position_history"],
    )

    print()
    print("=" * 70)
    print("✅ Strategy 3 Notion sync completed")
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
