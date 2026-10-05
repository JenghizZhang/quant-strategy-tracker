import json
import os
import urllib.request
import urllib.error


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_MAIN_PAGE_ID = os.environ["NOTION_MAIN_PAGE_ID"]
CURRENT_POSITIONS_DATA_SOURCE_ID = os.environ["CURRENT_POSITIONS_DATA_SOURCE_ID"]
STRATEGY1_STATUS_DATA_SOURCE_ID = os.environ["STRATEGY1_STATUS_DATA_SOURCE_ID"]


def notion_get(path):
    url = f"https://api.notion.com/v1/{path}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2025-09-03",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Notion API failed: HTTP {e.code} {e.reason}"
        ) from e


def main():
    print("Testing Notion connection...")

    notion_get(f"pages/{NOTION_MAIN_PAGE_ID}")
    print("✅ Main page accessible")

    notion_get(f"data_sources/{CURRENT_POSITIONS_DATA_SOURCE_ID}")
    print("✅ Current Positions accessible")

    notion_get(f"data_sources/{STRATEGY1_STATUS_DATA_SOURCE_ID}")
    print("✅ Strategy 1 Current Status accessible")

    print("✅ Notion connection test passed")


if __name__ == "__main__":
    main()
