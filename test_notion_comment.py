import json
import os
import urllib.error
import urllib.parse
import urllib.request


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_MAIN_PAGE_ID = os.environ["NOTION_MAIN_PAGE_ID"]
NOTION_MENTION_NAME = os.environ["NOTION_MENTION_NAME"]

NOTION_VERSION = "2025-09-03"


def notion_request(method, path, body=None):
    url = f"https://api.notion.com/v1/{path}"

    data = None

    if body is not None:
        data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            text = response.read().decode("utf-8")

            if not text:
                return {}

            return json.loads(text)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Notion API failed: "
            f"HTTP {e.code} {e.reason}\n"
            f"{error_body}"
        ) from e


def find_mention_user():
    """
    Find the Notion person whose displayed name
    exactly matches NOTION_MENTION_NAME.

    The actual Notion user ID is NOT printed.
    """

    start_cursor = None
    matches = []

    while True:
        params = {
            "page_size": 100,
        }

        if start_cursor:
            params["start_cursor"] = start_cursor

        query = urllib.parse.urlencode(params)

        result = notion_request(
            "GET",
            f"users?{query}",
        )

        for user in result.get(
            "results",
            [],
        ):
            if (
                user.get("type") == "person"
                and user.get("name")
                == NOTION_MENTION_NAME
            ):
                matches.append(user)

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

    if len(matches) == 0:
        raise RuntimeError(
            "Could not find a Notion user "
            "matching NOTION_MENTION_NAME."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "More than one Notion user "
            "has the same display name."
        )

    return matches[0]["id"]


def send_test_comment():
    user_id = find_mention_user()

    print(
        "✅ Mention user found"
    )

    rich_text = [
        {
            "type": "mention",
            "mention": {
                "type": "user",
                "user": {
                    "id": user_id
                },
            },
        },
        {
            "type": "text",
            "text": {
                "content": (
                    "\n\n"
                    "🔔 Quant Strategy Tracker Test\n\n"
                    "Strategy 1 comment + @mention "
                    "notification is working."
                )
            },
        },
    ]

    notion_request(
        "POST",
        "comments",
        {
            "parent": {
                "page_id": (
                    NOTION_MAIN_PAGE_ID
                )
            },
            "rich_text": rich_text,
        },
    )

    print(
        "✅ Test comment created"
    )


def main():
    print("=" * 70)
    print(
        "TEST NOTION COMMENT + @MENTION"
    )
    print("=" * 70)

    print(
        "Mention name:",
        NOTION_MENTION_NAME,
    )

    send_test_comment()

    print()
    print(
        "✅ Notion comment/@mention "
        "test passed"
    )


if __name__ == "__main__":
    main()
