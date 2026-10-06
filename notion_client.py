import json
import math
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


NOTION_VERSION = "2025-09-03"
PERFORMANCE_STORAGE_DECIMALS = 6
NUMBER_TOLERANCE = 1e-12


class NotionClient:
    def __init__(
        self,
        token,
        version=NOTION_VERSION,
    ):
        self.token = token
        self.version = version

    def request(
        self,
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
                    f"Bearer {self.token}"
                ),
                "Notion-Version": (
                    self.version
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
        self,
        data_source_id,
    ):
        pages = []
        cursor = None

        while True:
            body = {
                "page_size": 100,
            }

            if cursor:
                body[
                    "start_cursor"
                ] = cursor

            result = self.request(
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

            cursor = result.get(
                "next_cursor"
            )

            if not cursor:
                break

        return pages

    def find_user_id(
        self,
        display_name,
    ):
        """
        Find a Notion person by exact display name.

        The ID is returned to the program,
        but never printed.
        """

        cursor = None
        matches = []

        while True:
            params = {
                "page_size": 100,
            }

            if cursor:
                params[
                    "start_cursor"
                ] = cursor

            query = (
                urllib.parse.urlencode(
                    params
                )
            )

            result = self.request(
                "GET",
                f"users?{query}",
            )

            for user in result.get(
                "results",
                [],
            ):
                if (
                    user.get("type")
                    == "person"
                    and
                    user.get("name")
                    == display_name
                ):
                    matches.append(
                        user
                    )

            if not result.get(
                "has_more",
                False,
            ):
                break

            cursor = result.get(
                "next_cursor"
            )

            if not cursor:
                break

        if len(matches) == 0:
            raise RuntimeError(
                "Could not find configured "
                "Notion mention user."
            )

        if len(matches) > 1:
            raise RuntimeError(
                "More than one Notion user "
                "has the configured name."
            )

        return matches[0]["id"]

    def create_comment(
        self,
        page_id,
        rich_text,
    ):
        return self.request(
            "POST",
            "comments",
            {
                "parent": {
                    "page_id": page_id
                },
                "rich_text": rich_text,
            },
        )


# ============================================================
# PROPERTY BUILDERS
# ============================================================

def notion_title(value):
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


def notion_text(value):
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


def notion_select(value):
    return {
        "select": {
            "name": str(
                value
            )
        }
    }


def notion_number(value):
    return {
        "number": float(
            value
        )
    }


def notion_checkbox(value):
    return {
        "checkbox": bool(
            value
        )
    }


def notion_date(value):
    return {
        "date": {
            "start": str(
                value
            )
        }
    }


def normalize_performance(value):
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"Invalid performance "
            f"value: {value}"
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


def notion_user_mention(
    user_id,
):
    return {
        "type": "mention",
        "mention": {
            "type": "user",
            "user": {
                "id": user_id
            },
        },
    }


def notion_text_span(
    content,
):
    return {
        "type": "text",
        "text": {
            "content": str(
                content
            )
        },
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

    return "".join(
        item.get(
            "plain_text",
            "",
        )
        for item in prop.get(
            "title",
            [],
        )
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

    return "".join(
        item.get(
            "plain_text",
            "",
        )
        for item in prop.get(
            "rich_text",
            [],
        )
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
    return (
        page
        .get(
            "properties",
            {},
        )
        .get(
            property_name,
            {},
        )
        .get(
            "number"
        )
    )


def get_date_value(
    page,
    property_name,
):
    value = (
        page
        .get(
            "properties",
            {},
        )
        .get(
            property_name,
            {},
        )
        .get(
            "date"
        )
    )

    if not value:
        return None

    return value.get(
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
            ).strftime(
                "%Y-%m-%d"
            )
            ==
            pd.Timestamp(
                expected
            ).strftime(
                "%Y-%m-%d"
            )
        )

    except Exception:
        return False
