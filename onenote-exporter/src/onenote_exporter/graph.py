"""Thin Microsoft Graph helpers."""
from __future__ import annotations

import time
from collections.abc import Mapping

import requests

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"


def graph_get(get_token, url, params=None, stream=False):
    token = get_token()
    if not isinstance(token, str):
        raise TypeError(f"get_token() did not return a string: {type(token)}")

    headers = {"Authorization": f"Bearer {token}"}

    max_retries = 5

    for attempt in range(max_retries):
        resp = requests.get(
            url,
            headers=headers,
            params=params,
            stream=stream,
            timeout=30,
        )

        if resp.status_code != 429:
            if resp.status_code >= 400:
                raise RuntimeError(f"Graph error {resp.status_code}: {resp.text}")
            return resp

        retry_after = int(resp.headers.get("Retry-After", "5"))
        time.sleep(retry_after)

    time.sleep(30)
    resp = requests.get(
        url,
        headers=headers,
        params=params,
        stream=stream,
        timeout=30,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Graph error {resp.status_code}: {resp.text}")

    return resp


def list_notebooks(get_token):
    url = f"{GRAPH_ROOT}/me/onenote/notebooks"
    notebooks = []
    while url:
        r = graph_get(get_token, url)
        data = r.json()
        notebooks.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return notebooks


def list_all_sections_recursive(token, notebook_id=None, section_group_id=None):
    sections = []
    section_groups = []

    if notebook_id:
        base = f"https://graph.microsoft.com/v1.0/me/onenote/notebooks/{notebook_id}"
    else:
        base = f"https://graph.microsoft.com/v1.0/me/onenote/sectionGroups/{section_group_id}"

    r = graph_get(token, f"{base}/sections")
    sections.extend(r.json().get("value", []))

    r = graph_get(token, f"{base}/sectionGroups")
    for sg in r.json().get("value", []):
        section_groups.append(sg)
        sections.extend(
            list_all_sections_recursive(token, section_group_id=sg["id"])
        )

    return sections


def list_pages_in_section(token: str, section_id: str, pages_url: str | None = None):
    url = pages_url or f"{GRAPH_ROOT}/me/onenote/sections/{section_id}/pages"
    params: dict[str, str] | None = {"$top": "100"}
    pages: list[dict] = []
    while url:
        r = graph_get(token, url, params=params)
        data = r.json()
        pages.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = None
    return pages


def get_page_content_html(token: str, page_id: str) -> str:
    url = f"{GRAPH_ROOT}/me/onenote/pages/{page_id}/content"
    r = graph_get(token, url)
    return r.text
