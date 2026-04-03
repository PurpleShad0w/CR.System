import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests

GRAPH_ROOT = 'https://graph.microsoft.com/v1.0'


@dataclass
class GraphClient:
    token: str
    session: requests.Session

    @classmethod
    def create(cls, token: str) -> 'GraphClient':
        s = requests.Session()
        s.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
        })
        return cls(token=token, session=s)

    def _request(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None, stream: bool = False) -> requests.Response:
        # Basic retry policy for throttling & transient errors.
        for attempt in range(7):
            r = self.session.request(method, url, params=params, stream=stream, timeout=180)
            if r.status_code in (429, 503, 504, 500):
                retry_after = r.headers.get('Retry-After')
                sleep_s = float(retry_after) if retry_after else min(1.5 * (2 ** attempt), 20)
                time.sleep(sleep_s)
                continue
            return r
        return r

    def get_json(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = path if path.startswith('http') else f'{GRAPH_ROOT}{path}'
        r = self._request('GET', url, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f'Graph error {r.status_code} url={url} body={r.text[:2000]}')
        return r.json()

    def iter_paged(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        url = path if path.startswith('http') else f'{GRAPH_ROOT}{path}'
        while url:
            r = self._request('GET', url, params=params)
            if r.status_code >= 400:
                raise RuntimeError(f'Graph error {r.status_code} url={url} body={r.text[:2000]}')
            data = r.json()
            for item in data.get('value', []):
                yield item
            url = data.get('@odata.nextLink')
            params = None  # already encoded in nextLink

    def download(self, url: str) -> bytes:
        r = self._request('GET', url, stream=True)
        if r.status_code >= 400:
            raise RuntimeError(f'Download error {r.status_code} url={url} body={r.text[:2000]}')
        return r.content


def list_notebooks(gc: GraphClient) -> List[Dict[str, Any]]:
    return list(gc.iter_paged('/me/onenote/notebooks?$select=id,displayName'))


def list_sections(gc: GraphClient, notebook_id: str) -> List[Dict[str, Any]]:
    q = f"/me/onenote/notebooks/{notebook_id}/sections?$select=id,displayName"
    return list(gc.iter_paged(q))


def list_pages_in_section(gc: GraphClient, section_id: str) -> List[Dict[str, Any]]:
    # Keep fields that we need for frontmatter.
    q = (
        f"/me/onenote/sections/{section_id}/pages"
        "?$select=id,title,createdDateTime,lastModifiedDateTime,contentUrl,links"
    )
    return list(gc.iter_paged(q))


def get_page_content_html(gc: GraphClient, page_id: str) -> str:
    # Use /content endpoint (HTML). includeIDs helps stability.
    url = f"{GRAPH_ROOT}/me/onenote/pages/{page_id}/content"
    r = gc._request('GET', url, params={'includeIDs': 'true'})
    if r.status_code >= 400:
        raise RuntimeError(f'Graph page content error {r.status_code} page={page_id} body={r.text[:2000]}')
    return r.text
