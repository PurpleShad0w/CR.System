import os
from pathlib import Path
from typing import List, Optional

import msal


def _authority(tenant_id: str) -> str:
    tenant = tenant_id.strip() if tenant_id else 'common'
    return f"https://login.microsoftonline.com/{tenant}"


def build_cache(cache_path: Path) -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if cache_path.exists():
        cache.deserialize(cache_path.read_text(encoding='utf-8', errors='ignore'))
    return cache


def persist_cache(cache: msal.SerializableTokenCache, cache_path: Path) -> None:
    if cache.has_state_changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(cache.serialize(), encoding='utf-8')


def acquire_token_device_flow(
    client_id: str,
    tenant_id: str,
    scopes: List[str],
    cache_path: Path,
    prompt: bool = True,
) -> str:
    """Acquire an access token for Microsoft Graph using device-code flow.

    - Uses a persistent cache so colleagues only authenticate once.
    - Returns an access token string.
    """
    cache = build_cache(cache_path)
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=_authority(tenant_id),
        token_cache=cache,
    )

    # Try silent first
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(scopes=scopes, account=accounts[0])
        if result and 'access_token' in result:
            persist_cache(cache, cache_path)
            return result['access_token']

    # Device code
    flow = app.initiate_device_flow(scopes=scopes)
    if 'user_code' not in flow:
        raise RuntimeError(f"Failed to initiate device flow: {flow}")

    if prompt:
        print(flow['message'])

    result = app.acquire_token_by_device_flow(flow)
    persist_cache(cache, cache_path)

    if 'access_token' not in result:
        raise RuntimeError(f"Auth failed: {result}")

    return result['access_token']


def load_env_scopes(additional_scopes: Optional[str]) -> List[str]:
    base = ['Notes.Read', 'offline_access']
    extra = []
    if additional_scopes:
        extra = [s.strip() for s in additional_scopes.split(',') if s.strip()]
    # Dedup
    seen = set()
    out = []
    for s in base + extra:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
