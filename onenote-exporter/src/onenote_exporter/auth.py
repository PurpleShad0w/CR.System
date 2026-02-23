"""Authentication helpers using MSAL device code flow."""
from __future__ import annotations

import pathlib

import msal

from .config import AUTHORITY, CLIENT_ID, SCOPES, TOKEN_CACHE_PATH


def acquire_token_provider():
    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH and pathlib.Path(TOKEN_CACHE_PATH).exists():
        try:
            cache.deserialize(
                pathlib.Path(TOKEN_CACHE_PATH).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pass

    app = msal.PublicClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        token_cache=cache,
    )

    def get_token() -> str:
        accounts = app.get_accounts()
        result = (
            app.acquire_token_silent(scopes=SCOPES, account=accounts[0])
            if accounts
            else None
        )

        if not result:
            flow = app.initiate_device_flow(scopes=SCOPES)
            if "user_code" not in flow:
                raise RuntimeError("Failed to create device flow")
            print(flow["message"])
            result = app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            raise RuntimeError(
                f"Failed to acquire token: {result.get('error_description')}"
            )

        if TOKEN_CACHE_PATH:
            path = pathlib.Path(TOKEN_CACHE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(cache.serialize(), encoding="utf-8")

        return result["access_token"]

    return get_token
