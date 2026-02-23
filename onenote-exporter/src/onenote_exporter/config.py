import os
import pathlib

TENANT_ID = os.getenv("TENANT_ID", "").strip()
CLIENT_ID = os.getenv("CLIENT_ID", "").strip()

AUTHORITY = (
    f"https://login.microsoftonline.com/{TENANT_ID}"
    if TENANT_ID
    else "https://login.microsoftonline.com/common"
)

DEFAULT_SCOPES = ["Notes.Read"]
ADDITIONAL_SCOPES = [
    s.strip()
    for s in os.getenv("ADDITIONAL_SCOPES", "").split(",")
    if s.strip()
]
SCOPES = DEFAULT_SCOPES + ADDITIONAL_SCOPES

OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
TOKEN_CACHE_PATH = os.getenv("TOKEN_CACHE_PATH", "./.cache/token_cache.bin")
DB_PATH = os.getenv("DB_PATH", "./output/catalog.sqlite")
