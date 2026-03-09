import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .exporter import ExportConfig, export_notebook
from .graph import GraphClient, list_notebooks
from .auth import acquire_token_device_flow, load_env_scopes


def main():
    ap = argparse.ArgumentParser(prog='onenote-exporter-native', description='Export OneNote notebooks via Microsoft Graph into Markdown + assets (pipeline compatible, no Docker).')
    ap.add_argument('--config', default='.env', help='Path to .env config file')
    ap.add_argument('--list', action='store_true', help='List available notebooks')
    ap.add_argument('--notebook', default='', help='Notebook name (partial match)')
    ap.add_argument('--notebook-id', default='', help='Notebook id (exact match)')
    ap.add_argument('--merge', action='store_true', help='Produce merged.md')
    ap.add_argument('--formats', default='md', help='Output formats: md,jsonl (docx intentionally not supported here)')
    ap.add_argument('--output-dir', default='input/onenote-exporter/output', help='Output directory root')
    ap.add_argument('--token-cache', default='input/onenote-exporter/cache/token_cache.json', help='Token cache path')
    args = ap.parse_args()

    load_dotenv(args.config)
    tenant_id = os.environ.get('TENANT_ID', 'common')
    client_id = os.environ.get('CLIENT_ID')
    additional_scopes = os.environ.get('ADDITIONAL_SCOPES', '')

    if not client_id:
        raise SystemExit('CLIENT_ID is required (set in .env)')

    scopes = load_env_scopes(additional_scopes)
    delegated = [f"https://graph.microsoft.com/{s}" for s in scopes]

    token = acquire_token_device_flow(client_id, tenant_id, delegated, Path(args.token_cache))
    gc = GraphClient.create(token)

    if args.list:
        nbs = list_notebooks(gc)
        for nb in nbs:
            print(f"{nb.get('displayName')}\t{nb.get('id')}")
        return

    cfg = ExportConfig(
        tenant_id=tenant_id,
        client_id=client_id,
        additional_scopes=additional_scopes,
        output_dir=Path(args.output_dir),
        token_cache=Path(args.token_cache),
        notebook_name=args.notebook or None,
        notebook_id=args.notebook_id or None,
        merge=args.merge,
        formats=args.formats,
    )

    out = export_notebook(cfg)
    print(f"Wrote: {out}")


if __name__ == '__main__':
    main()
