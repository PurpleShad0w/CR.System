"""Console entrypoint for OneNote exporter."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import cast

from .auth import acquire_token_provider
from .config import CLIENT_ID, OUTPUT_DIR, DB_PATH
from .exporter import export_notebook
from .graph import list_notebooks
from .utils import slugify


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a Microsoft OneNote notebook to Markdown/DOCX and JSONL"
        )
    )
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--notebook", help="Notebook name (substring match)")
    group.add_argument("--notebook-id", help="Notebook ID (exact)")

    p.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory on host (default: ./output)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Create a single compiled Markdown",
    )
    p.add_argument(
        "--formats",
        default="md",
        help="Comma-separated: md,docx (default: md)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List notebooks and exit",
    )
    p.add_argument(
        "--build-db",
        action="store_true",
        help="Populate/update local SQLite catalog (incremental export)",
    )
    p.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help="Path to SQLite catalog (default: ./output/catalog.sqlite)",
    )
    p.add_argument(
        "--since",
        help="Only (re)export pages modified on/after this ISO8601 timestamp",
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Index existing output folder into DB without calling Graph",
    )

    return p.parse_args()


def resolve_notebook_choice(
    get_token: str, name_contains: str | None, notebook_id: str | None
):
    notebooks = list_notebooks(get_token)
    if notebook_id:
        for n in notebooks:
            if n.get("id") == notebook_id:
                return n
        raise SystemExit(f"Notebook with id {notebook_id} not found.")
    if name_contains:
        matches = [
            n
            for n in notebooks
            if name_contains.lower() in (n.get("displayName", "").lower())
        ]
        if not matches:
            names = ", ".join([n.get("displayName", "") for n in notebooks])
            raise SystemExit(
                "No notebook name contains '"
                + name_contains
                + "'. Available: "
                + names
            )
        if len(matches) > 1:
            print("Multiple notebooks matched; picking the first. Matches:")
            for n in matches:
                print(" -", n.get("displayName"), n.get("id"))
        return matches[0]
    if len(notebooks) == 1:
        return notebooks[0]
    raise SystemExit(
        "Please specify --notebook or --notebook-id. Available: "
        + ", ".join([n.get("displayName", "") for n in notebooks])
    )


def main() -> None:
    if not CLIENT_ID:
        print("Error: CLIENT_ID is required (env var).")
        sys.exit(1)

    get_token = acquire_token_provider()

    args = parse_args()

    if args.list:
        notebooks = list_notebooks(get_token)
        print("Available notebooks:")
        for n in notebooks:
            print(f"- {n.get('displayName')}  (id: {n.get('id')})")
        return

    out_dir = pathlib.Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nb = resolve_notebook_choice(get_token, args.notebook, args.notebook_id)
    nb_name = nb.get("displayName") or "Untitled Notebook"
    nb_id = cast(str, nb.get("id"))

    nb_out_root = out_dir / slugify(nb_name)
    nb_out_root.mkdir(parents=True, exist_ok=True)

    out_formats = {
        f.strip().lower()
        for f in args.formats.split(",")
        if f.strip()
    }
    export_notebook(
        token=get_token,
        notebook_id=nb_id,
        notebook_name=nb_name,
        out_root=nb_out_root,
        merge=args.merge,
        out_formats=out_formats,
        build_db=args.build_db,
        db_path=pathlib.Path(args.db_path),
        since=args.since,
        index_only=args.index_only,
    )

    print(f"\nDone. Outputs in: {nb_out_root}")
    print("Artifacts:")
    print("- Per-page Markdown in pages/")
    print("- Assets in assets/")
    print("- index.json (page listing)")
    print("- <notebook>-pages.jsonl (for LLM ingestion)")
    if args.merge:
        print("- <notebook>-compiled.md")
        if "docx" in out_formats:
            print("- <notebook>-compiled.docx")


if __name__ == "__main__":
    main()
