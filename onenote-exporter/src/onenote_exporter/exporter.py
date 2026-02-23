"""Export notebook to Markdown/DOCX and JSONL."""
from __future__ import annotations

import json
import datetime as _dt
import hashlib
import mimetypes
from typing import Any
import pathlib
import re
import uuid
import time
import imghdr

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

from .graph import (
    get_page_content_html,
    list_pages_in_section,
    list_all_sections_recursive,
)
from .utils import filename_from_url, slugify
try:  # optional at runtime
    from .db import DBManager  # type: ignore
except (ImportError, OSError):  # pragma: no cover
    DBManager = None  # type: ignore


def write_front_matter(info: dict) -> str:
    lines = ["---"]
    for k, v in info.items():
        if isinstance(v, str):
            v = v.replace("\n", " ").strip()
        lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines)


def onenote_html_to_markdown(
    token: str, html: str, assets_dir: pathlib.Path
) -> tuple[str, list[dict[str, Any]]]:
    soup = BeautifulSoup(html, "html.parser")

    collected_assets: list[dict[str, Any]] = []

    for tag in soup.find_all(["img", "object", "a"]):
        url_attr = None
        for attr in ["data-fullres-src", "data-src", "src", "data", "href"]:
            if tag.has_attr(attr):
                url_attr = attr
                break
        if not url_attr:
            continue

        res_url = tag.get(url_attr)
        if not res_url or not res_url.startswith("http"):
            continue

        ocr_text = None
        for a in ["alt", "title", "aria-label"]:
            if tag.has_attr(a) and str(tag.get(a)).strip():
                raw_ocr = str(tag.get(a))
                ocr_text = re.sub(r"\s+", " ", raw_ocr).strip()
                break

        for a in ["alt", "title", "aria-label"]:
            if tag.has_attr(a):
                del tag[a]

        if ocr_text:
            tag["data-ocr-text"] = ocr_text

        fallback_name = f"res-{uuid.uuid4().hex}"
        if res_url.endswith("/$value"):
            local_name = fallback_name
        else:
            local_name = filename_from_url(res_url, fallback_name)
        local_path = assets_dir / local_name

        try:
            from .graph import graph_get

            assets_dir.mkdir(parents=True, exist_ok=True)
            time.sleep(0.3)
            r = graph_get(token, res_url, stream=True)
            cd = r.headers.get("Content-Disposition", "")
            ctype = r.headers.get("Content-Type", "")
            is_audio = ctype.startswith("audio/")
            
            if "filename=" in cd:
                fn = cd.split("filename=")[-1].strip('"; ')
                if fn:
                    local_path = local_path.with_name(fn)
            else:
                if local_path.name == "$value" or res_url.endswith("/$value"):
                    local_path = local_path.with_name(fallback_name)

            ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) if ctype else None
            if ext:
                if local_path.suffix.lower() != ext:
                    local_path = local_path.with_suffix(ext)

            iterator = r.iter_content(chunk_size=8192)
            first = next(iterator, b"")

            ctype_main = (ctype.split(";")[0].strip().lower() if ctype else "")
            ext = mimetypes.guess_extension(ctype_main) if ctype_main else None

            if not ext or ext == ".bin" or ctype_main == "application/octet-stream":
                img_kind = imghdr.what(None, h=first)
                if img_kind == "jpeg":
                    ext = ".jpg"
                elif img_kind:
                    ext = "." + img_kind

                if (not ext or ext == ".bin") and first.startswith(b"%PDF"):
                    ext = ".pdf"
                if (not ext or ext == ".bin") and first.startswith(b"RIFF") and first[8:12] == b"WAVE":
                    ext = ".wav"
                if (not ext or ext == ".bin") and (first.startswith(b"ID3") or first[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
                    ext = ".mp3"
                if (not ext or ext == ".bin") and len(first) >= 12 and first[4:8] == b"ftyp":
                    major = first[8:12]
                    if major.startswith(b"3gp"):
                        ext = ".3gp"
                    else:
                        ext = ".m4a"

            if ext:
                if local_path.suffix.lower() in ("", ".bin", ".dat"):
                    local_path = local_path.with_suffix(ext)

            hasher = hashlib.sha256()
            size = 0

            with open(local_path, "wb") as f:
                if first:
                    f.write(first)
                    hasher.update(first)
                    size += len(first)
                for chunk in iterator:
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)

            rel_path = local_path.relative_to(assets_dir.parent).as_posix()

            is_audio = (ctype_main.startswith("audio/") or ctype_main.endswith("3gpp") or ext == ".3gp" or ext in {".m4a", ".mp3", ".wav"})

            if is_audio:
                if tag.name == "a":
                    tag["href"] = rel_path
                    label = tag.get_text(" ", strip=True) or "Audio recording"
                    tag.string = f"{label} ({rel_path})"
                else:
                    audio_tag = soup.new_tag("p")
                    audio_tag.append("[AUDIO RECORDING] ")
                    link = soup.new_tag("a", href=rel_path)
                    link.string = "Play"
                    audio_tag.append(link)
                    tag.replace_with(audio_tag)
                    continue

            if tag.name == "img":
                tag["src"] = rel_path
                for a in ["data-fullres-src", "data-src"]:
                    if a in tag.attrs:
                        del tag.attrs[a]
            elif tag.name == "object":
                tag["data"] = rel_path

            ocr_text = tag.attrs.pop("data-ocr-text", None)
            if ocr_text:
                caption = soup.new_tag("p")
                caption.string = f"[IMAGE OCR] {ocr_text}"
                tag.insert_after(caption)

            collected_assets.append(
                {
                    "rel_path": rel_path,
                    "mime_type": r.headers.get("Content-Type", ""),
                    "size_bytes": size,
                    "sha256": hasher.hexdigest(),
                }
            )
        except (OSError, ValueError):  # pragma: no cover - network edge
            # Leave absolute URL if download fails
            pass

    for meta in soup.find_all("meta"):
        meta.decompose()

    body_html = str(soup)

    markdown = md(
        body_html,
        heading_style="ATX",
        bullets="*",
        strip=["span", "div"],
    )

    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    return markdown, collected_assets


def export_notebook(
    token: str,
    notebook_id: str,
    notebook_name: str,
    out_root: pathlib.Path,
    merge: bool,
    out_formats: set[str],
    build_db: bool = False,
    db_path: pathlib.Path | None = None,
    since: str | None = None,
    index_only: bool = False,
):
    per_page_dir = out_root / "pages"
    per_page_dir.mkdir(parents=True, exist_ok=True)
    assets_root = out_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    all_pages_records: list[dict] = []
    compiled_md_parts: list[str] = []

    sections = []
    if not index_only:
        sections = list_all_sections_recursive(token, notebook_id=notebook_id)

    # Parse since timestamp if provided
    since_ts: _dt.datetime | None = None
    if since:
        try:
            since_ts = _dt.datetime.fromisoformat(
                since.replace("Z", "+00:00")
            )
        except ValueError:
            print(
                "[WARN] Could not parse --since value '"
                + since
                + "' (expected ISO8601). Ignoring."
            )

    db: Any = None
    if build_db and DBManager is not None:
        default_catalog = (out_root / ".." / "catalog.sqlite").resolve()
        catalog_path = db_path or default_catalog
        db = DBManager(catalog_path)
        # Upsert notebook metadata
        db.upsert_notebook(
            {
                "id": notebook_id,
                "name": notebook_name,
                "slug": slugify(notebook_name),
                "created": "",
                "modified": "",
            }
        )

    # For per-section JSONL aggregation
    section_page_records: dict[str, list[dict]] = {}

    created_count = 0
    updated_count = 0
    skipped_count = 0

    # Index-only mode: scan existing outputs and populate DB then return
    if index_only:
        if build_db and DBManager is not None:
            db = DBManager(db_path or (out_root / ".." / "catalog.sqlite"))
            db.upsert_notebook(
                {
                    "id": notebook_id,
                    "name": notebook_name,
                    "slug": slugify(notebook_name),
                    "created": "",
                    "modified": "",
                }
            )
            # scan pages
            for md_file in (out_root / "pages").glob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta: dict[str, str] = {}
                body = text
                if text.startswith("---\n"):
                    parts = text.split("\n---\n", 1)
                    if len(parts) == 2:
                        fm_block = parts[0][4:]
                        body = parts[1]
                        for line in fm_block.splitlines():
                            if ":" in line:
                                k, v = line.split(":", 1)
                                meta[k.strip()] = v.strip()
                title = meta.get("title") or "Unknown"
                section_name = meta.get("section") or "Unknown Section"
                page_id = meta.get("page_id") or md_file.stem
                section_id = meta.get("section_id") or (
                    "sec-" + slugify(section_name)
                )
                section_slug = slugify(section_name)
                db.upsert_section(
                    {
                        "id": section_id,
                        "notebook_id": notebook_id,
                        "name": section_name,
                        "slug": section_slug,
                        "created": meta.get("section_created", ""),
                        "modified": meta.get("section_modified", ""),
                    }
                )
                # Derive body content: drop first heading line if present
                body_lines = body.splitlines()
                if body_lines and body_lines[0].startswith("# "):
                    body_hash_source = "\n".join(body_lines[1:])
                else:
                    body_hash_source = body
                content_hash = DBManager.hash_content(body_hash_source)
                word_count = len(body_hash_source.split())
                db.upsert_page(
                    {
                        "id": page_id,
                        "section_id": section_id,
                        "notebook_id": notebook_id,
                        "title": title,
                        "slug": slugify(title),
                        "created": meta.get("created", ""),
                        "modified": meta.get("modified", ""),
                        "md_path": str(md_file),
                        "merged_path": "",
                        "jsonl_path": "",
                        "content_hash": content_hash,
                        "word_count": word_count,
                        "page_order": 0,
                    }
                )
                # Assets
                page_assets_dir = assets_root / page_id
                if page_assets_dir.exists():
                    asset_rows = []
                    for f in page_assets_dir.rglob("*"):
                        if f.is_file():
                            try:
                                data = f.read_bytes()
                            except OSError:
                                continue
                            asset_rows.append(
                                {
                                    "rel_path": f.relative_to(out_root)
                                    .as_posix(),
                                    "mime_type": (
                                        mimetypes.guess_type(f.name)[0] or ""
                                    ),
                                    "size_bytes": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                }
                            )
                    if asset_rows:
                        db.upsert_assets(page_id, asset_rows)
            db.commit()
            db.close()
        # Build index.json from existing pages
        index_records = []
        for md_file in (out_root / "pages").glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            meta = {}
            if content.startswith("---\n"):
                parts = content.split("\n---\n", 1)
                if len(parts) == 2:
                    fm_block = parts[0][4:]
                    for line in fm_block.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
            index_records.append(
                {
                    "notebook": notebook_name,
                    "section": meta.get("section", ""),
                    "title": meta.get("title", md_file.stem),
                    "page_id": meta.get("page_id", md_file.stem),
                    "created": meta.get("created", ""),
                    "modified": meta.get("modified", ""),
                    "path": str(md_file),
                    "web_url": meta.get("web_url", ""),
                    "client_url": meta.get("client_url", ""),
                }
            )
        (out_root / "index.json").write_text(
            json.dumps(index_records, indent=2), encoding="utf-8"
        )
        return index_records

    for section in tqdm(sections, desc=f"Sections in {notebook_name}"):
        section_name = section.get("displayName") or "Untitled Section"
        section_id = section.get("id")
        if not section_id:
            continue
        if db:
            db.upsert_section(
                {
                    "id": section_id,
                    "notebook_id": notebook_id,
                    "name": section_name,
                    "slug": slugify(section_name),
                    "created": section.get("createdDateTime", ""),
                    "modified": section.get("lastModifiedDateTime", ""),
                }
            )

        pages_url = section.get("pagesUrl")
        pages = (
            list_pages_in_section(token, str(section_id), pages_url)
            if not index_only
            else []
        )
        for page in tqdm(pages, desc=f"Pages in {section_name}", leave=False):
            page_id = page.get("id")
            if not page_id:
                continue
            page_id = str(page_id)
            title = page.get("title") or "Untitled Page"
            created = page.get("createdDateTime", "")
            modified = page.get("lastModifiedDateTime", "")
            web_url = (
                page.get("links", {})
                .get("oneNoteWebUrl", {})
                .get("href", "")
            )
            client_url = (
                page.get("links", {})
                .get("oneNoteClientUrl", {})
                .get("href", "")
            )

            safe_title = slugify(title)
            page_base = f"{safe_title}-{page_id[:8]}"
            md_path = per_page_dir / f"{page_base}.md"
            page_assets_dir = assets_root / page_id
            page_assets_dir.mkdir(parents=True, exist_ok=True)

            # Incremental skip check
            state = db.get_page_state(page_id) if db else None
            modified_dt: _dt.datetime | None = None
            try:
                if modified:
                    modified_dt = _dt.datetime.fromisoformat(
                        modified.replace("Z", "+00:00")
                    )
            except ValueError:
                modified_dt = None

            if (
                since_ts
                and modified_dt
                and modified_dt < since_ts
                and not state
            ):
                # Page older than window and not seen before: skip exporting
                skipped_count += 1
                continue

            skip_due_to_unchanged = False
            if state and state.modified == modified:
                # We'll still open file if missing; otherwise skip
                if state.content_hash and md_path.exists():
                    skipped_count += 1
                    continue
                skip_due_to_unchanged = True

            try:
                time.sleep(0.3)
                html = get_page_content_html(token, page_id)
                md_body, collected_assets = onenote_html_to_markdown(
                    token, html, page_assets_dir
                )
            except RuntimeError as e:
                print(f"[WARN] Skipping page {page_id} due to throttling: {e}")
                skipped_count += 1
                continue


            content_hash = (
                DBManager.hash_content(md_body)
                if db and DBManager is not None
                else ""
            )

            fm = write_front_matter(
                {
                    "notebook": notebook_name,
                    "section": section_name,
                    "section_id": section_id,
                    "title": title,
                    "page_id": page_id,
                    "created": created,
                    "modified": modified,
                    "web_url": web_url,
                    "client_url": client_url,
                    "content_hash": content_hash,
                }
            )

            md_full = fm + f"# {title}\n\n" + md_body
            md_path.write_text(md_full, encoding="utf-8")

            if db and DBManager is not None:
                if (
                    state
                    and state.content_hash
                    and state.content_hash != content_hash
                ):
                    updated_count += 1
                elif not state:
                    created_count += 1
                elif skip_due_to_unchanged:
                    skipped_count += 1
                db.upsert_page(
                    {
                        "id": page_id,
                        "section_id": section_id,
                        "notebook_id": notebook_id,
                        "title": title,
                        "slug": safe_title,
                        "created": created,
                        "modified": modified,
                        "md_path": str(md_path),
                        "merged_path": "",  # filled later if merge
                        "jsonl_path": "",  # per-page JSONL not tracked yet
                        "content_hash": content_hash,
                        "word_count": len(md_body.split()),
                        "page_order": 0,
                    }
                )
                if collected_assets:
                    db.upsert_assets(page_id, collected_assets)

            rec_obj = {
                "notebook": notebook_name,
                "section": section_name,
                "title": title,
                "page_id": page_id,
                "created": created,
                "modified": modified,
                "path": str(md_path),
                "web_url": web_url,
                "client_url": client_url,
            }
            all_pages_records.append(rec_obj)
            section_page_records.setdefault(section_id, []).append(rec_obj)

            if merge:
                compiled_md_parts.append(f"\n\n# {title}\n\n")
                compiled_md_parts.append(md_body)

    (out_root / "index.json").write_text(
        json.dumps(all_pages_records, indent=2), encoding="utf-8"
    )

    # Per-section JSONL files
    for section in sections:
        section_id = section.get("id")
        if not section_id:
            continue
        recs = section_page_records.get(section_id) or []
        if not recs:
            continue
        section_slug = slugify(section.get("displayName") or section_id)
        jsonl_file = out_root / section_slug / "section.jsonl"
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_file, "w", encoding="utf-8") as sf:
            for rec in recs:
                try:
                    content = pathlib.Path(rec["path"]).read_text(
                        encoding="utf-8"
                    )
                except OSError:
                    content = ""
                obj = {
                    "id": rec["page_id"],
                    "title": rec["title"],
                    "notebook": rec["notebook"],
                    "section": rec["section"],
                    "created": rec["created"],
                    "modified": rec["modified"],
                    "content": content,
                }
                sf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        # Update jsonl_path for pages in this section if DB present
        if db and DBManager is not None:
            for rec in recs:
                # Recompute content hash from body (exclude front matter)
                try:
                    raw = pathlib.Path(rec["path"]).read_text(encoding="utf-8")
                except OSError:
                    raw = ""
                body = raw
                if raw.startswith("---\n") and "\n---\n" in raw:
                    body = raw.split("\n---\n", 1)[1]
                if body.startswith("# "):
                    body_for_hash = "\n".join(body.splitlines()[1:])
                else:
                    body_for_hash = body
                content_hash = DBManager.hash_content(body_for_hash)
                word_count = len(body_for_hash.split())
                db.upsert_page(
                    {
                        "id": rec["page_id"],
                        "section_id": section_id,
                        "notebook_id": notebook_id,
                        "title": rec["title"],
                        "slug": slugify(rec["title"]),
                        "created": rec["created"],
                        "modified": rec["modified"],
                        "md_path": rec["path"],
                        "merged_path": "",  # set later if merge
                        "jsonl_path": str(jsonl_file),
                        "content_hash": content_hash,
                        "word_count": word_count,
                        "page_order": 0,
                    }
                )

    if merge:
        compiled_md = (
            f"# Notebook: {notebook_name}\n\n" + "\n".join(compiled_md_parts)
        )
        compiled_md_path = out_root / f"{slugify(notebook_name)}-compiled.md"
        compiled_md_path.write_text(compiled_md, encoding="utf-8")

    if "docx" in out_formats:
        try:
            import pypandoc  # type: ignore

            docx_path = out_root / (
                f"{slugify(notebook_name)}-compiled.docx"
            )
            pypandoc.convert_text(
                compiled_md,
                "docx",
                format="md",
                outputfile=str(docx_path),
            )
        except (OSError, RuntimeError) as e:
            print(
                f"[WARN] DOCX conversion failed (pandoc/pypandoc): {e}"
            )

    jsonl_path = out_root / f"{slugify(notebook_name)}-pages.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for rec in all_pages_records:
            try:
                content = pathlib.Path(rec["path"]).read_text(encoding="utf-8")
            except OSError:
                content = ""
            obj = {
                "id": rec["page_id"],
                "title": rec["title"],
                "notebook": rec["notebook"],
                "section": rec["section"],
                "created": rec["created"],
                "modified": rec["modified"],
                "content": content,
            }
            jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if db:
        # Update merged_path and jsonl_path for all pages if merge enabled
        if merge and DBManager is not None:
            merged_path_str = str(
                out_root / f"{slugify(notebook_name)}-compiled.md"
            )
            for rec in all_pages_records:
                try:
                    raw = pathlib.Path(rec["path"]).read_text(encoding="utf-8")
                except OSError:
                    raw = ""
                body = raw
                if raw.startswith("---\n") and "\n---\n" in raw:
                    body = raw.split("\n---\n", 1)[1]
                if body.startswith("# "):
                    body_for_hash = "\n".join(body.splitlines()[1:])
                else:
                    body_for_hash = body
                content_hash = DBManager.hash_content(body_for_hash)
                word_count = len(body_for_hash.split())
                db.upsert_page(
                    {
                        "id": rec["page_id"],
                        "section_id": "",  # keep existing
                        "notebook_id": notebook_id,
                        "title": rec["title"],
                        "slug": slugify(rec["title"]),
                        "created": rec["created"],
                        "modified": rec["modified"],
                        "md_path": rec["path"],
                        "merged_path": merged_path_str,
                        "jsonl_path": str(jsonl_path),
                        "content_hash": content_hash,
                        "word_count": word_count,
                        "page_order": 0,
                    }
                )
        db.commit()
        db.close()
        print(
            "[catalog] pages created: "
            + str(created_count)
            + ", updated: "
            + str(updated_count)
            + ", skipped: "
            + str(skipped_count)
        )
    return all_pages_records
