import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .auth import acquire_token_device_flow, load_env_scopes
from .graph import GraphClient, list_notebooks, list_sections, list_pages_in_section, get_page_content_html
from .markdown import slugify, md5_hex, render_markdown, html_to_blocks


def _sanitize_filename(name: str) -> str:
    name = (name or '').strip()
    name = re.sub(r"[\\/:*?\"<>|]", "-", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip() or 'page'


@dataclass
class ExportConfig:
    tenant_id: str
    client_id: str
    additional_scopes: str
    output_dir: Path
    token_cache: Path
    notebook_name: Optional[str] = None
    notebook_id: Optional[str] = None
    merge: bool = False
    formats: str = 'md'


def resolve_notebook(gc: GraphClient, notebook_name: Optional[str], notebook_id: Optional[str]) -> Dict[str, Any]:
    nbs = list_notebooks(gc)
    if notebook_id:
        for nb in nbs:
            if nb.get('id') == notebook_id:
                return nb
        raise RuntimeError(f"Notebook id not found: {notebook_id}")
    if not notebook_name:
        raise RuntimeError("Provide --notebook or --notebook-id")

    needle = notebook_name.strip().lower()
    # exact first
    for nb in nbs:
        if (nb.get('displayName') or '').strip().lower() == needle:
            return nb
    # partial match
    for nb in nbs:
        if needle in (nb.get('displayName') or '').strip().lower():
            return nb
    raise RuntimeError(f"Notebook not found (name match): {notebook_name}")


def export_notebook(cfg: ExportConfig) -> Path:
    scopes = ['https://graph.microsoft.com/.default']
    # For device flow on PublicClientApplication, .default is not always valid.
    # Use delegated scopes instead.
    delegated = load_env_scopes(cfg.additional_scopes)
    delegated = [f"https://graph.microsoft.com/{s}" for s in delegated]

    token = acquire_token_device_flow(cfg.client_id, cfg.tenant_id, delegated, cfg.token_cache)
    gc = GraphClient.create(token)

    nb = resolve_notebook(gc, cfg.notebook_name, cfg.notebook_id)
    nb_name = nb.get('displayName') or 'notebook'
    notebook_slug = nb_name if cfg.notebook_name else nb_name

    out_root = cfg.output_dir / notebook_slug
    pages_dir = out_root / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'notebook': notebook_slug,
        'notebook_id': nb.get('id'),
        'generated_at': None,
        'pages_written': [],
        'assets_written': [],
        'errors': [],
    }

    # Sections
    secs = list_sections(gc, nb['id'])

    merged_lines: List[str] = []
    jsonl_lines: List[str] = []

    for sec in secs:
        sec_name = sec.get('displayName') or ''
        sec_id = sec.get('id') or ''

        pages = list_pages_in_section(gc, sec_id)
        for p in pages:
            try:
                page_id = p.get('id')
                title = p.get('title') or 'Page'
                created = p.get('createdDateTime') or ''
                modified = p.get('lastModifiedDateTime') or ''
                links = p.get('links') or {}
                web_url = (links.get('oneNoteWebUrl') or {}).get('href') or ''
                client_url = (links.get('oneNoteClientUrl') or {}).get('href') or ''

                html = get_page_content_html(gc, page_id)
                blocks = html_to_blocks(html)

                # Download assets referenced by blocks
                asset_pairs: List[Tuple[str, str]] = []  # (src, rel_path)

                page_folder = out_root / page_id
                page_folder.mkdir(parents=True, exist_ok=True)

                for kind, payload in blocks:
                    if kind not in ('image', 'audio'):
                        continue
                    src = payload.get('src', '')
                    if not src:
                        continue
                    # Use md5 of src to mimic existing res-<32hex>.ext naming.
                    h = md5_hex(src)
                    # Try to keep extension from URL (if any)
                    ext = Path(src.split('?', 1)[0]).suffix
                    if not ext:
                        ext = '.jpg' if kind == 'image' else '.m4a'
                    fname = f"res-{h}{ext}" if kind == 'image' else f"aud-{h}{ext}"
                    rel = f"{page_id}/{fname}"
                    fpath = page_folder / fname
                    if not fpath.exists():
                        data = gc.download(src)
                        fpath.write_bytes(data)
                        manifest['assets_written'].append(rel)
                    asset_pairs.append((src, rel))

                # File naming: <slug(title)>-<shortid>.md
                short = 'page'
                try:
                    short = page_id.split('!')[0].split('-', 1)[1][:6]
                except Exception:
                    short = md5_hex(page_id)[:6]

                md_name = f"{slugify(title)}-{short}.md"
                md_path = pages_dir / md_name
                # Avoid collision
                if md_path.exists():
                    i = 2
                    while True:
                        alt = pages_dir / f"{slugify(title)}-{short}-{i}.md"
                        if not alt.exists():
                            md_path = alt
                            break
                        i += 1

                frontmatter = {
                    'notebook': notebook_slug,
                    'section': sec_name,
                    'section_id': f"1-{sec_id.split('-')[0]}" if sec_id else '',
                    'title': _sanitize_filename(title),
                    'page_id': page_id,
                    'created': created,
                    'modified': modified,
                    'web_url': web_url,
                    'client_url': client_url,
                }

                md_text = render_markdown(frontmatter, _sanitize_filename(title), blocks, asset_pairs)
                md_path.write_text(md_text, encoding='utf-8')

                manifest['pages_written'].append({
                    'page_id': page_id,
                    'file': str(md_path.relative_to(out_root)),
                    'section': sec_name,
                    'title': title,
                })

                if cfg.merge:
                    merged_lines.append(md_text)

                # JSONL record per page (LLM ingestion)
                if 'jsonl' in [f.strip() for f in cfg.formats.split(',')]:
                    rec = {
                        'notebook': notebook_slug,
                        'section': sec_name,
                        'section_id': sec_id,
                        'page_id': page_id,
                        'title': title,
                        'created': created,
                        'modified': modified,
                        'web_url': web_url,
                        'client_url': client_url,
                        'markdown_path': str(md_path.relative_to(out_root)),
                    }
                    jsonl_lines.append(json.dumps(rec, ensure_ascii=False))

            except Exception as e:
                manifest['errors'].append({'page_id': p.get('id'), 'error': str(e)})

    # Write merged outputs
    if cfg.merge:
        (out_root / 'merged.md').write_text('\n\n'.join(merged_lines).strip() + '\n', encoding='utf-8')

    if jsonl_lines:
        (out_root / 'merged.jsonl').write_text('\n'.join(jsonl_lines) + '\n', encoding='utf-8')

    (out_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    return out_root
