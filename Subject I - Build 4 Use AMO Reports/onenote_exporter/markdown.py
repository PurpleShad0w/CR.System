import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


def slugify(s: str) -> str:
    s = (s or '').strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip('-')
    return s or 'page'


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()


def guess_ext_from_content_type(ct: str) -> str:
    if not ct:
        return ''
    ct = ct.split(';', 1)[0].strip().lower()
    mapping = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'application/pdf': '.pdf',
        'audio/mpeg': '.mp3',
        'audio/mp4': '.m4a',
        'audio/x-m4a': '.m4a',
        'audio/wav': '.wav',
        'video/mp4': '.mp4',
    }
    return mapping.get(ct, '')


@dataclass
class ExtractedAsset:
    url: str
    rel_path: str  # relative path in markdown


def html_to_blocks(html: str) -> List[Tuple[str, Dict[str, str]]]:
    """Convert OneNote HTML to a sequence of simplified blocks.

    Returns list of (kind, payload) where kind in: heading, paragraph, image, audio.
    Payload for heading: {'level': '1', 'text': '...'}
    Payload for paragraph: {'text': '...'}
    Payload for image/audio: {'src': '...'}

    This is intentionally conservative to keep output stable.
    """
    soup = BeautifulSoup(html, 'html.parser')

    body = soup.body or soup

    blocks: List[Tuple[str, Dict[str, str]]] = []

    def add(kind: str, **kw):
        blocks.append((kind, {k: str(v) for k, v in kw.items()}))

    # Walk through relevant tags in document order
    for el in body.descendants:
        if getattr(el, 'name', None) is None:
            continue

        name = el.name.lower()

        if name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            txt = el.get_text(' ', strip=True)
            if txt:
                add('heading', level=str(int(name[1])), text=txt)

        elif name == 'p':
            # If the paragraph contains only an image/object, skip (handled separately)
            txt = el.get_text(' ', strip=True)
            if txt:
                add('paragraph', text=txt)

        elif name == 'li':
            txt = el.get_text(' ', strip=True)
            if txt:
                add('paragraph', text=f"- {txt}")

        elif name == 'img':
            src = el.get('src') or ''
            if src:
                add('image', src=src)

        elif name == 'object':
            data = el.get('data') or el.get('src') or ''
            if data:
                # OneNote uses <object data="..."> for attachments/audio.
                add('audio', src=data)

    # Deduplicate adjacent repeated headings/paragraphs (common in exports)
    deduped: List[Tuple[str, Dict[str, str]]] = []
    prev = None
    for b in blocks:
        sig = (b[0], tuple(sorted(b[1].items())))
        if sig == prev:
            continue
        deduped.append(b)
        prev = sig

    return deduped


def render_markdown(frontmatter: Dict[str, str], title: str, blocks: List[Tuple[str, Dict[str, str]]], assets: List[Tuple[str, str]]) -> str:
    """Render markdown in the exact style expected by process_onenote.py.

    - YAML-like frontmatter delimited by ---
    - First content heading: # <title>
    - Then one line per paragraph
    - Images as ![](<relpath>)

    NOTE: We keep OCR marker compatible with the current files: "[IMAGE OCR] ..." (single-bracket)
    because process_onenote currently does not parse OCR anyway.
    """
    lines: List[str] = []
    lines.append('---')
    for k, v in frontmatter.items():
        lines.append(f"{k}: {v}")
    lines.append('content_hash:')
    lines.append('---')

    lines.append(f"# {title}")

    for kind, payload in blocks:
        if kind == 'heading':
            lvl = int(payload.get('level', '1'))
            txt = payload.get('text', '').strip()
            if not txt:
                continue
            prefix = '#' * max(1, min(6, lvl))
            lines.append(f"{prefix} {txt}")

        elif kind == 'paragraph':
            txt = payload.get('text', '').strip()
            if txt:
                lines.append(txt)

        elif kind == 'image':
            src = payload.get('src', '')
            # asset mapping: src->rel_path
            rel = None
            for s, rp in assets:
                if s == src:
                    rel = rp
                    break
            if rel:
                lines.append(f"![]({rel})")

        elif kind == 'audio':
            src = payload.get('src', '')
            rel = None
            for s, rp in assets:
                if s == src:
                    rel = rp
                    break
            if rel:
                lines.append(f"[[AUDIO RECORDING]] [Play]({rel})")

    return "\n".join(lines).strip() + "\n"
