#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""render_report_pptx.py

Hotfix: use existing picture shapes as "slots" and replace their embedded image.

Context
-------
In "Templates Slides.pptx", the image locations are not real PowerPoint placeholders after
we clone the slide by copying XML. Your logs show placeholders=0 on every slide.
You also clarified that the template uses the **same dummy image** repeated for all slots.

Therefore, the most robust approach is:
- find the existing PICTURE shapes on the cloned slide (they are the slots)
- replace the picture's blip relationship (r:embed) to point to the new image

This keeps:
- the template picture style (shadow/gradient) because the shape stays the same
- the correct z-order (legends remain visible)

Legends
-------
Legends are updated only by replacing text in existing text boxes containing "Légende Photo".

Escape safety
-------------
- Regex patterns and backrefs use raw strings.
- Backslashes shown as "\\n".

"""

from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Pt
from pptx.oxml.ns import qn

from PIL import Image


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def normalize_whitespace(text: str) -> str:
    t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_markdown(text: str) -> str:
    if not text:
        return ''
    t = text
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace('•', '-')
    return t.strip()


def sanitize_client_body(text: str) -> str:
    if not text:
        return ''
    t = normalize_whitespace(strip_markdown(text))
    out: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            out.append('')
            continue
        low = s.lower()
        if 'page_id=' in low:
            continue
        if low.startswith('preuve:') or 'preuve:' in low:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ----------------------------
# Template slide cloning
# ----------------------------

def clone_slide(prs_out: Presentation, slide_in) -> Any:
    blank_layout = prs_out.slide_layouts[6]
    new_slide = prs_out.slides.add_slide(blank_layout)
    try:
        new_slide._element.get_or_add_bg()._set_bg(slide_in._element.get_or_add_bg())
    except Exception:
        pass
    for shape in slide_in.shapes:
        new_slide.shapes._spTree.insert_element_before(deepcopy(shape._element), 'p:extLst')
    return new_slide


def slide_text(slide) -> str:
    parts: List[str] = []
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            parts.append(shape.text_frame.text or '')
    return "\n".join(parts)


def detect_template_slide_types(tpl: Presentation, slide_types_cfg: Dict[str, Any]) -> Dict[str, Any]:
    types = (slide_types_cfg.get('types') or {})
    found: Dict[str, Any] = {}
    for sl in tpl.slides:
        stxt = slide_text(sl)
        for tname, tcfg in types.items():
            toks = tcfg.get('detect_tokens') or []
            if toks and all(tok in stxt for tok in toks):
                found.setdefault(tname, sl)
    return found


# ----------------------------
# Placeholder text replacement
# ----------------------------

def replace_text_in_shape(shape, mapping: Dict[str, str]) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    full = tf.text or ''
    replaced = full
    for k, v in mapping.items():
        if k in replaced:
            replaced = replaced.replace(k, v)
    if replaced != full:
        tf.text = replaced


def replace_placeholders(slide, mapping: Dict[str, str]) -> None:
    for shape in slide.shapes:
        replace_text_in_shape(shape, mapping)


def _set_title_any(slide_out, title: str) -> None:
    if not title:
        return
    for sh in slide_out.shapes:
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            txt = sh.text_frame.text or ''
            if '{{SLIDE_TITLE}}' in txt or 'Titre section' in txt:
                sh.text_frame.text = title
                return


def set_bullets_fit(shape, body: str, *, max_lines: int = 12, min_font_pt: int = 10) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    lines = [ln.strip() for ln in normalize_whitespace(strip_markdown(sanitize_client_body(body))).split('\n') if ln.strip()]
    out: List[str] = []
    for ln in lines:
        s = ln
        while s.startswith('- '):
            s = s[2:].strip()
        if len(s) > 190:
            s = s[:187].rstrip() + '…'
        out.append(s)
    if len(out) > max_lines:
        out = out[:max_lines-1] + ['…']
    for p in tf.paragraphs:
        p.text = ''
    if not out:
        return
    for i, s in enumerate(out):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = s
        p.level = 0
        p.alignment = PP_ALIGN.LEFT
    tf.word_wrap = True
    try:
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    except Exception:
        pass
    try:
        for p in tf.paragraphs:
            if p.font.size is not None and p.font.size < Pt(min_font_pt):
                p.font.size = Pt(min_font_pt)
    except Exception:
        pass


def _set_body_any(slide_out, body: str, *, max_lines: int) -> None:
    for sh in slide_out.shapes:
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            txt = sh.text_frame.text or ''
            if '{{TEXTE_BULLETS}}' in txt or 'Texte Texte' in txt:
                set_bullets_fit(sh, body, max_lines=max_lines)
                return


# ----------------------------
# Image resolution + conversion
# ----------------------------

_img_cache: Dict[str, Optional[Path]] = {}
_conv_cache: Dict[Path, Path] = {}


def infer_repo_root(assembled_path: Path) -> Path:
    start = assembled_path.resolve()
    for cand in [start.parent] + list(start.parents):
        if (cand / 'process').exists() and (cand / 'input').exists():
            return cand
        if cand.name.lower() == 'process' and (cand.parent / 'input').exists():
            return cand.parent
    return assembled_path.parent.parent.parent


def resolve_image_path(img: str, base_dir: Path, repo_root: Path) -> Optional[Path]:
    if not img:
        return None
    if img in _img_cache:
        return _img_cache[img]
    p = Path(img)
    if p.is_absolute() and p.exists():
        _img_cache[img] = p
        return p
    cand = (base_dir / img).resolve()
    if cand.exists():
        _img_cache[img] = cand
        return cand
    bn = p.name
    cand2 = (base_dir / bn).resolve()
    if cand2.exists():
        _img_cache[img] = cand2
        return cand2
    for sub in ('process', 'input'):
        base = repo_root / sub
        if not base.exists():
            continue
        try:
            for hit in base.rglob(bn):
                if hit.is_file():
                    _img_cache[img] = hit
                    return hit
        except Exception:
            continue
    _img_cache[img] = None
    return None


def magic_type(p: Path) -> str:
    try:
        b = p.read_bytes()[:16]
    except Exception:
        return 'unknown'
    if len(b) >= 3 and b[0:3] == b'\xFF\xD8\xFF':
        return 'jpeg'
    if len(b) >= 8 and b[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    return 'unknown'


def normalize_image_for_ppt(p: Path, tmp_dir: Path) -> Optional[Path]:
    if not p.exists():
        return None
    if p in _conv_cache:
        return _conv_cache[p]
    kind = magic_type(p)
    ext = p.suffix.lower()
    if kind == 'jpeg' and ext in ('.jpg', '.jpeg'):
        _conv_cache[p] = p
        return p
    if kind == 'png' and ext == '.png':
        _conv_cache[p] = p
        return p
    try:
        im = Image.open(p)
        if im.mode not in ('RGB', 'RGBA'):
            im = im.convert('RGB')
        out = tmp_dir / (p.stem + '.png')
        im.save(out, format='PNG', optimize=True)
        _conv_cache[p] = out
        return out
    except Exception:
        return None


def images_to_pairs(images: List[Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for im in images or []:
        if isinstance(im, str):
            out.append((im, ''))
        elif isinstance(im, dict):
            out.append(((im.get('path') or ''), (im.get('caption') or '')))
    return [(p, c) for p, c in out if p]


def _crop_to_fill(pic_shape, img_path: Path, box_w: int, box_h: int) -> None:
    try:
        im = Image.open(img_path)
        w, h = im.size
        if not w or not h or not box_h:
            return
        img_ar = w / h
        box_ar = float(box_w) / float(box_h)
        pic_shape.crop_left = 0.0
        pic_shape.crop_right = 0.0
        pic_shape.crop_top = 0.0
        pic_shape.crop_bottom = 0.0
        if img_ar > box_ar:
            new_w = h * box_ar
            excess = max(0.0, w - new_w)
            frac = (excess / w) / 2.0 if w else 0.0
            pic_shape.crop_left = frac
            pic_shape.crop_right = frac
        else:
            new_h = w / box_ar
            excess = max(0.0, h - new_h)
            frac = (excess / h) / 2.0 if h else 0.0
            pic_shape.crop_top = frac
            pic_shape.crop_bottom = frac
    except Exception:
        return


def pick_picture_slots(slide) -> List[Any]:
    """Return the existing PICTURE shapes (slots) ordered top/left."""
    pics = []
    for sh in slide.shapes:
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pics.append(sh)
        except Exception:
            continue
    pics.sort(key=lambda sh: (int(getattr(sh, 'top', 0)), int(getattr(sh, 'left', 0))))
    return pics


def replace_picture_image(slide, pic_shape, img_file: Path) -> Optional[Any]:
    """Replace the image embedded in an existing picture shape, preserving its style."""
    try:
        image_part, rId = slide.part.get_or_add_image_part(str(img_file))
    except Exception:
        return None
    try:
        blip = pic_shape._element.blipFill.blip
        blip.set(qn('r:embed'), rId)
        return pic_shape
    except Exception:
        # fallback: try different path to blip
        try:
            blip = pic_shape._element.xpath('.//a:blip')[0]
            blip.set(qn('r:embed'), rId)
            return pic_shape
        except Exception:
            return None


def fill_images_using_picture_shapes(slide, images: List[Any], base_dir: Path, repo_root: Path, tmp_dir: Path, stats: Dict[str, Any]) -> None:
    pairs = images_to_pairs(images)
    if not pairs:
        return
    slots = pick_picture_slots(slide)
    print(f"Image slots: pictures={len(slots)} images={len(pairs)}")
    if not slots:
        return
    count = min(len(pairs), len(slots))
    for i in range(count):
        path, _cap = pairs[i]
        stats['requested'] += 1
        ip = resolve_image_path(path, base_dir, repo_root)
        if not ip:
            stats['unresolved'].append(path)
            continue
        normp = normalize_image_for_ppt(ip, tmp_dir)
        if not normp:
            stats['unresolved'].append(path)
            continue
        pic = replace_picture_image(slide, slots[i], normp)
        if pic is None:
            stats['unresolved'].append(path)
            continue
        _crop_to_fill(pic, normp, int(getattr(pic, 'width', 0)), int(getattr(pic, 'height', 0)))
        stats['embedded'] += 1


def fill_legends_in_place(slide, images: List[Any]) -> None:
    pairs = images_to_pairs(images)
    legends = []
    for sh in slide.shapes:
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            if 'Légende Photo' in (sh.text_frame.text or ''):
                legends.append(sh)
    for idx, sh in enumerate(legends):
        cap = ''
        if idx < len(pairs):
            cap = (pairs[idx][1] or '').strip()
            if len(cap) > 80:
                cap = cap[:77].rstrip() + '…'
        try:
            sh.text_frame.text = cap
        except Exception:
            pass


def safe_save(prs: Presentation, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out_path))
        return out_path
    except PermissionError:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        alt = out_path.with_name(out_path.stem + f'_{stamp}' + out_path.suffix)
        prs.save(str(alt))
        print(f"WARNING: target PPTX locked, wrote instead: {alt}")
        return alt


def build_deck(base_template: Path, assembled_path: Path, out_path: Path, slide_types_path: Path, repo_root_override: Optional[Path] = None) -> None:
    base_tpl = Presentation(str(base_template))
    data = load_json(assembled_path)
    repo_root = repo_root_override or infer_repo_root(assembled_path)
    base_dir = assembled_path.parent
    tmp_dir = base_dir / '_tmp_img'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stats = {'requested': 0, 'embedded': 0, 'unresolved': []}
    cfg = load_json(slide_types_path) if slide_types_path.exists() else {'types': {}, 'defaults': {}}
    catalog = detect_template_slide_types(base_tpl, cfg)

    prs_out = Presentation()
    prs_out.slide_width = base_tpl.slide_width
    prs_out.slide_height = base_tpl.slide_height

    today = datetime.now().strftime('%d/%m/%Y')
    global_map = {
        '{{DATE}}': today,
        '{{VERSION}}': '1',
        '{{CONTACT_EMAIL}}': 'contact@build4use.eu',
    }

    def emit_content_slide(s: Dict[str, Any]):
        stype = (s.get('type') or 'CONTENT_TEXT').strip()
        title = (s.get('title') or '').strip()
        body = (s.get('bullets') or s.get('body') or '')
        imgs = s.get('images') or []

        slide_in = catalog.get(stype) or catalog.get('CONTENT_TEXT_IMAGES') or catalog.get('CONTENT_TEXT')
        if not slide_in:
            return
        slide_out = clone_slide(prs_out, slide_in)
        replace_placeholders(slide_out, global_map)
        _set_title_any(slide_out, title)
        _set_body_any(slide_out, body, max_lines=12 if imgs else 14)
        fill_legends_in_place(slide_out, imgs)
        fill_images_using_picture_shapes(slide_out, imgs, base_dir, repo_root, tmp_dir, stats)

    slides_spec = data.get('slides')
    if isinstance(slides_spec, list):
        for s in slides_spec:
            if isinstance(s, dict) and s.get('type') != 'PART_DIVIDER':
                emit_content_slide(s)

    final_path = safe_save(prs_out, out_path)
    print(f"Repo root: {repo_root}")
    print(f"Images: embedded {stats['embedded']} / requested {stats['requested']}")
    if stats['unresolved']:
        print('Unresolved image paths (sample):')
        for u in stats['unresolved'][:12]:
            print(' -', u)
    print(f"Wrote: {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--assembled', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slide-types', required=True)
    ap.add_argument('--project-root', default='')
    args = ap.parse_args()
    root_override = Path(args.project_root) if args.project_root else None
    build_deck(Path(args.template), Path(args.assembled), Path(args.out), Path(args.slide_types), repo_root_override=root_override)


if __name__ == '__main__':
    main()
