#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_report_pptx.py (Templates Slides focus)

Renderer for the Page Cards pipeline using "Templates Slides.pptx".

Patch (2026-03-26): fixes two recurring issues reported in production:
  1) Grey frames / dummy picture slots still visible.
     - If an image cannot be resolved/embedded, we now REMOVE (or neutralize) the slot instead
       of leaving the template dummy picture.
     - When a slot is removed, we delete a wider cluster of companion shapes (frames/gradients).
     - When a slot is filled, we also try to neutralize the picture outline/shadow and remove
       small overlay/frame shapes that overlap the slot.

  2) Missing legends.
     - Captions are read from assembled JSON (images: [{'path':..., 'caption':...}, ...]).
     - Legend shapes are detected more robustly (case-insensitive, whitespace-normalized), and
       also inside GROUP shapes.
     - Captions are assigned to the nearest legend for each kept slot (spatial pairing).

Patch v2 (2026-03-26): fix crash on unhashable Shape
  - Shapes are unhashable in python-pptx; we track used legends by id(shape).

CLI (unchanged)
----------------
python render_report_pptx.py --template <pptx> --assembled <json> --out <pptx> --slide-types <json> [--project-root <path>]

Notes
-----
- This file remains self-contained and avoids placeholder APIs because placeholders may not be
  preserved when cloning slides by XML copy.
- We keep XML clone strategy, but we no longer rely on leaving template dummy images as-is.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_VERTICAL_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Pt

from PIL import Image

# ----------------------------
# JSON helpers
# ----------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8', errors='replace'))

# ----------------------------
# Text helpers
# ----------------------------

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
# Slide cloning / type detection
# ----------------------------

def clone_slide(prs_out: Presentation, slide_in) -> Any:
    blank_layout = prs_out.slide_layouts[6]
    new_slide = prs_out.slides.add_slide(blank_layout)
    # background
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
# Shape iteration helpers (supports GROUP shapes)
# ----------------------------

def iter_shapes(obj) -> Iterable[Any]:
    """Yield shapes from a slide or group, recursively for groups."""
    for sh in getattr(obj, 'shapes', []):
        yield sh
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub in iter_shapes(sh):
                    yield sub
        except Exception:
            continue

# ----------------------------
# Text placement
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
    for shape in iter_shapes(slide):
        replace_text_in_shape(shape, mapping)


def _set_title_any(slide_out, title: str) -> None:
    if not title:
        return
    for sh in iter_shapes(slide_out):
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
    # clear existing
    for p in tf.paragraphs:
        p.text = ''
    if not out:
        return
    for i, s in enumerate(out):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = s
        p.level = 0
        p.alignment = PP_ALIGN.LEFT
    try:
        tf.word_wrap = True
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
    for sh in iter_shapes(slide_out):
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            txt = sh.text_frame.text or ''
            if '{{TEXTE_BULLETS}}' in txt or 'Texte Texte' in txt:
                set_bullets_fit(sh, body, max_lines=max_lines)
                return

# ----------------------------
# Image path resolution + conversion
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

# ----------------------------
# Geometry + deletion helpers
# ----------------------------

def _rect(sh) -> Tuple[int, int, int, int]:
    return (int(getattr(sh, 'left', 0)), int(getattr(sh, 'top', 0)), int(getattr(sh, 'width', 0)), int(getattr(sh, 'height', 0)))


def _area(r: Tuple[int, int, int, int]) -> int:
    return max(0, r[2]) * max(0, r[3])


def _intersection(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _center(r: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)


def _near_rect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        abs(ax - bx) <= max(60000, int(aw * 0.05)) and
        abs(ay - by) <= max(60000, int(ah * 0.05)) and
        abs(aw - bw) <= max(120000, int(aw * 0.08)) and
        abs(ah - bh) <= max(120000, int(ah * 0.08))
    )


def _remove_shape(slide, shape) -> bool:
    try:
        slide.shapes._spTree.remove(shape._element)
        return True
    except Exception:
        return False


def _remove_slot_cluster(slide, slot_shape) -> None:
    """Remove slot picture + any overlay/frame shapes tied to it."""
    slot_r = _rect(slot_shape)
    slot_a = _area(slot_r)
    if slot_a <= 0:
        _remove_shape(slide, slot_shape)
        return
    cx, cy = _center(slot_r)
    _remove_shape(slide, slot_shape)
    companions = []
    for sh in list(slide.shapes):
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            continue
        r = _rect(sh)
        a = _area(r)
        if a <= 0:
            continue
        if a > slot_a * 8.0:
            continue
        inter = _intersection(slot_r, r)
        over_slot = inter / slot_a
        over_min = inter / float(min(slot_a, a)) if min(slot_a, a) else 0.0
        rx, ry = _center(r)
        prox = (abs(rx - cx) <= slot_r[2] * 0.35 and abs(ry - cy) <= slot_r[3] * 0.35)
        near = _near_rect(slot_r, r)
        if over_slot >= 0.08 or over_min >= 0.18 or prox or near:
            companions.append(sh)
    for sh in companions:
        _remove_shape(slide, sh)


def _remove_overlays_for_filled_slot(slide, slot_shape) -> None:
    """When a slot is filled, remove small non-text overlay/frame shapes overlapping it."""
    slot_r = _rect(slot_shape)
    slot_a = _area(slot_r)
    if slot_a <= 0:
        return
    to_remove = []
    for sh in list(slide.shapes):
        if sh is slot_shape:
            continue
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            continue
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                continue
        except Exception:
            pass
        r = _rect(sh)
        a = _area(r)
        if a <= 0:
            continue
        if a > slot_a * 6.0:
            continue
        inter = _intersection(slot_r, r)
        if inter <= 0:
            continue
        if (inter / slot_a) >= 0.10 or (inter / float(min(slot_a, a))) >= 0.25:
            to_remove.append(sh)
    for sh in to_remove:
        _remove_shape(slide, sh)

# ----------------------------
# Slot ordering + legend pairing
# ----------------------------

def _center_distance(shape, slide_w: int, slide_h: int) -> float:
    cx = slide_w / 2.0
    cy = slide_h / 2.0
    x = float(getattr(shape, 'left', 0)) + float(getattr(shape, 'width', 0)) / 2.0
    y = float(getattr(shape, 'top', 0)) + float(getattr(shape, 'height', 0)) / 2.0
    return abs(cx - x) + abs(cy - y)


def _picture_slots(slide, slide_w: int, slide_h: int) -> List[Any]:
    pics = []
    for sh in slide.shapes:
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pics.append(sh)
        except Exception:
            continue
    pics.sort(key=lambda sh: _center_distance(sh, slide_w, slide_h))
    return pics


def _is_legend_text(t: str) -> bool:
    if not t:
        return False
    nt = re.sub(r"\s+", " ", t.strip()).lower()
    return ('légende photo' in nt) or ('legende photo' in nt) or ('caption photo' in nt)


def _legend_shapes(slide, slide_w: int, slide_h: int) -> List[Any]:
    leg = []
    for sh in iter_shapes(slide):
        try:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
                if _is_legend_text(sh.text_frame.text or ''):
                    leg.append(sh)
        except Exception:
            continue
    leg.sort(key=lambda sh: _center_distance(sh, slide_w, slide_h))
    return leg


def _shape_center(sh) -> Tuple[float, float]:
    r = _rect(sh)
    return _center(r)


def pair_legends_to_slots(slots: List[Any], legends: List[Any]) -> Dict[int, Any]:
    if not slots or not legends:
        return {}
    unused = set(range(len(legends)))
    map_idx: Dict[int, Any] = {}
    for si, s in enumerate(slots):
        sc = _shape_center(s)
        best_j = None
        best_d = None
        for j in list(unused):
            lc = _shape_center(legends[j])
            d = abs(sc[0] - lc[0]) + abs(sc[1] - lc[1])
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            unused.remove(best_j)
            map_idx[si] = legends[best_j]
    return map_idx

# ----------------------------
# Replace picture embed in-place
# ----------------------------

def replace_picture_image(slide, pic_shape, img_file: Path) -> bool:
    try:
        _image_part, rId = slide.part.get_or_add_image_part(str(img_file))
    except Exception:
        return False
    try:
        blip = pic_shape._element.blipFill.blip
        blip.set(qn('r:embed'), rId)
        return True
    except Exception:
        try:
            blip = pic_shape._element.xpath('.//a:blip')[0]
            blip.set(qn('r:embed'), rId)
            return True
        except Exception:
            return False

# ----------------------------
# Legends + images fill/prune
# ----------------------------

def fill_legends(slide, slots: List[Any], images: List[Any], slide_w: int, slide_h: int) -> None:
    pairs = images_to_pairs(images)
    leg = _legend_shapes(slide, slide_w, slide_h)
    if not leg:
        return

    # Associer chaque slot image à la légende la plus proche (stable quel que soit le layout)
    pairing = pair_legends_to_slots(slots, leg)
    used_legends = set(id(v) for v in pairing.values())

    WHITE = RGBColor(255, 255, 255)
    LEGEND_FONT_PT = 11  # optionnel (mets 10 si tu veux compacter)

    for i in range(min(len(pairs), len(slots))):
        cap = (pairs[i][1] or '').strip()
        if len(cap) > 120:
            cap = cap[:117].rstrip() + '…'

        sh = pairing.get(i)
        if sh is None:
            continue

        try:
            tf = sh.text_frame
            tf.text = cap

            # 1) Coller au BAS du cadre (important)
            # -> la dernière ligne reste sur le bas, et le texte "remonte" si plusieurs lignes
            try:
                tf.vertical_anchor = MSO_VERTICAL_ANCHOR.BOTTOM
            except Exception:
                pass

            # 2) Garder dans le cadre : auto-fit + wrap
            # -> si trop long, police se réduit plutôt que dépasser
            try:
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            except Exception:
                pass

            # 3) Marges minimales (évite le débordement bas)
            try:
                tf.margin_top = 0
                tf.margin_bottom = 0
                tf.margin_left = 0
                tf.margin_right = 0
            except Exception:
                pass

            # 4) Forcer le BLANC (python-pptx perd le style template quand on fait tf.text = ...)
            try:
                for p in tf.paragraphs:
                    p.alignment = PP_ALIGN.LEFT
                    # style au niveau paragraphe
                    try:
                        p.font.color.rgb = WHITE
                        p.font.size = Pt(LEGEND_FONT_PT)
                    except Exception:
                        pass
                    # style au niveau runs (plus robuste)
                    for run in p.runs:
                        try:
                            run.font.color.rgb = WHITE
                            run.font.size = Pt(LEGEND_FONT_PT)
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception:
            pass

    # Nettoyer les placeholders non utilisés (sans casser la mise en page)
    for sh in leg:
        if id(sh) in used_legends:
            continue
        try:
            if _is_legend_text(sh.text_frame.text or ''):
                sh.text_frame.text = ''
        except Exception:
            pass



def fill_images_and_cleanup(slide, images: List[Any], base_dir: Path, repo_root: Path, tmp_dir: Path, stats: Dict[str, Any], slide_w: int, slide_h: int) -> List[Any]:
    pairs = images_to_pairs(images)
    slots = _picture_slots(slide, slide_w, slide_h)
    k = len(pairs)
    if not slots:
        print(f"Image slots: pictures=0 images={k}")
        return []
    if k == 0:
        for sh in list(slots):
            _remove_slot_cluster(slide, sh)
        print("Image slots: pictures=0 images=0 (pruned all)")
        return []
    if k < len(slots):
        for sh in list(slots[k:]):
            _remove_slot_cluster(slide, sh)
        slots = slots[:k]
        print(f"Image slots: pictures={len(slots)} images={k} (kept center={len(slots)})")
    kept: List[Any] = []
    for i in range(min(k, len(slots))):
        path, _cap = pairs[i]
        stats['requested'] += 1
        ip = resolve_image_path(path, base_dir, repo_root)
        if not ip:
            stats['unresolved'].append(path)
            _remove_slot_cluster(slide, slots[i])
            continue
        normp = normalize_image_for_ppt(ip, tmp_dir)
        if not normp:
            stats['unresolved'].append(path)
            _remove_slot_cluster(slide, slots[i])
            continue
        ok = replace_picture_image(slide, slots[i], normp)
        if not ok:
            stats['unresolved'].append(path)
            _remove_slot_cluster(slide, slots[i])
            continue
        _crop_to_fill(slots[i], normp, int(getattr(slots[i], 'width', 0)), int(getattr(slots[i], 'height', 0)))
        try:
            slots[i].line.width = 0
            slots[i].line.fill.background()
        except Exception:
            pass
        try:
            slots[i].shadow.visible = False
        except Exception:
            pass
        _remove_overlays_for_filled_slot(slide, slots[i])
        stats['embedded'] += 1
        kept.append(slots[i])
    return kept

# ----------------------------
# Save helper
# ----------------------------

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

# ----------------------------
# Build deck
# ----------------------------

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
        sw, sh = int(prs_out.slide_width), int(prs_out.slide_height)
        kept_slots = fill_images_and_cleanup(slide_out, imgs, base_dir, repo_root, tmp_dir, stats, sw, sh)
        fill_legends(slide_out, kept_slots, imgs, sw, sh)
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
