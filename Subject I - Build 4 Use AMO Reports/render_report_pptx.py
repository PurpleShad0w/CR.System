#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""render_report_pptx.py (escape-safe)

Combined fix (base template + boss info template) with safe escaping.

- Base template: cover / TOC / part dividers / conclusion
- Info template (Templates Slides.pptx): content slides

This version avoids fragile escape sequences by using raw regex patterns and raw
replacement strings where applicable (e.g. r"\1").
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN


# -----------------------------
# IO
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


# -----------------------------
# Text helpers
# -----------------------------

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
    """Remove evidence traces from content that should not be client-visible."""
    if not text:
        return ''
    t = normalize_whitespace(strip_markdown(text))
    out: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            out.append('')
            continue
        if 'page_id=' in s:
            continue
        low = s.lower()
        if low.startswith('preuve:') or 'preuve:' in low:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def first_bullet_key(body: str) -> str:
    t = sanitize_client_body(body)
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith('- '):
            return s[2:].strip()[:90]
        if s.lower().startswith('constat'):
            return s[:90]
    for ln in t.splitlines():
        s = ln.strip()
        if s:
            return s[:90]
    return ''


# -----------------------------
# PPTX cloning
# -----------------------------

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


# -----------------------------
# Placeholder replacement (base template)
# -----------------------------

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


# -----------------------------
# Info template helpers
# -----------------------------

def find_shape_containing(slide, token: str):
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            if token in (shape.text_frame.text or ''):
                return shape
    return None


def find_shapes_exact(slide, exact: str) -> List[Any]:
    out = []
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            if (shape.text_frame.text or '').strip() == exact:
                out.append(shape)
    return out


def set_text(shape, text: str, *, font_size_pt: Optional[int] = None) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    if font_size_pt is not None:
        try:
            p.font.size = Pt(font_size_pt)
        except Exception:
            pass


def set_bullets(shape, body: str, *, font_size_pt: int = 16, max_lines: int = 14) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True

    body = sanitize_client_body(body)
    lines = [ln.strip() for ln in normalize_whitespace(strip_markdown(body)).split('\n') if ln.strip()]

    norm_lines: List[str] = []
    for ln in lines:
        norm_lines.append(ln if ln.startswith('- ') else ('- ' + ln))
    norm_lines = norm_lines[:max_lines]

    if not norm_lines:
        tf.paragraphs[0].text = ''
        return

    for i, ln in enumerate(norm_lines):
        content = ln[2:].strip()
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = content
        p.level = 0
        try:
            p.font.size = Pt(font_size_pt)
        except Exception:
            pass
        p.alignment = PP_ALIGN.LEFT


def _images_to_pairs(images: List[Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for im in images or []:
        if isinstance(im, str):
            out.append((im, ''))
        elif isinstance(im, dict):
            out.append((im.get('path') or '', im.get('caption') or ''))
    return [(p, c) for p, c in out if p]


def _resolve_image_path(img: str, base_dir: Path) -> Optional[Path]:
    p = Path(img)
    if p.is_absolute() and p.exists():
        return p
    cand = (base_dir / img).resolve()
    if cand.exists():
        return cand
    if p.exists():
        return p
    return None


def overlay_images(slide, images: List[Any], base_dir: Path) -> None:
    pairs = _images_to_pairs(images)
    pic_shapes = [sh for sh in slide.shapes if getattr(sh, 'shape_type', None) == 13]
    for idx, (path, _) in enumerate(pairs[:len(pic_shapes)]):
        ip = _resolve_image_path(path, base_dir)
        if not ip:
            continue
        ph = pic_shapes[idx]
        slide.shapes.add_picture(str(ip), ph.left, ph.top, width=ph.width, height=ph.height)


def fill_legends(slide, images: List[Any]) -> None:
    pairs = _images_to_pairs(images)
    legend_shapes = find_shapes_exact(slide, 'Légende Photo')
    pic_shapes = [sh for sh in slide.shapes if getattr(sh, 'shape_type', None) == 13]

    if not pairs:
        for sh in legend_shapes:
            set_text(sh, '')
        return

    assignments: Dict[int, int] = {}
    unused = set(range(len(legend_shapes)))

    for i, pic in enumerate(pic_shapes):
        best = None
        best_dist = None
        for j in list(unused):
            lg = legend_shapes[j]
            if lg.top < pic.top:
                continue
            dv = abs(int(lg.top) - int(pic.top + pic.height))
            dh = abs(int(lg.left) - int(pic.left))
            dist = dv + dh * 0.1
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = j
        if best is not None:
            assignments[i] = best
            unused.remove(best)

    for i, (path, cap) in enumerate(pairs):
        cap = (cap or '').strip() or Path(path).stem
        cap = cap.replace('_', ' ').strip()
        if len(cap) > 60:
            cap = cap[:57].rstrip() + '…'
        j = assignments.get(i)
        if j is None and i < len(legend_shapes):
            j = i
        if j is not None and j < len(legend_shapes):
            set_text(legend_shapes[j], cap)

    for j in unused:
        set_text(legend_shapes[j], '')


def fill_info_slide(slide, *, title: str, body: str, part_label: str, page_label: str,
                    images: List[Any], base_dir: Path) -> None:
    # Clear stray placeholders like "User flow" if present
    for sh in find_shapes_exact(slide, 'User flow'):
        set_text(sh, '')

    # Title
    ts = find_shape_containing(slide, 'Titre section')
    if ts:
        set_text(ts, title)

    # Part label
    ps = find_shape_containing(slide, 'Etat des lieux')
    if ps and part_label:
        set_text(ps, part_label)

    # Page label
    pg = find_shape_containing(slide, 'Page')
    if pg and page_label:
        set_text(pg, page_label)

    # Main body
    body_shape = None
    for sh in slide.shapes:
        if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
            if 'Texte Texte' in (sh.text_frame.text or ''):
                body_shape = sh
                break
    if body_shape:
        set_bullets(body_shape, body)

    # Info clé
    detail = find_shape_containing(slide, 'Texte Détail Info Clé')
    if detail:
        set_text(detail, first_bullet_key(body))

    fill_legends(slide, images)
    overlay_images(slide, images, base_dir)


# -----------------------------
# Build deck
# -----------------------------

def build_deck(base_template: Path, assembled_path: Path, out_path: Path, slide_types_path: Path,
               info_template: Optional[Path] = None) -> None:
    base_tpl = Presentation(str(base_template))
    data = load_json(assembled_path)

    slide_types_cfg = load_json(slide_types_path) if slide_types_path.exists() else {'types': {}, 'defaults': {}}
    catalog = detect_template_slide_types(base_tpl, slide_types_cfg)

    info_tpl = Presentation(str(info_template)) if info_template and info_template.exists() else None
    info_slide6 = info_tpl.slides[0] if info_tpl and len(info_tpl.slides) >= 1 else None
    info_slide4 = info_tpl.slides[1] if info_tpl and len(info_tpl.slides) >= 2 else None

    case_id = data.get('case_id') or assembled_path.stem
    report_type = data.get('report_type') or 'AUDIT'
    today = datetime.now().strftime('%d/%m/%Y')
    section_ctx = data.get('section_context') or {}
    site_label = section_ctx.get('onenote_section_name') or case_id

    prs_out = Presentation()
    prs_out.slide_width = base_tpl.slide_width
    prs_out.slide_height = base_tpl.slide_height

    # Part titles
    part_titles: Dict[int, str] = {1: 'Partie 1', 2: 'Partie 2', 3: 'Partie 3', 4: '', 5: ''}
    for mp in (data.get('macro_parts') or []):
        try:
            mp_num = int(mp.get('macro_part'))
            part_titles[mp_num] = mp.get('macro_part_name') or part_titles.get(mp_num, f'Partie {mp_num}')
        except Exception:
            pass

    global_map = {
        '{{PART1_TITLE}}': part_titles.get(1, 'Partie 1'),
        '{{PART2_TITLE}}': part_titles.get(2, 'Partie 2'),
        '{{PART3_TITLE}}': part_titles.get(3, 'Partie 3'),
        '{{PART4_TITLE}}': part_titles.get(4, ''),
        '{{PART5_TITLE}}': part_titles.get(5, ''),
        '{{DATE}}': today,
        '{{VERSION}}': '1',
        '{{CONTACT_EMAIL}}': 'contact@build4use.eu',
    }

    cover_map = {
        '{{AUDIT_TYPE}}': report_type,
        '{{CLIENT}}': 'N/A',
        '{{SITE}}': site_label,
        '{{VILLE}}': '',
        '{{CASE_CODE}}': case_id,
        **global_map,
    }

    # COVER
    if 'COVER' in catalog:
        s_cover = clone_slide(prs_out, catalog['COVER'])
        replace_placeholders(s_cover, cover_map)

    # TOC
    if 'TOC' in catalog:
        s_toc = clone_slide(prs_out, catalog['TOC'])
        replace_placeholders(s_toc, global_map)

    def apply_global(slide):
        replace_placeholders(slide, global_map)

    def emit_part_divider(part: int, title: str):
        slide_in = catalog.get('PART_DIVIDER')
        if not slide_in:
            return
        s_div = clone_slide(prs_out, slide_in)
        apply_global(s_div)
        replace_placeholders(s_div, {'{{PART1_TITLE}}': title})
        badge = f'{int(part):02d}'
        for sh in s_div.shapes:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
                if (sh.text_frame.text or '').strip() in {'01', '02', '03', '99'}:
                    sh.text_frame.text = badge

    def emit_content_slide(s: Dict[str, Any], idx_in_part: int):
        stype = (s.get('type') or 'CONTENT_TEXT').strip()
        title = (s.get('title') or '').strip()
        body = (s.get('bullets') or s.get('body') or '')
        images = s.get('images') or []
        part = int(s.get('part') or 0)

        if info_tpl and stype in ('CONTENT_TEXT', 'CONTENT_TEXT_IMAGES') and (info_slide6 or info_slide4):
            pairs = _images_to_pairs(images)
            chosen = info_slide6 if (len(pairs) > 4 and info_slide6 is not None) else (info_slide4 or info_slide6)
            if chosen is None:
                pass
            else:
                slide_out = clone_slide(prs_out, chosen)
                part_label_map = {1: 'Etat des lieux', 2: 'Scoring actuel', 3: 'Scoring projeté'}
                part_label = part_label_map.get(part, '')
                page_label = f'Page {idx_in_part}' if idx_in_part else ''
                fill_info_slide(slide_out, title=title, body=body, part_label=part_label,
                                page_label=page_label, images=images, base_dir=assembled_path.parent)
                return

        # Fallback base content slide
        slide_in = catalog.get(stype) or catalog.get('CONTENT_TEXT')
        if not slide_in:
            return
        slide_out = clone_slide(prs_out, slide_in)
        apply_global(slide_out)
        replace_placeholders(slide_out, {'{{SLIDE_TITLE}}': title})

        # Replace body placeholder at shape text level
        for sh in slide_out.shapes:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
                if '{{TEXTE_BULLETS}}' in (sh.text_frame.text or ''):
                    sh.text_frame.text = ''
                    set_bullets(sh, body)

    slides_spec = data.get('slides')
    per_part_counter = {1: 0, 2: 0, 3: 0}

    if isinstance(slides_spec, list) and slides_spec:
        for s in slides_spec:
            stype = (s.get('type') or 'CONTENT_TEXT').strip()
            if stype == 'PART_DIVIDER':
                part = int(s.get('part') or 0)
                emit_part_divider(part, (s.get('title') or '').strip() or f'Partie {part}')
                if part in per_part_counter:
                    per_part_counter[part] = 0
                continue
            part = int(s.get('part') or 0)
            if part in per_part_counter:
                per_part_counter[part] += 1
            emit_content_slide(s, per_part_counter.get(part, 0))

    # CONCLUSION (optional)
    if 'CONCLUSION' in catalog:
        s_conc = clone_slide(prs_out, catalog['CONCLUSION'])
        apply_global(s_conc)
        replace_placeholders(s_conc, {'{{CONCLUSION_TEXTE}}': sanitize_client_body(data.get('conclusion', '') or '')[:900] or 'Synthèse à compléter.'})

    # Guardrails
    offenders: List[Tuple[int, str]] = []
    for si, sl in enumerate(prs_out.slides, start=1):
        for sh in sl.shapes:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
                txt = sh.text_frame.text or ''
                if '{{' in txt or 'page_id=' in txt:
                    offenders.append((si, txt.strip().replace('\n', ' ')[:160]))
    if offenders:
        preview = ' | '.join([f'S{a}: {b}' for a, b in offenders[:6]])
        raise SystemExit('❌ Unreplaced template tokens or page_id leak detected. Offenders: ' + preview)

    prs_out.save(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--assembled', required=True)
    ap.add_argument('--out', default='Rapport_Audit.pptx')
    ap.add_argument('--slide-types', default='input/config/slide_types.json')
    ap.add_argument('--info-template', default='')
    args = ap.parse_args()

    build_deck(Path(args.template), Path(args.assembled), Path(args.out), Path(args.slide_types),
               Path(args.info_template) if args.info_template else None)
    print(f'Wrote: {args.out}')


if __name__ == '__main__':
    main()
