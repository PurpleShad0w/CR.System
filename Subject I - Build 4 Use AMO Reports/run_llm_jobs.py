#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_llm_jobs.py

Implements 4 improvement axes for dense OneNote sections:

A) Part 1 thematic interpretation (automatic):
   - Build multiple theme slides from the section aggregate (not only LLM text).

B) Deterministic "Inventaire / Liste des équipements" slide:
   - Extract from OneNote page titled like "Liste des équipements".

C) Evidence / image diversity:
   - Prefer covering all available pages (avoid repeating the same page_id).

D) Part 2/3 enriched (visuals + short explanations):
   - Add images per group when available and a short explanation sentence.

Notes:
- Client deck never contains raw OneNote ids (page_id=) nor 'Preuve:' lines.
- Internal trace keeps exhaustive evidence.

"""

from __future__ import annotations

import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import make_client
from quality_score import evaluate_quality
from section_context import require_section_context
from evidence_pack import build_evidence_pack
from bacs_scoring_from_bundle import (
    load_rules_json,
    build_digest_from_bundle,
    build_level_inference_prompt_from_digest,
    compute_group_scores_from_table6,
    parse_json_safely,
)

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


# -----------------------------
# IO
# -----------------------------

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


# -----------------------------
# LLM helpers
# -----------------------------

def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float = 1.0,
    retries: int = 4, base_sleep: float = 1.2) -> Tuple[Optional[Any], Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=False)
            return resp, None
        except Exception as e:
            msg = str(e)
            last_err = msg
            transient = (
                "HF error 500" in msg
                or "Internal Server Error" in msg
                or "Unknown error" in msg
                or "Model too busy" in msg
                or "unable to get response" in msg
            )
            if attempt >= retries or not transient:
                break
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.4))
    return None, last_err


def prompt_write_part1_from_pack(section: Dict[str, Any], pack: Dict[str, Any]) -> str:
    base_prompt = (section.get('prompt') or '').strip()
    payload = json.dumps(pack, ensure_ascii=False, indent=2)[:12000]
    return (
        f"{base_prompt}\n\n---\n"
        "EVIDENCE_PACK (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ce EVIDENCE_PACK pour rédiger la PARTIE 1 (État des lieux).\n"
        "Règles strictes:\n"
        "- Tu ne fais PAS de scoring (pas de classes A/B/C/D).\n"
        "- Tu ne proposes PAS de travaux.\n"
        "- Tu listes l'existant de manière structurée, en regroupant les faits.\n"
        "- Pour chaque constat important, ajoute une ligne 'Preuve:' (avec page_id + extrait).\n"
        "- Les questions/demandes vont dans '#### Points à confirmer'.\n\n"
        f"{payload}\n---\n"
    )


# -----------------------------
# Evidence extraction (client vs internal)
# -----------------------------

RE_EVIDENCE_LINE = re.compile(r"^(\s*[-•]?\s*)?(preuve\s*:|.*\bpage_id\s*=)", re.IGNORECASE)
RE_PAGE_ID = re.compile(r"\bpage_id\s*=\s*([^\s,;]+)", re.IGNORECASE)


def extract_client_text_and_trace(raw_text: str, *, origin: str, trace_start_index: int = 1) -> Tuple[str, List[Dict[str, Any]]]:
    if not raw_text:
        return '', []

    lines = raw_text.splitlines()
    out_lines: List[str] = []
    trace: List[Dict[str, Any]] = []
    p_idx = trace_start_index

    for ln in lines:
        s = ln.strip()
        if RE_EVIDENCE_LINE.search(s):
            ref = f"P{p_idx}"
            trace.append({"ref": ref, "origin": origin, "raw": s})
            p_idx += 1
            continue
        out_lines.append(ln)

    client_text = "\n".join(out_lines)
    client_text = re.sub(r"\s*Preuve\s*:\s*.*", "", client_text, flags=re.IGNORECASE)
    client_text = RE_PAGE_ID.sub("", client_text)
    client_text = re.sub(r"\[P\d+\]", "", client_text)
    client_text = re.sub(r"\n{3,}", "\n\n", client_text).strip()
    return client_text, trace


def strip_client_noise(text: str) -> str:
    """Remove automation smell tokens that still appear in generated text."""
    if not text:
        return ''
    t = re.sub(r"\bConstat\s*:\s*", "", text, flags=re.IGNORECASE)
    t = re.sub(r"\[P\d+\]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -----------------------------
# OneNote aggregate reader
# -----------------------------

def _norm(s: str) -> str:
    s = (s or '').lower().strip()
    s = s.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
    s = s.replace('à', 'a').replace('â', 'a').replace('ä', 'a')
    s = s.replace('î', 'i').replace('ï', 'i')
    s = s.replace('ô', 'o').replace('ö', 'o')
    s = s.replace('û', 'u').replace('ü', 'u')
    s = re.sub(r"\s+", " ", s)
    return s


def locate_section_aggregate(bundle_path: Path, bundle: Dict[str, Any]) -> Optional[Path]:
    ctx = bundle.get('section_context') or {}
    notebook = ctx.get('onenote_notebook')
    slug = ctx.get('section_slug')
    if not notebook or not slug:
        return None
    # project root = bundle_path.parent.parent.parent? actually bundle is process/drafts/<case>/draft_bundle.json
    project_root = bundle_path.parent.parent.parent.parent
    cand = project_root / 'process' / 'onenote_aggregates' / str(notebook) / f"{slug}.json"
    if cand.exists():
        return cand
    return None


def aggregate_pages(agg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = agg.get('pages') or []
    out = []
    for p in pages:
        pid = p.get('page_id') or p.get('id')
        title = p.get('title') or ''
        blocks = p.get('text_blocks') or []
        # Flatten blocks into lines
        lines = []
        for b in blocks:
            txt = (b.get('text') or '').strip()
            if not txt:
                continue
            for ln in txt.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        out.append({
            'page_id': pid,
            'title': title,
            'lines': lines,
            'num_images': int(p.get('num_images') or 0),
        })
    return out


def extract_equipment_list(pages: List[Dict[str, Any]]) -> List[str]:
    """Find the equipment list page and extract bullet-like items."""
    best = None
    for p in pages:
        if 'liste des equipements' in _norm(p.get('title')):
            best = p
            break
    if best is None:
        # fallback: any page with many equipment-like tokens
        for p in pages:
            if 'equipement' in _norm(' '.join(p.get('lines') or [])):
                best = p
                break
    if best is None:
        return []

    items: List[str] = []
    for ln in best.get('lines') or []:
        s = ln.strip()
        if not s:
            continue
        # bullet forms
        if s.startswith(('-', '•')):
            s = s.lstrip('-•').strip()
        # remove emoji numbering
        s = re.sub(r"^[0-9]+[)\.\-]\s*", "", s)
        s = re.sub(r"^[\U0001F300-\U0001FAFF]+\s*", "", s)
        # keep short-ish lines (equipment entries)
        if 2 <= len(s) <= 120:
            items.append(s)
    # de-dup preserving order
    seen = set()
    out = []
    for it in items:
        k = _norm(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def theme_definitions() -> List[Dict[str, Any]]:
    return [
        {'key': 'contexte', 'title': 'Contexte & objectifs', 'keywords': ['contexte', 'objectif', 'mission', 'decret', 'bacs', 'classe', 'em eis', 'idex', 'schneider', 'integrateur']},
        {'key': 'architecture', 'title': 'Architecture GTB / supervision', 'keywords': ['gtb', 'supervision', 'ebo', 'tac', 'vista', 'workstation', 'automate', 'bacnet', 'lon']},
        {'key': 'comptage', 'title': 'Comptage & suivi des consommations', 'keywords': ['comptage', 'diris', 'tgbt', 'kwh', 'compteur', 'energie', 'eau', 'calorie', 'frigorie']},
        {'key': 'alarmes', 'title': 'Alarmes & historisation', 'keywords': ['alarme', 'defaut', 'histor', 'derive', 'reporting', 'diagnostic']},
    ]


def extract_theme_bullets(pages: List[Dict[str, Any]], theme: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (bullets, page_ids_used) for a theme."""
    kws = [_norm(k) for k in (theme.get('keywords') or [])]
    bullets: List[str] = []
    used_pages: List[str] = []

    for p in pages:
        pid = p.get('page_id')
        lines = p.get('lines') or []
        for ln in lines:
            n = _norm(ln)
            if any(k in n for k in kws):
                # transform into a concise bullet
                b = ln.strip()
                if len(b) > 170:
                    b = b[:167].rstrip() + '…'
                bullets.append(b)
                if pid and pid not in used_pages:
                    used_pages.append(pid)

    # de-dup bullets
    seen = set()
    out = []
    for b in bullets:
        k = _norm(b)
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out, used_pages


# -----------------------------
# Image resolution and diversity
# -----------------------------

def load_page_index(onenote_root: Path) -> Dict[str, Dict[str, Any]]:
    pages_dir = onenote_root / 'pages'
    idx: Dict[str, Dict[str, Any]] = {}
    if not pages_dir.exists():
        return idx
    for p in pages_dir.glob('*.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        pid = obj.get('page_id') or obj.get('metadata', {}).get('page_id')
        if pid:
            idx[str(pid)] = obj
    return idx


def short_caption(text: str) -> str:
    t = (text or '').strip()
    if not t:
        return ''
    t = t.split('\n', 1)[0].strip()
    # keep manual-like captions short
    if len(t) > 60:
        t = t[:57].rstrip() + '…'
    return t


def resolve_images_for_page_ids(page_index: Dict[str, Dict[str, Any]], page_ids: List[str], *, max_images: int = 6,
    per_page_cap: int = 2) -> List[Dict[str, Any]]:
    """Diverse selection: max per_page_cap images per page_id."""
    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for pid in page_ids:
        obj = page_index.get(pid)
        if not obj:
            continue
        page_title = (obj.get('title') or obj.get('metadata', {}).get('title') or '').strip()
        assets = obj.get('assets') or {}
        images = assets.get('images') or []
        for im in images:
            if len(out) >= max_images:
                return out
            if counts.get(pid, 0) >= per_page_cap:
                break
            path = None
            cap = None
            if isinstance(im, str):
                path = im
            elif isinstance(im, dict):
                path = im.get('path') or im.get('file') or im.get('name')
                cap = im.get('caption') or im.get('alt') or im.get('title')
            if not cap:
                cap = page_title
            if path:
                out.append({'path': path, 'caption': short_caption(cap), 'page_id': pid, 'page_title': page_title})
                counts[pid] = counts.get(pid, 0) + 1
    return out


def diverse_page_order(pages: List[Dict[str, Any]], preferred: List[str]) -> List[str]:
    """Return page_ids starting with preferred, then remaining, preserving uniqueness."""
    all_ids = [p.get('page_id') for p in pages if p.get('page_id')]
    seen = set()
    out = []
    for pid in preferred + all_ids:
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


# -----------------------------
# Slide building utilities
# -----------------------------

def bullets_from_text(text: str) -> List[str]:
    text = strip_client_noise(text)
    text = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    bullets = []
    for ln in lines:
        if ln.startswith('- '):
            bullets.append(ln)
        elif ln.startswith('•'):
            bullets.append('- ' + ln.lstrip('•').strip())
        else:
            # keep short sentences as bullets
            if len(ln) <= 170:
                bullets.append('- ' + ln)
    return bullets


def chunk_bullets(bullets: List[str], max_bullets: int = 10) -> List[str]:
    if not bullets:
        return ['']
    chunks = []
    for i in range(0, len(bullets), max_bullets):
        chunks.append('\n'.join(bullets[i:i + max_bullets]))
    return chunks


def split_by_h4(md: str) -> List[Tuple[str, str]]:
    md = (md or '').replace('\r\n', '\n').replace('\r', '\n')
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if not md:
        return []
    cur_title: Optional[str] = None
    cur_lines: List[str] = []
    out: List[Tuple[str, str]] = []
    for ln in md.splitlines():
        s = ln.strip()
        if s.startswith('#### '):
            if cur_title is not None:
                out.append((cur_title, '\n'.join(cur_lines).strip()))
            cur_title = s.replace('#### ', '').strip()
            cur_lines = []
        else:
            cur_lines.append(ln)
    if cur_title is not None:
        out.append((cur_title, '\n'.join(cur_lines).strip()))
    return out


# -----------------------------
# Part 2/3 enrichment helpers
# -----------------------------

def group_keywords() -> Dict[str, List[str]]:
    return {
        'Chauffage': ['chaufferie', 'chaudiere', 'radiateur', 'pac', 'loi d', 'pompe'],
        'ECS': ['ecs', 'eau chaude', 'ballon', 'bouclage'],
        'Refroidissement': ['refroid', 'groupe froid', 'eau glacee', 'pac'],
        'Ventilation': ['cta', 'ventilation', 'vmc', 'soufflage', 'reprise', 'co2'],
        'Éclairage': ['eclairage', 'luminaire', 'detecteur de presence'],
        'Stores': ['store'],
        'GTB': ['gtb', 'supervision', 'ebo', 'tac', 'vista', 'automate', 'bacnet', 'lon'],
    }


def choose_group_images(pages: List[Dict[str, Any]], page_index: Dict[str, Dict[str, Any]], group: str, *, max_images: int = 2) -> List[Dict[str, Any]]:
    kws = [_norm(k) for k in group_keywords().get(group, [])]
    preferred = []
    for p in pages:
        pid = p.get('page_id')
        if not pid:
            continue
        blob = _norm(p.get('title', '') + ' ' + ' '.join(p.get('lines') or []))
        if any(k in blob for k in kws):
            preferred.append(pid)
    order = diverse_page_order(pages, preferred)
    return resolve_images_for_page_ids(page_index, order, max_images=max_images, per_page_cap=1)


def group_explanation(group: str, group_score: Dict[str, Any]) -> str:
    ach = (group_score or {}).get('achieved_class') or ''
    block_b = ((group_score or {}).get('blockers') or {}).get('B') or []
    nb = len(block_b)
    if not ach:
        return ''
    if nb:
        return f"Classe {ach} : {nb} exigence(s) minimales vers la classe B ne sont pas démontrées dans les notes / preuves disponibles."
    return f"Classe {ach} : les exigences minimales vers la classe B semblent satisfaites sur la base des preuves disponibles."


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max_tokens', type=int, default=1500)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--mode', choices=['single', 'multistep'], default='multistep')
    ap.add_argument('--min_quality', type=float, default=0.0)
    ap.add_argument('--quality', default='')
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry_sleep', type=float, default=1.2)
    ap.add_argument('--onenote', default='', help='Path to process/onenote/<notebook>')
    ap.add_argument('--bacs_rules', default='')
    ap.add_argument('--bacs_building_scope', default='Non résidentiel', choices=['Résidentiel', 'Non résidentiel'])
    ap.add_argument('--bacs_targets', default='')
    ap.add_argument('--bacs_part2_slides', action='store_true')
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    if bundle.get('section_context'):
        require_section_context(bundle, 'draft_bundle.json')

    client = make_client()
    generated = dict(bundle)

    page_index = load_page_index(Path(args.onenote)) if args.onenote else {}

    # Load section aggregate if available
    agg_path = locate_section_aggregate(bundle_path, bundle)
    agg_obj = load_json(agg_path) if agg_path else None
    agg_pages = aggregate_pages(agg_obj) if isinstance(agg_obj, dict) else []

    # BACS scoring (deterministic)
    bacs_meta = None
    rules_doc = None
    observed_levels: Dict[str, Any] = {}
    group_scores: Dict[str, Any] = {}

    if (generated.get('report_type') == 'BACS_SCORING') and args.bacs_rules:
        rules_doc = load_rules_json(args.bacs_rules)
        digest = build_digest_from_bundle(generated)
        prompt = build_level_inference_prompt_from_digest(digest, rules_doc, args.bacs_building_scope)
        resp, err = safe_chat(client, build_messages(prompt), temperature=0.0, max_tokens=1900, top_p=1.0,
            retries=args.retries, base_sleep=args.retry_sleep)
        if resp is None:
            inferred = {'levels': {}, 'evidence': {}, 'unknowns': [f'LLM error: {err}']}
        else:
            obj, perr = parse_json_safely(resp.text or '')
            inferred = obj if obj is not None else {'levels': {}, 'evidence': {}, 'unknowns': [f'Parse error: {perr}']}
        observed_levels = inferred.get('levels') or {}
        group_scores = compute_group_scores_from_table6(rules_doc, args.bacs_building_scope, observed_levels)
        bacs_meta = {'building_scope': args.bacs_building_scope, 'inferred': inferred, 'group_scores': group_scores}

    # LLM generation for Part 1 only (kept)
    for sec in generated.get('sections', []):
        mp = sec.get('macro_part')
        base_prompt = (sec.get('prompt') or '').strip()
        if not base_prompt:
            sec['final_text'] = ''
            continue

        if generated.get('report_type') == 'BACS_SCORING' and mp in (2, 3) and rules_doc is not None:
            sec['final_text'] = ''
            sec['skipped_reason'] = 'deterministic_table6'
            continue

        pack = build_evidence_pack(sec, max_facts_per_topic=14)
        sec['evidence_pack'] = pack
        write_prompt = prompt_write_part1_from_pack(sec, pack) if mp == 1 else base_prompt
        resp, err = safe_chat(client, build_messages(write_prompt), temperature=args.temperature, max_tokens=args.max_tokens,
            top_p=args.top_p, retries=args.retries, base_sleep=args.retry_sleep)
        sec['final_text'] = (resp.text or '').strip() if resp else ''
        if err:
            sec['llm_error'] = err

    save_json(out_dir / 'generated_bundle.json', generated)

    # Assemble macro parts
    assembled: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': generated.get('section_context'),
        'macro_parts': [],
        'slides': [],
        'evidence_trace': [],
    }

    internal_trace: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': generated.get('section_context'),
        'created_from': str(bundle_path),
        'evidence_trace': [],
        'full_text_by_section': [],
    }

    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get('sections', []):
        by_mp.setdefault(int(sec.get('macro_part') or 0), []).append(sec)

    trace_cursor = 1
    for mp in sorted(by_mp.keys()):
        mp_name = by_mp[mp][0].get('macro_part_name')
        mp_obj = {'macro_part': mp, 'macro_part_name': mp_name, 'sections': []}
        for s in by_mp[mp]:
            raw = (s.get('final_text') or '').strip()
            origin = f"mp{mp}:{s.get('bucket_id') or 'SECTION'}"
            client_text, trace_items = extract_client_text_and_trace(raw, origin=origin, trace_start_index=trace_cursor)
            trace_cursor += len(trace_items)
            client_text = strip_client_noise(client_text)
            mp_obj['sections'].append({'bucket_id': s.get('bucket_id'), 'text': client_text})
            assembled['evidence_trace'].extend(trace_items)
            internal_trace['evidence_trace'].extend(trace_items)
            internal_trace['full_text_by_section'].append({'macro_part': mp, 'bucket_id': s.get('bucket_id'), 'raw_text': raw})
        assembled['macro_parts'].append(mp_obj)

    if bacs_meta:
        assembled['bacs_table6'] = bacs_meta
        internal_trace['bacs_table6'] = bacs_meta

    # -----------------------------
    # Build slides[] (4 axes)
    # -----------------------------
    slides: List[Dict[str, Any]] = []

    def add_div(part: int, title: str):
        slides.append({'type': 'PART_DIVIDER', 'part': part, 'title': title})

    ptitle = {1: 'Etat des lieux GTB', 2: 'Scoring GTB actuel', 3: 'Scoring projeté'}
    for mp in assembled['macro_parts']:
        try:
            ptitle[int(mp['macro_part'])] = mp.get('macro_part_name') or ptitle.get(int(mp['macro_part']))
        except Exception:
            pass

    # ---- Part 1 ----
    mp1 = next((x for x in assembled['macro_parts'] if int(x.get('macro_part', -1)) == 1), None)
    if mp1:
        add_div(1, ptitle.get(1, 'Etat des lieux GTB'))

        # B) Deterministic equipment inventory slide
        equipment_items = extract_equipment_list(agg_pages) if agg_pages else []
        inv_bullets = ['- ' + it for it in equipment_items[:24]]
        if not inv_bullets:
            inv_bullets = ['- À confirmer (liste d\'équipements non trouvée dans les artefacts)']

        # Diverse images: prefer equipment page then others
        preferred_pids = []
        for p in agg_pages:
            if 'liste des equipements' in _norm(p.get('title')):
                if p.get('page_id'):
                    preferred_pids.append(p.get('page_id'))
        order = diverse_page_order(agg_pages, preferred_pids)
        inv_imgs = resolve_images_for_page_ids(page_index, order, max_images=4, per_page_cap=1) if page_index else []

        slides.append({
            'type': 'CONTENT_TEXT_IMAGES' if inv_imgs else 'CONTENT_TEXT',
            'part': 1,
            'title': 'Inventaire / équipements mentionnés',
            'bullets': '\n'.join(inv_bullets),
            'images': inv_imgs,
        })

        # A) Automatic theme slides from aggregate
        if agg_pages:
            for th in theme_definitions():
                bullets_raw, used_pids = extract_theme_bullets(agg_pages, th)
                if not bullets_raw:
                    continue
                # limit and chunk
                bullets = ['- ' + b.lstrip('-• ').strip() for b in bullets_raw][:30]
                chunks = chunk_bullets(bullets, max_bullets=10)
                pid_order = diverse_page_order(agg_pages, used_pids)
                imgs = resolve_images_for_page_ids(page_index, pid_order, max_images=4, per_page_cap=1) if page_index else []
                for ci, chunk in enumerate(chunks):
                    slides.append({
                        'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
                        'part': 1,
                        'title': th.get('title') + ('' if ci == 0 else f" ({ci+1})"),
                        'bullets': chunk,
                        'images': imgs if ci == 0 else [],
                    })

        # Existing LLM section text (fallback, but chunked)
        for sec in mp1.get('sections', []):
            bucket = sec.get('bucket_id') or 'SECTION'
            text = sec.get('text') or ''
            blocks = split_by_h4(text)
            if not blocks:
                blocks = [(bucket, text)]
            orig = next((s for s in (generated.get('sections') or []) if int(s.get('macro_part') or 0) == 1 and s.get('bucket_id') == bucket), None)
            pref = [p.get('page_id') for p in (orig.get('top_pages') or []) if p.get('page_id')] if orig else []
            pid_order = diverse_page_order(agg_pages, pref) if agg_pages else pref
            imgs = resolve_images_for_page_ids(page_index, pid_order, max_images=4, per_page_cap=1) if page_index else []
            for h, body in blocks:
                buls = bullets_from_text(body)
                for ci, chunk in enumerate(chunk_bullets(buls, max_bullets=10)):
                    slides.append({
                        'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
                        'part': 1,
                        'title': h or bucket,
                        'bullets': chunk,
                        'images': imgs if ci == 0 else [],
                    })
                    imgs = []

    # ---- Part 2 (D: visuals + explanation) ----
    if bacs_meta and isinstance(group_scores, dict) and group_scores:
        add_div(2, ptitle.get(2, 'Scoring GTB actuel'))
        # Scorecard
        lines = []
        for grp, sc in group_scores.items():
            lines.append(f"- {grp}: classe {sc.get('achieved_class')}")
        slides.append({'type': 'CONTENT_TEXT', 'part': 2, 'title': 'Scorecard (synthèse)', 'bullets': '\n'.join(lines[:12]), 'images': []})
        # per-group
        for grp, sc in group_scores.items():
            expl = group_explanation(grp, sc)
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            bul = [f"- {expl}" ] if expl else []
            bul.append(f"- Classe atteinte : {sc.get('achieved_class')}")
            if block_b:
                bul.append("- Top 3 manquants bloquants (vers classe B):")
                for b in block_b[:3]:
                    bul.append(f"  - {b.get('rule_id')} — {b.get('title')}")
            imgs = choose_group_images(agg_pages, page_index, grp, max_images=2) if (agg_pages and page_index) else []
            slides.append({'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT', 'part': 2, 'title': grp, 'bullets': '\n'.join(bul), 'images': imgs})

    # ---- Part 3 (D: visuals + better framing) ----
    if bacs_meta and isinstance(group_scores, dict) and group_scores:
        add_div(3, ptitle.get(3, 'Scoring projeté'))
        # Synthesis
        synth = ["- Synthèse des écarts vers la cible (classe B)"]
        for grp, sc in group_scores.items():
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            synth.append(f"- {grp}: {len(block_b)} écart(s) vers classe B")
        slides.append({'type': 'CONTENT_TEXT', 'part': 3, 'title': 'Synthèse (chemin critique)', 'bullets': '\n'.join(synth), 'images': []})
        # Action plan per group (short)
        for grp, sc in group_scores.items():
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            if not block_b:
                continue
            bul = ["- Priorité 1 (bloquants):"]
            for b in block_b[:6]:
                bul.append(f"  - {b.get('rule_id')} — {b.get('title')}")
            bul.append("- Priorité 2 (optimisation):")
            for b in block_b[6:10]:
                bul.append(f"  - {b.get('rule_id')} — {b.get('title')}")
            imgs = choose_group_images(agg_pages, page_index, grp, max_images=2) if (agg_pages and page_index) else []
            slides.append({'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT', 'part': 3, 'title': f"Plan d'action — {grp}", 'bullets': '\n'.join(bul), 'images': imgs})

    assembled['slides'] = slides

    save_json(out_dir / 'assembled_report.json', assembled)
    save_json(out_dir / 'internal_trace.json', internal_trace)

    quality = evaluate_quality(generated, assembled)
    save_json(out_dir / 'quality_report.json', quality)

    print('Wrote:', out_dir / 'assembled_report.json')


if __name__ == '__main__':
    main()
