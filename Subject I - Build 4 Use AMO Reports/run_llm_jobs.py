#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""run_llm_jobs.py (escape-safe)

This patch fixes escaping issues by using raw regex strings and avoids emitting
client-visible evidence references.

Key fixes vs the previous patch:
- Slide splitting: Part 1 is split per #### heading, then chunked into multiple slides.
- Evidence: lines containing 'Preuve:' or 'page_id=' are moved to evidence_trace, not to client.
- Client deck: strips any remaining [P#] references.
- Images: attaches dict entries {path, caption} so legends can be filled.
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


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


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


# Evidence extraction
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
        if RE_EVIDENCE_LINE.search(ln.strip()):
            ref = f"P{p_idx}"
            trace.append({"ref": ref, "origin": origin, "raw": ln.strip()})
            p_idx += 1
            continue
        out_lines.append(ln)

    client_text = "\n".join(out_lines)
    client_text = re.sub(r"\s*Preuve\s*:\s*.*", "", client_text, flags=re.IGNORECASE)
    client_text = RE_PAGE_ID.sub("", client_text)
    client_text = re.sub(r"\[P\d+\]", "", client_text)
    client_text = re.sub(r"\n{3,}", "\n\n", client_text).strip()
    return client_text, trace


# OneNote page index + captions

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
    if len(t) > 60:
        t = t[:57].rstrip() + '…'
    return t


def resolve_images_for_page_ids(page_index: Dict[str, Dict[str, Any]], page_ids: List[str], *, max_images: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid in page_ids:
        obj = page_index.get(pid)
        if not obj:
            continue
        page_title = (obj.get('title') or obj.get('metadata', {}).get('title') or '').strip()
        assets = obj.get('assets') or {}
        images = assets.get('images') or []
        for im in images:
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
            if len(out) >= max_images:
                return out
    return out


# Slide splitting

def normalize_ws(s: str) -> str:
    s = (s or '').replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_by_h4(md: str) -> List[Tuple[str, str]]:
    md = normalize_ws(md)
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


def to_bullets(body: str) -> List[str]:
    body = normalize_ws(body)
    bullets: List[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith('- '):
            bullets.append(s)
        elif s.lower().startswith('constat'):
            bullets.append('- ' + s)
        else:
            if len(s) <= 160:
                bullets.append('- ' + s)
    return bullets


def chunk_bullets(bullets: List[str], max_bullets: int = 10) -> List[str]:
    if not bullets:
        return ['']
    chunks: List[str] = []
    for i in range(0, len(bullets), max_bullets):
        chunks.append('\n'.join(bullets[i:i + max_bullets]))
    return chunks


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

    # LLM generation for Part 1
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
            mp_obj['sections'].append({'bucket_id': s.get('bucket_id'), 'text': client_text})
            assembled['evidence_trace'].extend(trace_items)
            internal_trace['evidence_trace'].extend(trace_items)
            internal_trace['full_text_by_section'].append({'macro_part': mp, 'bucket_id': s.get('bucket_id'), 'raw_text': raw})
        assembled['macro_parts'].append(mp_obj)

    if bacs_meta:
        assembled['bacs_table6'] = bacs_meta
        internal_trace['bacs_table6'] = bacs_meta

    # Build slides: Part 1 only here (extend similarly for Part 2/3 if desired)
    slides: List[Dict[str, Any]] = []

    def add_div(part: int, title: str):
        slides.append({'type': 'PART_DIVIDER', 'part': part, 'title': title})

    ptitle = {1: 'Etat des lieux GTB', 2: 'Scoring GTB actuel', 3: 'Scoring projeté'}
    for mp in assembled['macro_parts']:
        try:
            ptitle[int(mp['macro_part'])] = mp.get('macro_part_name') or ptitle.get(int(mp['macro_part']))
        except Exception:
            pass

    mp1 = next((x for x in assembled['macro_parts'] if int(x.get('macro_part', -1)) == 1), None)
    if mp1:
        add_div(1, ptitle.get(1, 'Etat des lieux GTB'))
        for sec in mp1.get('sections', []):
            bucket = sec.get('bucket_id') or 'SECTION'
            text = sec.get('text') or ''

            blocks = split_by_h4(text)
            if not blocks:
                blocks = [(bucket, text)]

            orig = next((s for s in (generated.get('sections') or []) if int(s.get('macro_part') or 0) == 1 and s.get('bucket_id') == bucket), None)
            imgs: List[Dict[str, Any]] = []
            if orig and page_index:
                pids = [p.get('page_id') for p in (orig.get('top_pages') or []) if p.get('page_id')]
                imgs = resolve_images_for_page_ids(page_index, pids, max_images=6)

            for h, body in blocks:
                bullets = to_bullets(body)
                for chunk in chunk_bullets(bullets, max_bullets=10):
                    slides.append({
                        'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
                        'part': 1,
                        'title': h or bucket,
                        'bullets': chunk,
                        'images': imgs,
                    })
                    imgs = []  # images only on first slide of the block

    assembled['slides'] = slides

    save_json(out_dir / 'assembled_report.json', assembled)
    save_json(out_dir / 'internal_trace.json', internal_trace)

    quality = evaluate_quality(generated, assembled)
    save_json(out_dir / 'quality_report.json', quality)

    print('Wrote:', out_dir / 'assembled_report.json')


if __name__ == '__main__':
    main()
