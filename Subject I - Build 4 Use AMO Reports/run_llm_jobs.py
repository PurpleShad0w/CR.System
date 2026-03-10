#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_llm_jobs.py

Reworked to match the intended 3-step process:

(1) Part 1 (État des lieux):
    - Use OneNote evidence -> deterministic evidence_pack (rich, non-truncating)
    - LLM renders a coherent descriptive section (no scoring, no works)

(2) Part 2 (Scoring actuel):
    - Use Tableau 6 rules + OneNote evidence digest
    - LLM infers levels per rule (JSON) -> deterministic compute classes
    - Deterministic rendering (optionally LLM slide formatting)

(3) Part 3 (Scoring projeté / travaux):
    - Deterministic derive missing rules to reach target class (default B)
    - Render as works list aligned with ISO 52120-1 rule titles

This avoids the prior failure mode where aggressive sanitization + over-conservative briefs
produced almost no text.
"""

import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from llm_client import make_client
from quality_score import evaluate_quality
import quality_score as qs
from section_context import require_section_context

from evidence_pack import build_evidence_pack
from bacs_scoring_from_bundle import (
    load_rules_json,
    build_digest_from_bundle,
    build_level_inference_prompt_from_digest,
    compute_group_scores_from_table6,
    render_part2_markdown,
    render_part3_markdown,
    parse_json_safely,
)

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float = 1.0, retries: int = 4, base_sleep: float = 1.2) -> Tuple[Optional[Any], Optional[str]]:
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
        f"{base_prompt}\n\n"
        "---\n"
        "EVIDENCE_PACK (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ce EVIDENCE_PACK pour rédiger la PARTIE 1 (État des lieux).\n"
        "Règles strictes:\n"
        "- Tu ne fais PAS de scoring (pas de classes A/B/C/D).\n"
        "- Tu ne proposes PAS de travaux.\n"
        "- Tu listes l'existant de manière structurée, en regroupant les faits.\n"
        "- Pour chaque constat, ajoute une ligne 'Preuve: page_id=... + extrait'.\n"
        "- Les questions/demandes vont dans '#### Points à confirmer'.\n\n"
        "Format attendu:\n"
        "- Une phrase d'introduction (2-3 lignes) synthétisant l'état global.\n"
        "- Ensuite, pour chaque rubrique #### du plan: 3 à 12 puces 'Constat:' (si data), sinon 'À confirmer'.\n"
        "- Ajouter une sous-section '#### Inventaire / équipements mentionnés' si equipment_facts n'est pas vide.\n\n"
        f"{payload}\n"
        "---\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='Path to process/drafts/<case_id>/draft_bundle.json')
    ap.add_argument('--out', default='', help='Output folder (defaults to bundle folder)')

    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max_tokens', type=int, default=1500)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--mode', choices=['single','multistep'], default='multistep')

    ap.add_argument('--min_quality', type=float, default=0.0)
    ap.add_argument('--quality', default='')

    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry_sleep', type=float, default=1.2)

    # BACS scoring inputs (now supported even without section_aggregate)
    ap.add_argument('--bacs_rules', default='', help='Path to Tableau 6 rules JSON')
    ap.add_argument('--bacs_building_scope', default='Non résidentiel', choices=['Résidentiel','Non résidentiel'])
    ap.add_argument('--bacs_targets', default='', help='Optional JSON mapping group->target class for Part 3')
    ap.add_argument('--bacs_part2_slides', action='store_true', help='If set, Part 2 slides are formatted by LLM')

    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    if bundle.get('section_context'):
        require_section_context(bundle, 'draft_bundle.json')

    client = make_client()
    generated = dict(bundle)

    # Precompute BACS scoring from bundle if rules provided
    bacs_part2_text = None
    bacs_part3_text = None
    bacs_meta = None

    if (generated.get('report_type') == 'BACS_SCORING') and args.bacs_rules:
        try:
            rules_doc = load_rules_json(args.bacs_rules)
            digest = build_digest_from_bundle(generated)
            prompt = build_level_inference_prompt_from_digest(digest, rules_doc, args.bacs_building_scope)
            resp, err = safe_chat(client, build_messages(prompt), temperature=0.0, max_tokens=1900, top_p=1.0, retries=args.retries, base_sleep=args.retry_sleep)
            if resp is None:
                inferred = {'levels': {}, 'evidence': {}, 'unknowns': [f'LLM error: {err}']}
            else:
                obj, perr = parse_json_safely(resp.text or '')
                inferred = obj if obj is not None else {'levels': {}, 'evidence': {}, 'unknowns': [f'Parse error: {perr}']}

            observed_levels = inferred.get('levels') or {}
            group_scores = compute_group_scores_from_table6(rules_doc, args.bacs_building_scope, observed_levels)

            # Part2 rendering
            if args.bacs_part2_slides:
                # keep existing slide prompt from earlier versions if you want; here we stick to deterministic markdown
                bacs_part2_text = render_part2_markdown(group_scores)
            else:
                bacs_part2_text = render_part2_markdown(group_scores)

            # Part3 rendering
            target_by_group = {}
            if args.bacs_targets:
                try:
                    target_by_group = load_json(Path(args.bacs_targets))
                except Exception:
                    target_by_group = {}
            bacs_part3_text = render_part3_markdown(rules_doc, args.bacs_building_scope, observed_levels, target_by_group)

            bacs_meta = {
                'building_scope': args.bacs_building_scope,
                'inferred': inferred,
                'group_scores': group_scores,
            }
        except Exception as e:
            bacs_meta = {'error': str(e)}

    # 1) Generate sections (LLM) — primarily Part 1 (macro_part 1)
    for sec in generated.get('sections', []):
        mp = sec.get('macro_part')
        base_prompt = (sec.get('prompt') or '').strip()
        if not base_prompt:
            sec['generated_text'] = ''
            sec['final_text'] = ''
            continue

        # If BACS scoring and we have deterministic parts 2/3, we can skip LLM generation for them.
        if generated.get('report_type') == 'BACS_SCORING' and bacs_part2_text and bacs_part3_text and mp in (2,3):
            sec['generated_text'] = ''
            sec['final_text'] = ''
            sec['skipped_reason'] = 'deterministic_table6'
            continue

        # Build evidence pack (rich)
        pack = build_evidence_pack(sec, max_facts_per_topic=14)
        sec['evidence_pack'] = pack

        # Part 1 writer prompt
        if mp == 1:
            write_prompt = prompt_write_part1_from_pack(sec, pack)
        else:
            # Fallback: still write from evidence pack but less strict
            payload = json.dumps(pack, ensure_ascii=False, indent=2)[:12000]
            write_prompt = (
                f"{base_prompt}\n\n---\nEVIDENCE_PACK (JSON) = source UNIQUE\n"
                "Utilise uniquement ce pack. Respecte le plan ####.\n"
                f"{payload}\n---\n"
            )

        resp, err = safe_chat(client, build_messages(write_prompt), temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p, retries=args.retries, base_sleep=args.retry_sleep)
        if resp is None:
            sec['generated_text'] = ''
            sec['final_text'] = ''
            sec['llm_error'] = err
        else:
            text = (resp.text or '').strip()
            sec['generated_text'] = text
            sec['final_text'] = text

    gen_path = out_dir / 'generated_bundle.json'
    save_json(gen_path, generated)

    # 2) Assemble report
    assembled: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': generated.get('section_context'),
        'macro_parts': [],
    }

    # Group sections by macro_part
    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get('sections', []):
        by_mp.setdefault(sec.get('macro_part'), []).append(sec)

    for mp in sorted(by_mp.keys()):
        assembled['macro_parts'].append({
            'macro_part': mp,
            'macro_part_name': by_mp[mp][0].get('macro_part_name'),
            'sections': [
                {'bucket_id': s.get('bucket_id'), 'text': (s.get('final_text') or s.get('generated_text') or '').strip()}
                for s in by_mp[mp]
            ],
        })

    # 3) Inject deterministic BACS Part 2 & 3 if available
    if generated.get('report_type') == 'BACS_SCORING' and bacs_part2_text and bacs_part3_text:
        assembled['bacs_table6'] = bacs_meta

        def upsert_macro(mp_num: int, mp_name: str, text: str):
            for mp in assembled.get('macro_parts', []):
                if int(mp.get('macro_part', -1)) == mp_num:
                    mp['macro_part_name'] = mp_name
                    mp['sections'] = [{'bucket_id': 'AUTO_TABLE6', 'text': text}]
                    return
            assembled['macro_parts'].append({'macro_part': mp_num, 'macro_part_name': mp_name, 'sections': [{'bucket_id': 'AUTO_TABLE6', 'text': text}]})

        upsert_macro(2, 'Scoring GTB actuel selon ISO 52120-1', bacs_part2_text)
        upsert_macro(3, 'Futur : Scoring projeté et travaux à mettre en œuvre', bacs_part3_text)

    assembled_path = out_dir / 'assembled_report.json'
    save_json(assembled_path, assembled)

    quality = evaluate_quality(generated, assembled)
    q_path = Path(args.quality) if args.quality else (out_dir / 'quality_report.json')
    save_json(q_path, quality)

    print(f"Wrote: {gen_path}")
    print(f"Wrote: {assembled_path}")
    print(f"Wrote: {q_path}")
    print(f"Quality total: {quality.get('total')} / 100")

    if args.min_quality and (quality.get('total', 0) < args.min_quality):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
