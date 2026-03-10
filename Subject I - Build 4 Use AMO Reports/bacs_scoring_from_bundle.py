
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bacs_scoring_from_bundle.py

Compute Part 2 (scoring actuel) and Part 3 (travaux) from:
- Tableau 6 rules JSON
- Draft bundle evidence (no need for section_aggregate)

This matches the intended process:
- Part 1: describe observed status from OneNote (LLM writes from evidence pack)
- Part 2: infer Tableau-6 levels per rule from evidence, compute achieved classes
- Part 3: derive required works (missing rules) to reach target class

"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import make_client


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un auditeur GTB/BACS. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float = 1.0, retries: int = 4, base_sleep: float = 1.2):
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


def parse_json_safely(raw: str):
    if not raw:
        return None, "empty"
    s = raw.strip()
    try:
        obj = json.loads(s)
        return (obj if isinstance(obj, dict) else None), (None if isinstance(obj, dict) else "not_a_dict")
    except Exception:
        pass
    start = s.find('{')
    end = s.rfind('}')
    if start >= 0 and end > start:
        cand = s[start:end+1]
        try:
            obj = json.loads(cand)
            return (obj if isinstance(obj, dict) else None), (None if isinstance(obj, dict) else "extracted_not_a_dict")
        except Exception as e:
            return None, f"json_extract_parse_failed:{e}"
    return None, "no_json_object_found"


# ---- Table6 helpers (same logic as existing run_llm_jobs) ----

def load_rules_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _req_level(rule: Dict[str, Any], building_scope: str, target_class: str) -> Optional[int]:
    req = (rule.get('class_requirements') or {}).get(building_scope) or {}
    v = req.get(target_class)
    return int(v) if v is not None else None


def _meets(rule: Dict[str, Any], building_scope: str, target_class: str, observed_level: Optional[int]) -> bool:
    req = _req_level(rule, building_scope, target_class)
    if req is None:
        return True
    if observed_level is None:
        return False
    return int(observed_level) >= req


def _highest_class_for_rule(rule: Dict[str, Any], building_scope: str, observed_level: Optional[int]) -> str:
    if not _meets(rule, building_scope, 'C', observed_level):
        return 'D'
    if _meets(rule, building_scope, 'A', observed_level):
        return 'A'
    if _meets(rule, building_scope, 'B', observed_level):
        return 'B'
    return 'C'


def compute_group_scores_from_table6(rules_doc: Dict[str, Any], building_scope: str, observed_levels: Dict[str, Any]) -> Dict[str, Any]:
    by_group: Dict[str, List[Tuple[Dict[str, Any], Optional[int]]]] = {}
    for r in rules_doc.get('rules', []):
        grp = r.get('group') or 'Autres'
        rid = r.get('rule_id')
        lvl = observed_levels.get(rid)
        lvl = int(lvl) if isinstance(lvl, (int, float)) else None
        by_group.setdefault(grp, []).append((r, lvl))

    order = {'D': 0, 'C': 1, 'B': 2, 'A': 3}
    out: Dict[str, Any] = {}

    for grp, items in by_group.items():
        rule_classes = []
        blockers = {'C': [], 'B': [], 'A': []}

        for r, lvl in items:
            rc = _highest_class_for_rule(r, building_scope, lvl)
            rule_classes.append(rc)
            for tgt in ['C', 'B', 'A']:
                if not _meets(r, building_scope, tgt, lvl):
                    blockers[tgt].append({
                        'rule_id': r.get('rule_id'),
                        'title': r.get('title'),
                        'required_level': _req_level(r, building_scope, tgt),
                        'observed_level': lvl,
                    })

        achieved = min(rule_classes, key=lambda c: order.get(c, 0)) if rule_classes else 'D'
        out[grp] = {
            'achieved_class': achieved,
            'blockers': blockers,
            'rules_count': len(items),
        }

    return out


def build_digest_from_bundle(bundle: Dict[str, Any], max_pages: int = 20, max_lines_per_page: int = 12) -> str:
    """Compact digest from section evidences (top pages snippets)."""
    # Collect pages from all sections
    lines: List[str] = []
    lines.append(f"Case: {bundle.get('case_id')} | Report: {bundle.get('report_type')}")
    lines.append("")

    seen_pages = set()
    pages_count = 0

    for sec in (bundle.get('sections') or []):
        ev = sec.get('evidence') or ''
        # Keep only the OneNote '## Page:' blocks, ignore style exemplars to reduce noise
        in_page = False
        cur_pid = None
        cur_title = None
        cur_lines = []

        for raw in ev.splitlines():
            s = raw.strip()
            if s.startswith('## Exemples'):
                # skip style section
                in_page = False
                cur_pid = None
                cur_title = None
                cur_lines = []
                continue
            m = re.match(r"^##\s+Page:\s*(.*?)\s*\(page_id=([^,\)]+)", s)
            if m:
                # flush prev
                if cur_pid and cur_pid not in seen_pages:
                    seen_pages.add(cur_pid)
                    pages_count += 1
                    lines.append(f"- {cur_title} (page_id={cur_pid})")
                    for l in cur_lines[:max_lines_per_page]:
                        lines.append(f"  * {l}")
                    lines.append('')
                if pages_count >= max_pages:
                    break
                in_page = True
                cur_title = m.group(1).strip()
                cur_pid = m.group(2).strip()
                cur_lines = []
                continue
            if in_page and s.startswith('- '):
                cur_lines.append(s[2:].strip()[:240])

        if pages_count >= max_pages:
            break

        # flush last
        if cur_pid and cur_pid not in seen_pages and pages_count < max_pages:
            seen_pages.add(cur_pid)
            pages_count += 1
            lines.append(f"- {cur_title} (page_id={cur_pid})")
            for l in cur_lines[:max_lines_per_page]:
                lines.append(f"  * {l}")
            lines.append('')

    return '\n'.join(lines).strip()


def build_level_inference_prompt_from_digest(digest: str, rules_doc: Dict[str, Any], building_scope: str) -> str:
    rlines: List[str] = []
    rlines.append(f"Bâtiment: {building_scope}")
    rlines.append('Règles Tableau 6 (IDs + niveaux):')

    for r in rules_doc.get('rules', []):
        lv = r.get('levels') or {}
        keys = [k for k in lv.keys() if str(k).isdigit()]
        keys = sorted(keys, key=lambda x: int(x))
        lv_desc = ', '.join([f"{k}:{lv[k].get('label')}" for k in keys])
        rlines.append(f"- {r.get('rule_id')} [{r.get('group')}] {r.get('title')} => {lv_desc}")

    rules_digest = '\n'.join(rlines)

    return (
        "Tu es un auditeur GTB/BACS. À partir des preuves OneNote ci-dessous, identifie le niveau implémenté (0,1,2,3,4...) pour CHAQUE règle Tableau 6.\n"
        "Sortie attendue: UNIQUEMENT un JSON valide (sans markdown) avec la forme:\n"
        "{\"levels\": {\"<rule_id>\": <int|null>, ...}, \"evidence\": {\"<rule_id>\": [\"extrait 1\", ...]}, \"unknowns\": [\"points à confirmer\"]}\n\n"
        "Règles STRICTES:\n"
        "- Si tu ne peux pas établir un niveau, mets null.\n"
        "- Ne calcule PAS les classes A/B/C/D.\n"
        "- N'invente rien. Quand tu assignes un niveau, cite 1-3 extraits de preuve (champ evidence).\n\n"
        f"=== DIGEST PREUVES ONE NOTE ===\n{digest}\n\n"
        f"=== RÈGLES ===\n{rules_digest}\n"
    )


def render_part2_markdown(group_scores: Dict[str, Any]) -> str:
    lines = []
    lines.append('## Scoring GTB actuel selon ISO 52120-1')
    lines.append('')
    lines.append('### Synthèse')
    for grp, s in group_scores.items():
        lines.append(f"- {grp}: **classe {s.get('achieved_class')}**")
    lines.append('')

    for grp, s in group_scores.items():
        lines.append(f"#### {grp}")
        lines.append(f"Classe atteinte: **{s.get('achieved_class')}**")
        b_block = (s.get('blockers') or {}).get('B') or []
        if s.get('achieved_class') in ('C', 'D') and b_block:
            lines.append('Écarts bloquants vers la classe B:')
            for b in b_block[:12]:
                lines.append(f"- {b.get('rule_id')} — {b.get('title')} (requis: {b.get('required_level')}, observé: {b.get('observed_level') if b.get('observed_level') is not None else 'non établi'})")
        lines.append('')

    return '\n'.join(lines).strip() + '\n'


def render_part3_markdown(rules_doc: Dict[str, Any], building_scope: str, observed_levels: Dict[str, Any], target_by_group: Dict[str, str]) -> str:
    lines = []
    lines.append('## Futur : Scoring projeté et travaux à mettre en œuvre')
    lines.append('')

    groups = sorted({r.get('group') or 'Autres' for r in rules_doc.get('rules', [])})
    for grp in groups:
        tgt = target_by_group.get(grp, 'B')
        missing = []
        for r in rules_doc.get('rules', []):
            if (r.get('group') or 'Autres') != grp:
                continue
            rid = r.get('rule_id')
            lvl = observed_levels.get(rid)
            lvl = int(lvl) if isinstance(lvl, (int, float)) else None
            if not _meets(r, building_scope, tgt, lvl):
                missing.append({
                    'rule_id': rid,
                    'title': r.get('title'),
                    'required_level': _req_level(r, building_scope, tgt),
                    'observed_level': lvl,
                })
        if not missing:
            continue
        lines.append(f"#### {grp} — Objectif classe {tgt}")
        lines.append('Travaux / évolutions fonctionnelles à prévoir (alignement ISO 52120-1):')
        for m in missing:
            lines.append(f"- Mettre en œuvre {m['rule_id']} — {m['title']} (requis: {m['required_level']}, observé: {m['observed_level'] if m['observed_level'] is not None else 'non établi'})")
        lines.append('')

    return '\n'.join(lines).strip() + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='Path to draft_bundle.json')
    ap.add_argument('--bacs_rules', required=True, help='Path to Tableau 6 rules JSON')
    ap.add_argument('--bacs_building_scope', default='Non résidentiel', choices=['Résidentiel','Non résidentiel'])
    ap.add_argument('--bacs_targets', default='', help='Optional JSON mapping group->target class (default: B)')
    ap.add_argument('--out', default='', help='Output json path (default: alongside bundle as bacs_table6_from_bundle.json)')
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry_sleep', type=float, default=1.2)
    args = ap.parse_args()

    bundle = load_json(Path(args.bundle))
    rules_doc = load_rules_json(args.bacs_rules)

    target_by_group = {}
    if args.bacs_targets:
        try:
            target_by_group = load_json(Path(args.bacs_targets))
        except Exception:
            target_by_group = {}

    digest = build_digest_from_bundle(bundle)
    client = make_client()

    prompt = build_level_inference_prompt_from_digest(digest, rules_doc, args.bacs_building_scope)
    resp, err = safe_chat(client, build_messages(prompt), temperature=0.0, max_tokens=1900, retries=args.retries, base_sleep=args.retry_sleep)
    if resp is None:
        inferred = {"levels": {}, "evidence": {}, "unknowns": [f"LLM error: {err}"]}
    else:
        obj, perr = parse_json_safely(resp.text or '')
        inferred = obj if obj is not None else {"levels": {}, "evidence": {}, "unknowns": [f"Parse error: {perr}"]}

    observed_levels = inferred.get('levels') or {}
    group_scores = compute_group_scores_from_table6(rules_doc, args.bacs_building_scope, observed_levels)

    part2 = render_part2_markdown(group_scores)
    part3 = render_part3_markdown(rules_doc, args.bacs_building_scope, observed_levels, target_by_group)

    out_obj = {
        'case_id': bundle.get('case_id'),
        'report_type': bundle.get('report_type'),
        'building_scope': args.bacs_building_scope,
        'inferred': inferred,
        'group_scores': group_scores,
        'part2_markdown': part2,
        'part3_markdown': part3,
    }

    out_path = Path(args.out) if args.out else Path(args.bundle).parent / 'bacs_table6_from_bundle.json'
    save_json(out_path, out_obj)
    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
