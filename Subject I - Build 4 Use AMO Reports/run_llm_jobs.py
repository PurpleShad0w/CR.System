#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from llm_client import make_client
from quality_score import evaluate_quality  # deterministic full-report scorer
import quality_score as qs  # reuse exact rule lists + helpers to avoid divergence
from section_context import require_section_context

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# -----------------------------
# IO helpers
# -----------------------------

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# LLM message builder
# -----------------------------

def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


# -----------------------------
# Robust chat wrapper (retries + backoff)
# -----------------------------

def safe_chat(
    client,
    messages,
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
    retries: int = 4,
    base_sleep: float = 1.2,
) -> Tuple[Optional[Any], Optional[str]]:
    """Returns (resp, err). Never raises."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )
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
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
            time.sleep(sleep_s)
    return None, last_err


# -----------------------------
# Multistep prompts (internal)
# -----------------------------

def prompt_extract_facts(section: Dict[str, Any]) -> str:
    mp = section.get("macro_part")
    mp_name = section.get("macro_part_name") or f"Macro {mp}"
    bucket = section.get("bucket_id") or "SECTION"
    evidence = (section.get("evidence") or "").strip()

    schema = {
        "observations": ["string"],
        "entities": ["string"],
        "quantities": ["string"],
        "constraints_from_evidence": ["string"],
        "unknowns": ["string"],
        "norm_refs": ["string"],
        "do_not_say": ["string"],
    }

    return (
        "Tu dois produire UNIQUEMENT un objet JSON valide (sans texte autour, sans markdown).\n"
        "Tâche: extraire un 'fact sheet' à partir des preuves OneNote.\n\n"
        f"Contexte:\n- Macro-partie: {mp} - {mp_name}\n- Bucket: {bucket}\n\n"
        "Règles STRICTES:\n"
        "- Ne rédige PAS de section de rapport.\n"
        "- Ne propose PAS de travaux/actions (sauf si c'est explicitement énoncé comme fait/élément dans les preuves).\n"
        "- Ne déduis pas de faits non présents.\n"
        "- Si une information manque, mets-la dans unknowns.\n"
        "- Si une phrase des preuves est une QUESTION/DEMANDE (ex: 'peux-tu...'), ne la transforme pas en fait (mets-la dans unknowns).\n\n"
        f"Format JSON attendu (mêmes clés):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Preuves (texte brut):\n{evidence}\n"
    )


def prompt_write_from_facts(section: Dict[str, Any], facts: Dict[str, Any]) -> str:
    base_prompt = (section.get("prompt") or "").strip()
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    return (
        f"{base_prompt}\n\n"
        "---\n"
        "FACTS (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ces FACTS pour rédiger. Ignore les preuves brutes.\n"
        "Pour chaque constat important, ajoute une ligne 'Preuve: page_id=... + extrait court'.\n"
        f"{facts_json}\n"
        "---\n\n"
        "Rappel critique:\n"
        "- N'invente rien.\n"
        "- Si quelque chose est dans unknowns, indique-le explicitement dans '#### Points à confirmer'.\n"
        "- Respecte strictement l'intention de la macro-partie.\n"
    )


def prompt_repair(section: Dict[str, Any], facts: Dict[str, Any], draft_text: str, issues: List[str]) -> str:
    mp = section.get("macro_part")
    mp_name = section.get("macro_part_name") or f"Macro {mp}"
    bucket = section.get("bucket_id") or "SECTION"
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    issues_txt = "\n- ".join(issues) if issues else "(aucun)"

    return (
        "Tu dois CORRIGER une section de rapport pour supprimer des violations.\n"
        f"Contexte: Macro-partie {mp} - {mp_name} Bucket {bucket}\n\n"
        "Contraintes:\n"
        "- Ne change pas le fond: reste fidèle aux FACTS.\n"
        "- Ne rajoute aucun fait.\n"
        "- Supprime toute formulation qui viole les règles listées.\n"
        "- Si info manquante: mets-la dans '#### Points à confirmer'.\n"
        "- Si les preuves contiennent des questions/demandes, ne les transforme pas en faits: mets-les dans '#### Points à confirmer'.\n"
        "- Conserver le plan (####) et le format Constats/Preuve.\n"
        "- Sortie: UNIQUEMENT le texte corrigé.\n\n"
        f"Issues détectées:\n- {issues_txt}\n\n"
        f"FACTS (source unique):\n{facts_json}\n\n"
        f"TEXTE À CORRIGER:\n{draft_text}\n"
    )


# -----------------------------
# Helpers: parsing + deterministic fallbacks
# -----------------------------

def parse_json_safely(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw:
        return None, "empty"
    s = raw.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "not_a_dict"
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            return None, "extracted_not_a_dict"
        except Exception as e:
            return None, f"json_extract_parse_failed:{e}"
    return None, "no_json_object_found"


def default_empty_facts() -> Dict[str, Any]:
    return {
        "observations": [],
        "entities": [],
        "quantities": [],
        "constraints_from_evidence": [],
        "unknowns": [],
        "norm_refs": [],
        "do_not_say": [],
    }


def heuristic_facts_from_evidence(section: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback if HF is down."""
    evidence = (section.get("evidence") or "")
    lines = [ln.strip() for ln in evidence.splitlines() if ln.strip()]
    facts = default_empty_facts()

    for ln in lines:
        if ln.startswith("## Page:"):
            facts["entities"].append(ln.replace("## Page:", "").strip())
            continue
        if ln.startswith("- [") and "]" in ln:
            content = ln.split("]", 1)[1].strip()
            if "?" in content or content.lower().startswith(("peux-tu", "peux tu", "pouvez-vous", "pouvez vous")):
                facts["unknowns"].append(content)
            else:
                facts["observations"].append(content)
            for tok in content.split():
                if tok.isdigit():
                    facts["quantities"].append(content)
                    break
            low = content.lower()
            if "iso" in low or "52120" in low or "décret bacs" in low or "decret bacs" in low:
                facts["norm_refs"].append(content)
            continue
        if ln.startswith("- Bucket:") or ln.startswith("- Mots-clés:"):
            facts["constraints_from_evidence"].append(ln)

    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            k = (x or "").strip()
            if not k:
                continue
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    for k in list(facts.keys()):
        facts[k] = dedup(facts[k])

    return facts


# -----------------------------
# NEW: traceability + sanitization
# -----------------------------

def has_evidence_markers(text: str) -> bool:
    t = (text or "")
    return ("page_id=" in t) or ("Preuve:" in t) or ("[preuve" in t.lower())


def sanitize_text_for_intent(section: Dict[str, Any], text: str) -> str:
    """Deterministic last-resort sanitizer.

    Removes lines that violate intent constraints.
    Uses the same forbidden/prescriptive lists as quality_score.
    """
    mp = section.get("macro_part")
    if not text:
        return text

    lines = text.splitlines()
    out = []

    forb = qs.INTENT_FORBIDDEN.get(mp, [])
    presc = qs.PRESCRIPTIVE_MP1 if mp == 1 else []

    for ln in lines:
        lnorm = qs.norm(ln)
        if lnorm.startswith("####") or lnorm.startswith("###"):
            out.append(ln)
            continue
        if "points à confirmer" in lnorm:
            out.append(ln)
            continue
        if any(qs.norm(f) in lnorm for f in forb if f):
            continue
        if mp == 1 and any(qs.norm(p) in lnorm for p in presc if p):
            continue
        if any(qs.norm(p) in lnorm for p in qs.PLACEHOLDERS if p):
            continue
        out.append(ln)

    return "\n".join(out).strip()


def keyword_hit_ratio(section: Dict[str, Any], text: str) -> float:
    kws = [qs.norm(k) for k in (section.get("keywords") or []) if k]
    t = qs.norm(text or "")
    if not kws:
        return 0.0
    hits = sum(1 for k in kws if k and k in t)
    return hits / len(kws)


def detect_section_issues(section: Dict[str, Any], text: str, strict_traceability: bool = False) -> Tuple[List[str], Dict[str, Any]]:
    mp = section.get("macro_part")
    t = text or ""

    forb = qs.INTENT_FORBIDDEN.get(mp, [])
    forb_hits = qs.count_hits(t, forb)
    ph_hits = qs.count_hits(t, qs.PLACEHOLDERS)
    mp1_presc_hits = qs.count_hits(t, qs.PRESCRIPTIVE_MP1) if mp == 1 else 0
    khr = keyword_hit_ratio(section, t)

    issues: List[str] = []
    if forb_hits:
        issues.append(f"forbidden_terms_hits={forb_hits}")
    if ph_hits:
        issues.append(f"placeholders_hits={ph_hits}")
    if mp1_presc_hits:
        issues.append(f"mp1_prescriptive_hits={mp1_presc_hits}")
    if (section.get("keywords") or []) and khr < 0.25:
        issues.append(f"low_keyword_alignment={khr:.2f}")

    if strict_traceability:
        tp = section.get("top_pages") or []
        if tp and not has_evidence_markers(t):
            issues.append("missing_evidence_markers")

    metrics = {
        "forbidden_hits": forb_hits,
        "placeholder_hits": ph_hits,
        "mp1_prescriptive_hits": mp1_presc_hits,
        "keyword_hit_ratio": round(khr, 3),
        "has_evidence_markers": bool(has_evidence_markers(t)),
    }

    return issues, metrics


# -----------------------------
# Tableau 6 scoring (ISO 52120-1) — copied from your current pipeline
# -----------------------------

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


def _digest_from_section_aggregate(agg: Dict[str, Any], max_pages: int = 22, max_blocks_per_page: int = 4) -> str:
    sec = agg.get('onenote_section') or (agg.get('section_context') or {}).get('onenote_section_name') or ''
    inv = agg.get('inventory') or []
    pages = agg.get('pages') or []

    lines: List[str] = []
    lines.append(f"Section OneNote: {sec}")
    lines.append(f"Pages agrégées: {agg.get('page_count')}")
    lines.append('')

    lines.append('Inventaire (familles):')
    for it in inv[:20]:
        lines.append(f"- {it.get('family')}: {it.get('count')} élément(s)")

    lines.append('')
    lines.append('Index des pages (titres + extraits):')

    for p in pages[:max_pages]:
        lines.append(f"- {p.get('title')} (images={p.get('num_images')}, audio={p.get('num_audio')})")
        tbs = p.get('text_blocks') or []
        for tb in tbs[:max_blocks_per_page]:
            txt = (tb.get('text') or '').strip()
            if txt:
                lines.append(f"  * {tb.get('type')}: {txt[:220]}")

    return '\n'.join(lines)


def build_level_inference_prompt(agg: Dict[str, Any], rules_doc: Dict[str, Any], building_scope: str) -> str:
    digest = _digest_from_section_aggregate(agg)
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


def infer_levels_with_llm(client, agg: Dict[str, Any], rules_doc: Dict[str, Any], building_scope: str, retries: int, retry_sleep: float) -> Dict[str, Any]:
    prompt = build_level_inference_prompt(agg, rules_doc, building_scope)
    resp, err = safe_chat(client, build_messages(prompt), temperature=0.0, max_tokens=1800, top_p=1.0, retries=retries, base_sleep=retry_sleep)
    if resp is None:
        return {"levels": {}, "evidence": {}, "unknowns": [f"LLM error: {err}"]}

    obj, perr = parse_json_safely(resp.text or '')
    if obj is None:
        return {"levels": {}, "evidence": {}, "unknowns": [f"Parse error: {perr}"]}
    if 'levels' not in obj:
        return {"levels": {}, "evidence": {}, "unknowns": ["No levels field in LLM output"]}
    return obj


def prompt_part2_scoring_slides(agg: Dict[str, Any], rules_doc: Dict[str, Any], building_scope: str, inferred: Dict[str, Any], group_scores: Dict[str, Any]) -> str:
    digest = _digest_from_section_aggregate(agg, max_pages=12, max_blocks_per_page=2)
    payload = {
        "building_scope": building_scope,
        "inferred_levels": inferred.get('levels') or {},
        "inferred_evidence": inferred.get('evidence') or {},
        "unknowns": inferred.get('unknowns') or [],
        "group_scores": group_scores,
    }

    rules_index = []
    for r in rules_doc.get('rules', []):
        rules_index.append({
            "rule_id": r.get('rule_id'),
            "group": r.get('group'),
            "title": r.get('title'),
            "class_requirements": (r.get('class_requirements') or {}).get(building_scope) or {},
        })

    return (
        "Tu dois produire UNIQUEMENT du texte (pas de JSON), au format Markdown.\n"
        "Objectif: générer des SLIDES de scoring (Partie 2) prêts à être paginés.\n\n"
        "CONTRAINTES STRICTES:\n"
        "- Tu n'inventes aucun fait.\n"
        "- Tu n'utilises QUE les données JSON fournies ci-dessous (scores + preuves extraites).\n"
        "- Si une information est inconnue, tu l'écris dans une slide 'Points à confirmer'.\n"
        "- Format: chaque slide commence par une ligne '#### <Titre de slide>' puis des puces '- ...' (max 8 puces).\n"
        "- Ne produis pas de macro-partie 4.\n\n"
        "Contenu attendu (minimum):\n"
        "1) Une slide 'Synthèse' (classes atteintes par usage/groupe)\n"
        "2) Une slide par groupe\n"
        " - rappeler la classe atteinte\n"
        " - lister 3-6 règles bloquantes vers B (si classe < B), avec (rule_id, niveau requis, niveau observé)\n"
        " - ajouter 1-2 extraits de preuves si disponibles\n"
        "3) Une slide 'Points à confirmer'\n\n"
        "=== CONTEXTE ONE NOTE (digest) ===\n"
        f"{digest}\n\n"
        "=== RÈGLES (index) ===\n"
        f"{json.dumps(rules_index, ensure_ascii=False, indent=2)[:6000]}\n\n"
        "=== DONNÉES DE SCORING (source unique) ===\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to process/drafts/<case_id>/draft_bundle.json")
    ap.add_argument("--out", default="", help="Output folder (defaults to bundle folder)")

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--mode", choices=["single", "multistep"], default="multistep")

    ap.add_argument("--facts_max_tokens", type=int, default=650)
    ap.add_argument("--facts_temperature", type=float, default=0.0)

    ap.add_argument("--repair_max_tokens", type=int, default=1000)
    ap.add_argument("--repair_temperature", type=float, default=0.2)
    ap.add_argument("--max_repairs", type=int, default=3)

    ap.add_argument("--min_quality", type=float, default=0.0)
    ap.add_argument("--quality", default="", help="Path to write quality_report.json")

    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry_sleep", type=float, default=1.2)

    ap.add_argument("--strict_traceability", action="store_true", help="Require evidence markers when evidence exists.")

    # Tableau 6 inputs
    ap.add_argument("--section_aggregate", default="", help="Path to process/onenote_aggregates/<notebook>/<section_slug>.json")
    ap.add_argument("--bacs_rules", default="", help="Path to Tableau 6 rules JSON")
    ap.add_argument("--bacs_building_scope", default="Non résidentiel", choices=["Résidentiel", "Non résidentiel"])
    ap.add_argument("--bacs_targets", default="", help="Optional JSON mapping group->target class for Part 3")
    ap.add_argument("--bacs_part2_slides", action="store_true", help="If set, Part 2 is generated by LLaMA as slide-ready markdown.")

    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    if bundle.get("section_context"):
        require_section_context(bundle, "draft_bundle.json")

    client = make_client()
    generated = dict(bundle)

    # 1) Generate sections via LLM
    for sec in generated.get("sections", []):
        base_prompt = (sec.get("prompt") or "").strip()
        if not base_prompt:
            sec["generated_text"] = ""
            sec["final_text"] = ""
            sec["llm_raw"] = None
            continue

        if args.mode == "single":
            resp, err = safe_chat(
                client,
                build_messages(base_prompt),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                retries=args.retries,
                base_sleep=args.retry_sleep,
            )
            if resp is None:
                sec["generated_text"] = ""
                sec["final_text"] = ""
                sec["llm_error"] = err
                sec["llm_raw"] = None
            else:
                text = (resp.text or "").strip()
                sec["generated_text"] = text
                sec["final_text"] = text
                sec["llm_raw"] = resp.raw
            continue

        # MULTISTEP
        facts_prompt = prompt_extract_facts(sec)
        facts_resp, facts_err = safe_chat(
            client,
            build_messages(facts_prompt),
            temperature=args.facts_temperature,
            max_tokens=args.facts_max_tokens,
            top_p=1.0,
            retries=args.retries,
            base_sleep=args.retry_sleep,
        )

        if facts_resp is None:
            sec["facts_llm_error"] = facts_err
            facts_obj = heuristic_facts_from_evidence(sec)
            sec["facts_source"] = "heuristic"
            sec["facts_json"] = facts_obj
            sec["llm_raw_facts"] = None
        else:
            facts_obj, parse_err = parse_json_safely((facts_resp.text or ""))
            if facts_obj is None:
                sec["facts_parse_error"] = parse_err
                facts_obj = heuristic_facts_from_evidence(sec)
                sec["facts_source"] = "heuristic_parse_fallback"
            else:
                sec["facts_source"] = "llm"
                sec["facts_json"] = facts_obj
                sec["llm_raw_facts"] = facts_resp.raw

        write_prompt = prompt_write_from_facts(sec, facts_obj)
        write_resp, write_err = safe_chat(
            client,
            build_messages(write_prompt),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            retries=args.retries,
            base_sleep=args.retry_sleep,
        )

        if write_resp is None:
            sec["write_llm_error"] = write_err
            sec["generated_text"] = ""
            sec["final_text"] = ""
            sec["llm_raw"] = None
            continue

        draft_text = (write_resp.text or "").strip()
        sec["draft_text"] = draft_text
        sec["llm_raw_write"] = write_resp.raw

        final_text = draft_text

        # Repair loop
        repairs_done = 0
        while repairs_done < max(0, args.max_repairs):
            issues, metrics = detect_section_issues(sec, final_text, strict_traceability=args.strict_traceability)
            sec["section_metrics"] = metrics
            if not issues:
                break
            rep_prompt = prompt_repair(sec, facts_obj, final_text, issues)
            rep_resp, rep_err = safe_chat(
                client,
                build_messages(rep_prompt),
                temperature=args.repair_temperature,
                max_tokens=args.repair_max_tokens,
                top_p=1.0,
                retries=args.retries,
                base_sleep=args.retry_sleep,
            )
            if rep_resp is None:
                sec.setdefault("repair_errors", []).append({"issues": issues, "error": rep_err})
                break
            repaired = (rep_resp.text or "").strip()
            sec.setdefault("repair_attempts", []).append({
                "issues": issues,
                "repaired_text": repaired,
                "llm_raw_repair": rep_resp.raw,
            })
            final_text = repaired
            repairs_done += 1

        # Deterministic sanitizer (last resort)
        sanitized = sanitize_text_for_intent(sec, final_text)
        sec["sanitized"] = (sanitized != final_text)
        final_text = sanitized

        sec["final_text"] = final_text
        sec["generated_text"] = final_text
        sec["llm_raw"] = {"facts": sec.get("llm_raw_facts"), "write": sec.get("llm_raw_write")}

    gen_path = out_dir / "generated_bundle.json"
    save_json(gen_path, generated)

    # 2) Assemble report
    assembled: Dict[str, Any] = {
        "case_id": generated.get("case_id"),
        "report_type": generated.get("report_type"),
        "section_context": generated.get("section_context"),
        "macro_parts": [],
    }

    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get("sections", []):
        mp = sec.get("macro_part")
        by_mp.setdefault(mp, []).append(sec)

    for mp in sorted(by_mp.keys()):
        assembled["macro_parts"].append({
            "macro_part": mp,
            "macro_part_name": by_mp[mp][0].get("macro_part_name"),
            "sections": [
                {"bucket_id": s.get("bucket_id"), "text": (s.get("final_text") or s.get("generated_text") or "").strip()}
                for s in by_mp[mp]
            ],
        })

    # 3) Tableau 6 scoring injection for Part 2 & Part 3 (ISO 52120-1)
    if (generated.get('report_type') == 'BACS_SCORING') and args.section_aggregate and args.bacs_rules:
        try:
            rules_doc = load_rules_json(args.bacs_rules)
            agg = load_json(Path(args.section_aggregate))
            inferred = infer_levels_with_llm(client, agg, rules_doc, args.bacs_building_scope, args.retries, args.retry_sleep)
            observed_levels = inferred.get('levels') or {}
            group_scores = compute_group_scores_from_table6(rules_doc, args.bacs_building_scope, observed_levels)

            # Part 2
            if args.bacs_part2_slides:
                p = prompt_part2_scoring_slides(agg, rules_doc, args.bacs_building_scope, inferred, group_scores)
                resp, err = safe_chat(client, build_messages(p), temperature=0.2, max_tokens=1400, top_p=1.0, retries=args.retries, base_sleep=args.retry_sleep)
                if resp is None:
                    part2_text = "## Scoring GTB actuel selon ISO 52120-1\n\n- À confirmer sur site / données manquantes (erreur LLM slides)\n"
                    assembled.setdefault('bacs_table6_errors', []).append(f"Part2 slides LLM error: {err}")
                else:
                    part2_text = (resp.text or '').strip() + "\n"
            else:
                lines = []
                lines.append('## Scoring GTB actuel selon ISO 52120-1')
                lines.append('')
                for grp, s in group_scores.items():
                    lines.append(f"#### {grp}")
                    lines.append(f"Classe atteinte (fonctionnalités BAC/GTB): **{s.get('achieved_class')}**")
                    b_block = (s.get('blockers') or {}).get('B') or []
                    if s.get('achieved_class') in ('C', 'D') and b_block:
                        lines.append('Principaux écarts bloquants vers la classe B:')
                        for b in b_block[:10]:
                            lines.append(f"- {b.get('rule_id')} — {b.get('title')} (niveau requis: {b.get('required_level')}, observé: {b.get('observed_level') if b.get('observed_level') is not None else 'non établi'})")
                    lines.append('')
                part2_text = '\n'.join(lines).strip() + '\n'

            # Part 3
            target_by_group = {}
            if args.bacs_targets:
                try:
                    target_by_group = load_json(Path(args.bacs_targets))
                except Exception:
                    target_by_group = {}

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
                    if not _meets(r, args.bacs_building_scope, tgt, lvl):
                        missing.append({
                            'rule_id': rid,
                            'title': r.get('title'),
                            'required_level': _req_level(r, args.bacs_building_scope, tgt),
                            'observed_level': lvl,
                        })
                if not missing:
                    continue
                lines.append(f"#### {grp} — Objectif classe {tgt}")
                lines.append('Travaux / évolutions fonctionnelles à prévoir (au strict sens ISO 52120-1):')
                for m in missing:
                    lines.append(f"- Mettre en œuvre {m['rule_id']} — {m['title']} (niveau requis: {m['required_level']}, observé: {m['observed_level'] if m['observed_level'] is not None else 'non établi'})")
                lines.append('')
            part3_text = '\n'.join(lines).strip() + '\n'

            assembled['bacs_table6'] = {
                'building_scope': args.bacs_building_scope,
                'inferred_levels': inferred,
                'group_scores': group_scores,
            }

            def upsert_macro(mp_num: int, mp_name: str, text: str):
                for mp in assembled.get('macro_parts', []):
                    if int(mp.get('macro_part', -1)) == mp_num:
                        mp['macro_part_name'] = mp_name
                        mp['sections'] = [{'bucket_id': 'AUTO_TABLE6', 'text': text}]
                        return
                assembled['macro_parts'].append({
                    'macro_part': mp_num,
                    'macro_part_name': mp_name,
                    'sections': [{'bucket_id': 'AUTO_TABLE6', 'text': text}],
                })

            upsert_macro(2, 'Scoring GTB actuel selon ISO 52120-1', part2_text)
            upsert_macro(3, 'Futur : Scoring projeté et travaux à mettre en œuvre', part3_text)

        except Exception as e:
            assembled.setdefault('bacs_table6_errors', []).append(str(e))

    assembled_path = out_dir / "assembled_report.json"
    save_json(assembled_path, assembled)

    print(f"Wrote: {gen_path}")
    print(f"Wrote: {assembled_path}")

    quality = evaluate_quality(generated, assembled)
    q_path = Path(args.quality) if args.quality else (out_dir / "quality_report.json")
    save_json(q_path, quality)
    print(f"Wrote: {q_path}")
    print(f"Quality total: {quality.get('total')} / 100")

    if args.min_quality and (quality.get("total", 0) < args.min_quality):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
