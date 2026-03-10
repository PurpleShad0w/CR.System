#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_draft.py

PATCH (Dynamic sections / slots)
-------------------------------
- Supports plan['selected_slots_by_macro_part'] and plan['routing_slots'].
- When dynamic slots are enabled, we DO NOT force a fixed scaffold of #### headings.
  Instead, we instruct the model to:
  - produce 1..N slides (markdown) depending on evidence density,
  - merge sparse subtopics,
  - keep clear headings and bullets,
  - keep questions as "Points à confirmer".

Backward compatible:
- If slots are absent, falls back to legacy selected_buckets_by_macro_part + routing.

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from section_context import build_section_context, maybe_ctx_from_legacy_fields

# Intent mapping (unchanged)
BACX_INTENT_BY_MACRO = {
    "État des lieux GTB": "DESCRIBE_EXISTING",
    "Scoring GTB actuel": "SCORE_EXISTING",
    "Scoring projeté": "PROJECT_FUTURE",
    "Généralités": "DESCRIBE_EXISTING",
    "État du système GTB": "DESCRIBE_EXISTING",
    "Diagnostics": "DESCRIBE_EXISTING",
}

INTENT_RULES = {
    "DESCRIBE_EXISTING": {
        "forbidden": [
            "classe", "conforme", "non conforme",
            "iso", "52120",
            "objectif", "à atteindre",
            "projeté", "futur", "travaux",
            "mise en œuvre", "mettre en place", "il faut", "il est nécessaire", "doit", "devra",
        ],
        "note": "Décrire strictement l’existant à partir des preuves. Aucune recommandation ni projection.",
    },
    "SCORE_EXISTING": {
        "forbidden": [
            "travaux", "mise en œuvre",
            "objectif", "futur", "projeté",
        ],
        "note": "Décrire le scoring actuel/capacités observées. Ne pas parler du futur.",
    },
    "PROJECT_FUTURE": {
        "forbidden": [],
        "note": "Décrire une cible/projection. Les travaux ne sont permis que s’ils sont dans les preuves ou explicitement justifiés.",
    },
}


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_onenote_pages(onenote_root: Path) -> List[Dict[str, Any]]:
    pages_dir = onenote_root / "pages"
    if not pages_dir.exists():
        raise SystemExit(f"pages/ not found under {onenote_root} (expected {pages_dir})")
    pages: List[Dict[str, Any]] = []
    for p in sorted(pages_dir.glob("*.json")):
        try:
            pages.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pages


def load_section_aggregate(agg_path: Path) -> Dict[str, Any]:
    if not agg_path or not agg_path.exists():
        return {}
    try:
        return json.loads(agg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------- Evidence extraction (unchanged) ----------------

def is_question_or_request(text: str) -> bool:
    t = norm(text)
    if '?' in text:
        return True
    starters = ("peux-tu", "peux tu", "pouvez-vous", "pouvez vous", "merci de", "svp", "stp", "à faire", "todo")
    if t.startswith(starters):
        return True
    if t.startswith(("il faut", "il est nécessaire", "doit", "devra")):
        return True
    return False


def page_text_blocks(page: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    title = page.get("title") or ""
    if title:
        out.append(("title", title))
    for b in page.get("blocks", []) or []:
        t = b.get("type")
        if t in ("heading", "paragraph", "image_ocr"):
            txt = (b.get("text") or "").strip()
            if txt:
                out.append((t, txt))
        elif t == "audio":
            txt = (b.get("transcript") or "").strip()
            if txt:
                out.append(("audio_transcript", txt))
    return out


def extract_snippets(page: Dict[str, Any], keywords: List[str], max_snips: int = 14) -> Tuple[List[str], List[str]]:
    kws = [norm(k) for k in keywords if k]
    fact_snips: List[str] = []
    req_snips: List[str] = []

    for t, txt in page_text_blocks(page):
        nt = norm(txt)
        hit = sum(1 for k in kws if k and k in nt)
        if hit <= 0:
            continue
        snippet = f"[{t}] {txt[:700]}"
        if is_question_or_request(txt):
            req_snips.append(snippet)
        else:
            fact_snips.append(snippet)
        if len(fact_snips) + len(req_snips) >= max_snips:
            break

    return fact_snips, req_snips


def build_evidence_block(label: str, keywords: List[str], routed_pages: List[Dict[str, Any]], pages_by_id: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    requests: List[str] = []

    lines.append(f"- Slot/Bucket: {label}")
    lines.append(f"- Mots-clés: {', '.join(keywords)}")
    lines.append("")

    for rp in routed_pages:
        pid = rp.get("page_id")
        score = rp.get("score")
        page = pages_by_id.get(pid)
        if not page:
            continue
        title = page.get("title") or pid
        lines.append(f"## Page: {title} (page_id={pid}, score={score})")
        fact_snips, req_snips = extract_snippets(page, keywords, max_snips=14)

        if not fact_snips and not req_snips:
            lines.append("(Aucun extrait direct trouvé; vérifier la page complète.)")
        else:
            for s in fact_snips:
                lines.append(f"- {s}")
            for s in req_snips:
                requests.append(f"page_id={pid} {s}")
        lines.append("")

    if requests:
        lines.append("## Questions / demandes présentes dans les notes (non factuelles)")
        lines.append("⚠️ À traiter comme 'Points à confirmer' (ne pas les transformer en faits).")
        for r in requests[:18]:
            lines.append(f"- {r[:800]}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------- Prompt helpers ----------------

def build_intent_block(intent: str) -> str:
    rules = INTENT_RULES.get(intent) or {}
    forb = rules.get("forbidden") or []
    note = rules.get("note") or ""

    lines = ["### Intention & contraintes (contrat)"]
    if note:
        lines.append(f"- {note}")
    if forb:
        lines.append("- Mots / thèmes interdits (à ne pas écrire, même si évoqués dans une question des notes) :")
        for f in forb:
            lines.append(f"  - {f}")
    lines.append("- Chaque constat doit inclure une preuve: 'Preuve: page_id=... + extrait'.")
    lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_dynamic_structure_instructions(mp: int) -> str:
    return (
        "### Structure dynamique (slides adaptatives)\n"
        "- Tu peux produire 1 à N sous-sections (####) selon la densité des preuves.\n"
        "- Si 2 catégories ont très peu de matière, fusionne-les dans une même sous-section.\n"
        "- Reste fidèle aux preuves: pas d'invention.\n"
        "- Termine par '#### Points à confirmer' si des demandes/questions existent.\n\n"
    )


def render_prompt(template_cfg: Dict[str, Any], **kwargs) -> str:
    fmt_lines = template_cfg["section_prompt_format"]
    txt = "\n".join(fmt_lines)
    return txt.format(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to process/plans/<case_id>.json")
    ap.add_argument("--onenote", required=True, help="Path to process/onenote/<notebook> folder (contains pages/)")
    ap.add_argument("--templates", default="input/config/prompt_templates.json", help="Path to prompt_templates.json")
    ap.add_argument("--out", default="process/drafts", help="Output root under process/")
    ap.add_argument("--section-aggregate", default="", help="Optional path to process/onenote_aggregates/<notebook>/<section_slug>.json")

    args = ap.parse_args()

    plan = load_json(Path(args.plan))
    tpl = load_json(Path(args.templates))
    agg = load_section_aggregate(Path(args.section_aggregate)) if args.section_aggregate else {}

    onenote_root = Path(args.onenote)
    pages = load_onenote_pages(onenote_root)

    case_id = plan.get("case_id") or Path(args.plan).stem
    report_type = plan.get("report_type")
    macro_parts = plan.get("macro_parts") or {}
    gen_parts = plan.get("generate_macro_parts") or [1, 2, 3]

    pages_by_id: Dict[str, Dict[str, Any]] = {}
    for p in pages:
        pid = p.get("page_id") or p.get("title")
        if pid:
            pages_by_id[pid] = p

    # Section context propagation
    section_ctx = plan.get("section_context")
    if not section_ctx and agg:
        legacy = maybe_ctx_from_legacy_fields(agg)
        if legacy:
            section_ctx = legacy
    if plan.get("onenote_section") and not section_ctx:
        section_ctx = build_section_context(onenote_root.name, plan.get("onenote_section"))

    out_root = Path(args.out) / case_id
    prompts_dir = out_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, Any] = {
        "case_id": case_id,
        "report_type": report_type,
        "generate_macro_parts": gen_parts,
        "macro_parts": macro_parts,
        "section_context": section_ctx,
        "sections": [],
    }

    all_prompts: List[str] = []
    rt_templates = tpl.get("templates", {}).get(report_type, {})

    # Prefer slots if present
    slots_by_part = plan.get("selected_slots_by_macro_part") or {}
    routing_slots = plan.get("routing_slots") or {}

    selected_by_part = plan.get("selected_buckets_by_macro_part") or {}
    routing = plan.get("routing") or {}

    dynamic_enabled = bool((plan.get("dynamic_sections") or {}).get("enabled", False)) and bool(slots_by_part)

    for mp_str, mp_cfg in (macro_parts or {}).items():
        mp = int(mp_str)
        if mp not in gen_parts:
            continue

        mp_name = mp_cfg.get("name") if isinstance(mp_cfg, dict) else str(mp_cfg)
        mp_template = rt_templates.get(f"macro_part_{mp}", {})
        mp_instructions = mp_template.get("instructions", [])

        if dynamic_enabled:
            slot_items = slots_by_part.get(str(mp)) or []
            for si in slot_items:
                slot_id = si.get("slot_id")
                slot_title = si.get("slot_title") or slot_id
                r = routing_slots.get(slot_id) or {}
                keywords = r.get("keywords") or []
                top_pages = r.get("top_pages") or []

                evidence_full = build_evidence_block(slot_title, keywords, top_pages, pages_by_id)

                intent = BACX_INTENT_BY_MACRO.get(mp_name) or ("PROJECT_FUTURE" if mp == 3 else "DESCRIBE_EXISTING")

                prompt = render_prompt(
                    tpl,
                    case_id=case_id,
                    report_type=report_type,
                    macro_part_num=mp,
                    macro_part_name=mp_name,
                    bucket_id=slot_title,
                    evidence_block=evidence_full,
                )

                if mp_instructions:
                    header = "### Instructions macro-partie\n- " + "\n- ".join(mp_instructions) + "\n\n"
                    prompt = header + prompt

                prompt = build_dynamic_structure_instructions(mp) + build_intent_block(intent) + prompt

                if isinstance(section_ctx, dict) and section_ctx.get("onenote_section_name"):
                    sec_hdr = (
                        "### Contexte OneNote\n"
                        f"- Notebook: {section_ctx.get('onenote_notebook')}\n"
                        f"- Section: {section_ctx.get('onenote_section_name')} (slug: {section_ctx.get('section_slug')})\n\n"
                    )
                    prompt = sec_hdr + prompt

                section_obj = {
                    "macro_part": mp,
                    "macro_part_name": mp_name,
                    "bucket_id": slot_id,
                    "bucket_label": slot_title,
                    "component_buckets": r.get("component_buckets") or si.get("component_buckets") or [],
                    "keywords": keywords,
                    "top_pages": top_pages,
                    "evidence": evidence_full,
                    "prompt": prompt,
                    "section_context": section_ctx,
                }

                bundle["sections"].append(section_obj)
                p_out = prompts_dir / f"{slot_id}.txt"
                p_out.write_text(prompt, encoding="utf-8")
                all_prompts.append(f"\n\n===== {slot_id} (Macro {mp}: {mp_name}) =====\n\n{prompt}")

        else:
            bucket_items = selected_by_part.get(str(mp)) or selected_by_part.get(mp) or []
            for bi in bucket_items:
                bucket_id = bi.get("bucket_id")
                if not bucket_id:
                    continue

                r = routing.get(bucket_id) or {}
                keywords = r.get("keywords") or []
                top_pages = r.get("top_pages") or []

                evidence_full = build_evidence_block(bucket_id, keywords, top_pages, pages_by_id)

                intent = BACX_INTENT_BY_MACRO.get(mp_name) or ("PROJECT_FUTURE" if mp == 3 else "DESCRIBE_EXISTING")
                prompt = render_prompt(
                    tpl,
                    case_id=case_id,
                    report_type=report_type,
                    macro_part_num=mp,
                    macro_part_name=mp_name,
                    bucket_id=bucket_id,
                    evidence_block=evidence_full,
                )

                if mp_instructions:
                    header = "### Instructions macro-partie\n- " + "\n- ".join(mp_instructions) + "\n\n"
                    prompt = header + prompt

                prompt = build_intent_block(intent) + prompt

                section_obj = {
                    "macro_part": mp,
                    "macro_part_name": mp_name,
                    "bucket_id": bucket_id,
                    "bucket_score": bi.get("score"),
                    "keywords": keywords,
                    "top_pages": top_pages,
                    "evidence": evidence_full,
                    "prompt": prompt,
                    "section_context": section_ctx,
                }

                bundle["sections"].append(section_obj)
                p_out = prompts_dir / f"{bucket_id}.txt"
                p_out.write_text(prompt, encoding="utf-8")
                all_prompts.append(f"\n\n===== {bucket_id} (Macro {mp}: {mp_name}) =====\n\n{prompt}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "draft_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "prompts.txt").write_text("\n".join(all_prompts).strip() + "\n", encoding="utf-8")

    print(f"Wrote: {out_root / 'draft_bundle.json'}")
    print(f"Wrote: {out_root / 'prompts.txt'}")
    print(f"Prompts directory: {prompts_dir}")


if __name__ == "__main__":
    main()
