#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_draft.py

PATCH: Clean Questions/Requests formatting + remove internal block-type prefixes.

User intent
-----------
- Only correct things that should not appear in a report slide (e.g., raw `page_id=... [paragraph]` lines)
- Keep exhaustivity: do NOT reduce the amount of evidence.

What this patch does
--------------------
1) Evidence snippets no longer include the internal OneNote block type prefix like "[paragraph]".
2) Questions/requests are emitted in a dedicated block as:
   - Question: ...
     Preuve: page_id=... + extrait
   (instead of a single raw line prefixed by `page_id=... [paragraph]`)

Everything else (routing, prompts, intent/scaffold blocks, style exemplars) is unchanged.

NOTE
----
This file is a drop-in replacement for your current generate_draft.py.
It is based on the version currently in your repo (quality & fidelity patch) with minimal edits.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from section_context import build_section_context, maybe_ctx_from_legacy_fields

# Intent mapping
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

# Style mapping: bucket->canonical historical section names
STYLE_SECTION_BY_BUCKET = {
    "BACS_SCORING": {
        "ETAT_DES_LIEUX_GTB": ["État des lieux du BACS", "Présentation du site et application du Décret BACS", "État des lieux GTB"],
        "SCORING_ACTUEL": ["Scoring GTB selon ISO 52120-1", "SCORING GTB"],
        "SCORING_PROJETE": ["Travaux et TRI", "Scénarios", "Scoring GTB selon ISO 52120-1"],
    },
    "ETAT_GTB_AUDIT": {
        "SITE_CONTEXTE": ["Contexte et objectifs", "Présentation"],
        "ARCHI_GTB": ["État des lieux GTB", "Architecture", "Supervision"],
        "COMMUNICATIONS": ["Communications", "Réseau"],
        "DIAGNOSTICS": ["Diagnostics", "Dysfonctionnements"],
    }
}

# Structure scaffold per macro-part
SCAFFOLD = {
    "BACS_SCORING": {
        1: [
            "#### Comptage énergétique",
            "#### Supervision",
            "#### Historisation",
            "#### Alarmes",
            "#### Couverture des usages",
            "#### Limites",
            "#### Points à confirmer",
        ],
        2: [
            "#### Synthèse des capacités observées",
            "#### Chauffage",
            "#### Refroidissement",
            "#### Ventilation",
            "#### Éclairage",
            "#### Points à confirmer",
        ],
        3: [
            "#### Hypothèses",
            "#### Cible / classe visée",
            "#### Travaux & fonctions",
            "#### Points à confirmer",
        ],
    },
    "ETAT_GTB_AUDIT": {
        1: [
            "#### Contexte",
            "#### Périmètre",
            "#### Méthode",
            "#### Points à confirmer",
        ],
        2: [
            "#### Supervision",
            "#### Réseau & protocoles",
            "#### Comptage & historisation",
            "#### Alarmes",
            "#### Limites",
            "#### Points à confirmer",
        ],
        3: [
            "#### Constats / anomalies",
            "#### Impacts exploitation",
            "#### Points à confirmer",
        ],
    }
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


# ---------------- Style corpus ----------------

def load_style_corpus(corpus_dir: Path, max_files: int = 120) -> List[Dict[str, Any]]:
    if not corpus_dir or not corpus_dir.exists():
        return []
    docs = []
    for p in sorted(corpus_dir.glob("*.json"))[:max_files]:
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
    return docs


def _flatten_text_from_doc(doc: Dict[str, Any]) -> str:
    parts = (doc.get("parts_first3") or {})
    chunks = []
    for k in ("part_1", "part_2", "part_3"):
        for it in (parts.get(k) or []):
            t = (it.get("text") or "").strip()
            if t:
                chunks.append(t)
    if chunks:
        return "\n".join(chunks)
    return json.dumps(doc, ensure_ascii=False)


def select_style_exemplars(report_type: str, bucket_id: str, docs: List[Dict[str, Any]], max_snips: int = 2, max_chars: int = 850) -> List[str]:
    wanted = (STYLE_SECTION_BY_BUCKET.get(report_type, {}) or {}).get(bucket_id, [])
    if not wanted:
        return []
    targets = [norm(x) for x in wanted]
    snips: List[str] = []
    for doc in docs:
        txt = _flatten_text_from_doc(doc)
        nt = norm(txt)
        if not any(t in nt for t in targets):
            continue
        for t in targets:
            pos = nt.find(t)
            if pos < 0:
                continue
            start = max(0, pos - 380)
            end = min(len(txt), pos + 950)
            chunk = txt[start:end].strip()
            chunk = re.sub(r"\n{3,}", "\n\n", chunk)
            chunk = chunk[:max_chars]
            if chunk and chunk not in snips:
                snips.append(chunk)
            if len(snips) >= max_snips:
                return snips
    return snips


def build_style_block(exemplars: List[str]) -> str:
    if not exemplars:
        return ""
    lines = [
        "## Exemples (anciens rapports) — STYLE UNIQUEMENT",
        "⚠️ Ne copie aucun fait/valeur/matériel depuis ces exemples. Ils ne servent qu’à reproduire le ton et la structure.",
        "",
    ]
    for i, ex in enumerate(exemplars, start=1):
        lines.append(f"### Exemple {i}")
        lines.append(ex.strip())
        lines.append("")
    return "\n".join(lines).strip()


# ---------------- OneNote synthesis ----------------

def build_section_context_block(agg: Dict[str, Any], keywords: List[str], max_titles: int = 12) -> str:
    if not agg:
        return ""
    sec_name = agg.get("onenote_section") or ""
    page_count = agg.get("page_count") or 0
    inv = agg.get("inventory") or []
    kw_norm = [norm(k) for k in (keywords or []) if k]

    lines: List[str] = []
    lines.append(f"## Synthèse OneNote (section: {sec_name}, pages: {page_count})")
    lines.append("")
    lines.append("### Inventaire (par familles)")
    for it in inv[:12]:
        fam = it.get("family")
        cnt = it.get("count")
        lines.append(f"- {fam}: {cnt} élément(s)")
    lines.append("")
    lines.append("### Pages probablement pertinentes pour ce bucket (titres)")
    picked: List[str] = []
    for it in inv:
        for t in (it.get("titles") or []):
            nt = norm(t)
            if (not kw_norm) or any(k in nt for k in kw_norm):
                picked.append(t)
            if len(picked) >= max_titles:
                break
        if len(picked) >= max_titles:
            break
    if not picked:
        lines.append("(Aucun titre détecté via mots-clés; se baser sur les preuves ci-dessous.)")
    else:
        for t in picked:
            lines.append(f"- {t}")
    lines.append("")
    lines.append("### Consigne de rédaction")
    lines.append("- Regrouper les observations répétées; éviter une narration page-par-page.")
    lines.append("- Utiliser le plan fourni (####) et écrire des puces 'Constat:' + 'Preuve:' (page_id + extrait).")
    lines.append("")
    return "\n".join(lines).strip()


# ---------------- Evidence extraction ----------------

def is_question_or_request(text: str) -> bool:
    """Detect questions/requests. (Keep broad detection; this is a formatting patch only.)"""
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


def _clean_block_prefix(snippet: str) -> str:
    """Remove internal block type prefixes like '[paragraph]' if present."""
    return re.sub(r"^\[[^\]]+\]\s*", "", (snippet or "").strip())


def extract_snippets(page: Dict[str, Any], keywords: List[str], max_snips: int = 14) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Return fact snippets and request snippets.

    Each snippet is a dict with:
      - block_type
      - text (clean)
    """
    kws = [norm(k) for k in keywords if k]
    fact_snips: List[Dict[str, str]] = []
    req_snips: List[Dict[str, str]] = []

    for t, txt in page_text_blocks(page):
        nt = norm(txt)
        hit = sum(1 for k in kws if k and k in nt)
        if hit <= 0:
            continue

        clean_txt = txt.strip()

        if is_question_or_request(clean_txt):
            req_snips.append({"block_type": t, "text": clean_txt[:900]})
        else:
            fact_snips.append({"block_type": t, "text": clean_txt[:900]})

        if len(fact_snips) + len(req_snips) >= max_snips:
            break

    return fact_snips, req_snips


def build_evidence_block(bucket_id: str, keywords: List[str], routed_pages: List[Dict[str, Any]], pages_by_id: Dict[str, Dict[str, Any]]) -> str:
    """Build evidence block without leaking raw internal prefixes like 'page_id=... [paragraph]' in the question list."""
    lines: List[str] = []
    requests: List[Tuple[str, str]] = []  # (page_id, question_text)

    lines.append(f"- Bucket: {bucket_id}")
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
            # FACTS: keep exhaustive text, just remove internal [type] wrappers
            for s in fact_snips:
                # Keep block type info but do not use bracketed prefix
                bt = s.get('block_type')
                txt = s.get('text')
                if bt and bt != 'paragraph':
                    lines.append(f"- ({bt}) {txt}")
                else:
                    lines.append(f"- {txt}")

            # REQUESTS: store clean question text, to be rendered in dedicated block
            for s in req_snips:
                qtxt = s.get('text')
                if qtxt:
                    requests.append((pid, qtxt))

        lines.append("")

    if requests:
        lines.append("## Questions / demandes présentes dans les notes (non factuelles)")
        lines.append("⚠️ À traiter comme 'Points à confirmer' (ne pas les transformer en faits).")
        lines.append("")
        for pid, qtxt in requests[:30]:
            # Important: do NOT prefix with page_id; keep it as a separate 'Preuve' line
            lines.append(f"- Question: {qtxt}")
            lines.append(f"  Preuve: page_id={pid}")
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
            lines.append(f" - {f}")
    lines.append("- Les constats doivent être ancrés : pour chaque 'Constat', ajouter une ligne 'Preuve: page_id=... + extrait'.")
    lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_scaffold_block(report_type: str, macro_part_num: int) -> str:
    plan = (SCAFFOLD.get(report_type) or {}).get(macro_part_num) or []
    if not plan:
        return ""
    lines = ["### Plan attendu (structure)"]
    lines += plan
    lines.append("")
    lines.append("Consigne: respecter ce plan. Si aucune preuve pour une rubrique, écrire 'À confirmer sur site / données manquantes'.")
    return "\n".join(lines).strip() + "\n\n"


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
    ap.add_argument("--style-corpus", default="process/learning/processed_reports/Rapports_d_audit/docs", help="Optional folder of historical docs JSON")

    args = ap.parse_args()

    plan = load_json(Path(args.plan))
    agg = load_section_aggregate(Path(args.section_aggregate)) if args.section_aggregate else {}
    tpl = load_json(Path(args.templates))

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

    style_docs = load_style_corpus(Path(args.style_corpus))

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

    routing = plan.get("routing") or {}
    selected_by_part = plan.get("selected_buckets_by_macro_part") or {}

    # Evidence reuse guard
    used_page_ids_by_mp: Dict[int, set] = {}

    rt_templates = tpl.get("templates", {}).get(report_type, {})

    for mp_str, mp_cfg in (macro_parts or {}).items():
        mp = int(mp_str)
        if mp not in gen_parts:
            continue

        mp_name = mp_cfg.get("name") if isinstance(mp_cfg, dict) else str(mp_cfg)
        mp_template = rt_templates.get(f"macro_part_{mp}", {})
        mp_instructions = mp_template.get("instructions", [])

        bucket_items = selected_by_part.get(str(mp)) or selected_by_part.get(mp) or []

        for bi in bucket_items:
            bucket_id = bi.get("bucket_id")
            if not bucket_id:
                continue

            r = routing.get(bucket_id) or {}
            keywords = r.get("keywords") or []
            top_pages = r.get("top_pages") or []

            # Reduce overlap for mp3 when possible
            if mp == 3 and used_page_ids_by_mp.get(1) and len(top_pages) > 2:
                filt = [p for p in top_pages if p.get("page_id") not in used_page_ids_by_mp[1]]
                if len(filt) >= 2:
                    top_pages = filt

            used_page_ids_by_mp.setdefault(mp, set())
            for p in top_pages:
                if p.get("page_id"):
                    used_page_ids_by_mp[mp].add(p["page_id"])

            section_ctx_block = build_section_context_block(agg, keywords)
            exemplars = select_style_exemplars(report_type, bucket_id, style_docs, max_snips=2)
            style_block = build_style_block(exemplars)

            evidence = build_evidence_block(bucket_id, keywords, top_pages, pages_by_id)
            evidence_blocks = [b for b in [section_ctx_block, style_block, evidence] if b]
            evidence_full = "\n\n".join(evidence_blocks).strip()

            intent = BACX_INTENT_BY_MACRO.get(mp_name) or ("PROJECT_FUTURE" if mp == 3 else "DESCRIBE_EXISTING")
            intent_block = build_intent_block(intent)
            scaffold_block = build_scaffold_block(report_type, mp)

            prompt = render_prompt(
                tpl,
                case_id=case_id,
                report_type=report_type,
                macro_part_num=mp,
                macro_part_name=mp_name,
                bucket_id=bucket_id,
                evidence_block=evidence_full,
            )

            # Prepend macro-part instructions
            if mp_instructions:
                header = "### Instructions macro-partie\n- " + "\n- ".join(mp_instructions) + "\n\n"
                prompt = header + prompt

            # Add scaffold + intent
            prompt = scaffold_block + intent_block + prompt

            # OneNote identity header
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
