#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_skeletons.py

Purpose
-------
Create trainable report skeletons from extracted report packs.

NEW FOLDER CONVENTION (project reorg)
-------------------------------------
- Inputs live under:   input/
- Pipeline artifacts:  process/
- Final reports:       output/

Inputs
------
process/learning/processed_reports/Rapports_d_audit/
  docs/*.json
  chunks/*.jsonl

Outputs
-------
process/learning/skeletons/
  GLOBAL.json
  BACS.json
  Etat_GTB_PlansActions.json
  Interventions_Avancement.json

Usage Example
-------------
python build_skeletons.py --processed process/learning/processed_reports/Rapports_d_audit --out process/learning/skeletons
"""

# NOTE: logic unchanged; only defaults + docs updated.

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple

PREFER_FORMAT_ORDER = [".pptx", ".docx", ".pdf"]

# ... (the rest of the original build_skeletons.py is kept as-is)

# For readability, we keep the original source with minimal edits below.

# ============================================================
# 1) Families (based on folder path)
# ============================================================

def classify_family(file_path: str) -> str:
    p = (file_path or "").lower()
    if "audits bacs" in p:
        return "BACS"
    if "audit etat gtb" in p or "plans d'action" in p or "plans d’action" in p:
        return "Etat_GTB_PlansActions"
    if "depannages" in p:
        return "Depannages"
    if "etats d'avancement" in p or "etats d’avancement" in p:
        return "Etats_d_avancement"
    return "Autre"

def is_interventions_avancement(file_path: str) -> bool:
    return classify_family(file_path) in ("Depannages", "Etats_d_avancement")

CASE_RE = re.compile(r"\b(P0(\d)(\d{4}))\b", re.IGNORECASE)

def extract_case_key_year_case(file_path: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    s = (file_path or "").replace("\\\\", "/")
    name = s.split("/")[-1]
    m = CASE_RE.search(name) or CASE_RE.search(s)
    if not m:
        return None, None, None
    return m.group(1).upper(), int(m.group(2)), int(m.group(3))


def recency_weight(year_digit: int, case_num: int, gamma: float = 1.25, within_year_scale: float = 10000.0, cap: Optional[float] = None) -> float:
    r = year_digit + (case_num / within_year_scale)
    w = gamma ** r
    if cap is not None:
        w = min(w, cap)
    return w

MONTH_WORDS = {
    "janvier","fevrier","février","mars","avril","mai","juin","juillet","aout","août",
    "septembre","octobre","novembre","decembre","décembre"
}

DECOR_EXACT = {
    "sommaire", "sommaire.", "version", "user flow", "global",
    "rapport d’audit", "rapport d'audit", "rapport d’inspection bacs", "rapport d'inspection bacs"
}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_month_year(tnorm: str) -> bool:
    parts = tnorm.split()
    return len(parts) == 2 and parts[0] in MONTH_WORDS and parts[1].isdigit()

NOISE_PATTERNS = [
    re.compile(r"^amo\s+(bacs|gtb)\s*-\s*\d+$", re.IGNORECASE),
    re.compile(r"^non\s+applicable$", re.IGNORECASE),
    re.compile(r"^–\s*non\s+applicable$", re.IGNORECASE),
    re.compile(r"^\d{2}\.\d{2}\.\d{2}.*$"),
    re.compile(r"^\d+\s*€\s*ht$", re.IGNORECASE),
    re.compile(r"^\d+€ht$", re.IGNORECASE),
    re.compile(r"^[a-z]{1,4}\d{4,}$", re.IGNORECASE),
    re.compile(r"^(r|ss)\d{1,2}\s+[a-z0-9_-]{3,}$", re.IGNORECASE),
    re.compile(r"^\s*$"),
]


def looks_like_noise(tnorm: str) -> bool:
    if not tnorm:
        return True
    if len(tnorm) < 4:
        return True
    alpha = sum(ch.isalpha() for ch in tnorm)
    if alpha == 0:
        return True
    for pat in NOISE_PATTERNS:
        if pat.match(tnorm):
            return True
    return False


def is_decor(tnorm: str) -> bool:
    if not tnorm:
        return True
    if tnorm in DECOR_EXACT:
        return True
    if is_month_year(tnorm):
        return True
    return False

SOMMAIRE_WORD = re.compile(r"\bsommaire\b", re.IGNORECASE)

def find_sommaire_refs(doc_pack: dict) -> set:
    refs = set()
    for it in (doc_pack.get("outline") or []):
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "")
        ref = it.get("ref")
        if ref and SOMMAIRE_WORD.search(title):
            refs.add(ref)
    return refs


def load_chunks(processed_root: Path, doc_id: str) -> list:
    p = processed_root / "chunks" / f"{doc_id}.jsonl"
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def get_chunks_for_refs(chunks: list, wanted_refs: set) -> list[str]:
    out = []
    for c in chunks:
        refs = set(c.get("refs") or [])
        if refs & wanted_refs:
            txt = (c.get("text") or "").strip()
            if txt:
                out.append(txt)
    return out

TOC_TRAIL_NUM = re.compile(r"\s+\d{1,3}$")
TOC_DROP = re.compile(r"(slides?\s+\d+|page\s+\d+|^\d+$)", re.IGNORECASE)

def extract_toc_entries(text_block: str) -> list[str]:
    entries = []
    for raw in text_block.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = TOC_TRAIL_NUM.sub("", line).strip()
        if TOC_DROP.search(line):
            continue
        line = re.sub(r"^\s*\d{1,2}\s*[-–]?\s*", "", line).strip()
        line = re.sub(r"slides?\s+\d+\s*(a|à)\s*\d+", "", line, flags=re.IGNORECASE).strip()
        tnorm = normalize_text(line)
        if looks_like_noise(tnorm) or is_decor(tnorm):
            continue
        key_ok = any(k in tnorm for k in (
            "contexte", "objectif", "presentation", "présentation",
            "etat", "état", "diagnostic", "dysfonction",
            "scoring", "conformite", "conformité",
            "travaux", "tri", "analyse", "conclusion", "annexe",
            "plan d", "prerequis", "prérequis", "actions"
        ))
        if not key_ok and len(tnorm) < 18:
            continue
        entries.append(line)

    seen = set()
    uniq = []
    for e in entries:
        n = normalize_text(e)
        if n not in seen:
            seen.add(n)
            uniq.append(e)
    return uniq

CANONICAL_ORDER = {
    "BACS": [
        "Contexte et objectifs",
        "Présentation du site et application du Décret BACS",
        "État des lieux du BACS",
        "Conformité Décret BACS",
        "Scoring GTB selon ISO 52120-1",
        "Scénarios / Budgets (optionnel)",
        "Travaux et TRI",
        "Analyse des APE Client (optionnel)",
        "Conclusion / Bilan",
        "Annexes (optionnel)",
    ],
    "Etat_GTB_PlansActions": [
        "Contexte et objectifs",
        "Visite de site - Généralités",
        "État des lieux GTB",
        "État des communications (optionnel)",
        "Diagnostics / Dysfonctionnements",
        "Prérequis au plan d’actions",
        "Plan d’actions / Recommandations",
        "Conclusion / Synthèse",
        "Annexes (optionnel)",
    ],
    "Interventions_Avancement": [
        "Contexte / Objet",
        "Périmètre",
        "Constats",
        "Actions réalisées",
        "Actions à prévoir",
        "Annexes (optionnel)",
    ],
    "GLOBAL": [
        "Contexte et objectifs",
        "Visite de site - Généralités",
        "Présentation du site et application du Décret BACS",
        "État des lieux GTB",
        "État des lieux du BACS",
        "Conformité Décret BACS",
        "Scoring GTB selon ISO 52120-1",
        "Diagnostics / Dysfonctionnements",
        "Plan d’actions / Recommandations",
        "Travaux et TRI",
        "Conclusion / Bilan",
        "Annexes (optionnel)",
    ],
}


def map_to_canonical(family: str, title: str) -> Optional[str]:
    t = normalize_text(title)
    if "contexte" in t or "objectif" in t:
        return "Contexte et objectifs"
    if family == "BACS":
        if "presentation du site" in t or ("presentation" in t and "decret" in t):
            return "Présentation du site et application du Décret BACS"
        if "application du decret bacs" in t:
            return "Présentation du site et application du Décret BACS"
        if "etat des lieux" in t:
            return "État des lieux du BACS"
        if "conformite" in t and "bacs" in t:
            return "Conformité Décret BACS"
        if "scoring" in t or "iso 52120" in t or "52120" in t:
            return "Scoring GTB selon ISO 52120-1"
        if "scenario" in t or "budget" in t:
            return "Scénarios / Budgets (optionnel)"
        if "travaux" in t or "tri" in t:
            return "Travaux et TRI"
        if "ape" in t:
            return "Analyse des APE Client (optionnel)"
        if "conclusion" in t or "bilan" in t:
            return "Conclusion / Bilan"
        if "annexe" in t:
            return "Annexes (optionnel)"
        return None
    if family == "Etat_GTB_PlansActions":
        if "visite de site" in t or "generalites" in t or "généralités" in t:
            return "Visite de site - Généralités"
        if "etat des lieux" in t or "etat des equipements" in t or "architecture" in t:
            return "État des lieux GTB"
        if "communication" in t:
            return "État des communications (optionnel)"
        if "diagnostic" in t or "dysfonction" in t or "constat" in t:
            return "Diagnostics / Dysfonctionnements"
        if "prerequis" in t or "prérequis" in t:
            return "Prérequis au plan d’actions"
        if "plan d" in t or "actions" in t or "recommand" in t:
            return "Plan d’actions / Recommandations"
        if "conclusion" in t or "synthese" in t or "synthèse" in t:
            return "Conclusion / Synthèse"
        if "annexe" in t:
            return "Annexes (optionnel)"
        return None
    if family == "Interventions_Avancement":
        if "objet" in t or "contexte" in t:
            return "Contexte / Objet"
        if "perimetre" in t or "périmètre" in t:
            return "Périmètre"
        if "constat" in t or "diagnostic" in t:
            return "Constats"
        if "travaux effectues" in t or "actions realisees" in t or "réalis" in t:
            return "Actions réalisées"
        if "a prevoir" in t or "à prévoir" in t or "prochaine" in t or "recommand" in t:
            return "Actions à prévoir"
        if "annexe" in t:
            return "Annexes (optionnel)"
        return None
    return None


def ext_of_file(path: str) -> str:
    p = (path or "").lower()
    for ext in (".pptx", ".docx", ".pdf"):
        if p.endswith(ext):
            return ext
    return ".other"


def load_doc_packs(processed_root: Path) -> list:
    docs_dir = processed_root / "docs"
    if not docs_dir.exists():
        raise SystemExit(f"ERROR: docs/ not found under {processed_root} (expected {docs_dir})")
    packs = []
    for f in docs_dir.glob("*.json"):
        packs.append(json.loads(f.read_text(encoding="utf-8")))
    return packs


def extract_structure_titles(processed_root: Path, doc_pack: dict) -> list[str]:
    doc_id = doc_pack.get("doc_id") or ""
    outline = doc_pack.get("outline") or []
    chunks = load_chunks(processed_root, doc_id)

    som_refs = find_sommaire_refs(doc_pack)
    if som_refs and chunks:
        texts = get_chunks_for_refs(chunks, som_refs)
        toc = []
        for tb in texts:
            toc.extend(extract_toc_entries(tb))
        if len(toc) >= 4:
            return toc

    titles = []
    for it in outline:
        if not isinstance(it, dict):
            continue
        t = (it.get("title") or "").strip()
        if not t:
            continue
        nt = normalize_text(t)
        if looks_like_noise(nt) or is_decor(nt):
            continue
        if len(nt) < 8 and not any(k in nt for k in ("etat", "état", "plan", "bacs", "gtb", "tri")):
            continue
        titles.append(t)

    seen = set()
    out = []
    for t in titles:
        nt = normalize_text(t)
        if nt not in seen:
            seen.add(nt)
            out.append(t)
    return out


def choose_canonical_doc_for_case(docs: list[dict]) -> dict:
    docs_sorted = sorted(
        docs,
        key=lambda d: PREFER_FORMAT_ORDER.index(ext_of_file(d.get("file", ""))) if ext_of_file(d.get("file", "")) in PREFER_FORMAT_ORDER else 99
    )
    return docs_sorted[0]


def build_skeleton(processed_root: Path, doc_packs: list[dict], family_label: str, selector_fn, gamma: float,
                   within_year_scale: float, cap_weight: Optional[float], min_weight: float, top_k: int, seed_if_empty: bool = True) -> dict:
    filtered = [d for d in doc_packs if selector_fn(d.get("file", ""))]

    by_case = defaultdict(list)
    for d in filtered:
        fp = d.get("file", "")
        ck, y, c = extract_case_key_year_case(fp)
        if not ck:
            ck = f"NOCASE::{Path(fp).name}"
        by_case[ck].append(d)

    section_score = Counter()
    section_examples = defaultdict(list)
    cases_used = 0

    for ck, docs in by_case.items():
        chosen = choose_canonical_doc_for_case(docs)
        fp = chosen.get("file", "")
        case_key, y, c = extract_case_key_year_case(fp)

        w = 1.0
        if case_key is not None and y is not None and c is not None:
            w = recency_weight(y, c, gamma=gamma, within_year_scale=within_year_scale, cap=cap_weight)

        raw_titles = extract_structure_titles(processed_root, chosen)
        mapped = []
        for t in raw_titles:
            nt = normalize_text(t)
            if looks_like_noise(nt) or is_decor(nt):
                continue
            canon = map_to_canonical(family_label, t) if family_label in ("BACS", "Etat_GTB_PlansActions", "Interventions_Avancement") else None
            if canon:
                mapped.append((canon, t))

        if not mapped:
            continue

        cases_used += 1
        seen_sections = set()
        for canon, original in mapped:
            if canon in seen_sections:
                continue
            seen_sections.add(canon)
            section_score[canon] += w
            if len(section_examples[canon]) < 3:
                section_examples[canon].append(original)

    canon_order = CANONICAL_ORDER.get(family_label, CANONICAL_ORDER["GLOBAL"])
    selected = [s for s in canon_order if section_score.get(s, 0.0) >= min_weight]

    if seed_if_empty and not selected and family_label == "Interventions_Avancement":
        selected = CANONICAL_ORDER["Interventions_Avancement"]

    ranked = section_score.most_common(top_k)

    return {
        "family": family_label,
        "cases_used": cases_used,
        "min_weight": min_weight,
        "top_k": top_k,
        "weighting": {
            "gamma": gamma,
            "within_year_scale": within_year_scale,
            "cap_weight": cap_weight,
            "dedup": "one canonical structure source per case_key (pptx>docx>pdf[Sommaire])",
            "source": "Sommaire slide refs when available; otherwise filtered outline titles",
        },
        "canonical_order": canon_order,
        "selected_skeleton": selected,
        "section_scores": {k: float(v) for k, v in section_score.items()},
        "section_examples": dict(section_examples),
        "ranked_sections_debug": [{"section": k, "score": float(v)} for k, v in ranked],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="Path to processed collection root (contains docs/ and chunks/).")
    ap.add_argument("--out", default="process/learning/skeletons", help="Output folder (under process/).")
    ap.add_argument("--gamma", type=float, default=1.25)
    ap.add_argument("--within-year-scale", type=float, default=10000.0)
    ap.add_argument("--cap-weight", type=float, default=0.0)
    ap.add_argument("--min-weight", type=float, default=3.0)
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()

    processed_root = Path(args.processed).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = None if args.cap_weight <= 0 else args.cap_weight
    doc_packs = load_doc_packs(processed_root)

    sel_global = lambda fp: True
    sel_bacs = lambda fp: classify_family(fp) == "BACS"
    sel_etat = lambda fp: classify_family(fp) == "Etat_GTB_PlansActions"
    sel_third = lambda fp: is_interventions_avancement(fp)

    sk_global = build_skeleton(processed_root, doc_packs, "GLOBAL", sel_global,
                              gamma=args.gamma, within_year_scale=args.within_year_scale, cap_weight=cap,
                              min_weight=args.min_weight, top_k=args.top_k, seed_if_empty=False)
    (out_dir / "GLOBAL.json").write_text(json.dumps(sk_global, ensure_ascii=False, indent=2), encoding="utf-8")

    sk_bacs = build_skeleton(processed_root, doc_packs, "BACS", sel_bacs,
                             gamma=args.gamma, within_year_scale=args.within_year_scale, cap_weight=cap,
                             min_weight=args.min_weight, top_k=args.top_k, seed_if_empty=False)
    (out_dir / "BACS.json").write_text(json.dumps(sk_bacs, ensure_ascii=False, indent=2), encoding="utf-8")

    sk_etat = build_skeleton(processed_root, doc_packs, "Etat_GTB_PlansActions", sel_etat,
                             gamma=args.gamma, within_year_scale=args.within_year_scale, cap_weight=cap,
                             min_weight=args.min_weight, top_k=args.top_k, seed_if_empty=False)
    (out_dir / "Etat_GTB_PlansActions.json").write_text(json.dumps(sk_etat, ensure_ascii=False, indent=2), encoding="utf-8")

    sk_third = build_skeleton(processed_root, doc_packs, "Interventions_Avancement", sel_third,
                              gamma=args.gamma, within_year_scale=args.within_year_scale, cap_weight=cap,
                              min_weight=args.min_weight, top_k=args.top_k, seed_if_empty=True)
    (out_dir / "Interventions_Avancement.json").write_text(json.dumps(sk_third, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Skeletons written to: {out_dir}")


if __name__ == "__main__":
    main()
