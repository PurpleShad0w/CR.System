#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_score.py

Deterministic quality scoring for Build4Use GTB/BACS reports.

Designed to be run immediately after run_llm_jobs.py, using:
- generated_bundle.json (has evidence routing from generate_draft.py)
- assembled_report.json (final text for PPTX rendering)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict


# ---- Intent rules (mirrors generate_draft.py) -----------------------------
# We keep these lists local so quality_score.py is drop-in and stable.
# Source of truth lives in generate_draft.py (forbidden terms per intent).
INTENT_FORBIDDEN = {
    1: [  # DESCRIBE_EXISTING
        "classe", "conforme", "non conforme",
        "iso", "52120",
        "objectif", "à atteindre",
        "projeté", "futur", "travaux",
    ],
    2: [  # SCORE_EXISTING
        "travaux", "mise en œuvre",
        "objectif", "futur", "projeté",
    ],
    3: [  # PROJECT_FUTURE
        # allowed to be prescriptive/projection; keep light checks only
    ],
}

PLACEHOLDERS = [
    "à confirmer", "a confirmer", "à calculer", "a calculer",
    "selon les preuves", "dans les preuves", "voir preuves",
    "à compléter", "a completer",
]

PRESCRIPTIVE_MP1 = [
    "il est nécessaire", "il faut", "doit", "devra",
    "mettre en place", "installer", "ajouter", "prévoir", "prevoir",
]


# ---- Utils ---------------------------------------------------------------
def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = norm(s)
    return re.findall(r"[a-z0-9àâçéèêëîïôûùüÿñæœ\-]+", s, flags=re.IGNORECASE)

def count_hits(text: str, patterns: List[str]) -> int:
    t = norm(text)
    return sum(t.count(norm(p)) for p in patterns if p)

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# ---- Core scoring --------------------------------------------------------
def evaluate_quality(
    generated_bundle: Dict[str, Any],
    assembled_report: Dict[str, Any],
    *,
    weights: Dict[str, int] = None,
) -> Dict[str, Any]:
    """
    Returns a dict:
    {
      "total": 0..100,
      "components": {...},
      "per_section": [...],
      "flags": [...],
      "gate": {"min_total": x, "pass": bool}
    }
    """
    # Default weights sum to 100
    if weights is None:
        weights = {
            "structure": 25,
            "evidence_coverage": 25,
            "evidence_alignment": 20,
            "semantic_compliance": 20,
            "clarity": 10,
        }

    # Index assembled text by (macro_part, bucket_id)
    assembled_text = {}
    macro_parts = assembled_report.get("macro_parts", []) or []
    for mp in macro_parts:
        mp_num = mp.get("macro_part")
        for sec in (mp.get("sections") or []):
            key = (mp_num, sec.get("bucket_id"))
            assembled_text[key] = (sec.get("text") or "").strip()

    # Sections live in generated_bundle (draft_bundle + LLM outputs)
    sections = generated_bundle.get("sections", []) or []

    # ---- A) Structure (0..weights["structure"]) --------------------------
    structure_max = weights["structure"]
    structure_score = 0.0
    flags = []

    mp_ids = sorted({m.get("macro_part") for m in macro_parts if m.get("macro_part") is not None})
    if set(mp_ids) == {1, 2, 3} and len(mp_ids) == 3:
        structure_score += structure_max * 0.45
    else:
        # partial credit
        structure_score += structure_max * 0.45 * (len(set(mp_ids).intersection({1,2,3})) / 3.0)
        flags.append(f"structure:unexpected_macro_parts:{mp_ids}")

    if 4 not in mp_ids:
        structure_score += structure_max * 0.15
    else:
        flags.append("structure:macro_part_4_present")

    # completeness: each generated section must map to assembled text
    mapped = 0
    for s in sections:
        key = (s.get("macro_part"), s.get("bucket_id"))
        if (assembled_text.get(key) or "").strip():
            mapped += 1
    structure_score += structure_max * 0.40 * safe_div(mapped, max(1, len(sections)))
    if mapped < len(sections):
        flags.append(f"structure:missing_assembled_text:{len(sections)-mapped}")

    # ---- B) Evidence coverage (0..weights["evidence_coverage"]) ----------
    cov_max = weights["evidence_coverage"]
    cov_score = 0.0

    # Per-section evidence presence & density from top_pages
    present_points = 0.0
    density_points = 0.0

    # For reuse discipline across macro parts
    pages_by_mp = defaultdict(set)

    per_section = []
    for s in sections:
        mp = s.get("macro_part")
        bucket = s.get("bucket_id")
        tp = s.get("top_pages") or []
        kws = s.get("keywords") or []
        text = assembled_text.get((mp, bucket), "")

        page_ids = [p.get("page_id") for p in tp if p.get("page_id")]
        for pid in page_ids:
            pages_by_mp[mp].add(pid)

        # Presence: at least 1 page_id
        has_ev = 1 if page_ids else 0
        present_points += has_ev

        # Density: sum of top_page scores capped
        score_sum = 0.0
        for p in tp:
            try:
                score_sum += float(p.get("score") or 0.0)
            except Exception:
                pass
        # normalize: full density if >= 8 total (tunable), cap at 1
        density_points += min(1.0, score_sum / 8.0)

        per_section.append({
            "macro_part": mp,
            "bucket_id": bucket,
            "has_evidence": bool(page_ids),
            "evidence_page_count": len(page_ids),
            "evidence_score_sum": round(score_sum, 2),
            "keywords_count": len(kws),
        })

    n = max(1, len(sections))
    # presence contributes 40% of coverage
    cov_score += cov_max * 0.40 * (present_points / n)
    # density contributes 35% of coverage
    cov_score += cov_max * 0.35 * (density_points / n)

    # reuse discipline contributes 25% of coverage: penalize heavy overlap mp1 vs mp3
    mp1 = pages_by_mp.get(1, set())
    mp3 = pages_by_mp.get(3, set())
    overlap = len(mp1.intersection(mp3))
    union = len(mp1.union(mp3))
    overlap_ratio = safe_div(overlap, union)
    reuse_component = max(0.0, 1.0 - overlap_ratio)  # higher is better (less overlap)
    cov_score += cov_max * 0.25 * reuse_component
    if overlap_ratio > 0.35:
        flags.append(f"evidence:high_reuse_mp1_mp3:{overlap_ratio:.2f}")

    # ---- C) Evidence-text alignment (0..weights["evidence_alignment"]) ---
    align_max = weights["evidence_alignment"]
    align_score = 0.0

    # Keyword hit ratio per section: fraction of keywords appearing in final text
    # (Deterministic and cheap, and keywords come from routing.)
    align_acc = 0.0
    for i, s in enumerate(sections):
        mp = s.get("macro_part")
        bucket = s.get("bucket_id")
        kws = [norm(k) for k in (s.get("keywords") or []) if k]
        text = norm(assembled_text.get((mp, bucket), ""))

        if not kws:
            hit_ratio = 0.0
        else:
            hits = sum(1 for k in kws if k and k in text)
            hit_ratio = hits / len(kws)

        per_section[i]["keyword_hit_ratio"] = round(hit_ratio, 3)
        align_acc += min(1.0, hit_ratio / 0.6)  # full credit if >=60% keywords hit
        if hit_ratio < 0.25 and kws:
            flags.append(f"alignment:low_keyword_hit:mp{mp}:{bucket}:{hit_ratio:.2f}")

    align_score = align_max * (align_acc / n)

    # ---- D) Semantic compliance (0..weights["semantic_compliance"]) ------
    sem_max = weights["semantic_compliance"]
    sem_score = sem_max  # start full and subtract penalties

    # Penalize forbidden terms per macro-part intent (mirrors generate_draft intent rules)
    forb_hits_total = 0
    ph_hits_total = 0
    mp1_presc_hits = 0

    for i, s in enumerate(sections):
        mp = s.get("macro_part")
        bucket = s.get("bucket_id")
        text = assembled_text.get((mp, bucket), "")

        forb = INTENT_FORBIDDEN.get(mp, [])
        forb_hits = count_hits(text, forb)
        ph_hits = count_hits(text, PLACEHOLDERS)

        forb_hits_total += forb_hits
        ph_hits_total += ph_hits

        if mp == 1:
            mp1_presc_hits += count_hits(text, PRESCRIPTIVE_MP1)

        per_section[i]["forbidden_hits"] = forb_hits
        per_section[i]["placeholder_hits"] = ph_hits

    # Apply penalties (tunable)
    sem_score -= min(0.55 * sem_max, forb_hits_total * 0.8)
    sem_score -= min(0.40 * sem_max, ph_hits_total * 1.2)
    sem_score -= min(0.35 * sem_max, mp1_presc_hits * 0.9)

    sem_score = max(0.0, sem_score)

    if forb_hits_total:
        flags.append(f"semantics:forbidden_hits:{forb_hits_total}")
    if ph_hits_total:
        flags.append(f"semantics:placeholders:{ph_hits_total}")
    if mp1_presc_hits:
        flags.append(f"semantics:mp1_prescriptive:{mp1_presc_hits}")

    # ---- E) Clarity (0..weights["clarity"]) ------------------------------
    clar_max = weights["clarity"]
    all_text = "\n".join(assembled_text.values())
    toks = [t for t in tokenize(all_text) if len(t) > 2]

    if not toks:
        clarity_score = 0.0
        flags.append("clarity:no_text")
    else:
        c = Counter(toks)
        top10 = sum(cnt for _, cnt in c.most_common(10))
        top_frac = top10 / len(toks)

        # avg sentence length
        sents = [s.strip() for s in re.split(r"[.!?]+", all_text) if s.strip()]
        avg_len = safe_div(sum(len(tokenize(s)) for s in sents), len(sents))

        clarity = 1.0
        if top_frac > 0.32:
            clarity -= min(0.6, (top_frac - 0.32) * 2.0)
        if avg_len > 28:
            clarity -= min(0.4, (avg_len - 28) * 0.02)

        clarity_score = clar_max * max(0.0, min(1.0, clarity))

    # ---- Total -----------------------------------------------------------
    total = structure_score + cov_score + align_score + sem_score + clarity_score
    total = max(0.0, min(100.0, total))

    components = {
        "structure": round(structure_score, 2),
        "evidence_coverage": round(cov_score, 2),
        "evidence_alignment": round(align_score, 2),
        "semantic_compliance": round(sem_score, 2),
        "clarity": round(clarity_score, 2),
    }

    return {
        "case_id": assembled_report.get("case_id") or generated_bundle.get("case_id"),
        "report_type": assembled_report.get("report_type") or generated_bundle.get("report_type"),
        "total": round(total, 2),
        "components": components,
        "per_section": per_section,
        "flags": flags,
        "weights": weights,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="Path to generated_bundle.json (output of run_llm_jobs.py)")
    ap.add_argument("--assembled", required=True, help="Path to assembled_report.json (output of run_llm_jobs.py)")
    ap.add_argument("--out", default="", help="Path to write quality_report.json (default: alongside assembled)")
    ap.add_argument("--min_total", type=float, default=0.0, help="If >0, exit with code 2 when total < min_total")
    args = ap.parse_args()

    gen_p = Path(args.generated)
    asm_p = Path(args.assembled)
    out_p = Path(args.out) if args.out else asm_p.parent / "quality_report.json"

    generated = load_json(gen_p)
    assembled = load_json(asm_p)

    rep = evaluate_quality(generated, assembled)
    rep["gate"] = {"min_total": args.min_total, "pass": (rep["total"] >= args.min_total if args.min_total else True)}

    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_p}")
    print(f"Quality total: {rep['total']} / 100")

    if args.min_total and rep["total"] < args.min_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()