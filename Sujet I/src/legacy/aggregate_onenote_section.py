#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_onenote_section.py

Purpose
-------
Aggregate OneNote pages at the *OneNote section* level (e.g., "Oseraie - OSNY"),
to produce a structured synthesis artifact used for prompting.

This addresses: one OneNote page != one slide, and report text must be synthesized
from the whole section.

Inputs
------
process/onenote/<notebook>/pages/*.json (output of process_onenote.py)

Outputs
-------
process/onenote_aggregates/<notebook>/<section_slug>.json

HARDENING (small addition)
--------------------------
Adds a canonical `section_context` object at the top-level, while keeping legacy keys:
- notebook
- onenote_section
- section_slug

This makes downstream artifacts self-describing and prevents ambiguity.

Usage
-----
python aggregate_onenote_section.py --onenote process/onenote/test --section "Oseraie - OSNY"
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from section_context import build_section_context, slugify


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pages(onenote_root: Path) -> List[Dict[str, Any]]:
    pages_dir = onenote_root / "pages"
    if not pages_dir.exists():
        raise SystemExit(f"pages/ not found under {onenote_root} (expected {pages_dir})")
    pages = []
    for p in sorted(pages_dir.glob("*.json")):
        try:
            pages.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pages


def extract_text_blocks(page: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (type, text) from heading/paragraph/image_ocr/audio blocks."""
    out: List[Tuple[str, str]] = []
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


def equipment_family(title: str) -> str:
    """Very lightweight equipment clustering based on page title."""
    t = norm(title)
    # HVAC / air handling
    if re.search(r"cta", t) or "centrale de traitement d'air" in t or "centrale de traitement d air" in t:
        return "CTA / Traitement d'air"
    if "extracteur" in t or "extraction" in t:
        return "Extracteurs / Ventilation"
    # Cooling / heating plants
    if "groupe froid" in t or "groupefroid" in t:
        return "Production froid"
    if "chaudi" in t:
        return "Production chaud"
    # Electrical
    if "tgbt" in t:
        return "Électrique / TGBT"
    if re.search(r"td", t) or "tableau divisionnaire" in t:
        return "Électrique / TD"
    # Control / GTB
    if "supervision" in t:
        return "Supervision"
    if "automate" in t or "regulateur" in t or "régulateur" in t or "local autocom" in t:
        return "Automates / Régulation"
    if "gtb" in t or "gma" in t:
        return "GTB (général)"
    # Default
    return "Autres équipements"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onenote", required=True, help="Path to process/onenote/<notebook> (contains pages/)")
    ap.add_argument("--section", required=True, help="OneNote section name (e.g., 'Oseraie - OSNY')")
    ap.add_argument("--out", default="process/onenote_aggregates", help="Output root under process/")
    args = ap.parse_args()

    onenote_root = Path(args.onenote).resolve()
    notebook = onenote_root.name
    target_section = args.section.strip()

    pages = load_pages(onenote_root)

    # Filter pages by OneNote section name (page packs contain 'section')
    selected = [p for p in pages if (p.get("section") or p.get("metadata", {}).get("section")) == target_section]
    if not selected:
        # helpful error message with available sections
        secs = sorted({(p.get("section") or p.get("metadata", {}).get("section") or "") for p in pages if (p.get("section") or p.get("metadata", {}).get("section"))})
        raise SystemExit(
            f"No pages found for section='{target_section}'. Available sections: {secs[:30]}{' ...' if len(secs)>30 else ''}"
        )

    # Build aggregation
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    page_index: List[Dict[str, Any]] = []
    for p in selected:
        pid = p.get("page_id") or p.get("metadata", {}).get("page_id") or ""
        title = p.get("title") or p.get("metadata", {}).get("title") or pid
        fam = equipment_family(title)

        # count assets
        imgs = (p.get("assets", {}) or {}).get("images", []) or []
        auds = (p.get("assets", {}) or {}).get("audio", []) or []
        text_blocks = extract_text_blocks(p)

        page_index.append(
            {
                "page_id": pid,
                "title": title,
                "family": fam,
                "num_images": len(imgs),
                "num_audio": len(auds),
                "text_blocks": [{"type": t, "text": txt[:800]} for t, txt in text_blocks],
                "images": imgs[:12],
            }
        )

        clusters[fam].append(
            {
                "page_id": pid,
                "title": title,
                "num_images": len(imgs),
                "notable_text": [txt for _, txt in text_blocks if len(txt) > 3][:20],
            }
        )

    inventory: List[Dict[str, Any]] = []
    for fam, items in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        inventory.append(
            {
                "family": fam,
                "count": len(items),
                "titles": [it["title"] for it in items][:30],
            }
        )

    # HARDENING: canonical section identity
    ctx = build_section_context(notebook, target_section)

    agg = {
        "section_context": ctx,
        "aggregation_version": "1.0",

        # Legacy keys kept for backward compatibility
        "notebook": ctx["onenote_notebook"],
        "onenote_section": ctx["onenote_section_name"],
        "section_slug": ctx["section_slug"],

        "page_count": len(selected),
        "inventory": inventory,
        "pages": page_index,
        "notes": {
            "principle": "OneNote pages are aggregated at section level; pages are not mapped 1:1 to slides.",
        },
    }

    out_root = Path(args.out).resolve() / notebook
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{ctx['section_slug']}.json"
    out_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
