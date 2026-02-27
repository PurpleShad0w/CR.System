#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_regen.py

Purpose
-------
Automatically regenerate non‑BACX‑compliant sections using the LLM.

This is a HARD corrective loop, not advisory.

Usage
-----
python auto_regen.py input.json output.json
"""

import json
import sys
from pathlib import Path

from llm_client import run_llm_completion  # you already have this


MAX_RETRIES = 2


REPAIR_INSTRUCTIONS = {
    "État des lieux GTB": (
        "Réécris cette section en décrivant STRICTEMENT l’existant.\n"
        "Interdictions absolues : ISO, classe, conformité, objectif, futur, travaux.\n"
        "Si une information n’est pas présente dans les preuves, indique-le explicitement."
    ),
    "Scoring GTB actuel": (
        "Réécris cette section pour expliquer UNIQUEMENT le scoring actuel "
        "selon l’ISO 52120‑1.\n"
        "Ne parle pas de travaux ni de projection future."
    ),
    "Scoring projeté": (
        "Réécris cette section en décrivant une cible normative.\n"
        "La cible minimale est la classe B. L’inférence est autorisée."
    ),
}


def regenerate_section(section: dict, reason: str) -> str:
    macro = section["macro_part_name"]
    original = section["text"]

    instruction = REPAIR_INSTRUCTIONS.get(macro, "")
    prompt = (
        "Tu as produit un texte NON conforme au cadre BACX.\n\n"
        f"Raison : {reason}\n\n"
        f"{instruction}\n\n"
        "Texte à corriger :\n"
        "------------------\n"
        f"{original}\n"
    )

    return run_llm_completion(prompt)


def main():
    if len(sys.argv) != 3:
        print("Usage: python auto_regen.py <input.json> <output.json>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    report = json.loads(inp.read_text(encoding="utf-8"))
    sections = report.get("sections", [])

    new_sections = []

    for sec in sections:
        status = sec.get("bacx_status")
        if status != "rejected":
            new_sections.append(sec)
            continue

        reason = sec.get("text", "")
        retries = 0
        fixed = False

        while retries < MAX_RETRIES:
            new_text = regenerate_section(sec, reason)
            sec["text"] = new_text
            sec.pop("bacx_status", None)

            # Re‑validate by re‑running policy
            from enforce_bacx_policy import enforce_section
            checked = enforce_section(dict(sec))

            if checked.get("bacx_status") != "rejected":
                new_sections.append(checked)
                fixed = True
                break

            retries += 1

        if not fixed:
            sec["text"] = (
                "[SECTION INCOMPLÈTE]\n\n"
                "Cette section n’a pas pu être générée de manière conforme "
                "au cadre BACX à partir des informations disponibles."
            )
            sec["bacx_status"] = "failed"
            new_sections.append(sec)

    report["sections"] = new_sections
    report["auto_regen_applied"] = True

    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[AUTO‑REGEN] Done → {out}")


if __name__ == "__main__":
    main()