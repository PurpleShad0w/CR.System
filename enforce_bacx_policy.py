#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enforce_bacx_policy.py

Purpose
-------
Enforce BACX semantic rules on generated report sections.
This is a HARD gate. If a section violates the policy, it is rejected
or rewritten with explicit placeholders.

Inputs
------
assembled_report.json (or equivalent structure with macro_part_name + text)

Outputs
-------
validated_report.json (same structure, policy-compliant)

This script is REQUIRED for BACX correctness.
"""

import json
import sys
from pathlib import Path

# -------------------------------------------------------------------
# BACX POLICY
# -------------------------------------------------------------------

POLICY = {
    "État des lieux GTB": {
        "forbidden": [
            "classe", "iso", "52120",
            "conforme", "non conforme",
            "objectif", "à atteindre",
            "travaux", "mise en œuvre",
            "futur", "projeté",
        ],
        "mode": "reject",
        "message": (
            "Cette section doit décrire strictement l’existant. "
            "Les éléments normatifs, objectifs ou projetés ont été supprimés."
        ),
    },
    "Scoring GTB actuel": {
        "forbidden": [
            "travaux", "mise en œuvre",
            "futur", "projeté",
        ],
        "mode": "allow_with_warning",
        "message": (
            "Cette section explique le scoring actuel uniquement. "
            "Les éléments de projection ont été retirés."
        ),
    },
    "Scoring projeté": {
        "forbidden": [],
        "mode": "allow",
        "message": "",
    },
}


def violates(text: str, forbidden: list[str]) -> list[str]:
    t = (text or "").lower()
    return [w for w in forbidden if w in t]


def enforce_section(section: dict) -> dict:
    name = section.get("macro_part_name")
    text = section.get("text", "")

    rule = POLICY.get(name)
    if not rule:
        return section

    hits = violates(text, rule["forbidden"])
    if not hits:
        return section

    mode = rule["mode"]

    if mode == "reject":
        section["text"] = (
            f"[SECTION BLOQUÉE – NON CONFORME BACX]\n\n"
            f"{rule['message']}\n\n"
            f"(Termes interdits détectés : {', '.join(hits)})"
        )
        section["bacx_status"] = "rejected"
        return section

    if mode == "allow_with_warning":
        section["text"] = (
            f"[AVERTISSEMENT BACX]\n\n"
            f"{rule['message']}\n\n"
            + text
        )
        section["bacx_status"] = "warning"
        return section

    return section


def main():
    if len(sys.argv) != 3:
        print("Usage: python enforce_bacx_policy.py <input.json> <output.json>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    report = json.loads(inp.read_text(encoding="utf-8"))

    sections = report.get("sections") or []
    new_sections = []

    for s in sections:
        new_sections.append(enforce_section(dict(s)))

    report["sections"] = new_sections
    report["bacx_policy_enforced"] = True

    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[BACX] Policy enforced → {out}")


if __name__ == "__main__":
    main()