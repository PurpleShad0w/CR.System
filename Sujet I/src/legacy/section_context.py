#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""section_context.py

Canonical section identity helpers.

Goal
----
Provide a single, explicit, immutable identity object for the OneNote report unit:

    section_context = {
      "onenote_notebook": "test",
      "onenote_section_name": "Oseraie – OSNY",
      "section_slug": "oseraie_osny"
    }

This is propagated across: aggregates, plans, drafts, generated bundle,
assembled report, and finally the PPTX metadata.

This module is deliberately tiny and dependency-free.
"""

import re
from typing import Any, Dict, Optional


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"


def build_section_context(notebook: str, section_name: str) -> Dict[str, str]:
    section_name = (section_name or "").strip()
    return {
        "onenote_notebook": (notebook or "").strip(),
        "onenote_section_name": section_name,
        "section_slug": slugify(section_name),
    }


def require_section_context(obj: Dict[str, Any], label: str) -> Dict[str, str]:
    ctx = obj.get("section_context")
    if not isinstance(ctx, dict):
        raise SystemExit(f"❌ Missing section_context in {label}")
    for k in ("onenote_notebook", "onenote_section_name", "section_slug"):
        if not ctx.get(k):
            raise SystemExit(f"❌ Invalid section_context.{k} in {label}")
    return ctx  # type: ignore


def assert_same_section_context(expected: Dict[str, str], actual: Dict[str, str], label: str):
    if expected != actual:
        raise SystemExit(
            "❌ section_context mismatch in "
            f"{label}\nExpected: {expected}\nActual:   {actual}"
        )


def maybe_ctx_from_legacy_fields(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Backward-compat for older artifacts that still have notebook/onenote_section/section_slug."""
    nb = obj.get("notebook")
    sec = obj.get("onenote_section")
    slug = obj.get("section_slug")
    if nb and sec and slug:
        return {
            "onenote_notebook": nb,
            "onenote_section_name": sec,
            "section_slug": slug,
        }
    return None
