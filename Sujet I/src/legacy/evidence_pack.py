
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evidence_pack.py

A richer (non-truncating) deterministic representation of OneNote evidence.

Why this exists
---------------
The previous section_brief builder was intentionally conservative and kept only
one "best" fact per rubric; combined with sanitization this could yield too little text.

This module keeps *multiple* facts per rubric, separates questions/requests,
and extracts equipment-level assertions (e.g. "chaudière n'est pas connectée").

Output
------
{
  "scaffold": ["Comptage énergétique", ...],
  "by_topic": {
      "Comptage énergétique": {"facts": [...], "normative": [...], "headings": [...], "requests": [...]},
      ...
  },
  "equipment_facts": [{"equipment":..., "status":..., "statement":..., "page_id":..., "excerpt":...}],
  "requests": [{"text":..., "page_id":...}],
  "stats": {...}
}

"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

RE_PAGE = re.compile(r"^##\s+Page:\s*(.*?)\s*\(page_id=([^,\)]+)", re.IGNORECASE)
RE_QBLOCK = re.compile(r"^##\s+Questions\s*/\s*demandes", re.IGNORECASE)
RE_BULLET = re.compile(r"^\-\s+(.*)$")
RE_SNIPTYPE = re.compile(r"^\[(\w+?)\]\s+(.*)$")
RE_SCAFFOLD_HDR = re.compile(r"^####\s+(.*)$")
RE_PLAN_BLOCK_START = re.compile(r"^###\s+Plan\s+attendu", re.IGNORECASE)

# Equipment detection (very simple heuristics; can be extended)
EQUIPMENT_PATTERNS = [
    (re.compile(r"\bchaudi(è|e)re(s)?\b", re.IGNORECASE), "chaudiere"),
    (re.compile(r"\bgroupe\s+froid\b", re.IGNORECASE), "groupe_froid"),
    (re.compile(r"\bcta\b|centrale(s)?\s+de\s+traitement\s+d['’]?air", re.IGNORECASE), "cta"),
    (re.compile(r"\bventilo[-\s]?convecteur(s)?\b", re.IGNORECASE), "ventilo_convecteur"),
    (re.compile(r"\btgbt\b", re.IGNORECASE), "tgbt"),
    (re.compile(r"\btd\b|tableau(x)?\s+divisionnaire(s)?\b", re.IGNORECASE), "tableaux_divisionnaires"),
    (re.compile(r"\becl(airage|aire|airages)\b|\bluminaire(s)?\b", re.IGNORECASE), "eclairage"),
]

NEGATION_PATTERNS = [
    re.compile(r"\bn['’]?est\s+pas\s+connect(é|ee|ée|és|ées)\b", re.IGNORECASE),
    re.compile(r"\bpas\s+connect(é|ee|ée|és|ées)\b", re.IGNORECASE),
    re.compile(r"\bhors\s+communication\b", re.IGNORECASE),
    re.compile(r"\baucun(e)?\s+supervision\b", re.IGNORECASE),
    re.compile(r"\baucun(e)?\s+comptage\b", re.IGNORECASE),
    re.compile(r"\bsans\s+report\b|\bsans\s+retour\b", re.IGNORECASE),
]


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_scaffold_from_prompt(prompt: str) -> List[str]:
    if not prompt:
        return []
    lines = prompt.splitlines()
    in_plan = False
    headings: List[str] = []
    for ln in lines:
        if RE_PLAN_BLOCK_START.match(ln.strip()):
            in_plan = True
            continue
        if in_plan:
            m = RE_SCAFFOLD_HDR.match(ln.strip())
            if m:
                headings.append(m.group(1).strip())
                continue
            if headings and not ln.strip():
                break
    return headings


def parse_evidence(evidence: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pages: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    if not evidence:
        return pages, questions

    cur: Optional[Dict[str, Any]] = None
    in_q = False

    for raw_ln in evidence.splitlines():
        ln = raw_ln.rstrip("\r\n")
        s = ln.strip()
        if not s:
            continue

        if RE_QBLOCK.match(s):
            in_q = True
            cur = None
            continue

        m = RE_PAGE.match(s)
        if m:
            in_q = False
            title = m.group(1).strip()
            page_id = m.group(2).strip()
            cur = {"page_id": page_id, "title": title, "snippets": []}
            pages.append(cur)
            continue

        bm = RE_BULLET.match(s)
        if bm:
            content = bm.group(1).strip()
            if in_q:
                pid = None
                m_pid = re.search(r"page_id=([^\s]+)", content)
                if m_pid:
                    pid = m_pid.group(1).strip().rstrip(',')
                questions.append({"page_id": pid, "raw": content})
                continue
            if cur is None:
                continue
            st = None
            sm = RE_SNIPTYPE.match(content)
            if sm:
                st = sm.group(1)
                content = sm.group(2).strip()
            cur["snippets"].append({"raw": content, "snip_type": st})

    return pages, questions


def classify_snippet(text: str) -> str:
    t = (text or "").strip()
    nt = norm(t)
    if not nt:
        return "empty"

    if "?" in t or nt.startswith(("peux-tu", "peux tu", "pouvez-vous", "pouvez vous", "merci de", "svp", "stp", "à faire", "todo")):
        return "request"

    # Normative commentary (keep it, but separated)
    if any(k in nt for k in ("iso", "52120", "décret bacs", "decret bacs")):
        return "normative"
    if re.search(r"\bclasse\s*[a-d]\b", nt):
        return "normative"
    if nt.startswith(("il faut", "il est nécessaire", "doit", "devra")):
        return "normative"

    # Heading-like: short and noun-ish
    words = nt.split()
    if len(words) <= 6 and not any(v in nt for v in ("est", "sont", "dispose", "présent", "present", "aucun", "aucune", "pas", "n'") ):
        return "heading"

    return "fact"


def topic_for_snippet(snippet: str, scaffold_topics: List[str]) -> Optional[str]:
    nt = norm(snippet)
    if not scaffold_topics:
        return None

    best = None
    best_score = 0
    for topic in scaffold_topics:
        tt = norm(topic)
        toks = [w for w in tt.split() if len(w) > 3]
        score = sum(1 for w in toks if w in nt)
        if score > best_score:
            best_score = score
            best = topic

    if best_score <= 0:
        return None
    return best


def extract_equipment_facts(text: str) -> List[Tuple[str, str]]:
    """Return list of (equipment_key, status) for a text if detectable."""
    matches = []
    eqs = []
    for rx, key in EQUIPMENT_PATTERNS:
        if rx.search(text or ""):
            eqs.append(key)

    if not eqs:
        return []

    status = None
    for nrx in NEGATION_PATTERNS:
        if nrx.search(text or ""):
            status = "not_connected_or_missing"
            break

    # if no explicit negation, keep unknown
    if status is None:
        status = "mentioned"

    for e in sorted(set(eqs)):
        matches.append((e, status))

    return matches


def build_evidence_pack(section: Dict[str, Any], max_facts_per_topic: int = 12) -> Dict[str, Any]:
    prompt = section.get("prompt") or ""
    evidence = section.get("evidence") or ""
    scaffold = extract_scaffold_from_prompt(prompt)

    pages, qlist = parse_evidence(evidence)

    # Collect items
    items: List[Dict[str, Any]] = []
    for p in pages:
        pid = p.get("page_id")
        for sn in p.get("snippets", []):
            raw = (sn.get("raw") or "").strip()
            if not raw:
                continue
            cls = classify_snippet(raw)
            items.append({
                "class": cls,
                "text": raw,
                "page_id": pid,
                "page_title": p.get("title"),
            })

    requests: List[Dict[str, Any]] = []
    for q in qlist:
        raw = (q.get("raw") or "").strip()
        if raw:
            requests.append({"text": raw, "page_id": q.get("page_id")})

    # Group by topics
    by_topic: Dict[str, Dict[str, List[Dict[str, Any]]]] = {t: {"facts": [], "normative": [], "headings": [], "requests": []} for t in scaffold}
    unassigned: List[Dict[str, Any]] = []

    for it in items:
        if it["class"] == "empty":
            continue
        t = topic_for_snippet(it["text"], scaffold)
        if t is None:
            unassigned.append(it)
            continue
        bucket = by_topic.setdefault(t, {"facts": [], "normative": [], "headings": [], "requests": []})
        bucket_key = it["class"] + "s" if it["class"] != "fact" else "facts"
        # normalize key mapping
        if it["class"] == "normative":
            bucket["normative"].append(it)
        elif it["class"] == "heading":
            bucket["headings"].append(it)
        elif it["class"] == "request":
            bucket["requests"].append(it)
        else:
            bucket["facts"].append(it)

    # Truncate only for size safety, but keep multiple
    for t, buckets in by_topic.items():
        buckets["facts"] = buckets["facts"][:max_facts_per_topic]
        buckets["normative"] = buckets["normative"][:max_facts_per_topic]
        buckets["headings"] = buckets["headings"][:max_facts_per_topic]
        buckets["requests"] = buckets["requests"][:max_facts_per_topic]

    # Equipment facts across all items
    equipment_facts: List[Dict[str, Any]] = []
    for it in items:
        if it["class"] not in ("fact", "normative"):
            continue
        eqs = extract_equipment_facts(it["text"])
        for eq_key, status in eqs:
            equipment_facts.append({
                "equipment": eq_key,
                "status": status,
                "statement": it["text"],
                "page_id": it.get("page_id"),
                "excerpt": (it["text"] or "")[:240],
            })

    # Dedup equipment facts
    seen = set()
    dedup_eq = []
    for e in equipment_facts:
        k = (e["equipment"], e["status"], e.get("page_id"), e["excerpt"])
        if k in seen:
            continue
        seen.add(k)
        dedup_eq.append(e)

    stats = {
        "total_items": len(items),
        "unassigned": len(unassigned),
        "requests": len(requests),
        "equipment_facts": len(dedup_eq),
        "topics": len(scaffold),
    }

    return {
        "scaffold": scaffold,
        "by_topic": by_topic,
        "equipment_facts": dedup_eq,
        "requests": requests,
        "unassigned": unassigned[:40],
        "stats": stats,
    }
