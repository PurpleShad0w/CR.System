#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from llm_client import make_client

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_messages(prompt: str) -> List[Dict[str, str]]:
    # Keep it simple and consistent with your prompts
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt}
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to drafts/<id>/draft_bundle.json")
    ap.add_argument("--out", default="", help="Output folder (defaults to bundle folder)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    client = make_client()

    generated = dict(bundle)
    for sec in generated.get("sections", []):
        prompt = (sec.get("prompt") or "").strip()
        if not prompt:
            sec["generated_text"] = ""
            sec["llm_raw"] = None
            continue

        messages = build_messages(prompt)
        resp = client.chat(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            stream=False
        )
        sec["generated_text"] = resp.text
        sec["llm_raw"] = resp.raw

    gen_path = out_dir / "generated_bundle.json"
    save_json(gen_path, generated)

    # Assemble macro-parts 1–3
    assembled = {
        "case_id": generated.get("case_id"),
        "report_type": generated.get("report_type"),
        "macro_parts": []
    }

    by_mp = {}
    for sec in generated.get("sections", []):
        mp = sec.get("macro_part")
        by_mp.setdefault(mp, []).append(sec)

    for mp in sorted(by_mp.keys()):
        assembled["macro_parts"].append({
            "macro_part": mp,
            "macro_part_name": by_mp[mp][0].get("macro_part_name"),
            "sections": [
                {"bucket_id": s.get("bucket_id"), "text": (s.get("generated_text") or "").strip()}
                for s in by_mp[mp]
            ]
        })

    assembled_path = out_dir / "assembled_report.json"
    save_json(assembled_path, assembled)

    print(f"Wrote: {gen_path}")
    print(f"Wrote: {assembled_path}")

if __name__ == "__main__":
    main()