#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline.py

Single entrypoint to run the full GTB / BACS report generation pipeline.

DEFAULTS PATCH
--------------
- Keeps default execution with no CLI args.
- BUT: always passes BACS rules to run_llm_jobs.py if the rules file exists.
  This avoids the current behavior where Tableau 6 scoring is only activated
  when --onenote-section is set.

This matches the intended process:
- Part 1: use OneNote evidence
- Part 2/3: scoring/travaux driven by rules (when rules file is available)

"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from section_context import build_section_context, assert_same_section_context

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CASE_ID = "P050011"
DEFAULT_NOTEBOOK_NAME = "test"
DEFAULT_ONENOTE_SECTION = ""  # keep empty by default (case mode)

ROOT = Path(__file__).resolve().parent

REPORT_TYPES_CONFIG = ROOT / "input" / "config" / "report_types.json"
PROMPT_TEMPLATES = ROOT / "input" / "config" / "prompt_templates.json"
PPTX_TEMPLATE = ROOT / "input" / "templates" / "TEMPLATE_AUDIT_BUILD4USE.pptx"

DEFAULT_BACS_RULES = ROOT / "input" / "rules" / "bacs_table6_rules_structured_clean.json"

ONENOTE_EXPORT_ROOT = ROOT / "input" / "onenote-exporter" / "output"
PROCESS_ROOT = ROOT / "process"
ONENOTE_PROCESSED_ROOT = PROCESS_ROOT / "onenote"
PLANS_ROOT = PROCESS_ROOT / "plans"
DRAFTS_ROOT = PROCESS_ROOT / "drafts"
SKELETONS_DIR = PROCESS_ROOT / "learning" / "skeletons"
OUTPUT_ROOT = ROOT / "output"

DEFAULT_MIN_QUALITY_SCORE = 0.0  # dev-friendly default; can be raised by CLI
LLM_TEMPERATURE = "0.2"
LLM_MAX_TOKENS = "1500"
DEFAULT_LLM_MODE = "multistep"


try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n▶", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\n❌ Pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run the GTB/BACS generation pipeline end-to-end.")
    ap.add_argument("--case-id", default=DEFAULT_CASE_ID)
    ap.add_argument("--notebook", default=DEFAULT_NOTEBOOK_NAME)
    ap.add_argument("--onenote-section", default=DEFAULT_ONENOTE_SECTION)
    ap.add_argument("--mode", choices=["single", "multistep"], default=DEFAULT_LLM_MODE)
    ap.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY_SCORE)

    ap.add_argument("--bacs-rules", default=str(DEFAULT_BACS_RULES), help="Path to Tableau 6 rules JSON")
    ap.add_argument("--bacs-building-scope", default="Non résidentiel", choices=["Résidentiel", "Non résidentiel"])
    ap.add_argument("--bacs-targets", default="", help="Optional JSON mapping group->target class")
    ap.add_argument("--bacs-part2-slides", action="store_true")

    args = ap.parse_args()

    def slugify_local(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "section"

    notebook_name = args.notebook
    onenote_section = (args.onenote_section or "").strip()
    case_id = slugify_local(onenote_section) if onenote_section else args.case_id

    llm_mode = args.mode
    min_quality = args.min_quality

    notebook_processed = ONENOTE_PROCESSED_ROOT / notebook_name
    plan_path = PLANS_ROOT / f"{case_id}.json"
    draft_dir = DRAFTS_ROOT / case_id

    output_reports = OUTPUT_ROOT / "reports" / case_id
    output_reports.mkdir(parents=True, exist_ok=True)

    expected_ctx = build_section_context(notebook_name, onenote_section) if onenote_section else None

    # 1) OneNote processing
    run([
        sys.executable,
        "process_onenote.py",
        notebook_name,
        "--input",
        str(ONENOTE_EXPORT_ROOT),
        "--out",
        str(ONENOTE_PROCESSED_ROOT),
        "--transcribe",
    ])

    # 2) Section aggregation (only when section mode)
    agg_path = None
    if onenote_section:
        run([
            sys.executable,
            "aggregate_onenote_section.py",
            "--onenote",
            str(notebook_processed),
            "--section",
            onenote_section,
            "--out",
            str((ROOT / "process" / "onenote_aggregates")),
        ])
        agg_path = (ROOT / "process" / "onenote_aggregates" / notebook_name / f"{case_id}.json").resolve()
        agg_obj = json.loads(agg_path.read_text(encoding="utf-8"))
        if expected_ctx and isinstance(agg_obj.get("section_context"), dict):
            assert_same_section_context(expected_ctx, agg_obj["section_context"], "onenote_aggregate")

    # 3) Planning
    run([
        sys.executable,
        "plan_generation.py",
        "--config",
        str(REPORT_TYPES_CONFIG),
        "--skeletons",
        str(SKELETONS_DIR),
        "--onenote",
        str(notebook_processed),
        "--case-id",
        case_id,
        "--out",
        str(PLANS_ROOT),
        "--onenote-section",
        onenote_section,
    ])

    # 4) Draft generation
    cmd = [
        sys.executable,
        "generate_draft.py",
        "--plan",
        str(plan_path),
        "--onenote",
        str(notebook_processed),
        "--templates",
        str(PROMPT_TEMPLATES),
        "--out",
        str(DRAFTS_ROOT),
    ]
    if agg_path:
        cmd += ["--section-aggregate", str(agg_path)]
    run(cmd)

    # 5) LLM + assembly + quality
    llm_cmd = [
        sys.executable,
        "run_llm_jobs.py",
        "--bundle",
        str(draft_dir / "draft_bundle.json"),
        "--mode",
        llm_mode,
        "--temperature",
        LLM_TEMPERATURE,
        "--max_tokens",
        LLM_MAX_TOKENS,
        "--min_quality",
        str(min_quality),
    ]

    # KEY CHANGE: pass rules whenever file exists
    rules_path = Path(args.bacs_rules)
    if rules_path.exists():
        llm_cmd += [
            "--bacs_rules",
            str(rules_path),
            "--bacs_building_scope",
            args.bacs_building_scope,
        ]
        if args.bacs_targets:
            llm_cmd += ["--bacs_targets", args.bacs_targets]
        if args.bacs_part2_slides:
            llm_cmd += ["--bacs_part2_slides"]

    # still pass section_aggregate if we have one
    if agg_path and agg_path.exists():
        llm_cmd += ["--section_aggregate", str(agg_path)]

    run(llm_cmd)

    # 6) Render PPTX
    out_pptx = output_reports / "Rapport_Audit.pptx"
    run([
        sys.executable,
        "render_report_pptx.py",
        "--template",
        str(PPTX_TEMPLATE),
        "--assembled",
        str(draft_dir / "assembled_report.json"),
        "--out",
        str(out_pptx),
    ])

    print("\n✅ Pipeline completed successfully")
    print(f"📄 Output: {out_pptx}")


if __name__ == "__main__":
    main()
