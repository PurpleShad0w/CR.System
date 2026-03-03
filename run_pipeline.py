#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py

Single entrypoint to run the full GTB / BACS report generation pipeline.

Pipeline (as per README):
1. process_onenote.py
2. plan_generation.py
3. generate_draft.py
4. run_llm_jobs.py
5. quality gating (quality_score.py)
6. render_report_pptx.py

This script is intentionally simple and deterministic.
All parameters are defined below.
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION (EDIT HERE)
# ============================================================================

# Case / input identifiers
CASE_ID = "P050011"
NOTEBOOK_NAME = "test"              # matches processed/<notebook>/
REPORT_TYPES_CONFIG = "report_types.json"

# Paths
ROOT = Path(__file__).resolve().parent
ONENOTE_EXPORT_ROOT = ROOT / "onenote-exporter" / "output"
PROCESSED_ROOT = ROOT / "processed"
PLANS_ROOT = ROOT / "plans"
DRAFTS_ROOT = ROOT / "drafts"

PROMPT_TEMPLATES = ROOT / "prompts" / "prompt_templates.json"
SKELETONS_DIR = ROOT / "skeletons"

PPTX_TEMPLATE = ROOT / "TEMPLATE_AUDIT_BUILD4USE.pptx"

# Quality gate
MIN_QUALITY_SCORE = 90.0

# LLM parameters
LLM_TEMPERATURE = "0.2"
LLM_MAX_TOKENS = "1200"

# ============================================================================
# UTIL
# ============================================================================

def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n▶", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\n❌ Pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


# ============================================================================
# PIPELINE
# ============================================================================

def main():
    notebook_processed = PROCESSED_ROOT / NOTEBOOK_NAME
    plan_path = PLANS_ROOT / f"{CASE_ID}.json"
    draft_dir = DRAFTS_ROOT / CASE_ID

    # 1. OneNote → structured JSON
    run([
        sys.executable,
        "process_onenote.py",
        NOTEBOOK_NAME,
        "--transcribe",
    ])

    # 2. Planning
    run([
        sys.executable,
        "plan_generation.py",
        "--config", REPORT_TYPES_CONFIG,
        "--skeletons", str(SKELETONS_DIR),
        "--onenote", str(notebook_processed),
    ])

    # 3. Draft generation (LLM prompts + evidence)
    run([
        sys.executable,
        "generate_draft.py",
        "--plan", str(plan_path),
        "--onenote", str(notebook_processed),
        "--templates", str(PROMPT_TEMPLATES),
        "--out", str(DRAFTS_ROOT),
    ])

    # 4. LLM execution + assembly + quality gate
    run([
        sys.executable,
        "run_llm_jobs.py",
        "--bundle", str(draft_dir / "draft_bundle.json"),
        "--temperature", LLM_TEMPERATURE,
        "--max_tokens", LLM_MAX_TOKENS,
        "--min_quality", str(MIN_QUALITY_SCORE),
    ])

    # 5. PPTX rendering (only if quality gate passed)
    run([
        sys.executable,
        "render_report_pptx.py",
        "--template", str(PPTX_TEMPLATE),
        "--assembled", str(draft_dir / "assembled_report.json"),
        "--out", str(draft_dir / "Rapport_Audit.pptx"),
    ])

    print("\n✅ Pipeline completed successfully")
    print(f"📄 Output: {draft_dir / 'Rapport_Audit.pptx'}")


if __name__ == "__main__":
    main()