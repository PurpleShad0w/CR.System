#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""run_pipeline.py (escape-safe)

Only change: passes the boss info template if present under
input/templates/Templates Slides.pptx.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from section_context import build_section_context, assert_same_section_context

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_ID = 'P060011'
DEFAULT_NOTEBOOK_NAME = 'test'
DEFAULT_ONENOTE_SECTION = 'Clinique - Goussonville'

REPORT_TYPES_CONFIG = ROOT / 'input' / 'config' / 'report_types.json'
PROMPT_TEMPLATES = ROOT / 'input' / 'config' / 'prompt_templates.json'
PPTX_TEMPLATE = ROOT / 'input' / 'templates' / 'TEMPLATE_AUDIT_BUILD4USE.pptx'
INFO_TEMPLATE = ROOT / 'input' / 'templates' / 'Templates Slides.pptx'
DEFAULT_BACS_RULES = ROOT / 'input' / 'rules' / 'bacs_table6_rules_structured_clean.json'

ONENOTE_EXPORT_ROOT = ROOT / 'input' / 'onenote-exporter' / 'output'
PROCESS_ROOT = ROOT / 'process'
ONENOTE_PROCESSED_ROOT = PROCESS_ROOT / 'onenote'
PLANS_ROOT = PROCESS_ROOT / 'plans'
DRAFTS_ROOT = PROCESS_ROOT / 'drafts'
SKELETONS_DIR = PROCESS_ROOT / 'learning' / 'skeletons'
OUTPUT_ROOT = ROOT / 'output'

LLM_TEMPERATURE = '0.2'
LLM_MAX_TOKENS = '1500'
DEFAULT_LLM_MODE = 'multistep'


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print('\n>', ' '.join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\nERROR: Pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case-id', default=DEFAULT_CASE_ID)
    ap.add_argument('--notebook', default=DEFAULT_NOTEBOOK_NAME)
    ap.add_argument('--onenote-section', default=DEFAULT_ONENOTE_SECTION)
    ap.add_argument('--mode', choices=['single', 'multistep'], default=DEFAULT_LLM_MODE)
    ap.add_argument('--min-quality', type=float, default=0.0)
    ap.add_argument('--bacs-rules', default=str(DEFAULT_BACS_RULES))
    ap.add_argument('--bacs-building-scope', default='Non résidentiel', choices=['Résidentiel', 'Non résidentiel'])
    ap.add_argument('--bacs-targets', default='')
    args = ap.parse_args()

    notebook = args.notebook
    section = (args.onenote_section or '').strip()
    case_id = (args.case_id or '').strip() or DEFAULT_CASE_ID

    notebook_processed = ONENOTE_PROCESSED_ROOT / notebook
    plan_path = PLANS_ROOT / f'{case_id}.json'
    draft_dir = DRAFTS_ROOT / case_id
    out_reports = OUTPUT_ROOT / 'reports' / case_id
    out_reports.mkdir(parents=True, exist_ok=True)

    expected_ctx = build_section_context(notebook, section) if section else None

    run([sys.executable, 'process_onenote.py', notebook, '--input', str(ONENOTE_EXPORT_ROOT), '--out', str(ONENOTE_PROCESSED_ROOT), '--transcribe'])

    if section:
        run([sys.executable, 'aggregate_onenote_section.py', '--onenote', str(notebook_processed), '--section', section, '--out', str(ROOT / 'process' / 'onenote_aggregates')])
        slug = expected_ctx.get('section_slug') if expected_ctx else ''
        agg_path = (ROOT / 'process' / 'onenote_aggregates' / notebook / f'{slug}.json').resolve()
        agg_obj = json.loads(agg_path.read_text(encoding='utf-8'))
        if expected_ctx and isinstance(agg_obj.get('section_context'), dict):
            assert_same_section_context(expected_ctx, agg_obj['section_context'], 'onenote_aggregate')

    run([sys.executable, 'plan_generation.py', '--config', str(REPORT_TYPES_CONFIG), '--skeletons', str(SKELETONS_DIR), '--onenote', str(notebook_processed), '--case-id', case_id, '--out', str(PLANS_ROOT), '--onenote-section', section])

    run([sys.executable, 'generate_draft.py', '--plan', str(plan_path), '--onenote', str(notebook_processed), '--templates', str(PROMPT_TEMPLATES), '--out', str(DRAFTS_ROOT)])

    llm_cmd = [
        sys.executable, 'run_llm_jobs.py',
        '--bundle', str(draft_dir / 'draft_bundle.json'),
        '--mode', args.mode,
        '--temperature', LLM_TEMPERATURE,
        '--max_tokens', LLM_MAX_TOKENS,
        '--min_quality', str(args.min_quality),
        '--onenote', str(notebook_processed),
        '--bacs_rules', str(Path(args.bacs_rules)),
        '--bacs_building_scope', args.bacs_building_scope,
    ]
    if args.bacs_targets:
        llm_cmd += ['--bacs_targets', args.bacs_targets]
    run(llm_cmd)

    out_pptx = out_reports / 'Rapport_Audit.pptx'
    render_cmd = [
        sys.executable, 'render_report_pptx.py',
        '--template', str(PPTX_TEMPLATE),
        '--assembled', str(draft_dir / 'assembled_report.json'),
        '--out', str(out_pptx),
    ]
    if INFO_TEMPLATE.exists():
        render_cmd += ['--info-template', str(INFO_TEMPLATE)]
    run(render_cmd)

    print('\nOK: Pipeline completed successfully')
    print('Output:', out_pptx)


if __name__ == '__main__':
    main()
