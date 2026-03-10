from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class PipelineConfig:
    project_root: Path
    notebook: str
    onenote_section: str
    case_id: str
    mode: str
    min_quality: float
    bacs_rules: str
    bacs_building_scope: str
    bacs_targets: str
    bacs_part2_slides: bool


def _exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def build_command(cfg: PipelineConfig) -> List[str]:
    """Build run_pipeline.py command using the project's CLI (as currently implemented)."""
    py = sys.executable
    cmd = [py, str(cfg.project_root / 'run_pipeline.py')]

    if cfg.case_id:
        cmd += ['--case-id', cfg.case_id]
    if cfg.notebook:
        cmd += ['--notebook', cfg.notebook]
    # Only pass onenote-section if provided
    if (cfg.onenote_section or '').strip():
        cmd += ['--onenote-section', cfg.onenote_section.strip()]

    if cfg.mode:
        cmd += ['--mode', cfg.mode]

    cmd += ['--min-quality', str(cfg.min_quality)]

    # Tableau 6 rules
    if cfg.bacs_rules and _exists(cfg.bacs_rules):
        cmd += ['--bacs-rules', cfg.bacs_rules]
        cmd += ['--bacs-building-scope', cfg.bacs_building_scope]
        if cfg.bacs_targets and _exists(cfg.bacs_targets):
            cmd += ['--bacs-targets', cfg.bacs_targets]
        if cfg.bacs_part2_slides:
            cmd += ['--bacs-part2-slides']

    return cmd


def run_streaming(cfg: PipelineConfig) -> Tuple[int, Iterable[str]]:
    """Run pipeline and yield stdout lines."""
    cmd = build_command(cfg)

    env = os.environ.copy()
    # Force unbuffered Python output for live streaming
    env['PYTHONUNBUFFERED'] = '1'

    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    def _iter_lines():
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip('\n')
        proc.wait()

    return proc.pid, _iter_lines()


def compute_expected_outputs(project_root: Path, case_id: str) -> dict:
    """Best-effort paths based on current run_pipeline.py conventions."""
    out_pptx = project_root / 'output' / 'reports' / case_id / 'Rapport_Audit.pptx'
    assembled = project_root / 'process' / 'drafts' / case_id / 'assembled_report.json'
    generated = project_root / 'process' / 'drafts' / case_id / 'generated_bundle.json'
    quality = project_root / 'process' / 'drafts' / case_id / 'quality_report.json'

    return {
        'pptx': out_pptx,
        'assembled': assembled,
        'generated_bundle': generated,
        'quality_report': quality,
    }
