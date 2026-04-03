from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


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


@dataclass
class OneNoteExportConfig:
    project_root: Path
    env_path: str
    list_only: bool
    notebook_name: str
    notebook_id: str
    merge: bool
    formats: str
    output_dir: str
    token_cache: str


@dataclass
class OneNoteProcessConfig:
    project_root: Path
    notebook: str
    input_root: str
    out_root: str
    transcribe: bool
    copy_assets: bool


@dataclass
class LearningPipelineConfig:
    project_root: Path


def _exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def _base_env() -> dict:
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def build_pipeline_command(cfg: PipelineConfig) -> List[str]:
    py = sys.executable
    cmd = [py, str(cfg.project_root / 'run_pipeline.py')]

    if cfg.case_id:
        cmd += ['--case-id', cfg.case_id]
    if cfg.notebook:
        cmd += ['--notebook', cfg.notebook]
    if (cfg.onenote_section or '').strip():
        cmd += ['--onenote-section', cfg.onenote_section.strip()]

    if cfg.mode:
        cmd += ['--mode', cfg.mode]

    cmd += ['--min-quality', str(cfg.min_quality)]

    if cfg.bacs_rules and _exists(cfg.bacs_rules):
        cmd += ['--bacs-rules', cfg.bacs_rules]
        cmd += ['--bacs-building-scope', cfg.bacs_building_scope]
        if cfg.bacs_targets and _exists(cfg.bacs_targets):
            cmd += ['--bacs-targets', cfg.bacs_targets]
        if cfg.bacs_part2_slides:
            cmd += ['--bacs-part2-slides']

    return cmd


def build_onenote_export_command(cfg: OneNoteExportConfig) -> List[str]:
    py = sys.executable
    cmd = [py, '-m', 'onenote_exporter.cli', '--config', cfg.env_path]

    if cfg.list_only:
        cmd += ['--list']
        return cmd

    if cfg.notebook_name:
        cmd += ['--notebook', cfg.notebook_name]
    if cfg.notebook_id:
        cmd += ['--notebook-id', cfg.notebook_id]

    if cfg.merge:
        cmd += ['--merge']
    if cfg.formats:
        cmd += ['--formats', cfg.formats]
    if cfg.output_dir:
        cmd += ['--output-dir', cfg.output_dir]
    if cfg.token_cache:
        cmd += ['--token-cache', cfg.token_cache]

    return cmd


def build_onenote_process_command(cfg: OneNoteProcessConfig) -> List[str]:
    py = sys.executable
    cmd = [py, str(cfg.project_root / 'process_onenote.py'), cfg.notebook]

    if cfg.input_root:
        cmd += ['--input', cfg.input_root]
    if cfg.out_root:
        cmd += ['--out', cfg.out_root]
    if cfg.transcribe:
        cmd += ['--transcribe']
    if cfg.copy_assets:
        cmd += ['--copy-assets']

    return cmd


def build_learning_pipeline_command(cfg: LearningPipelineConfig) -> List[str]:
    py = sys.executable
    return [py, str(cfg.project_root / 'run_learning_pipeline.py')]


def run_streaming(cmd: List[str], *, cwd: Path) -> Tuple[int, Iterable[str]]:
    env = _base_env()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    def _iter_lines():
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip('\n')
        proc.wait()

    return proc.pid, _iter_lines()


def run_pipeline_streaming(cfg: PipelineConfig) -> Tuple[int, Iterable[str]]:
    return run_streaming(build_pipeline_command(cfg), cwd=cfg.project_root)


def run_onenote_export_streaming(cfg: OneNoteExportConfig) -> Tuple[int, Iterable[str]]:
    return run_streaming(build_onenote_export_command(cfg), cwd=cfg.project_root)


def run_onenote_process_streaming(cfg: OneNoteProcessConfig) -> Tuple[int, Iterable[str]]:
    return run_streaming(build_onenote_process_command(cfg), cwd=cfg.project_root)


def run_learning_pipeline_streaming(cfg: LearningPipelineConfig) -> Tuple[int, Iterable[str]]:
    return run_streaming(build_learning_pipeline_command(cfg), cwd=cfg.project_root)


def compute_expected_outputs(project_root: Path, case_id: str) -> dict:
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
