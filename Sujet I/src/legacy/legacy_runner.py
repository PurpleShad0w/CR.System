#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from legacy.section_names import section_name_variants


def repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def python_exe() -> str:
	return sys.executable or 'python'


def resolve_run_pipeline(root: Optional[Path] = None) -> Path:
	root = root or repo_root()
	cand = root / 'src' / 'legacy' / 'run_pipeline.py'
	if cand.exists():
		return cand
	cand2 = root / 'run_pipeline.py'
	if cand2.exists():
		return cand2
	return cand


def resolve_page_cards(root: Optional[Path] = None) -> Path:
	root = root or repo_root()
	return root / 'src' / 'page_cards' / 'run_page_cards.py'


def run_script(script: Path, argv: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
	cmd = [python_exe(), str(script)] + list(argv)
	proc = subprocess.run(cmd, cwd=str(cwd or repo_root()), capture_output=True, text=True)
	out = (proc.stdout or '') + (proc.stderr or '')
	return proc.returncode, out


def run_page_cards(argv: List[str], root: Optional[Path] = None) -> Tuple[int, str]:
	root = root or repo_root()
	entry = resolve_page_cards(root)
	return run_script(entry, argv, cwd=root)


def _replace_arg(argv: List[str], flag: str, value: str) -> List[str]:
	out = list(argv)
	if flag in out:
		i = out.index(flag)
		if i + 1 < len(out):
			out[i + 1] = value
			return out
		out.append(value)
		return out
	out.extend([flag, value])
	return out


def run_legacy(argv: List[str], root: Optional[Path] = None) -> Tuple[int, str]:
	root = root or repo_root()
	entry = resolve_run_pipeline(root)

	if '--onenote-section' not in argv:
		return run_script(entry, argv, cwd=root)

	idx = argv.index('--onenote-section')
	section = argv[idx + 1] if idx + 1 < len(argv) else ''
	cands = section_name_variants(section)
	if not cands:
		return run_script(entry, argv, cwd=root)

	combined = []
	last = 1
	for cand in cands:
		argv_try = _replace_arg(argv, '--onenote-section', cand)
		code, out = run_script(entry, argv_try, cwd=root)
		combined.append("\n=== Tried onenote-section: " + cand + " (exit=" + str(code) + ") ===\n")
		combined.append(out)
		last = code
		if code == 0:
			return 0, ''.join(combined)

	return last, ''.join(combined)
