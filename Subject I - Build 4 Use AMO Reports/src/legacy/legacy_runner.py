#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def python_exe() -> str:
	return sys.executable or 'python'


def resolve_run_pipeline(root: Optional[Path] = None) -> Path:
	root = root or repo_root()
	# Preferred if you moved legacy under src/legacy
	cand = root / 'src' / 'legacy' / 'run_pipeline.py'
	if cand.exists():
		return cand
	# Fallbacks
	cand2 = root / 'run_pipeline.py'
	if cand2.exists():
		return cand2
	# Last resort
	return cand


def resolve_page_cards(root: Optional[Path] = None) -> Path:
	root = root or repo_root()
	return root / 'src' / 'page_cards' / 'run_page_cards.py'


def run_script(script: Path, argv: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
	cmd = [python_exe(), str(script)] + list(argv)
	proc = subprocess.run(cmd, cwd=str(cwd or (repo_root())), capture_output=True, text=True)
	out = (proc.stdout or '') + (proc.stderr or '')
	return proc.returncode, out


def run_legacy(argv: List[str], root: Optional[Path] = None) -> Tuple[int, str]:
	root = root or repo_root()
	entry = resolve_run_pipeline(root)
	return run_script(entry, argv, cwd=root)


def run_page_cards(argv: List[str], root: Optional[Path] = None) -> Tuple[int, str]:
	root = root or repo_root()
	entry = resolve_page_cards(root)
	return run_script(entry, argv, cwd=root)
