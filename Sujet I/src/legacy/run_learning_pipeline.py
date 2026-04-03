#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_learning_pipeline.py (patched)

- Adds Windows console encoding hardening to avoid UnicodeEncodeError on symbols.
- Replaces unicode markers in prints with ASCII.

The functional behavior is unchanged: it runs process_reports.py then build_skeletons.py.
"""

import subprocess
import sys
from pathlib import Path

# --- Console encoding hardening (Windows cp1252) ---
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

INPUT_REPORTS_DIR = ROOT / 'input' / 'reports'
PROCESSED_REPORTS_ROOT = ROOT / 'process' / 'learning' / 'processed_reports'
PROCESSED_COLLECTION = PROCESSED_REPORTS_ROOT / 'Rapports_d_audit'
SKELETONS_OUT_DIR = ROOT / 'process' / 'learning' / 'skeletons'

EXTRACT_IMAGES = True


def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\nERROR: Learning pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main():
    INPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    SKELETONS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd1 = [
        sys.executable,
        'process_reports.py',
        '--root', '.',
        '--audits', str(INPUT_REPORTS_DIR),
        '--out', str(PROCESSED_REPORTS_ROOT),
    ]
    if EXTRACT_IMAGES:
        cmd1.append('--extract-images')
    run(cmd1)

    run([
        sys.executable,
        'build_skeletons.py',
        '--processed', str(PROCESSED_COLLECTION),
        '--out', str(SKELETONS_OUT_DIR),
    ])

    print("\nOK: Learning pipeline completed successfully")
    print(f"Corpus: {PROCESSED_COLLECTION}")
    print(f"Skeletons: {SKELETONS_OUT_DIR}")


if __name__ == '__main__':
    main()
