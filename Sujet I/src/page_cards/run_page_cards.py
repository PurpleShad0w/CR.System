#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from legacy.section_names import DEFAULT_SECTION_NAME, normalize_section_name


def _run(cmd: list[str]) -> None:
	print('>', ' '.join(cmd))
	subprocess.check_call(cmd)


def maybe_run_process_onenote(args) -> Path:
	manifest = REPO_ROOT / 'process' / 'onenote' / args.notebook / 'manifest.json'
	if not args.ensure_onenote:
		return manifest
	if manifest.exists():
		return manifest
	cmd = [
		args.python_exe,
		str(REPO_ROOT / 'process_onenote.py'),
		args.notebook,
		'--input', str(Path(args.onenote_input)),
		'--out', str(Path(args.onenote_out)),
	]
	if args.audio_transcribe:
		cmd.append('--transcribe')
	if args.copy_assets:
		cmd.append('--copy-assets')
	_run(cmd)
	return manifest


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument('--case-id', required=True)
	ap.add_argument('--section-name', default=DEFAULT_SECTION_NAME)

	# process_onenote wiring
	ap.add_argument('--ensure-onenote', action='store_true', default=True)
	ap.add_argument('--no-ensure-onenote', action='store_false', dest='ensure_onenote')
	ap.add_argument('--notebook', default='test')
	ap.add_argument('--onenote-input', default='input/onenote-exporter/output')
	ap.add_argument('--onenote-out', default='process/onenote')
	ap.add_argument('--audio-transcribe', action='store_true', default=True)
	ap.add_argument('--copy-assets', action='store_true', default=False)
	ap.add_argument('--python-exe', default=sys.executable or 'python')

	# page cards
	ap.add_argument('--pages-index', default='', help='Override pages source (manifest.json). If empty, uses process/onenote/<notebook>/manifest.json.')
	ap.add_argument('--template', default='input/templates/Templates Slides.pptx')
	ap.add_argument('--slide-types', default='input/config/slide_types_template_slides.json')
	ap.add_argument('--renderer', default='render_report_pptx.py')
	ap.add_argument('--out', default='')
	ap.add_argument('--max-images', type=int, default=6)
	ap.add_argument('--max-bullets', type=int, default=10)

	# per-slide HF rewrite (default ON)
	ap.add_argument('--humanize', dest='humanize', action='store_true')
	ap.add_argument('--no-humanize', dest='humanize', action='store_false')
	ap.set_defaults(humanize=True)
	ap.add_argument('--humanize-temperature', type=float, default=0.2)
	ap.add_argument('--humanize-max-tokens', type=int, default=380)
	ap.add_argument('--humanize-top-p', type=float, default=1.0)
	ap.add_argument('--humanize-sleep', type=float, default=0.0)

	args = ap.parse_args()

	manifest = maybe_run_process_onenote(args)
	pages_src = Path(args.pages_index) if args.pages_index else manifest
	section = normalize_section_name(args.section_name or DEFAULT_SECTION_NAME)

	assembled = Path('process/page_cards/assembled_page_cards.json')
	out_pptx = Path(args.out) if args.out else Path(f'output/reports/{args.case_id}/Part1_PageCards.pptx')
	out_pptx.parent.mkdir(parents=True, exist_ok=True)

	_run([
		args.python_exe,
		'src/page_cards/build_page_cards_assembled.py',
		'--pages-index', str(pages_src),
		'--out', str(assembled),
		'--case-id', args.case_id,
		'--section-name', section,
		'--max-images', str(args.max_images),
		'--max-bullets', str(args.max_bullets),
	])

	if args.humanize:
		_run([
			args.python_exe,
			'src/page_cards/humanize_page_cards.py',
			'--assembled', str(assembled),
			'--out', str(assembled),
			'--temperature', str(args.humanize_temperature),
			'--max-tokens', str(args.humanize_max_tokens),
			'--top-p', str(args.humanize_top_p),
			'--sleep', str(args.humanize_sleep),
		])

	_run([
		args.python_exe,
		args.renderer,
		'--template', args.template,
		'--assembled', str(assembled),
		'--out', str(out_pptx),
		'--slide-types', args.slide_types,
	])

	print('OK:', out_pptx)


if __name__ == '__main__':
	main()
