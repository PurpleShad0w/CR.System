#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
	print('>', ' '.join(cmd))
	subprocess.check_call(cmd)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument('--pages-index', required=True)
	ap.add_argument('--case-id', required=True)
	ap.add_argument('--section-name', default='')
	ap.add_argument('--template', default='input/templates/Templates Slides.pptx')
	ap.add_argument('--slide-types', default='input/config/slide_types.json')
	ap.add_argument('--renderer', default='render_report_pptx.py')
	ap.add_argument('--out', default='')
	ap.add_argument('--max-images', type=int, default=3)
	ap.add_argument('--max-bullets', type=int, default=10)
	args = ap.parse_args()

	assembled = Path('process/page_cards/assembled_page_cards.json')
	out_pptx = Path(args.out) if args.out else Path(f'output/reports/{args.case_id}/Part1_PageCards.pptx')
	out_pptx.parent.mkdir(parents=True, exist_ok=True)

	section = args.section_name or args.case_id

	_run([
		'python',
		str(Path('src/page_cards/build_page_cards_assembled.py')),
		'--pages-index', args.pages_index,
		'--out', str(assembled),
		'--case-id', args.case_id,
		'--section-name', section,
		'--max-images', str(args.max_images),
		'--max-bullets', str(args.max_bullets),
	])

	_run([
		'python',
		args.renderer,
		'--template', args.template,
		'--assembled', str(assembled),
		'--out', str(out_pptx),
		'--slide-types', args.slide_types,
	])

	print('OK:', out_pptx)


if __name__ == '__main__':
	main()
