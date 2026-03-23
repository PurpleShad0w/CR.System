#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from page_text import collect_text, to_bullets
from page_images import collect_images, pick_best


def load_json(p: Path) -> Any:
	return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Any) -> None:
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def iter_pages(obj: Any) -> List[Dict[str, Any]]:
	if isinstance(obj, list):
		return [p for p in obj if isinstance(p, dict)]
	if isinstance(obj, dict):
		pages = obj.get('pages')
		if isinstance(pages, list):
			return [p for p in pages if isinstance(p, dict)]
	return []


def build(pages_index: Path, out_json: Path, *, case_id: str, section_name: str, max_images: int, max_bullets: int) -> None:
	data = load_json(pages_index)
	pages = iter_pages(data)
	if not pages:
		raise SystemExit('No pages found in pages-index JSON (expected list or {"pages": [...]})')

	slides: List[Dict[str, Any]] = []
	slides.append({'type': 'PART_DIVIDER', 'part': 1, 'title': 'Etat des lieux (Pages OneNote)'})

	for pg in pages:
		title = (pg.get('title') or pg.get('name') or pg.get('display_name') or 'Page').strip()
		text = collect_text(pg)
		bul = to_bullets(text, max_lines=max_bullets)
		imgs = pick_best(collect_images(pg), max_images=max_images)

		slides.append({
			'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
			'part': 1,
			'title': title,
			'bullets': bul,
			'images': imgs,
		})

	out = {
		'case_id': case_id,
		'report_type': 'PAGE_CARDS_PART1',
		'section_context': {'onenote_section_name': section_name},
		'macro_parts': [{'macro_part': 1, 'macro_part_name': 'Etat des lieux (Pages OneNote)'}],
		'slides': slides,
	}
	save_json(out_json, out)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument('--pages-index', required=True)
	ap.add_argument('--out', default='process/page_cards/assembled_page_cards.json')
	ap.add_argument('--case-id', required=True)
	ap.add_argument('--section-name', default='')
	ap.add_argument('--max-images', type=int, default=3)
	ap.add_argument('--max-bullets', type=int, default=10)
	args = ap.parse_args()

	section = args.section_name or args.case_id
	build(Path(args.pages_index), Path(args.out), case_id=args.case_id, section_name=section, max_images=args.max_images, max_bullets=args.max_bullets)
	print('Wrote:', args.out)


if __name__ == '__main__':
	main()
