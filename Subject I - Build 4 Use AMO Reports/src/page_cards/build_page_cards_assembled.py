#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `src/` importable so `import legacy...` works when running as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from legacy.section_names import DEFAULT_SECTION_NAME, normalize_section_name

from page_text import collect_text, to_bullets
from page_images import collect_images, pick_best


def load_json(p: Path) -> Any:
	try:
		return json.loads(p.read_text(encoding='utf-8'))
	except UnicodeDecodeError:
		return json.loads(p.read_text(encoding='utf-8-sig', errors='replace'))


def save_json(p: Path, obj: Any) -> None:
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def repo_root_from_here() -> Path:
	return REPO_ROOT


def find_existing_pages_source(hint: Path) -> Optional[Path]:
	root = repo_root_from_here()
	candidates = [
		hint,
		root / 'process' / 'onenote' / 'pages_index.json',
		root / 'process' / 'onenote' / 'manifest.json',
		root / 'process' / 'onenote' / 'manifest' / 'manifest.json',
		root / 'process' / 'onenote' / 'manifest' / 'index.json',
		root / 'input' / 'onenote-exporter' / 'output' / 'manifest.json',
	]
	for c in candidates:
		try:
			if c and c.exists() and c.is_file():
				return c
		except Exception:
			continue
	base = root / 'process' / 'onenote'
	if base.exists():
		for cand in base.rglob('manifest.json'):
			if cand.is_file():
				return cand
	return None


def iter_pages_from_pages_index(obj: Any) -> List[Dict[str, Any]]:
	if isinstance(obj, list):
		return [p for p in obj if isinstance(p, dict)]
	if isinstance(obj, dict) and isinstance(obj.get('pages'), list):
		return [p for p in obj['pages'] if isinstance(p, dict)]
	return []


def _find_page_json_file(page_id: str) -> Optional[Path]:
	root = repo_root_from_here()
	candidates = [
		root / 'process' / 'onenote' / 'pages' / f'{page_id}.json',
		root / 'process' / 'onenote' / f'{page_id}.json',
	]
	for c in candidates:
		if c.exists() and c.is_file():
			return c
	base = root / 'process' / 'onenote'
	if base.exists():
		for cand in base.rglob(f'{page_id}.json'):
			if cand.is_file():
				return cand
	return None


def iter_pages_from_manifest(obj: Any) -> List[Dict[str, Any]]:
	if not isinstance(obj, dict):
		return []
	page_ids = obj.get('processed_pages')
	if not isinstance(page_ids, list) or not page_ids:
		return []
	pages = []
	for pid in page_ids:
		if not isinstance(pid, str) or not pid.strip():
			continue
		pfile = _find_page_json_file(pid.strip())
		if pfile and pfile.exists():
			try:
				pages.append(load_json(pfile))
			except Exception:
				pages.append({'title': pid.strip(), 'blocks': [], 'assets': {}})
		else:
			pages.append({'title': pid.strip(), 'blocks': [], 'assets': {}})
	return [p for p in pages if isinstance(p, dict)]


def build(pages_source: Path, out_json: Path, *, case_id: str, section_name: str, max_images: int, max_bullets: int) -> None:
	if not pages_source.exists():
		found = find_existing_pages_source(pages_source)
		if not found:
			raise FileNotFoundError(f"pages source not found: {pages_source}")
		pages_source = found

	obj = load_json(pages_source)
	pages = iter_pages_from_pages_index(obj)
	if not pages:
		pages = iter_pages_from_manifest(obj)
	if not pages:
		raise SystemExit('No pages found (need pages_index.json with pages, or manifest.json with processed_pages + per-page JSON files).')

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
	ap.add_argument('--pages-index', required=True, help='pages_index.json or manifest.json')
	ap.add_argument('--out', default='process/page_cards/assembled_page_cards.json')
	ap.add_argument('--case-id', required=True)
	ap.add_argument('--section-name', default=DEFAULT_SECTION_NAME)
	ap.add_argument('--max-images', type=int, default=3)
	ap.add_argument('--max-bullets', type=int, default=10)
	args = ap.parse_args()

	section = normalize_section_name(args.section_name or DEFAULT_SECTION_NAME)
	build(Path(args.pages_index), Path(args.out), case_id=args.case_id, section_name=section, max_images=args.max_images, max_bullets=args.max_bullets)
	print('Wrote:', args.out)


if __name__ == '__main__':
	main()
