#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from legacy.section_names import DEFAULT_SECTION_NAME, normalize_section_name
from page_text import collect_text, to_bullets
from page_images import collect_images
from image_selection import select_best_images


def load_json(p: Path) -> Any:
	try:
		return json.loads(p.read_text(encoding='utf-8'))
	except UnicodeDecodeError:
		return json.loads(p.read_text(encoding='utf-8-sig', errors='replace'))


def save_json(p: Path, obj: Any) -> None:
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def find_existing_pages_source(hint: Path) -> Optional[Path]:
	# Accept either process/onenote/<notebook>/manifest.json or older paths
	candidates = [
		hint,
		REPO_ROOT / 'process' / 'onenote' / 'manifest.json',
		REPO_ROOT / 'process' / 'onenote' / 'pages_index.json',
	]
	# Also search under process/onenote/*/manifest.json
	base = REPO_ROOT / 'process' / 'onenote'
	if base.exists():
		for cand in base.rglob('manifest.json'):
			candidates.append(cand)
	for c in candidates:
		try:
			if c and c.exists() and c.is_file():
				return c
		except Exception:
			continue
	return None


def iter_pages_from_pages_index(obj: Any) -> List[Dict[str, Any]]:
	if isinstance(obj, list):
		return [p for p in obj if isinstance(p, dict)]
	if isinstance(obj, dict) and isinstance(obj.get('pages'), list):
		return [p for p in obj['pages'] if isinstance(p, dict)]
	return []


def _safe_page_filename(page_id: str) -> str:
	# process_onenote.py writes: pages/<page_id>.json with '/' replaced by '_'
	return (page_id or '').replace('/', '_')


def _find_page_json_file(page_id: str, pages_dir: Path) -> Optional[Path]:
	pid = (page_id or '').strip()
	if not pid:
		return None
	cand = pages_dir / f"{pid}.json"
	if cand.exists():
		return cand
	cand2 = pages_dir / f"{_safe_page_filename(pid)}.json"
	if cand2.exists():
		return cand2
	# fallback: rglob within manifest folder
	try:
		for hit in pages_dir.rglob(f"{_safe_page_filename(pid)}.json"):
			if hit.is_file():
				return hit
	except Exception:
		pass
	return None


def iter_pages_from_manifest(obj: Any, pages_dir: Path) -> List[Dict[str, Any]]:
	if not isinstance(obj, dict):
		return []
	page_ids = obj.get('processed_pages')
	if not isinstance(page_ids, list) or not page_ids:
		return []
	pages: List[Dict[str, Any]] = []
	seen = set()
	for pid in page_ids:
		if not isinstance(pid, str):
			continue
		pid = pid.strip()
		if not pid or pid in seen:
			continue
		seen.add(pid)
		pfile = _find_page_json_file(pid, pages_dir)
		if pfile and pfile.exists():
			try:
				pages.append(load_json(pfile))
			except Exception:
				pages.append({'metadata': {'page_id': pid}, 'title': pid, 'blocks': [], 'assets': {}})
		else:
			pages.append({'metadata': {'page_id': pid}, 'title': pid, 'blocks': [], 'assets': {}})
	return pages


def _page_id(page: Dict[str, Any]) -> str:
	m = page.get('metadata')
	if isinstance(m, dict) and isinstance(m.get('page_id'), str):
		return m['page_id']
	pid = page.get('page_id')
	return pid if isinstance(pid, str) else ''


def _page_section(page: Dict[str, Any]) -> str:
	m = page.get('metadata')
	if isinstance(m, dict) and isinstance(m.get('section'), str):
		return m['section']
	sec = page.get('section')
	return sec if isinstance(sec, str) else ''


def build(pages_source: Path, out_json: Path, *, case_id: str, section_name: str, max_images: int, max_bullets: int) -> None:
	if not pages_source.exists():
		found = find_existing_pages_source(pages_source)
		if not found:
			raise FileNotFoundError(f"pages source not found: {pages_source}")
		pages_source = found

	obj = load_json(pages_source)
	pages = iter_pages_from_pages_index(obj)

	# process_onenote output: manifest.json lives in process/onenote/<notebook>/manifest.json
	pages_dir = pages_source.parent / 'pages'
	if not pages_dir.exists():
		# fallback to older layout
		pages_dir = REPO_ROOT / 'process' / 'onenote' / 'pages'

	if not pages:
		pages = iter_pages_from_manifest(obj, pages_dir)
	if not pages:
		raise SystemExit('No pages found (need pages_index.json with pages, or manifest.json with processed_pages + per-page JSON files).')

	section_norm = normalize_section_name(section_name)
	filtered: List[Dict[str, Any]] = []
	seen_pid = set()
	for pg in pages:
		pid = (_page_id(pg) or '').strip()
		if pid and pid in seen_pid:
			continue
		sec = normalize_section_name(_page_section(pg))
		if sec and section_norm and sec != section_norm:
			continue
		if pid:
			seen_pid.add(pid)
		filtered.append(pg)
	pages = filtered

	slides: List[Dict[str, Any]] = []
	slides.append({'type': 'PART_DIVIDER', 'part': 1, 'title': 'Etat des lieux (Pages OneNote)'})

	for pg in pages:
		title = (pg.get('title') or 'Page').strip()
		text = collect_text(pg)
		bul = to_bullets(text, max_lines=max_bullets)
		imgs = collect_images(pg)
		imgs = select_best_images(pg, imgs, title=title, bullets=bul, max_images=max_images)
		slides.append({
			'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
			'part': 1,
			'title': title,
			'bullets': bul,
			'images': imgs,
			'page_id': _page_id(pg),
		})

	out = {
		'case_id': case_id,
		'report_type': 'PAGE_CARDS_PART1',
		'section_context': {'onenote_section_name': section_norm},
		'macro_parts': [{'macro_part': 1, 'macro_part_name': 'Etat des lieux (Pages OneNote)'}],
		'slides': slides,
	}
	save_json(out_json, out)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument('--pages-index', required=True)
	ap.add_argument('--out', default='process/page_cards/assembled_page_cards.json')
	ap.add_argument('--case-id', required=True)
	ap.add_argument('--section-name', default=DEFAULT_SECTION_NAME)
	ap.add_argument('--max-images', type=int, default=6)
	ap.add_argument('--max-bullets', type=int, default=10)
	args = ap.parse_args()

	section = normalize_section_name(args.section_name or DEFAULT_SECTION_NAME)
	build(Path(args.pages_index), Path(args.out), case_id=args.case_id, section_name=section, max_images=args.max_images, max_bullets=args.max_bullets)
	print('Wrote:', args.out)


if __name__ == '__main__':
	main()
