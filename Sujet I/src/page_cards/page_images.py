#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff'}


def _is_image_path(p: str) -> bool:
	try:
		return Path(p).suffix.lower() in IMG_EXTS
	except Exception:
		return False


def _as_image_dict(it: Any) -> Dict[str, str]:
	if isinstance(it, str):
		return {'path': it, 'caption': ''}
	if isinstance(it, dict):
		p = it.get('path') or it.get('file') or it.get('relpath') or it.get('name') or ''
		c = it.get('caption') or it.get('legend') or it.get('title') or ''
		return {'path': p, 'caption': c}
	return {'path': '', 'caption': ''}


def _push(out: List[Dict[str, str]], p: str, c: str = '') -> None:
	p = (p or '').strip()
	if not p:
		return
	if Path(p).suffix and not _is_image_path(p):
		return
	out.append({'path': p, 'caption': (c or '')})


def collect_images(page: Dict[str, Any]) -> List[Dict[str, str]]:
	out: List[Dict[str, str]] = []

	for key in ('images', 'pics', 'pictures', 'media', 'attachments'):
		raw = page.get(key)
		if isinstance(raw, list):
			for it in raw:
				d = _as_image_dict(it)
				if d.get('path'):
					_push(out, d['path'], d.get('caption', ''))

	assets = page.get('assets')
	if isinstance(assets, dict):
		imgs = assets.get('images')
		if isinstance(imgs, list):
			for it in imgs:
				d = _as_image_dict(it)
				if d.get('path'):
					_push(out, d['path'], d.get('caption', ''))

	blocks = page.get('blocks')
	if isinstance(blocks, list):
		for b in blocks:
			if not isinstance(b, dict):
				continue
			btype = (b.get('type') or '').lower()
			p = b.get('path') or ''
			if btype == 'image' and p:
				_push(out, p, '')
			elif p and _is_image_path(p):
				_push(out, p, '')

	seen = set(); uniq = []
	for d in out:
		p = d.get('path')
		if not p or p in seen:
			continue
		seen.add(p)
		uniq.append(d)
	return uniq


def pick_best(images: List[Dict[str, str]], *, max_images: int = 3) -> List[Dict[str, str]]:
	if not images:
		return []
	imgs = sorted(images, key=lambda d: (0 if (d.get('caption') or '').strip() else 1, len(d.get('path') or '')))
	return imgs[:max_images]
