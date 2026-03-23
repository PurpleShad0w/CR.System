#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff'}


def _as_image_dict(it: Any) -> Dict[str, str]:
	if isinstance(it, str):
		return {'path': it, 'caption': ''}
	if isinstance(it, dict):
		p = it.get('path') or it.get('file') or it.get('relpath') or it.get('name') or ''
		c = it.get('caption') or it.get('legend') or it.get('title') or ''
		return {'path': p, 'caption': c}
	return {'path': '', 'caption': ''}


def collect_images(page: Dict[str, Any]) -> List[Dict[str, str]]:
	"""Schema-flexible image collection from a page dict."""
	out: List[Dict[str, str]] = []
	for key in ('images', 'pics', 'pictures', 'assets', 'media'):
		raw = page.get(key)
		if isinstance(raw, list):
			for it in raw:
				d = _as_image_dict(it)
				if d.get('path'):
					out.append(d)
	att = page.get('attachments')
	if isinstance(att, list):
		for it in att:
			d = _as_image_dict(it)
			if d.get('path'):
				out.append(d)
	filtered: List[Dict[str, str]] = []
	for d in out:
		p = d.get('path') or ''
		suf = Path(p).suffix.lower()
		if suf and suf not in IMG_EXTS:
			continue
		filtered.append(d)
	return filtered


def pick_best(images: List[Dict[str, str]], *, max_images: int = 3) -> List[Dict[str, str]]:
	"""Deterministic policy (no ML): prefer items with captions."""
	if not images:
		return []
	imgs = sorted(images, key=lambda d: (0 if (d.get('caption') or '').strip() else 1, len(d.get('path') or '')))
	return imgs[:max_images]
