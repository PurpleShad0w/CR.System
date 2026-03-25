#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _norm(s: str) -> str:
	return re.sub(r"\s+", " ", (s or '').strip()).lower()


def extract_keywords(title: str, bullets: str) -> List[str]:
	"""Lightweight keyword extraction (no dependencies).

	- Keeps alphanum tokens length>=4
	- Adds a few domain anchors
	"""
	base = _norm(title) + ' ' + _norm(bullets)
	tokens = re.findall(r"[a-zàâçéèêëîïôùûüÿœæ0-9]{4,}", base)
	anchors = [
		"chaufferie", "tgbt", "cta", "groupe froid", "groupe", "local", "autocom",
		"tableau", "armoire", "ventilo", "ventilo-convecteur", "ecs", "cvc",
		"compteur", "supervision", "gtb", "bacs",
	]
	out: List[str] = []
	seen = set()
	for t in tokens + anchors:
		t = _norm(t)
		if not t or t in seen:
			continue
		seen.add(t)
		out.append(t)
	return out


def _blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
	b = page.get('blocks')
	return b if isinstance(b, list) else []


def _find_image_block_id(blocks: List[Dict[str, Any]], img_path: str) -> str:
	img_path = (img_path or '').strip()
	if not img_path:
		return ''
	for b in blocks:
		if not isinstance(b, dict):
			continue
		if (b.get('type') or '').lower() != 'image':
			continue
		if (b.get('path') or '').strip() == img_path:
			return (b.get('block_id') or '').strip()
	return ''


def _image_ocr_text(blocks: List[Dict[str, Any]], image_block_id: str) -> str:
	if not image_block_id:
		return ''
	texts = []
	for b in blocks:
		if not isinstance(b, dict):
			continue
		if (b.get('type') or '').lower() != 'image_ocr':
			continue
		if (b.get('image_block_id') or '').strip() != image_block_id:
			continue
		t = b.get('text')
		if isinstance(t, str) and t.strip():
			texts.append(t.strip())
	return "\n".join(texts).strip()


def caption_from_blocks(page: Dict[str, Any], img_path: str) -> str:
	"""Infer caption for an image using process_onenote blocks.

	Priority:
	1) image_ocr blocks referencing the image block
	2) nearest paragraph/text block AFTER the image (window=3)
	3) nearest paragraph/text block BEFORE the image (window=3)
	"""
	blocks = _blocks(page)
	img_path = (img_path or '').strip()
	if not img_path:
		return ''

	# 1) OCR block(s)
	img_id = _find_image_block_id(blocks, img_path)
	ocr = _image_ocr_text(blocks, img_id)
	if ocr:
		return ocr

	# locate image index
	idx = None
	for i, b in enumerate(blocks):
		if not isinstance(b, dict):
			continue
		if (b.get('type') or '').lower() == 'image' and (b.get('path') or '').strip() == img_path:
			idx = i
			break
	if idx is None:
		return ''

	# 2) forward paragraphs
	for j in range(idx + 1, min(len(blocks), idx + 4)):
		b = blocks[j]
		if not isinstance(b, dict):
			continue
		btype = (b.get('type') or '').lower()
		if btype in ('paragraph', 'text', 'heading', 'list', 'bullet'):
			t = b.get('text')
			if isinstance(t, str) and t.strip():
				return t.strip()

	# 3) backward paragraphs
	for j in range(idx - 1, max(-1, idx - 4), -1):
		b = blocks[j]
		if not isinstance(b, dict):
			continue
		btype = (b.get('type') or '').lower()
		if btype in ('paragraph', 'text'):
			t = b.get('text')
			if isinstance(t, str) and t.strip():
				return t.strip()

	return ''


def score_images(images: List[Dict[str, str]], *, keywords: List[str]) -> List[Tuple[int, Dict[str, str]]]:
	"""Deterministic scoring by keyword overlap with caption/path."""
	out: List[Tuple[int, Dict[str, str]]] = []
	for im in images:
		cap = _norm(im.get('caption') or '')
		path = _norm(im.get('path') or '')
		score = 0
		for kw in keywords:
			if not kw:
				continue
			if kw in cap:
				score += 3
			elif kw in path:
				score += 1
		out.append((score, im))
	out.sort(key=lambda t: (-t[0], len(t[1].get('path') or '')))
	return out


def select_best_images(page: Dict[str, Any], images: List[Dict[str, str]], *, title: str, bullets: str, max_images: int) -> List[Dict[str, str]]:
	# Fill missing captions from OCR / neighboring text
	for im in images:
		if not (im.get('caption') or '').strip():
			cap = caption_from_blocks(page, im.get('path') or '')
			if cap:
				im['caption'] = cap

	kws = extract_keywords(title, bullets)
	scored = score_images(images, keywords=kws)
	picked: List[Dict[str, str]] = []
	seen = set()
	for score, im in scored:
		p = (im.get('path') or '').strip()
		if not p or p in seen:
			continue
		seen.add(p)
		picked.append(im)
		if len(picked) >= max_images:
			break
	return picked
