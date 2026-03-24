#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import List

DEFAULT_SECTION_NAME = 'Oseraie - OSNY'

KNOWN_UPPER = {'osny': 'OSNY'}
_DASH_RE = re.compile(r"\s*[‒–—−\-]+\s*")
_WS_RE = re.compile(r"\s+")


def normalize_section_name(name: str) -> str:
	name = (name or '').strip()
	if not name:
		return ''
	name = _DASH_RE.sub(' - ', name)
	name = _WS_RE.sub(' ', name).strip()
	parts = [p.strip() for p in name.split(' - ') if p.strip()]
	if len(parts) >= 2:
		left = parts[0]
		right = parts[1]
		mapped = KNOWN_UPPER.get(right.lower())
		if mapped:
			right = mapped
		return left + ' - ' + right
	return name


def section_name_variants(name: str) -> List[str]:
	name = (name or '').strip()
	if not name:
		return []
	norm = normalize_section_name(name)
	out: List[str] = []
	for cand in (name, norm):
		c = (cand or '').strip()
		if c and c not in out:
			out.append(c)
	parts = [p.strip() for p in norm.split(' - ') if p.strip()]
	if len(parts) >= 2:
		left, right = parts[0], parts[1]
		for cand in (left + ' - ' + right.upper(), left + ' - ' + right.title()):
			if cand not in out:
				out.append(cand)
	return out
