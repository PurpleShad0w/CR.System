#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re


def normalize_whitespace(text: str) -> str:
	t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
	t = re.sub(r"\n{3,}", "\n\n", t)
	return t.strip()


def strip_markdown(text: str) -> str:
	if not text:
		return ''
	t = text
	t = re.sub(r"\*\*(.*?)\*\*", r"\\1", t)
	t = re.sub(r"\*(.*?)\*", r"\\1", t)
	t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
	t = t.replace('•', '-')
	return t.strip()


def sanitize_client_body(text: str) -> str:
	if not text:
		return ''
	t = normalize_whitespace(strip_markdown(text))
	out = []
	for ln in t.splitlines():
		s = ln.strip()
		if not s:
			out.append('')
			continue
		low = s.lower()
		if 'page_id=' in low:
			continue
		if low.startswith('preuve:') or 'preuve:' in low:
			continue
		out.append(ln)
	return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def collect_text(page: dict) -> str:
	"""Schema-flexible text extraction from a OneNote-processed page dict."""
	parts = []
	for k in ('text', 'body', 'content', 'markdown', 'md', 'ocr', 'transcript', 'dictation', 'audio_transcript'):
		v = page.get(k)
		if isinstance(v, str) and v.strip():
			parts.append(v)
	for k in ('paragraphs', 'blocks', 'elements'):
		v = page.get(k)
		if isinstance(v, list):
			for it in v:
				if isinstance(it, str) and it.strip():
					parts.append(it)
				elif isinstance(it, dict):
					txt = it.get('text') or it.get('content') or it.get('markdown') or ''
					if isinstance(txt, str) and txt.strip():
						parts.append(txt)
	return normalize_whitespace("\n\n".join(parts))


def to_bullets(text: str, *, max_lines: int = 10) -> str:
	body = sanitize_client_body(text)
	lines = [ln.strip() for ln in normalize_whitespace(strip_markdown(body)).split('\n') if ln.strip()]
	out = []
	for ln in lines:
		s = ln
		while s.startswith('- '):
			s = s[2:].strip()
		if len(s) > 190:
			s = s[:187].rstrip() + '…'
		out.append(s)
	if len(out) > max_lines:
		out = out[:max_lines-1] + ['…']
	if not out:
		return ''
	return "\n".join([f"- {x}" for x in out])
