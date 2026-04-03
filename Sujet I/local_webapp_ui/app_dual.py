#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
# Ensure repo + src are importable
for p in (REPO_ROOT, SRC):
	sp = str(p)
	if sp not in sys.path:
		sys.path.insert(0, sp)

from legacy.legacy_runner import run_legacy, run_page_cards
from legacy.section_names import DEFAULT_SECTION_NAME, normalize_section_name
from ui.onenote_cloud import onenote_cloud_ui


def load_presets() -> list[dict]:
	for p in (REPO_ROOT / 'input' / 'config' / 'presets.json', REPO_ROOT / 'presets.json'):
		if p.exists():
			try:
				obj = json.loads(p.read_text(encoding='utf-8'))
			except Exception:
				continue
			if isinstance(obj, dict) and isinstance(obj.get('presets'), list):
				return [x for x in obj['presets'] if isinstance(x, dict)]
			if isinstance(obj, list):
				return [x for x in obj if isinstance(x, dict)]
	return []


def main():
	st.set_page_config(page_title='AMO Reports', layout='wide')
	st.title('AMO Reports Generator')
	st.caption('Dual app: Main (Page Cards), OneNote (Microsoft Graph download), Legacy pipeline.')

	# Tabs order requested: Main | OneNote | Legacy
	tab_main, tab_onenote, tab_legacy = st.tabs(['Main', 'OneNote', 'Legacy'])

	with st.sidebar:
		st.header('Common')
		case_id = st.text_input('case_id', value='')
		section_name = st.text_input('OneNote section name', value=DEFAULT_SECTION_NAME)
		section_name = normalize_section_name(section_name)
		st.markdown('---')
		st.write('Outputs go to output/reports/<case_id>/')

	log_area = st.empty()

	# -------------------------
	# MAIN TAB (Page Cards)
	# -------------------------
	with tab_main:
		st.subheader('Page Cards (Part 1 only)')
		col1, col2 = st.columns(2)
		with col1:
			pages_index = st.text_input('pages-index JSON path', value='process/onenote/pages_index.json')
			max_images = st.number_input('max images per page', min_value=0, max_value=6, value=3, step=1)
		with col2:
			max_bullets = st.number_input('max bullets per page', min_value=1, max_value=25, value=10, step=1)
			out_path = st.text_input('output pptx (optional)', value='')

		if st.button('Run Page Cards', type='primary'):
			argv = ['--pages-index', pages_index]
			if not case_id:
				st.error('case_id is required')
				st.stop()
			argv += ['--case-id', case_id]
			if section_name:
				argv += ['--section-name', section_name]
			argv += ['--max-images', str(int(max_images))]
			argv += ['--max-bullets', str(int(max_bullets))]
			if out_path.strip():
				argv += ['--out', out_path.strip()]
			code, out = run_page_cards(argv)
			log_area.code(out or '(no output)')
			st.success('Page Cards completed') if code == 0 else st.error(f'Page Cards failed (exit={code})')

	# -------------------------
	# ONENOTE TAB (Graph list + download)
	# -------------------------
	with tab_onenote:
		onenote_cloud_ui(REPO_ROOT)

	# -------------------------
	# LEGACY TAB
	# -------------------------
	with tab_legacy:
		st.subheader('Legacy pipeline (full)')
		st.caption('Runs run_pipeline.py. Auto-tries section name variants to avoid casing/dash mismatches.')

		presets = load_presets()
		selected = None
		if presets:
			name = st.selectbox('Preset', [x.get('name','(unnamed)') for x in presets])
			selected = next((x for x in presets if x.get('name') == name), None)
		else:
			st.info('No presets.json found/readable. Fill fields manually.')

		col1, col2, col3 = st.columns(3)
		with col1:
			notebook = st.text_input('notebook', value=(selected.get('notebook') if selected else ''))
		with col2:
			onenote_section = st.text_input('onenote_section', value=(selected.get('onenote_section') if selected else section_name))
		with col3:
			mode_ = st.text_input('mode', value=(selected.get('mode') if selected else ''))

		min_quality = st.text_input('min_quality', value=(str(selected.get('min_quality')) if selected and selected.get('min_quality') is not None else ''))
		bacs_building_scope = st.text_input('bacs_building_scope', value=(selected.get('bacs_building_scope') if selected else ''))
		bacs_part2_slides = st.checkbox('bacs_part2_slides', value=bool(selected.get('bacs_part2_slides')) if selected else False)

		if st.button('Run Legacy', type='primary'):
			argv = []
			if notebook:
				argv += ['--notebook', notebook]
			argv += ['--onenote-section', normalize_section_name(onenote_section or section_name)]
			if case_id:
				argv += ['--case-id', case_id]
			if mode_:
				argv += ['--mode', mode_]
			if min_quality.strip():
				argv += ['--min-quality', min_quality.strip()]
			if bacs_building_scope.strip():
				argv += ['--bacs-building-scope', bacs_building_scope.strip()]
			if bacs_part2_slides:
				argv += ['--bacs-part2-slides']

			code, out = run_legacy(argv)
			log_area.code(out or '(no output)')
			st.success('Legacy pipeline completed') if code == 0 else st.error(f'Legacy pipeline failed (exit={code})')


if __name__ == '__main__':
	main()
