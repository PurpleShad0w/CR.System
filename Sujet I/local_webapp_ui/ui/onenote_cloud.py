#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Streamlit: OneNote (Microsoft Graph) import

This tab lists OneNote notebooks available in the current user's Microsoft account
(via Microsoft Graph) using the existing onenote_exporter package, and downloads
(export) the selected notebook to local disk.

It relies on onenote_exporter scripts (device-code auth + export pipeline).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Use onenote_exporter package from repo root
from onenote_exporter.auth import build_cache, persist_cache, load_env_scopes, _authority  # type: ignore
from onenote_exporter.graph import GraphClient, list_notebooks  # type: ignore
from onenote_exporter.exporter import ExportConfig, export_notebook  # type: ignore

import msal
from dotenv import load_dotenv


def _default_env_path(repo_root: Path) -> Path:
	# Prefer a dedicated exporter env if present, else repo .env
	cand = repo_root / 'onenote_exporter' / '.env'
	if cand.exists():
		return cand
	cand2 = repo_root / '.env'
	if cand2.exists():
		return cand2
	return repo_root / 'onenote_exporter' / 'README.md'  # dummy, load_dotenv will ignore


def _token_cache_path(repo_root: Path) -> Path:
	# match onenote_exporter CLI default
	return repo_root / 'input' / 'onenote-exporter' / 'cache' / 'token_cache.json'


def _auth_get_token(*, client_id: str, tenant_id: str, scopes: List[str], cache_path: Path) -> str:
	"""Device-flow auth with UI prompt (no console printing)."""
	cache = build_cache(cache_path)
	app = msal.PublicClientApplication(
		client_id=client_id,
		authority=_authority(tenant_id),
		token_cache=cache,
	)
	accounts = app.get_accounts()
	if accounts:
		res = app.acquire_token_silent(scopes=scopes, account=accounts[0])
		if res and 'access_token' in res:
			persist_cache(cache, cache_path)
			return res['access_token']

	flow = app.initiate_device_flow(scopes=scopes)
	if 'user_code' not in flow:
		raise RuntimeError(f'Failed to initiate device flow: {flow}')

	# Show message in Streamlit
	st.info(flow.get('message') or f"Open {flow.get('verification_uri')} and enter code {flow.get('user_code')}")
	with st.spinner('Waiting for Microsoft sign-in to complete...'):
		res = app.acquire_token_by_device_flow(flow)
	persist_cache(cache, cache_path)
	if 'access_token' not in res:
		raise RuntimeError(f'Auth failed: {res}')
	return res['access_token']


def onenote_cloud_ui(repo_root: Path) -> None:
	st.subheader('OneNote (Microsoft Graph)')
	st.caption('List notebooks from Microsoft Graph and download (export) one locally using onenote_exporter.')

	# Config
	default_env = _default_env_path(repo_root)
	env_path = st.text_input('Config .env path (CLIENT_ID / TENANT_ID)', value=str(default_env))
	cache_path = st.text_input('Token cache path', value=str(_token_cache_path(repo_root)))
	out_dir = st.text_input('Output directory', value=str(repo_root / 'input' / 'onenote-exporter' / 'output'))

	load_dotenv(env_path, override=False)
	tenant_id = os.environ.get('TENANT_ID', 'common')
	client_id = os.environ.get('CLIENT_ID')
	additional_scopes = os.environ.get('ADDITIONAL_SCOPES', '')

	if not client_id:
		st.error('CLIENT_ID is not set. Put it in the .env used by onenote_exporter (CLIENT_ID=...).')
		st.stop()

	scopes = load_env_scopes(additional_scopes)
	delegated = [f'https://graph.microsoft.com/{s}' for s in scopes]

	# Auth + list notebooks
	if 'graph_token' not in st.session_state:
		st.session_state['graph_token'] = None

	colA, colB = st.columns(2)
	with colA:
		if st.button('Authenticate / Refresh token'):
			try:
				tok = _auth_get_token(client_id=client_id, tenant_id=tenant_id, scopes=delegated, cache_path=Path(cache_path))
				st.session_state['graph_token'] = tok
				st.success('Authenticated.')
			except Exception as e:
				st.error(str(e))
	with colB:
		st.write('Scopes:')
		st.code(', '.join(scopes))

	token = st.session_state.get('graph_token')
	if not token:
		st.info('Authenticate to list notebooks.')
		return

	try:
		gc = GraphClient.create(token)
		nbs = list_notebooks(gc)
	except Exception as e:
		st.error(f'Failed to list notebooks: {e}')
		return

	if not nbs:
		st.info('No notebooks returned by Graph.')
		return

	# Build display list
	options = [(nb.get('displayName') or '(unnamed)', nb.get('id') or '') for nb in nbs]
	labels = [f"{name}  ({nid[:8]})" if nid else name for name, nid in options]

	idx = st.selectbox('Available notebooks', list(range(len(labels))), format_func=lambda i: labels[i])
	sel_name, sel_id = options[idx]

	st.markdown('---')
	st.write('Download / export selected notebook to local disk:')
	merge = st.checkbox('Produce merged.md', value=False)
	formats = st.text_input('Formats (comma-separated)', value='md')

	if st.button('Download selected notebook', type='primary'):
		try:
			cfg = ExportConfig(
				tenant_id=tenant_id,
				client_id=client_id,
				additional_scopes=additional_scopes,
				output_dir=Path(out_dir),
				token_cache=Path(cache_path),
				notebook_name=None,
				notebook_id=sel_id,
				merge=bool(merge),
				formats=formats,
			)
			with st.spinner('Exporting notebook via Microsoft Graph...'):
				out_root = export_notebook(cfg)
			st.success(f'Export completed: {out_root}')
			st.code(str(out_root))
		except Exception as e:
			st.error(str(e))
