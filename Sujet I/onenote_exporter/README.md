# onenote-exporter-native (internal)

This is a **no-Docker**, internally-owned OneNote exporter designed to produce outputs compatible with the existing pipeline step `process_onenote.py`.

## Why
- Keep full ownership (no external GitHub dependency)
- Keep outputs stable so the GTB/BACS pipeline is not disturbed

## Outputs (compatible)
Default output root: `input/onenote-exporter/output/<notebook>/`

- `pages/*.md` : one markdown per OneNote page
- `<page_id>/res-<md5>.<ext>` : downloaded assets (images/audio)
- `manifest.json`
- optionally `merged.md` (`--merge`)
- optionally `merged.jsonl` (include `jsonl` in `--formats`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit CLIENT_ID / TENANT_ID
```

## Usage
List notebooks:
```bash
python -m onenote_exporter --config .env --list
```

Export a notebook (partial name match):
```bash
python -m onenote_exporter --config .env --notebook "test"
```

Export and produce merged.md + merged.jsonl:
```bash
python -m onenote_exporter --config .env --notebook "test" --merge --formats md,jsonl
```

## Notes
- Uses Microsoft Graph OneNote API.
- Device-code login caches tokens to `input/onenote-exporter/cache/token_cache.json` by default.
